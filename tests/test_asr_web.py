"""서버 바인딩 없이 웹 데이터 생성·검토·내보내기의 정합성을 검증한다."""

import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from asr_corpus.local import LlamaCppConfig
from asr_corpus.web import DatasetRequestHandler, DatasetWebApplication
from asr_corpus.web_dataset import DatasetAccumulator, DatasetJobService, GenerationCancelledError, PairedLlamaClient, validate_settings


def pair(spoken="사과 세 개 주세요.", normalized="사과 3개 주세요."):
    """숫자 정규화가 포함된 테스트용 발화·전사문 쌍을 만든다."""
    return {"spoken_text": spoken, "normalized_text": normalized, "protected_terms": []}


class AsrWebRegressionTest(unittest.TestCase):
    def setUp(self):
        """모델 실행과 네트워크를 차단하고 자동 정리할 작업 폴더를 준비한다."""
        self.directory = tempfile.TemporaryDirectory(prefix="asr-web-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "test.gguf"
        self.model.write_bytes(b"GGUFtest")
        self.config = LlamaCppConfig(self.model, executable=sys.executable)
        self.process = patch("subprocess.Popen", side_effect=AssertionError("실제 모델 실행 금지")).start()
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("네트워크 연결 금지")).start()
        self.addCleanup(patch.stopall)

    def settings(self, **options):
        """정상적인 기본 설정에 테스트 조건을 덮어쓴다."""
        return {"prompt": "한국어 숫자를 아라비아 숫자로 변환하세요.", "count": 2, "batch_size": 2,
                "similarity_threshold": 0.95, "near_duplicate_policy": "review", **options}

    def service(self, responses):
        """미리 정한 응답만 반환하는 독립 작업 서비스를 만든다."""
        client = Mock()
        client.generate.side_effect = responses
        service = DatasetJobService(self.config, self.root / "jobs", lambda config, cancellation: client)
        self.addCleanup(service.close)
        return service

    def finish_job(self, service, settings=None):
        """짧은 모의 작업이 끝날 때까지 기다리고 최종 상태를 반환한다."""
        job = service.start_job(settings or self.settings())
        thread = service._threads.get(job["id"])
        if thread:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        return service.get_job(job["id"])

    def test_settings_reject_blank_prompt_boolean_count_and_nan_threshold(self):
        """프롬프트와 생성 범위가 잘못되면 작업을 시작하지 않는다."""
        invalid = [self.settings(prompt=" "), self.settings(count=True), self.settings(count=1001),
                   self.settings(batch_size=0), self.settings(similarity_threshold=float("nan"))]
        for settings in invalid:
            with self.assertRaises(ValueError):
                validate_settings(settings)

    def test_default_batch_size_is_one_and_explicit_choices_are_preserved(self):
        """배치 설정을 생략하면 1개를 사용하고 직접 지정한 크기는 유지한다."""
        self.assertEqual(validate_settings({"prompt": "발화를 생성하세요."})["batch_size"], 1)
        self.assertEqual(validate_settings({"prompt": "발화를 생성하세요.", "batch_size": 3})["batch_size"], 3)

    def test_default_generation_requests_one_example_per_batch(self):
        """기본 설정으로 목표 두 개를 생성하면 한 개씩 두 번 요청한다."""
        service = self.service([json.dumps({"examples": [pair()]}),
                                json.dumps({"examples": [pair("우유 두 개 주세요.", "우유 2개 주세요.")]})])
        job = self.finish_job(service, {"prompt": "발화를 생성하세요.", "count": 2})
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["settings"]["batch_size"], 1)
        self.assertEqual(job["request_count"], 2)
        requests = [json.loads(line) for line in (service.data_dir / job["id"] / "requests.jsonl").read_text().splitlines()]
        self.assertEqual([json.loads(record["prompt"].splitlines()[-1])["count"] for record in requests], [1, 1])

    def test_model_request_explains_date_units_and_preserves_date_particles(self):
        """실제 생성 요청에 날짜 단위 중복 방지와 조사 보존 규칙을 전달한다."""
        service = self.service([json.dumps({"examples": [pair()]})])
        job = self.finish_job(service, self.settings(count=1))
        self.assertEqual(job["status"], "completed")
        request = json.loads((self.root / "jobs" / job["id"] / "requests.jsonl").read_text())
        prompt = request["prompt"]
        for example in ("3일→삼일", "10일→십일", "11일→십일일", "15일→십오일",
                        "21일→이십일일", "25일→이십오일", "31일→삼십일일"):
            self.assertIn(example, prompt)
        self.assertIn("날짜 단위 '일'을 한 번만", prompt)
        self.assertIn("조사 '에'는 임의로 추가하거나 삭제하지 마세요", prompt)
        self.assertIn("정답이 '3일에'이면 발화는 '삼일에'", prompt)
        self.assertIn("정답이 '3일'이면 발화는 '삼일'", prompt)
        context = json.loads(prompt.splitlines()[-1])
        self.assertEqual(context["user_prompt"], self.settings()["prompt"])

    def test_pairs_preserve_original_spoken_and_normalized_fields(self):
        """TTS 발화와 ASR 정답을 분리해서 원문 그대로 보존한다."""
        accumulator = DatasetAccumulator(self.settings())
        row = accumulator.add(pair("  사과 세 개 주세요.  ", "사과 3개 주세요."))
        self.assertTrue(row["accepted"])
        self.assertEqual(row["spoken_text"], "  사과 세 개 주세요.  ")
        self.assertEqual(row["normalized_text"], "사과 3개 주세요.")
        self.assertEqual(len(row["id"]), 32)

    def test_same_pair_is_rejected_but_rows_have_unique_review_ids(self):
        """동일 예제는 제외하되 검토 행의 ID는 서로 구분한다."""
        accumulator = DatasetAccumulator(self.settings())
        first = accumulator.add(pair())
        duplicate = accumulator.add(pair())
        self.assertFalse(duplicate["accepted"])
        self.assertEqual(duplicate["reason"], "exact_duplicate")
        self.assertEqual(duplicate["matched_id"], first["id"])
        self.assertNotEqual(first["id"], duplicate["id"])

    def test_repeated_alignment_failure_is_excluded_after_the_first_review_row(self):
        """숫자 대응이 틀린 같은 예제는 최초 한 행만 검토 대상으로 남긴다."""
        accumulator = DatasetAccumulator(self.settings())
        candidate = pair("사과 세 개 주세요.", "사과 4개 주세요.")
        first = accumulator.add(candidate)
        repeated = [accumulator.add(candidate) for _ in range(3)]
        self.assertEqual(first["reason"], "numeric_alignment")
        self.assertTrue(first["needs_review"])
        self.assertTrue(all(row["reason"] == "exact_duplicate" for row in repeated))
        self.assertTrue(all(not row["accepted"] and not row["needs_review"] for row in repeated))
        self.assertTrue(all(row["matched_id"] == first["id"] for row in repeated))

    def test_repeated_near_duplicate_rejection_is_not_reconsidered_as_new(self):
        """유사 중복으로 제외한 후보도 다시 나오면 정확한 중복으로 제외한다."""
        accumulator = DatasetAccumulator(self.settings(near_duplicate_policy="drop", similarity_threshold=0.8))
        accumulator.add(pair("오늘 주문 내역을 확인해 주세요.", "오늘 주문 내역을 확인해 주세요."))
        candidate = pair("오늘 주문 내역을 확인해 주세요!", "오늘 주문 내역을 확인해 주세요!")
        first = accumulator.add(candidate)
        repeated = accumulator.add(candidate)
        self.assertEqual(first["reason"], "near_duplicate")
        self.assertFalse(first["accepted"])
        self.assertEqual(repeated["reason"], "exact_duplicate")
        self.assertEqual(repeated["matched_id"], first["id"])

    def test_repeated_invalid_candidate_is_excluded_without_blocking_a_corrected_candidate(self):
        """형식 오류 반복을 제외하되 보호어 형식을 고친 같은 문장은 다시 검증한다."""
        accumulator = DatasetAccumulator(self.settings())
        invalid = {**pair(), "protected_terms": "잘못된 배열 형식"}
        first = accumulator.add(invalid)
        repeated = accumulator.add(invalid)
        corrected = accumulator.add(pair())
        self.assertFalse(first["accepted"])
        self.assertEqual(repeated["reason"], "exact_duplicate")
        self.assertEqual(repeated["candidate"], invalid)
        self.assertTrue(corrected["accepted"])

    def test_conflicting_transcripts_exclude_both_rows_for_review(self):
        """같은 발화에 서로 다른 정답이 있으면 양쪽 모두 검토 전 채택하지 않는다."""
        accumulator = DatasetAccumulator(self.settings())
        first = accumulator.add(pair())
        second = accumulator.add(pair(normalized="사과 4개 주세요."))
        self.assertEqual([row["accepted"] for row in [first, second]], [False, False])
        self.assertTrue(all(row["needs_review"] for row in [first, second]))
        self.assertEqual(second["reason"], "transcript_conflict")

    def test_distinct_spoken_numbers_survive_near_duplicate_drop(self):
        """전사문이 비슷해도 발화 숫자가 다르면 각각 보존한다."""
        accumulator = DatasetAccumulator(self.settings(near_duplicate_policy="drop", similarity_threshold=0.8))
        first = accumulator.add(pair("오후 세 시로 예약해 주세요.", "오후 3시로 예약해 주세요."))
        second = accumulator.add(pair("오후 네 시로 예약해 주세요.", "오후 4시로 예약해 주세요."))
        self.assertTrue(first["accepted"] and second["accepted"])

    def test_invalid_transcript_is_rejected_but_hallucinated_term_does_not_discard_valid_pair(self):
        """잘못된 전사문은 제외하고 정상 쌍의 잘못된 보호어는 발화에서 다시 추출한다."""
        accumulator = DatasetAccumulator(self.settings())
        self.assertFalse(accumulator.add(pair(normalized=""))["accepted"])
        row = accumulator.add({**pair(), "protected_terms": ["없는 단어"]})
        self.assertTrue(row["accepted"])
        self.assertEqual(row["protected_terms"], ["세 개"])

    def test_valid_long_utterance_is_preserved_within_web_limit(self):
        """웹에서 허용한 길이의 긴 발화는 중복 판정에서도 동일하게 허용한다."""
        accumulator = DatasetAccumulator(self.settings())
        text = "긴 발화도 원문을 보존합니다. " * 35
        self.assertTrue(500 < len(text) <= 1000)
        row = accumulator.add(pair(text, text))
        self.assertTrue(row["accepted"])
        self.assertEqual(row["spoken_text"], text)

    def test_generation_persists_pairs_and_export_keeps_stable_ids(self):
        """실행 결과를 복원하거나 내보내도 발화·정답과 ID가 유지된다."""
        responses = [json.dumps({"examples": [pair(), pair("우유 두 개 주세요.", "우유 2개 주세요.")]})]
        service = self.service(responses)
        job = self.finish_job(service)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["counts"], {"accepted": 2, "review": 0, "rejected": 0, "total": 2})
        exported = [json.loads(line) for line in service.export_jsonl(job["id"]).splitlines()]
        self.assertEqual([row["id"] for row in exported], [row["id"] for row in job["examples"]])
        restored = DatasetJobService(self.config, self.root / "jobs")
        self.addCleanup(restored.close)
        self.assertEqual(restored.get_job(job["id"])["examples"], job["examples"])

    def test_corrected_protected_terms_are_saved_and_exported(self):
        """생성부터 저장·복원·내보내기까지 발화 기준 보호어가 유지된다."""
        candidate = {**pair(), "protected_terms": ["3개"]}
        service = self.service([json.dumps({"examples": [candidate]})])
        job = self.finish_job(service, self.settings(count=1))
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["counts"]["accepted"], 1)
        self.assertEqual(json.loads(service.export_jsonl(job["id"]))["protected_terms"], ["세 개"])
        restored = DatasetJobService(self.config, self.root / "jobs")
        self.addCleanup(restored.close)
        self.assertEqual(restored.get_job(job["id"])["examples"][0]["protected_terms"], ["세 개"])
        request = json.loads((self.root / "jobs" / job["id"] / "requests.jsonl").read_text())
        self.assertEqual(json.loads(request["raw_response"])["examples"][0]["protected_terms"], ["3개"])

    def test_partial_results_survive_a_later_model_failure(self):
        """후속 추론이 실패해도 앞 배치의 발화·전사문을 보존한다."""
        service = self.service([json.dumps({"examples": [pair()]}), RuntimeError("모의 모델 오류")])
        job = self.finish_job(service)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["counts"]["accepted"], 1)
        self.assertIn("모의 모델 오류", job["message"])
        self.assertEqual(len(service.export_jsonl(job["id"]).splitlines()), 1)

    def test_generation_passes_former_request_limit_and_stops_at_thirty_accepted(self):
        """중간에 진전이 있으면 중복을 포함해 총 요청 상한을 넘어도 목표까지 생성한다."""
        duplicate = json.dumps({"examples": [pair()]})
        candidates = [pair()] + [pair(f"주문 {index}번을 확인해 주세요.", f"주문 {index}번을 확인해 주세요.") for index in range(1, 30)]
        responses = []
        for start in range(0, 30, 5):
            if start:
                responses.extend([duplicate] * 3)
            responses.append(json.dumps({"examples": candidates[start:start + 5]}))
        service = self.service(responses)
        job = self.finish_job(service, self.settings(count=30, batch_size=5, near_duplicate_policy="drop"))
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["request_count"], 21)
        self.assertEqual(job["counts"], {"accepted": 30, "review": 0, "rejected": 15, "total": 45})
        self.assertEqual(len(service.export_jsonl(job["id"]).splitlines()), 30)

    def test_five_stalled_batches_stop_with_partial_results_and_keep_exportable_rows(self):
        """채택 수가 다섯 배치 연속 늘지 않으면 부분 결과를 보존하고 정상적으로 중단한다."""
        response = json.dumps({"examples": [pair()]})
        service = self.service([response] * 6)
        job = self.finish_job(service)
        self.assertEqual(job["status"], "partial")
        self.assertEqual(job["request_count"], 6)
        self.assertEqual(job["counts"], {"accepted": 1, "review": 0, "rejected": 5, "total": 6})
        self.assertIn("5개 배치 연속", job["message"])
        self.assertEqual(len(service.export_jsonl(job["id"]).splitlines()), 1)

    def test_repeated_bad_pairs_do_not_multiply_review_rows_after_reload(self):
        """반복된 오류 예제는 복원 후에도 최초 한 행만 검토 대상으로 유지한다."""
        response = json.dumps({"examples": [pair(normalized="사과 4개 주세요.")]})
        service = self.service([response] * 5)
        job = self.finish_job(service)
        self.assertEqual(job["status"], "partial")
        self.assertEqual(job["request_count"], 5)
        expected_counts = {"accepted": 0, "review": 1, "rejected": 4, "total": 5}
        self.assertEqual(job["counts"], expected_counts)
        restored = DatasetJobService(self.config, self.root / "jobs")
        self.addCleanup(restored.close)
        self.assertEqual(restored.get_job(job["id"])["counts"], expected_counts)
        self.assertEqual(restored.get_job(job["id"])["examples"], job["examples"])

    def test_each_batch_changes_the_actual_cli_seed_and_records_it(self):
        """배치마다 실제 CLI seed를 바꾸고 원본 설정과 사용자 조건은 유지한다."""
        responses = iter([json.dumps({"examples": [pair()]}),
                          json.dumps({"examples": [pair("배 두 개 주세요.", "배 2개 주세요.")]})])
        client = PairedLlamaClient(self.config, threading.Event())
        seeds = []
        def generate(prompt):
            """실제 프로세스를 실행하지 않고 CLI에 전달할 seed를 수집한다."""
            command = client._build_command(prompt)
            seeds.append(int(command[command.index("--seed") + 1]))
            return next(responses)
        client.generate = generate
        service = DatasetJobService(self.config, self.root / "jobs", lambda config, cancellation: client)
        self.addCleanup(service.close)
        settings = self.settings(batch_size=1)
        job = self.finish_job(service, settings)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(seeds, [42, 43])
        self.assertEqual(self.config.seed, 42)
        requests = [json.loads(line) for line in (self.root / "jobs" / job["id"] / "requests.jsonl").read_text().splitlines()]
        self.assertEqual([record["seed"] for record in requests], seeds)
        contexts = [json.loads(record["prompt"].splitlines()[-1]) for record in requests]
        self.assertEqual([context["batch_index"] for context in contexts], [1, 2])
        self.assertNotEqual(contexts[0]["diversity_hint"], contexts[1]["diversity_hint"])
        self.assertTrue(all(context["user_prompt"] == settings["prompt"] for context in contexts))
        self.process.assert_not_called()

    def test_next_request_contains_the_rejected_pair_and_its_alignment_reason(self):
        """채택되지 않은 발화와 실패 이유도 다음 생성 요청에 전달한다."""
        bad_pair = pair(normalized="사과 4개 주세요.")
        service = self.service([json.dumps({"examples": [bad_pair]}),
                                json.dumps({"examples": [pair("배 두 개 주세요.", "배 2개 주세요.")]})])
        job = self.finish_job(service, self.settings(count=1))
        self.assertEqual(job["status"], "completed")
        requests = [json.loads(line) for line in (self.root / "jobs" / job["id"] / "requests.jsonl").read_text().splitlines()]
        context = json.loads(requests[1]["prompt"].splitlines()[-1])
        self.assertEqual(context["previous_examples"], [])
        self.assertEqual(context["failed_examples"][0]["spoken_text"], bad_pair["spoken_text"])
        self.assertEqual(context["failed_examples"][0]["reason"], "numeric_alignment")
        self.assertIn("대응하지 않습니다", context["failed_examples"][0]["detail"])

    def test_generation_feedback_limits_history_and_removes_repeated_pairs(self):
        """실패 이력은 중복 없이 네 개로 제한하고 긴 원문은 그대로 보관한다."""
        service = self.service([])
        accumulator = DatasetAccumulator(self.settings())
        spoken = "원문을 보관합니다. " * 20 + "사과 세 개 주세요."
        transcript = "원문을 보관합니다. " * 20 + "사과 4개 주세요."
        for index in range(6):
            accumulator.add(pair(f"요청 {index}번입니다. " + spoken, f"요청 {index}번입니다. " + transcript))
        accumulator.add(pair("요청 5번입니다. " + spoken, "요청 5번입니다. " + transcript))
        feedback = service._generation_feedback(accumulator)
        self.assertEqual(len(feedback), 4)
        self.assertEqual(len({item["spoken_text"] for item in feedback}), 4)
        self.assertTrue(all(len(item["spoken_text"]) <= 160 for item in feedback))
        self.assertEqual(accumulator.rows[-1]["spoken_text"], "요청 5번입니다. " + spoken)

    def test_rejected_candidates_are_replaced_and_only_remaining_count_is_requested(self):
        """중복·숫자 불일치·잘못된 후보를 목표에서 빼고 남은 채택 수만 요청한다."""
        responses = [json.dumps({"examples": [pair(), pair(), pair("배 두 개 주세요.", "배 3개 주세요."), pair(normalized="")]}),
                     json.dumps({"examples": [pair("우유 두 개 주세요.", "우유 2개 주세요."), pair("배 네 개 주세요.", "배 4개 주세요.")]})]
        service = self.service(responses)
        job = self.finish_job(service, self.settings(count=3, batch_size=3))
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["counts"], {"accepted": 3, "review": 1, "rejected": 2, "total": 6})
        requests = [json.loads(line) for line in (self.root / "jobs" / job["id"] / "requests.jsonl").read_text().splitlines()]
        self.assertEqual([json.loads(record["prompt"].splitlines()[-1])["count"] for record in requests], [3, 2])

    def test_later_transcript_conflict_is_replaced_before_completing(self):
        """후속 배치의 정답 충돌로 기존 채택이 취소되면 감소한 수까지 보충한다."""
        responses = [json.dumps({"examples": [pair()]}),
                     json.dumps({"examples": [pair(normalized="사과 4개 주세요.")]}),
                     json.dumps({"examples": [pair("우유 두 개 주세요.", "우유 2개 주세요."), pair("배 네 개 주세요.", "배 4개 주세요.")]})]
        service = self.service(responses)
        job = self.finish_job(service)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["request_count"], 3)
        self.assertEqual(job["counts"], {"accepted": 2, "review": 2, "rejected": 0, "total": 4})

    def test_empty_or_malformed_batches_do_not_finish_below_target(self):
        """비어 있거나 JSON 형식이 잘못된 배치 후에도 목표까지 추가 생성한다."""
        service = self.service(["잘못된 JSON", '{"examples":[]}'] * 2 + [json.dumps({"examples": [pair()]})])
        job = self.finish_job(service, self.settings(count=1))
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["request_count"], 5)
        self.assertEqual(job["counts"]["accepted"], 1)

    def test_persistent_malformed_or_empty_responses_stop_and_explain_the_next_retry(self):
        """형식 오류와 빈 응답도 재시도에 이유를 전달하며 다섯 배치 뒤 중단한다."""
        for response in ("잘못된 JSON", '{"examples":[]}'):
            service = self.service([response] * 5)
            job = self.finish_job(service, self.settings(count=1))
            self.assertEqual(job["status"], "partial")
            self.assertEqual(job["request_count"], 5)
            self.assertEqual(job["counts"]["accepted"], 0)
            requests = [json.loads(line) for line in (self.root / "jobs" / job["id"] / "requests.jsonl").read_text().splitlines()]
            context = json.loads(requests[1]["prompt"].splitlines()[-1])
            self.assertEqual(context["failed_examples"][0]["reason"], "invalid_response")

    def test_review_resolves_conflicting_transcripts_without_changing_id(self):
        """충돌 정답을 직접 검토하면 하나만 채택하고 원래 ID를 유지한다."""
        response = json.dumps({"examples": [pair(), pair(normalized="사과 4개 주세요.")]})
        service = self.service([response, GenerationCancelledError()])
        job = self.finish_job(service)
        selected = job["examples"][0]
        reviewed = service.update_example(job["id"], selected["id"], {"accepted": True, "normalized_text": "사과 3개 주세요."})
        self.assertEqual(reviewed["counts"]["accepted"], 1)
        self.assertEqual(reviewed["counts"]["review"], 0)
        self.assertEqual(json.loads(service.export_jsonl(job["id"]))["id"], selected["id"])

    def test_invalid_review_does_not_change_saved_transcript(self):
        """검토 입력이 잘못되면 이전 정답을 변경하지 않는다."""
        service = self.service([json.dumps({"examples": [pair()]})])
        job = self.finish_job(service, self.settings(count=1))
        row = job["examples"][0]
        with self.assertRaises(ValueError):
            service.update_example(job["id"], row["id"], {"accepted": True, "normalized_text": "잘못된\n전사문"})
        self.assertEqual(service.get_job(job["id"])["examples"][0]["normalized_text"], row["normalized_text"])

    def test_cancel_stops_active_job_and_prevents_parallel_generation(self):
        """한 작업만 허용하며 중지 요청은 해당 작업에만 전달한다."""
        started = threading.Event()
        factory = self.blocking_factory(started)
        service = DatasetJobService(self.config, self.root / "jobs", factory)
        self.addCleanup(service.close)
        job = service.start_job(self.settings())
        self.assertTrue(started.wait(timeout=3))
        with self.assertRaises(RuntimeError):
            service.start_job(self.settings())
        with self.assertRaises(RuntimeError):
            service.update_example(job["id"], "missing", {"accepted": True})
        thread = service._threads[job["id"]]
        service.cancel_job(job["id"])
        thread.join(timeout=3)
        self.assertEqual(service.get_job(job["id"])["status"], "cancelled")

    def blocking_factory(self, started):
        """중지 신호를 기다리는 모의 로컬 클라이언트 팩터리를 반환한다."""
        def factory(config, cancellation):
            """서비스의 중지 이벤트를 사용하는 모의 응답을 만든다."""
            def generate(prompt):
                """추론 시작을 알린 뒤 중지를 기다린다."""
                started.set()
                cancellation.wait(timeout=5)
                raise GenerationCancelledError()
            return SimpleNamespace(generate=generate)
        return factory

    def test_web_routes_serve_ui_and_exports_without_binding_server(self):
        """화면과 데이터 API를 소켓을 열지 않고 직접 검증한다."""
        service = self.service([json.dumps({"examples": [pair()]})])
        job = self.finish_job(service, self.settings(count=1))
        application = DatasetWebApplication(service)
        self.assertEqual(application.dispatch("GET", "/").status, 200)
        self.assertIn("생성 프롬프트", application.dispatch("GET", "/").body.decode())
        exported = application.dispatch("GET", f'/api/jobs/{job["id"]}/export')
        self.assertEqual(exported.status, 200)
        self.assertTrue(exported.filename.endswith(".jsonl"))
        self.assertEqual(json.loads(exported.body)["normalized_text"], "사과 3개 주세요.")
        self.assertEqual(application.dispatch("GET", "/../../etc/passwd").status, 404)
        self.assertEqual(application.dispatch("GET", "/api/jobs/" + "0" * 32).status, 404)
        self.assertEqual(application.dispatch("POST", "/api/jobs", {}).status, 400)
        self.network.assert_not_called()
        self.process.assert_not_called()

    def test_http_origin_and_request_size_are_validated_without_socket(self):
        """다른 웹사이트와 과도한 요청 본문을 실제 네트워크 없이 거부한다."""
        handler = DatasetRequestHandler.__new__(DatasetRequestHandler)
        handler.server = SimpleNamespace(server_address=("127.0.0.1", 8765))
        handler.headers = Message()
        handler.headers["Host"] = "127.0.0.1:8765"
        handler.headers["Origin"] = "https://another-site.example"
        with self.assertRaises(ValueError):
            handler._validate_origin()
        handler.headers.replace_header("Origin", "http://127.0.0.1:8765")
        handler._validate_origin()
        handler.headers["Content-Type"] = "application/json"
        handler.headers["Content-Length"] = "65537"
        handler.rfile = io.BytesIO(b"{}")
        with self.assertRaises(ValueError):
            handler._read_payload()

    def test_pair_client_uses_pair_schema_and_cancel_does_not_launch_model(self):
        """발화·전사문 스키마를 적용하며 이미 중지된 작업은 실행하지 않는다."""
        cancellation = threading.Event()
        client = PairedLlamaClient(self.config, cancellation)
        command = client._build_command("테스트")
        schema = json.loads(command[command.index("--json-schema") + 1])
        self.assertIn("spoken_text", schema["properties"]["examples"]["items"]["properties"])
        self.assertIn("--offline", command)
        cancellation.set()
        with self.assertRaises(GenerationCancelledError):
            client.generate("테스트")
        self.process.assert_not_called()

    def test_cancellation_terminates_only_the_owned_process(self):
        """출력을 기다리는 중 사용자 중지가 오면 소유한 추론 프로세스를 종료한다."""
        cancellation = Mock()
        cancellation.is_set.side_effect = [False, False, True]
        process = MagicMock()
        process.__enter__.return_value = process
        process.communicate.side_effect = [subprocess.TimeoutExpired("fake", 0.3), (b"", b"")]
        self.process.side_effect = None
        self.process.return_value = process
        with self.assertRaises(GenerationCancelledError):
            PairedLlamaClient(self.config, cancellation).generate("테스트")
        process.terminate.assert_called_once()
        process.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
