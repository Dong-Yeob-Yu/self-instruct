"""서버 연결 없이 생성·중복 판정·결과 보존을 검증하는 회귀 테스트."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from asr_corpus.__main__ import main
from asr_corpus.examples import ExampleDeduplicator, normalize_text, parse_generated_examples
from asr_corpus.generation import CorpusGenerator, LocalLlmClient, LocalLlmConfig


class AsrCorpusRegressionTest(unittest.TestCase):
    def setUp(self):
        """모든 테스트의 실제 네트워크 접근을 차단한다."""
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("실제 네트워크 호출 금지")).start()
        self.addCleanup(patch.stopall)

    def make_client(self, backend="openai-compatible"):
        """실제 서버에 접속하지 않는 테스트용 클라이언트를 만든다."""
        return LocalLlmClient(LocalLlmConfig("http://localhost:12345/test", "test-model", backend=backend))

    def fake_http_response(self, payload):
        """HTTP 응답 객체를 메모리에서만 구성한다."""
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
        self.network.side_effect = None
        self.network.return_value = response

    def run_cli(self, arguments):
        """CLI 출력은 캡처하고 종료 코드만 반환한다."""
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(arguments)

    def read_jsonl(self, path):
        """검증할 JSONL 결과를 UTF-8로 읽는다."""
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_normalized_duplicate_keeps_first_original_text_and_id(self):
        """공백과 유니코드만 다른 중복은 제거하고 최초 원문과 ID를 보존한다."""
        deduplicator = ExampleDeduplicator()
        first = deduplicator.consider("  배송   언제 와요?  ")
        duplicate = deduplicator.consider("배송 언제 와요?")
        self.assertEqual(first.example.text, "  배송   언제 와요?  ")
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.matched_id, first.example.id)
        self.assertEqual(normalize_text("가"), normalize_text("가"))
        self.assertEqual(first.example.id, ExampleDeduplicator().consider(first.example.text).example.id)

    def test_near_duplicate_is_retained_for_review_by_default(self):
        """내용이 조금 다른 유사 발화는 기본적으로 채택하고 검토 근거를 남긴다."""
        deduplicator = ExampleDeduplicator()
        first = deduplicator.consider("오늘 주문한 상품의 배송이 언제 도착하는지 알려 주세요.")
        second = deduplicator.consider("오늘 주문한 상품의 배송이 언제 도착하는지 알려 주세요!")
        self.assertTrue(second.accepted)
        self.assertTrue(second.needs_review)
        self.assertEqual(second.matched_id, first.example.id)
        self.assertGreaterEqual(second.similarity, 0.95)

    def test_drop_policy_removes_near_duplicate(self):
        """유사 중복 제외 옵션을 선택하면 기준 이상의 후속 발화를 제외한다."""
        deduplicator = ExampleDeduplicator(near_duplicate_policy="drop")
        deduplicator.consider("오늘 주문한 상품의 배송이 언제 도착하는지 알려 주세요.")
        result = deduplicator.consider("오늘 주문한 상품의 배송이 언제 도착하는지 알려 주세요!")
        self.assertFalse(result.accepted)
        self.assertEqual(len(deduplicator.accepted_examples), 1)

    def test_different_numeric_and_korean_time_values_are_preserved(self):
        """유사 중복 제외 모드에서도 숫자와 한글 시간 값의 차이를 보존한다."""
        pairs = [
            ("오후 3시로 예약을 변경해 주세요", "오후 4시로 예약을 변경해 주세요"),
            ("오후 세 시로 예약을 변경해 주세요", "오후 네 시로 예약을 변경해 주세요"),
        ]
        for first, second in pairs:
            deduplicator = ExampleDeduplicator(0.8, "drop")
            deduplicator.consider(first)
            self.assertTrue(deduplicator.consider(second).accepted)

    def test_declared_place_names_are_preserved(self):
        """후보 메타데이터나 실행 옵션에 등록된 고유명사 차이를 보존한다."""
        deduplicator = ExampleDeduplicator(0.7, "drop")
        deduplicator.consider({"text": "서울역으로 가는 길 알려 주세요", "protected_terms": ["서울역"]})
        result = deduplicator.consider({"text": "수원역으로 가는 길 알려 주세요", "protected_terms": ["수원역"]})
        self.assertTrue(result.accepted)
        configured = ExampleDeduplicator(0.7, "drop", ["서울역", "수원역"])
        configured.consider("서울역으로 가는 길 알려 주세요")
        self.assertTrue(configured.consider("수원역으로 가는 길 알려 주세요").accepted)

    def test_rephrased_utterances_and_negation_are_preserved_in_review_mode(self):
        """같은 의도의 다른 표현과 부정 표현을 기본 모드에서 보존한다."""
        deduplicator = ExampleDeduplicator(0.5)
        for text in ["배송 언제 와요?", "택배 언제 도착하나요?", "예약을 취소해 주세요", "예약을 취소하지 말아 주세요"]:
            self.assertTrue(deduplicator.consider(text).accepted)
        self.assertEqual(len(deduplicator.accepted_examples), 4)

    def test_swapped_numeric_values_and_protected_entities_are_preserved(self):
        """같은 값이 등장해도 숫자나 보호어의 순서가 바뀌면 별도 발화로 보존한다."""
        numeric = ExampleDeduplicator(0.7, "drop")
        numeric.consider("1번에서 2번으로 예약 순서를 변경해 주세요")
        self.assertTrue(numeric.consider("2번에서 1번으로 예약 순서를 변경해 주세요").accepted)
        names = ExampleDeduplicator(0.7, "drop", ["서울역", "수원역"])
        names.consider("서울역에서 수원역까지 가는 경로를 알려 주세요")
        self.assertTrue(names.consider("수원역에서 서울역까지 가는 경로를 알려 주세요").accepted)

    def test_invalid_candidates_do_not_block_later_valid_candidates(self):
        """잘못된 후보를 제외해도 뒤의 유효한 후보는 처리한다."""
        deduplicator = ExampleDeduplicator(max_characters=20)
        invalid = [None, {}, "  ", "x" * 21, "첫 줄\n둘째 줄", {"text": "문장", "protected_terms": ["없는 단어"]}]
        for candidate in invalid:
            self.assertFalse(deduplicator.consider(candidate).accepted)
        self.assertTrue(deduplicator.consider("유효한 문장").accepted)

    def test_json_array_object_and_code_fence_are_supported(self):
        """지원하는 구조화 응답을 읽고 설명문이 섞인 응답은 거부한다."""
        self.assertEqual(parse_generated_examples('["발화"]'), ["발화"])
        self.assertEqual(parse_generated_examples('```json\n{"examples": ["발화"]}\n```'), ["발화"])
        with self.assertRaises(ValueError):
            parse_generated_examples('예제를 만들었습니다. {"examples": []}')

    def test_generation_filters_same_batch_duplicates_and_requests_remaining_count(self):
        """배치 내 중복을 즉시 제거하고 다음 요청은 부족한 개수만 생성한다."""
        client = Mock()
        client.generate.side_effect = ['{"examples": ["첫 발화", "첫 발화"]}', '{"examples": ["다른 발화", "초과 발화"]}']
        deduplicator = ExampleDeduplicator()
        batches = list(CorpusGenerator(client, deduplicator).generate("테스트용 요구사항", 2, 2, 5))
        self.assertEqual(len(batches), 2)
        self.assertEqual([item.text for item in deduplicator.accepted_examples], ["첫 발화", "다른 발화"])
        self.assertEqual(batches[0].decisions[1].reason, "exact_duplicate")
        self.assertEqual(batches[1].decisions[1].reason, "target_reached")
        context = json.loads(client.generate.call_args_list[1].args[0].splitlines()[-1])
        self.assertEqual(context["count"], 1)
        self.assertEqual(context["previous_examples"], ["첫 발화"])

    def test_generation_stops_at_request_limit_for_repeated_responses(self):
        """모델이 같은 예제만 내보내도 요청 상한에서 반드시 종료한다."""
        client = Mock()
        client.generate.return_value = '["같은 발화"]'
        deduplicator = ExampleDeduplicator()
        batches = list(CorpusGenerator(client, deduplicator).generate("테스트용 요구사항", 2, 1, 3))
        self.assertEqual(len(batches), 3)
        self.assertEqual(len(deduplicator.accepted_examples), 1)

    def test_invalid_response_is_archived_and_next_batch_can_succeed(self):
        """JSON 형식 오류를 배치 결과에 남기고 다음 응답을 계속 처리한다."""
        client = Mock()
        client.generate.side_effect = ["JSON이 아닙니다", '["유효한 발화"]']
        batches = list(CorpusGenerator(client, ExampleDeduplicator()).generate("테스트용 요구사항", 1, 1, 2))
        self.assertIsNotNone(batches[0].error)
        self.assertEqual(batches[0].raw_response, "JSON이 아닙니다")
        self.assertTrue(batches[1].decisions[0].accepted)

    def test_blank_requirement_fails_before_model_call(self):
        """요구사항이 없으면 모델 호출 없이 실제 생성을 거부한다."""
        client = Mock()
        with self.assertRaises(ValueError):
            list(CorpusGenerator(client, ExampleDeduplicator()).generate(" ", 1, 1, 1))
        client.generate.assert_not_called()

    def test_openai_compatible_request_and_response_without_sdk(self):
        """OpenAI 호환 로컬 API의 요청 구조와 응답 파싱을 모의 HTTP로 검증한다."""
        self.fake_http_response({"choices": [{"message": {"content": '["발화"]'}, "finish_reason": "stop"}]})
        self.assertEqual(self.make_client().generate("프롬프트"), '["발화"]')
        request = self.network.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "프롬프트"}])
        self.assertFalse(payload["stream"])
        self.assertNotIn("Authorization", request.headers)

    def test_native_ollama_request_and_response(self):
        """Ollama 비스트리밍 JSON 생성 규격을 모의 HTTP로 검증한다."""
        self.fake_http_response({"response": '{"examples": []}', "done_reason": "stop"})
        self.assertEqual(self.make_client("ollama").generate("프롬프트"), '{"examples": []}')
        payload = json.loads(self.network.call_args.args[0].data)
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["options"]["num_predict"], 2048)
        self.assertFalse(payload["stream"])

    def test_truncated_or_malformed_http_responses_are_rejected(self):
        """잘린 응답과 잘못된 API 응답을 정상 예제로 처리하지 않는다."""
        client = self.make_client()
        responses = [
            {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
            {"choices": []},
            {"choices": [{"message": {"content": None}}]},
        ]
        for response in responses:
            self.fake_http_response(response)
            with self.assertRaises(ValueError):
                client.generate("프롬프트")

    def test_client_initialization_and_prompt_command_do_not_connect(self):
        """설정 생성과 프롬프트 확인은 로컬 서버에 연결하지 않는다."""
        self.make_client()
        self.assertEqual(self.run_cli(["prompt", "--requirement", "테스트용 요구사항"]), 0)
        self.network.assert_not_called()

    def test_endpoint_rejects_inline_credentials(self):
        """설정과 실행 기록에 인증 정보가 섞이지 않도록 주소 내 자격 증명을 거부한다."""
        with self.assertRaises(ValueError):
            LocalLlmConfig("http://user:secret@localhost:12345/test", "test-model")

    def test_deduplication_cli_writes_fixed_expected_counts_without_network(self):
        """오프라인 필터가 채택·제외·검토 개수를 정확히 기록하고 원본을 보존한다."""
        with tempfile.TemporaryDirectory(prefix="asr-corpus-test-") as directory:
            self._verify_deduplication_files(Path(directory))
        self.network.assert_not_called()

    def _verify_deduplication_files(self, directory):
        """실제 임시 파일에서 필터 결과와 기존 출력 덮어쓰기 방지를 검증한다."""
        source = directory / "input.jsonl"
        original = '\n'.join([json.dumps(text, ensure_ascii=False) for text in [
            "오늘 주문한 상품의 배송이 언제 도착하는지 알려 주세요.",
            "오늘 주문한 상품의 배송이 언제 도착하는지 알려 주세요.",
            "오늘 주문한 상품의 배송이 언제 도착하는지 알려 주세요!",
        ]]) + "\nnot-json\n"
        source.write_text(original, encoding="utf-8")
        output = directory / "output"
        arguments = ["deduplicate", "--input-file", str(source), "--output-dir", str(output)]
        self.assertEqual(self.run_cli(arguments), 0)
        self.assertEqual(source.read_text(encoding="utf-8"), original)
        summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["counts"], {"accepted": 2, "rejected": 2, "review": 1, "requests": 0})
        self.assertEqual([row["needs_review"] for row in self.read_jsonl(output / "accepted.jsonl")], [False, True])
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        self.assertEqual(self.run_cli(arguments), 1)
        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)

    def generation_arguments(self, output, count="2", max_requests="2"):
        """실제 모델 대신 모의 응답을 사용할 생성 명령 인자를 구성한다."""
        return [
            "generate", "--requirement", "테스트용 요구사항", "--endpoint", "http://localhost:12345/test",
            "--model", "test-model", "--count", count, "--max-requests", max_requests, "--output-dir", str(output),
        ]

    def test_generation_cli_archives_responses_and_reports_incomplete_target(self):
        """목표 개수에 미달해도 원문과 부분 결과를 보존하고 종료 코드 2를 반환한다."""
        with tempfile.TemporaryDirectory(prefix="asr-corpus-test-") as directory:
            self._verify_incomplete_generation(Path(directory))

    def _verify_incomplete_generation(self, directory):
        """반복 응답만 있는 실행의 저장 결과와 미달성 상태를 검증한다."""
        output = directory / "output"
        with patch.object(LocalLlmClient, "generate", return_value='["반복 발화"]') as generate:
            self.assertEqual(self.run_cli(self.generation_arguments(output)), 2)
        summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "target_not_reached")
        self.assertEqual(summary["counts"], {"accepted": 1, "rejected": 1, "review": 0, "requests": 2})
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(self.read_jsonl(output / "requests.jsonl")), 2)

    def test_generation_cli_preserves_completed_batch_on_later_failure(self):
        """후속 모델 요청이 실패해도 앞서 채택한 예제를 보존하고 실패 상태를 기록한다."""
        with tempfile.TemporaryDirectory(prefix="asr-corpus-test-") as directory:
            self._verify_partial_failure(Path(directory))

    def _verify_partial_failure(self, directory):
        """첫 배치 성공 후 연결 오류가 발생하는 실행을 검증한다."""
        output = directory / "output"
        with patch.object(LocalLlmClient, "generate", side_effect=['["보존할 발화"]', RuntimeError("모의 연결 오류")]):
            self.assertEqual(self.run_cli(self.generation_arguments(output)), 1)
        self.assertEqual(self.read_jsonl(output / "accepted.jsonl")[0]["text"], "보존할 발화")
        self.assertEqual(json.loads((output / "run.json").read_text(encoding="utf-8"))["status"], "failed")

    def test_generation_cli_completes_with_filtered_examples(self):
        """생성부터 유효성 검사와 저장까지의 전체 경로가 모의 모델로 완료된다."""
        with tempfile.TemporaryDirectory(prefix="asr-corpus-test-") as directory:
            self._verify_successful_generation(Path(directory))

    def _verify_successful_generation(self, directory):
        """잘못된 후보가 섞인 성공 실행의 최종 결과를 검증한다."""
        output = directory / "output"
        with patch.object(LocalLlmClient, "generate", return_value='{"examples": [null, "첫 발화", "다른 발화"]}'):
            self.assertEqual(self.run_cli(self.generation_arguments(output)), 0)
        summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["counts"], {"accepted": 2, "rejected": 1, "review": 0, "requests": 1})


if __name__ == "__main__":
    unittest.main()
