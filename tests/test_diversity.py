"""숫자만 바꾼 형식 반복과 상황별 생성 계획의 회귀 검증."""

import json
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from asr_corpus.diversity import PLAN_SCHEMA, StructureIndex, select_plan_slots, structure_pattern, validate_generation_plan
from asr_corpus.local import LlamaCppConfig
from asr_corpus.prompt_generation import EXPRESSION_SCHEMA, PROMPT_SCHEMA, render_prompt_candidate
from asr_corpus.web_dataset import DatasetAccumulator, DatasetJobService, GenerationCancelledError, PairedLlamaClient, build_pair_prompt


def source(template, value=None, unit="원", plan_id=""):
    """한 숫자 슬롯을 가진 모의 모델 출력을 만든다."""
    if value is None:
        value = str(55000 + (int(plan_id[1:]) * 1000 if re.fullmatch(r"p[0-9]+", plan_id) else 0))
    return {"template": template, "plan_id": plan_id, "expressions": [
        {"value": value, "reading": "sino", "unit": unit, "spoken_unit": unit, "negative": "", "spoken": "", "role": "unspecified"}
    ], "protected_terms": []}


def plans(count):
    """검증용으로 서로 다른 상황·목적의 계획을 만든다."""
    return {"plans": [{"situation": f"사건 {index}", "purpose": f"정보 {index}", "form": f"전개 {index}"} for index in range(count)]}


def settings(**options):
    """새 프롬프트 경로에서 형식 반복 차단을 켠다."""
    return {"prompt": "사용자 주제", "prompt_mode": True, "count": 3, "batch_size": 3,
            "similarity_threshold": 0.95, "near_duplicate_policy": "review", **options}


class StructuralDiversityRegressionTest(unittest.TestCase):
    def test_refund_amounts_and_durations_do_not_bypass_structure_filter(self):
        """제보된 환불 안내는 금액·날짜를 바꿔도 같은 형식으로 제외한다."""
        accumulator = DatasetAccumulator(settings())
        first = accumulator.add(render_prompt_candidate(source("환불 금액은 {e0}이며 결제 수단으로 돌려드립니다.", "150000")))
        repeated = accumulator.add(render_prompt_candidate(source("환불 금액은 {e0}이며 결제 수단으로 돌려드립니다.", "120000")))
        self.assertTrue(first["accepted"])
        self.assertFalse(repeated["accepted"])
        self.assertEqual(repeated["reason"], "structure_duplicate")
        self.assertEqual(repeated["matched_id"], first["id"])
        self.assertFalse(repeated["needs_review"])

    def test_phone_changes_and_added_greetings_do_not_count_as_new_situations(self):
        """전화번호와 고객님 인사만 바꾼 연락 안내를 차단한다."""
        accumulator = DatasetAccumulator(settings())
        def phone(template, value):
            """유효한 전화 읽기를 쓰는 고정 형식 후보를 만든다."""
            item = source(template, value, "")
            item["expressions"][0]["reading"] = "phone"
            return render_prompt_candidate(item)
        first = accumulator.add(phone("예약 변경을 원하시면 {e0}로 연락 주세요.", "02-1234-5678"))
        repeated = accumulator.add(phone("고객님, 예약 변경을 원하시면 {e0}로 연락 주세요.", "02-7777-8888"))
        self.assertTrue(first["accepted"])
        self.assertEqual(repeated["reason"], "structure_duplicate")
        self.assertFalse(repeated["accepted"])

    def test_name_changes_cannot_bypass_structure_filter(self):
        """보호 대상 고유명사만 바뀐 문장도 같은 구조로 비교한다."""
        accumulator = DatasetAccumulator(settings())
        for name in ("가람", "나래"):
            item = source(name + " 고객님의 환불 금액은 {e0}입니다.")
            item["protected_terms"] = [name]
            row = accumulator.add(render_prompt_candidate(item))
        self.assertEqual(row["reason"], "structure_duplicate")

    def test_small_wording_changes_in_a_long_template_are_rejected(self):
        """긴 안내에서 문장 끝 표현만 바꾼 형식 반복도 제외한다."""
        index = StructureIndex()
        first = structure_pattern("환불 절차를 안내해 드리겠습니다. 구매하신 상품 금액은 150,000원이며 원래 결제 수단으로 3일 뒤 환불됩니다.")
        index.add(first, "first")
        match, score = index.find(structure_pattern("환불 절차를 안내해 드리겠습니다. 구매하신 상품 금액은 75,000원이며 원래 결제 수단으로 5일 뒤 환불됩니다!"))
        self.assertEqual(match, "first")
        self.assertGreaterEqual(score, 0.9)

    def test_different_events_conditions_and_numeral_readings_remain_valid(self):
        """다른 사건과 이내·이후 및 부정 차이를 가진 문장은 유지한다."""
        accumulator = DatasetAccumulator(settings())
        for item in (source("환불 금액은 {e0}입니다."), source("참가자 {e0}의 좌석을 붙여 주세요.", "15", "명"),
                     source("시작 {e0} 전에 접속해도 될까요?", "15", "분")):
            self.assertTrue(accumulator.add(render_prompt_candidate(item))["accepted"])
        for original, changed in (("이내", "이후"), ("가능합니다", "가능하지 않습니다")):
            index = StructureIndex()
            prefix = "안내된 절차와 조건을 확인하신 후 접수하시면 처리는 정해진 3일 "
            index.add(structure_pattern(prefix + original + "."), "first")
            self.assertIsNone(index.find(structure_pattern(prefix + changed + "."))[0])

    def test_legacy_numeric_variants_keep_existing_value_protection(self):
        """이전 생성 경로의 숫자별 학습 예제 보존 동작은 변경하지 않는다."""
        accumulator = DatasetAccumulator(settings(prompt_mode=False))
        for value, spoken in (("3", "세"), ("4", "네")):
            self.assertTrue(accumulator.add({"spoken_text": spoken + " 개 주세요.", "normalized_text": value + "개 주세요.", "protected_terms": []})["accepted"])

    def test_rejected_templates_do_not_seed_the_accepted_structure_index(self):
        """잘못된 숫자 대응으로 거절한 문장은 정상 후보의 형식 비교 기준으로 등록하지 않는다."""
        accumulator = DatasetAccumulator(settings())
        pair = render_prompt_candidate(source("환불 금액은 {e0}입니다."))
        bad = {**pair, "spoken_text": "환불 금액은 오만 원입니다."}
        self.assertFalse(accumulator.add(bad)["accepted"])
        self.assertTrue(accumulator.add(pair)["accepted"])

    def test_a_demoted_conflicting_answer_no_longer_blocks_new_valid_structures(self):
        """채택 후 정답 충돌로 제외된 행을 형식 인덱스에서도 제거한다."""
        accumulator = DatasetAccumulator(settings())
        first = render_prompt_candidate(source("환불 금액은 {e0}입니다."))
        original = accumulator.add(first)
        accumulator.add({**first, "normalized_text": "환불 금액은 50000원입니다."})
        self.assertFalse(original["accepted"])
        self.assertTrue(accumulator.add(render_prompt_candidate(source("환불 금액은 {e0}입니다.", "120000")))["accepted"])

    def test_prompt_uses_exclusion_patterns_and_plans_instead_of_positive_examples(self):
        """이전 문장은 복제 예시가 아니라 값이 가려진 금지 형식으로 전달한다."""
        slots = validate_generation_plan(plans(1), 1)
        prompt = build_pair_prompt("온도에 관한 발화", 1, [{"normalized_text": "냉장고는 3℃입니다."}], prompt_mode=True, planned_slots=slots)
        context = json.loads(prompt.splitlines()[-1])
        self.assertEqual(context["previous_examples"], [])
        self.assertEqual(context["used_patterns"], ["냉장고는 〈값〉℃입니다."])
        self.assertEqual(context["planned_slots"], slots)
        self.assertIn("복제할 예시가 아닌 금지 형식", prompt)

    def test_numeric_schema_cannot_emit_compound_dates_or_empty_custom_pronunciations(self):
        """계획 도입 뒤 실제 모델이 내던 복합 날짜·단위 값과 빈 별도 발음을 생성 제약에서 막는다."""
        numeric, custom, digits, phone = EXPRESSION_SCHEMA["oneOf"]
        pattern = numeric["properties"]["value"]["pattern"]
        for value in ("2024년 10월 25일", "2024-10-25", "오전 2시", "3개월"):
            self.assertIsNone(re.fullmatch(pattern, value))
        for value in ("2024", "10", "25", "2", "3", "-3.5", "55,000"):
            self.assertIsNotNone(re.fullmatch(pattern, value))
        self.assertNotIn("custom", numeric["properties"]["reading"]["enum"])
        self.assertEqual(numeric["properties"]["spoken"]["enum"], [""])
        self.assertEqual(custom["properties"]["spoken"]["minLength"], 1)
        self.assertIsNotNone(re.fullmatch(phone["properties"]["value"]["pattern"], "010-1234-5678"))
        self.assertIsNone(re.fullmatch(digits["properties"]["value"]["pattern"], "2024-10-25"))
        self.assertIsNone(re.fullmatch(phone["properties"]["value"]["pattern"], "2024-10-25"))
        template_pattern = PROMPT_SCHEMA["properties"]["examples"]["items"]["properties"]["parts"]["items"]["oneOf"][0]["properties"]["text"]["pattern"]
        self.assertIsNone(re.fullmatch(template_pattern, "수령 후 7일 이내에 신청하세요."))
        self.assertIsNotNone(re.fullmatch(template_pattern, "수령 후 이내에 신청하세요."))
        self.assertIsNone(re.fullmatch(template_pattern, "수령 후 {e0} 이내에 신청하세요."))


class GenerationPlanRegressionTest(unittest.TestCase):
    def test_plan_quotas_cover_the_target_and_are_balanced_without_domain_lists(self):
        """대량 생성도 계획별 할당량 합이 목표와 같고 차이는 최대 하나다."""
        result = validate_generation_plan(plans(16), 1000)
        self.assertEqual(sum(item["quota"] for item in result), 1000)
        self.assertLessEqual(max(item["quota"] for item in result) - min(item["quota"] for item in result), 1)
        self.assertEqual(result[0]["id"], "p1")
        self.assertEqual(len(validate_generation_plan(plans(3), 3)), 3)

    def test_missing_duplicate_and_malformed_plans_are_rejected(self):
        """계획 개수 부족과 같은 상황·목적의 반복 및 잘못된 형식을 거부한다."""
        cases = [{}, plans(1), {"plans": [plans(1)["plans"][0]] * 2},
                 {"plans": [{"situation": "", "purpose": "확인", "form": "질문"}] * 2}]
        for response in cases:
            with self.assertRaises(ValueError):
                validate_generation_plan(response, 2)

    def test_untried_slots_take_priority_so_a_failed_slot_does_not_starve_others(self):
        """첫 계획이 실패해도 다음 배치에서 아직 시도하지 않은 상황을 먼저 선택한다."""
        result = validate_generation_plan(plans(3), 3)
        self.assertEqual(select_plan_slots(result, [], {}, 1)[0]["id"], "p1")
        self.assertEqual(select_plan_slots(result, [], {"p1": 1}, 1)[0]["id"], "p2")
        rows = [{"accepted": True, "plan_id": "p2"}]
        self.assertEqual(select_plan_slots(result, rows, {"p1": 1, "p2": 1}, 2)[0]["id"], "p3")


class PlannedGenerationServiceRegressionTest(unittest.TestCase):
    def setUp(self):
        """실제 서버·모델 없이 계획과 배치 생성의 저장 흐름을 준비한다."""
        temporary = tempfile.TemporaryDirectory(prefix="asr-diversity-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        model = self.root / "model.gguf"
        model.write_bytes(b"GGUFtest")
        self.config = LlamaCppConfig(model, executable=sys.executable)
        self.contexts = []

    def service(self, generate, plan_responses=None):
        """계획 기능이 있는 추론 클라이언트로 실제 작업 서비스를 구성한다."""
        responses = iter(plan_responses or [json.dumps(plans(3), ensure_ascii=False)])
        client = SimpleNamespace(supports_planning=True, generate=generate, generate_plan=lambda prompt: next(responses))
        service = DatasetJobService(self.config, self.root / "jobs", lambda config, cancel: client)
        self.addCleanup(service.close)
        return service

    def finish(self, service, **options):
        """모의 생성 종료를 기다리고 작업 상태를 반환한다."""
        job = service.start_job(settings(**options))
        thread = service._threads.get(job["id"])
        if thread:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        return service.get_job(job["id"])

    def test_planned_slots_are_covered_and_numeric_only_repeats_are_replaced(self):
        """상황별 계획을 모두 사용하며 반복 형식은 제외 후 새 사건으로 보충한다."""
        attempts = {}
        def generate(prompt):
            """세 번째 계획의 첫 응답만 반복시켜 보충 생성 경로를 재현한다."""
            context = json.loads(prompt.splitlines()[-1])
            self.contexts.append(context)
            examples = []
            for slot in context["planned_slots"]:
                identifier = slot["id"]
                attempts[identifier] = attempts.get(identifier, 0) + 1
                template = {"p1": "환불 금액은 {e0}입니다.", "p2": "배송비 {e0}을 결제하셨나요?", "p3": "추가 비용 {e0}은 왜 청구됐나요?"}[identifier]
                if identifier == "p3" and attempts[identifier] == 1:
                    template = "환불 금액은 {e0}입니다."
                examples.append(source(template, {"p1": "55000", "p2": "120000", "p3": "130000"}[identifier], plan_id=identifier))
            return json.dumps({"examples": examples})
        service = self.service(generate)
        job = self.finish(service)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["counts"]["accepted"], 3)
        self.assertEqual(job["counts"]["rejected"], 1)
        self.assertEqual({row["plan_id"] for row in job["examples"] if row["accepted"]}, {"p1", "p2", "p3"})
        self.assertEqual(job["request_count"], 2)
        self.assertEqual(len(job["generation_plan"]), 3)
        feedback = self.contexts[-1]["failed_examples"][0]
        self.assertEqual(feedback["reason"], "structure_duplicate")
        self.assertIn("pattern", feedback)
        self.assertNotIn("normalized_text", feedback)
        self.assertTrue(all(context["previous_examples"] == [] for context in self.contexts))
        records = [json.loads(line) for line in (service.data_dir / job["id"] / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(records[0]["phase"], "planning")
        restored = DatasetJobService(self.config, service.data_dir)
        self.addCleanup(restored.close)
        self.assertEqual(restored.get_job(job["id"])["generation_plan"], job["generation_plan"])

    def test_wrong_plan_ids_and_filled_slots_cannot_bypass_plan_quotas(self):
        """다른 계획 id와 한 상황의 과다 생성을 거부하고 목표를 부풀리지 않는다."""
        calls = []
        def generate(prompt):
            """첫 응답은 같은 계획을 두 번 제출하고 이후에는 계획별 새 문장을 반환한다."""
            context = json.loads(prompt.splitlines()[-1])
            calls.append(context)
            if len(calls) == 1:
                return json.dumps({"examples": [source("비용은 {e0}입니다.", plan_id="p1"),
                                                source("금액 {e0}을 확인하셨나요?", plan_id="p1"),
                                                source("계좌 금액 {e0}을 확인해 주세요.", plan_id="unrequested")]})
            return json.dumps({"examples": [source("다음 처리 비용 {e0}은 별도입니다.", plan_id="p2")]})
        service = self.service(generate, [json.dumps(plans(2))])
        job = self.finish(service, count=2)
        self.assertEqual(job["counts"]["accepted"], 2)
        self.assertEqual(job["counts"]["rejected"], 2)
        self.assertEqual([slot["id"] for slot in calls[1]["planned_slots"]], ["p2"])

    def test_repeated_formats_end_as_partial_without_counting_rejections_as_success(self):
        """같은 형식만 반환하면 목표를 채운 척하지 않고 제한된 재시도 후 일부 완료한다."""
        def generate(prompt):
            """모든 상황의 응답을 숫자만 바꾼 같은 형식으로 고정한다."""
            context = json.loads(prompt.splitlines()[-1])
            return json.dumps({"examples": [source("환불 금액은 {e0}입니다.", str(5000 + len(self.contexts)), plan_id=slot["id"])
                                            for slot in context["planned_slots"]]})
        service = self.service(generate, [json.dumps(plans(2))])
        job = self.finish(service, count=2)
        self.assertEqual(job["status"], "partial")
        self.assertEqual(job["counts"]["accepted"], 1)
        self.assertLessEqual(job["request_count"], 6)

    def test_bad_plans_retry_once_and_never_fall_back_to_unplanned_generation(self):
        """계획이 계속 잘못되면 실제 문장 생성을 시작하지 않는다."""
        calls = []
        service = self.service(lambda prompt: calls.append(prompt), ["{}", "{}"])
        job = self.finish(service)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(calls, [])
        self.assertIn("생성 계획", job["message"])
        self.assertEqual(job["counts"]["accepted"], 0)

    def test_invalid_plan_can_be_repaired_and_cancellation_remains_supported(self):
        """계획 수정 후 생성을 진행하고 계획 도중 중지 요청도 정상 처리한다."""
        def generate(prompt):
            """요청받은 상황 id에 맞는 서로 다른 문장을 반환한다."""
            context = json.loads(prompt.splitlines()[-1])
            templates = {"p1": "금액은 {e0}입니다.", "p2": "접수 비용 {e0}을 결제해 주세요.", "p3": "변경 수수료 {e0}은 언제 환불되나요?"}
            return json.dumps({"examples": [source(templates[slot["id"]], plan_id=slot["id"]) for slot in context["planned_slots"]]})
        service = self.service(generate, ["{}", json.dumps(plans(3))])
        self.assertEqual(self.finish(service)["status"], "completed")
        client = PairedLlamaClient(self.config, threading.Event())
        with patch.object(client, "generate", side_effect=GenerationCancelledError()):
            with self.assertRaises(GenerationCancelledError):
                client.generate_plan("계획")
        self.assertFalse(client.planning_mode)

    def test_real_client_decode_failure_retries_three_plans_at_a_time(self):
        """디코딩 실패 뒤에도 고정된 3개 상황으로 재시도한다."""
        calls = []
        def generate(prompt):
            """첫 디코딩만 실패시키고 이후에는 요청받은 계획에 맞게 응답한다."""
            context = json.loads(prompt.splitlines()[-1])
            calls.append(context)
            if len(calls) == 1:
                raise ValueError("로컬 모델 출력이 완전한 UTF-8 JSON이 아닙니다.")
            templates = {"p1": "금액은 {e0}입니다.", "p2": "접수 비용 {e0}을 확인해 주세요.", "p3": "수수료 {e0}은 언제 돌려받나요?"}
            return json.dumps({"examples": [source(templates[slot["id"]], plan_id=slot["id"]) for slot in context["planned_slots"]]})
        service = self.service(generate)
        job = self.finish(service)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(calls[0]["count"], 3)
        self.assertTrue(all(context["count"] == 3 for context in calls[1:]))

    def test_avoidance_history_is_bounded_even_for_large_jobs(self):
        """생성 결과가 많아져도 금지 형식 이력이 모델 문맥을 과도하게 차지하지 않는다."""
        service = self.service(lambda prompt: "")
        accumulator = SimpleNamespace(rows=[{"accepted": True, "structure_pattern": str(index) + "서로 다른 문장 형식 " * 30} for index in range(100)])
        patterns = service._used_structure_patterns(accumulator)
        self.assertLessEqual(len(patterns), 12)
        self.assertLessEqual(sum(map(len, patterns)), 1200)
        self.assertTrue(patterns[-1].startswith("99"))

    def test_real_client_uses_separate_schemas_for_planning_and_sentences(self):
        """실제 실행 도구의 계획 스키마와 문장 스키마를 분리한다."""
        client = PairedLlamaClient(self.config, threading.Event())
        client.planning_mode = True
        command = client._build_command("계획")
        self.assertEqual(json.loads(command[command.index("--json-schema") + 1]), PLAN_SCHEMA)
        client.planning_mode = False
        client.prompt_mode = True
        command = client._build_command("문장")
        self.assertEqual(json.loads(command[command.index("--json-schema") + 1]), PROMPT_SCHEMA)

    def test_plan_output_decoding_does_not_require_an_examples_array(self):
        """실제 계획 JSON은 examples 배열 없이 디코딩하고 CLI 종료 표식 검증은 유지한다."""
        client = PairedLlamaClient(self.config, threading.Event())
        raw = (json.dumps(plans(3)) + "\n[end of text]").encode()
        self.assertEqual(json.loads(client._extract_response(raw, validate_examples=False)), plans(3))
        with self.assertRaises(ValueError):
            client._extract_response(raw)
        with self.assertRaises(ValueError):
            client._extract_response(b'{"plans":[]} unexpected', validate_examples=False)

    def test_output_schema_enforces_the_requested_count_and_current_plan_ids(self):
        """모델이 다른 계획을 반복하거나 요청 개수를 넘겨 응답하지 못하게 제약한다."""
        client = PairedLlamaClient(self.config, threading.Event())
        client.prompt_mode = True
        prompt = build_pair_prompt("고객 상담", 2, [], prompt_mode=True, planned_slots=[{"id": "p3"}, {"id": "p4"}])
        schema = client._response_schema(prompt)
        self.assertEqual(schema["properties"]["examples"]["minItems"], 2)
        self.assertEqual(schema["properties"]["examples"]["maxItems"], 2)
        self.assertEqual(schema["properties"]["examples"]["items"]["properties"]["plan_id"]["enum"], ["p3", "p4"])
        self.assertNotIn("minItems", PROMPT_SCHEMA["properties"]["examples"])
        self.assertNotIn("enum", PROMPT_SCHEMA["properties"]["examples"]["items"]["properties"]["plan_id"])

    def test_structured_cli_is_selected_only_when_available_next_to_completion(self):
        """같은 설치 위치에 CLI가 있으면 사용하고 없으면 지정 실행 도구를 유지한다."""
        completion = self.root / "llama-completion"
        cli = self.root / "llama-cli"
        completion.write_bytes(b"test")
        cli.write_bytes(b"test")
        with patch("asr_corpus.web_dataset.shutil.which", return_value=str(completion)):
            client = PairedLlamaClient(LlamaCppConfig(self.config.model_file, executable=str(completion)), threading.Event())
            command = client._build_command("계획")
        self.assertEqual(Path(command[0]), cli.resolve())
        self.assertEqual(command[command.index("--reasoning-budget") + 1], "0")
        with patch("asr_corpus.web_dataset.shutil.which", return_value=None):
            self.assertEqual(client._build_command("계획")[0], str(completion))

    def test_cli_parser_excludes_echoed_examples_and_rejects_unknown_trailing_text(self):
        """CLI 안내·프롬프트 속 예시를 실제 답으로 오인하지 않고 종료 형식도 검증한다."""
        client = PairedLlamaClient(self.config, threading.Event())
        client._uses_cli_output = True
        answer = json.dumps({"examples": [source("금액은 {e0}입니다.")]})
        trailer = "\n\n[ Prompt: 500.0 t/s | Generation: 20.0 t/s ]\n\nExiting...\n"
        output = 'Loading model...\n{"examples":[]}\n> echoed prompt\n' + answer + trailer
        self.assertEqual(client._extract_response(output.encode()), answer)
        self.assertEqual(client._extract_response(b"\xff\n" + (answer + trailer).encode()), answer)
        with self.assertRaises(ValueError):
            client._extract_response((answer + "\n알 수 없는 추가 출력").encode())
        with self.assertRaises(ValueError):
            client._extract_response(b"\xff")
        plan = json.dumps(plans(3))
        self.assertEqual(client._extract_response((plan + trailer).encode(), validate_examples=False), plan)
