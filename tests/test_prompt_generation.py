"""프롬프트 기반 도메인 확장과 공통 설정·검토 선택의 정합성을 검증한다."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from asr_corpus.local import LlamaCppConfig
from asr_corpus.number_validation import numeric_alignment_issue
from asr_corpus.prompt_generation import DEFAULT_COMMON_PROMPT, PROMPT_SCHEMA, render_prompt_candidate
from asr_corpus.web import DatasetWebApplication
from asr_corpus.web_dataset import DatasetAccumulator, DatasetJobService, build_pair_prompt, validate_settings
from asr_corpus.reading_policy import validate_reading_rules


def expression(value, reading="sino", unit="", spoken_unit="", negative="", spoken=""):
    """도메인 이름 없이 읽기와 단위 표기를 정의한다."""
    return {"value": value, "reading": reading, "unit": unit, "spoken_unit": spoken_unit, "negative": negative, "spoken": spoken}


def source(template, *expressions):
    """같은 문장 틀에서 발화와 전사문을 만드는 입력을 반환한다."""
    return {"template": template, "expressions": list(expressions), "protected_terms": []}


class PromptGenerationRegressionTest(unittest.TestCase):
    def test_money_format_and_tts_unit_boundary_are_consistent(self):
        """입력 쉼표 유무와 무관하게 금액을 세 자리로 묶고 원 뒤 TTS 경계를 유지한다."""
        for value, expected in (("4500", "4,500"), ("4,500", "4,500"), ("12010", "12,010"),
                                ("50607", "50,607"), ("100000000", "100,000,000"), ("9", "9")):
            for suffix in ("입니다.", "을 결제합니다.", " 입니다.", "."):
                with self.subTest(value=value, suffix=suffix):
                    pair = render_prompt_candidate({"parts": [{"value": value, "unit": "원"}, {"text": suffix}]})
                    self.assertEqual(pair["normalized_text"], expected + "원" + suffix)
                    self.assertNotRegex(pair["spoken_text"], r"원[가-힣]")
                    self.assertNotIn("  ", pair["spoken_text"])
                    self.assertIsNone(numeric_alignment_issue(pair["spoken_text"], pair["normalized_text"], pair["reading_aliases"], pair["reading_policy"]))

    def test_same_amount_in_a_different_sentence_requires_review_and_is_sent_to_next_batch(self):
        """문장이 달라도 같은 금액의 자동 채택을 막고 다음 요청에 재사용 금액을 전달한다."""
        accumulator = DatasetAccumulator(validate_settings({"prompt": "금액 발화", "prompt_mode": True}))
        first = accumulator.add(render_prompt_candidate({"parts": [{"text": "배송비는 "}, {"value": "4500", "unit": "원"}]}))
        repeated = accumulator.add(render_prompt_candidate({"parts": [{"text": "취소할 때 차감되는 금액 "}, {"value": "4,500", "unit": "원"}, {"text": "을 확인하세요."}]}))
        self.assertTrue(first["accepted"])
        self.assertFalse(repeated["accepted"])
        self.assertTrue(repeated["needs_review"])
        self.assertEqual(repeated["reason"], "repeated_amount")
        prompt = build_pair_prompt("새 금액", 3, [], prompt_mode=True, used_amounts=sorted(accumulator.used_amounts))
        self.assertEqual(json.loads(prompt.splitlines()[-1])["used_amounts"], ["4500"])

    def test_unconverted_korean_quantities_cannot_be_followed_by_another_numeric_value(self):
        """화면에서 확인된 금액 중복·값 불일치와 텍스트 분할 우회를 차단한다."""
        cases = [
            [{"text": "환불된 금액은 칠만 오천 원입니다."}, {"value": "75000", "unit": "원"}],
            [{"text": "추가로 만 이백 원이 결제됩니다."}, {"value": "1200", "unit": "원"}],
            [{"text": "금액은 십만 "}, {"text": "오천 원입니다."}],
            [{"text": "배송비는 칠천원입니다."}],
            [{"text": "참석자는 열다섯 명입니다."}],
            [{"text": "예약일은 십오일입니다."}],
            [{"text": "신규 수치는 삼점오 새단위입니다."},
             {"value": "3.5", "unit": "X", "spoken_unit": "새단위"}],
        ]
        for parts in cases:
            with self.subTest(parts=parts), self.assertRaisesRegex(ValueError, "미변환 한글 수치"):
                render_prompt_candidate({"parts": parts})

    def test_quantity_validation_preserves_general_words_and_correctly_placed_values(self):
        """지원·회원·일시 같은 일반 표현과 정상적인 숫자 객체는 보존한다."""
        text = "회원 지원과 할인 신청입니다. 예약 일시를 변경하고 이 시간에 방문하세요."
        self.assertEqual(render_prompt_candidate({"parts": [{"text": text}]})["spoken_text"], text)
        pair = render_prompt_candidate({"parts": [{"text": "추가로 "}, {"value": "10200", "unit": "원"},
                                                  {"text": "이 결제됩니다."}]})
        self.assertEqual(pair["spoken_text"], "추가로 만 이백 원 이 결제됩니다.")
        self.assertEqual(pair["normalized_text"], "추가로 10,200원이 결제됩니다.")

    def test_prompt_generation_preserves_batch_choices_and_uses_internal_rules(self):
        """배치는 선택값을 유지하고 생략하면 3개를 쓰며 공통 규칙·검토 정책은 내부에서 적용한다."""
        settings = validate_settings({"prompt": "금액 발화", "prompt_mode": True, "batch_size": 10,
                                      "common_prompt": "이전 규칙", "reading_rules": [],
                                      "near_duplicate_policy": "drop", "similarity_threshold": 0.8})
        self.assertEqual(settings["batch_size"], 10)
        self.assertEqual(validate_settings({"prompt": "발화", "prompt_mode": True})["batch_size"], 3)
        for size in (1, 3, 5, 10):
            self.assertEqual(validate_settings({"prompt": "발화", "prompt_mode": True, "batch_size": size})["batch_size"], size)
        with self.assertRaises(ValueError):
            validate_settings({"prompt": "발화", "prompt_mode": True, "batch_size": 2})
        self.assertEqual(settings["near_duplicate_policy"], "review")
        self.assertEqual(settings["similarity_threshold"], 0.95)
        self.assertEqual(settings["common_prompt"], DEFAULT_COMMON_PROMPT)
        self.assertEqual(settings["reading_rules"], validate_reading_rules())

    def test_final_reading_requests_use_one_parts_contract(self):
        """한글 발화·숫자 정답 요청도 단일 parts 지시와 같은 값으로 연결한다."""
        request = "금액을 반드시 포함하고 발화는 한글로, 정답은 아라비아 숫자로 작성해줘."
        prompt = build_pair_prompt(request, 1, [], prompt_mode=True,
                                   common_prompt="발화에는 TTS가 읽을 표현을, 정답에는 ASR 표기를 사용하세요.")
        context = json.loads(prompt[prompt.index('{"user_prompt"'):])
        self.assertEqual(context["user_prompt"], request)
        self.assertIn("두 문장을 직접 작성하지 마세요", prompt)
        self.assertIn("각 예제에 해당 표현 객체를 포함하세요", prompt)
        pair = render_prompt_candidate({"parts": [{"text": "환불 금액은 "},
                                                  {"value": "105000", "unit": "원"},
                                                  {"text": "입니다."}]})
        self.assertEqual(pair["spoken_text"], "환불 금액은 십만 오천 원 입니다.")
        self.assertEqual(pair["normalized_text"], "환불 금액은 105,000원입니다.")

    def test_temperature_notation_and_subzero_reading_follow_the_prompt(self):
        """온도 전용 코드 없이 ℃ 표기와 영하 읽기를 같은 값으로 연결한다."""
        pair = render_prompt_candidate(source("오늘 최저 기온은 {e0}입니다.", expression("-3.5", unit="℃", spoken_unit="도씨", negative="영하")))
        self.assertEqual(pair["spoken_text"], "오늘 최저 기온은 영하 삼점오 도씨입니다.")
        self.assertEqual(pair["normalized_text"], "오늘 최저 기온은 -3.5℃입니다.")
        self.assertIsNone(numeric_alignment_issue(pair["spoken_text"], pair["normalized_text"], pair["reading_aliases"]))
        self.assertIsNotNone(numeric_alignment_issue(pair["spoken_text"], pair["normalized_text"].replace("-3.5", "-5.3"), pair["reading_aliases"]))
        self.assertIsNotNone(numeric_alignment_issue(pair["spoken_text"], pair["normalized_text"].replace("-3.5", "3.5"), pair["reading_aliases"]))
        self.assertIn("℃", pair["review_note"])

    def test_unregistered_domains_and_units_require_no_source_or_schema_changes(self):
        """목록에 없는 물리량·단위도 요청에서 선언한 읽기로 생성한다."""
        for unit, spoken_unit in (("kPa", "킬로파스칼"), ("μSv/h", "마이크로시버트 매시간"), ("신규단위", "새 단위")):
            with self.subTest(unit=unit):
                pair = render_prompt_candidate(source("측정값은 {e0}입니다.", expression("0.25", unit=unit, spoken_unit=spoken_unit)))
                self.assertEqual(pair["normalized_text"], "측정값은 0.25" + unit + "입니다.")
                self.assertIsNone(numeric_alignment_issue(pair["spoken_text"], pair["normalized_text"], pair["reading_aliases"]))
        for branch in PROMPT_SCHEMA["properties"]["examples"]["items"]["properties"]["parts"]["items"]["oneOf"][1:]:
            self.assertNotIn("enum", branch["properties"]["unit"])
            self.assertNotIn("enum", branch["properties"]["spoken_unit"])

    def test_common_reading_primitives_preserve_natural_money_and_people_readings(self):
        """새 생성 경로도 천이백오십 원·열 명 규칙을 유지한다."""
        pair = render_prompt_candidate(source("참석자는 {e0}이고 참가비는 {e1}입니다.", expression("10", "native", "명", "명"), expression("1,250", unit="원", spoken_unit="원")))
        self.assertEqual(pair["spoken_text"], "참석자는 열 명이고 참가비는 천이백오십 원 입니다.")
        self.assertEqual(pair["normalized_text"], "참석자는 10명이고 참가비는 1,250원입니다.")
        self.assertFalse(pair["review_note"])

    def test_general_dialogue_without_numbers_uses_the_same_generation_path(self):
        """숫자가 없는 일반 상담 발화도 별도 모드 선택 없이 생성한다."""
        pair = render_prompt_candidate(source("예약 날짜를 변경하고 싶습니다."))
        self.assertEqual(pair["spoken_text"], pair["normalized_text"])
        self.assertIsNone(numeric_alignment_issue(pair["spoken_text"], pair["normalized_text"], pair["reading_aliases"]))
        self.assertIsNotNone(numeric_alignment_issue(pair["spoken_text"], "예약을 취소하고 싶습니다.", pair["reading_aliases"]))

    def test_custom_abbreviations_are_visible_for_review_and_do_not_rewrite_other_text(self):
        """낯선 약어 발음은 검토 표시하며 문장 내용 변경은 감지한다."""
        pair = render_prompt_candidate(source("{e0} 연결을 확인하세요.", expression("API", "custom", spoken="에이피아이")))
        self.assertEqual(pair["spoken_text"], "에이피아이 연결을 확인하세요.")
        self.assertIsNone(numeric_alignment_issue(pair["spoken_text"], pair["normalized_text"], pair["reading_aliases"]))
        self.assertTrue(pair["review_note"])
        self.assertIsNotNone(numeric_alignment_issue(pair["spoken_text"], "API 연결을 삭제하세요.", pair["reading_aliases"]))

    def test_bad_slot_values_are_rejected_instead_of_becoming_freeform_numeric_readings(self):
        """잘못된 숫자와 자리표시자는 자유 발음으로 우회하여 채택하지 않는다."""
        invalid = [source("값은 {e0}입니다.", expression("3.5", "custom", spoken="오점삼")),
                   source("값은 {e1}입니다.", expression("3.5")), source("값은 3과 {e0}입니다.", expression("3.5")),
                   source("값은 {e0}입니다.", expression("3", unit="℃")),
                   source("값은 {e0}입니다.", expression("3", unit="L", spoken_unit="1리터"))]
        for candidate in invalid:
            with self.assertRaises(ValueError):
                render_prompt_candidate(candidate)

    def test_particles_follow_the_spoken_unit_without_changing_numeric_values(self):
        """낯선 단위도 한글 발음의 받침에 맞게 같은 문장 조사를 사용한다."""
        pair = render_prompt_candidate(source("측정값 {e0}를 넘지 않습니다.", expression("1.5", unit="μSv/h", spoken_unit="마이크로시버트 매시간")))
        self.assertEqual(pair["spoken_text"], "측정값 일점오 마이크로시버트 매시간을 넘지 않습니다.")
        self.assertEqual(pair["normalized_text"], "측정값 1.5μSv/h을 넘지 않습니다.")
        self.assertIsNone(numeric_alignment_issue(pair["spoken_text"], pair["normalized_text"], pair["reading_aliases"]))
        pair = render_prompt_candidate(source("기온 {e0}이상입니다.", expression("3", unit="도", spoken_unit="도")))
        self.assertTrue(pair["spoken_text"].endswith("도이상입니다."))

    def test_conflicting_pronunciations_for_the_same_notation_are_not_silently_chosen(self):
        """한 예제에서 같은 표기의 발음이 다르면 임의로 하나를 선택하지 않는다."""
        with self.assertRaisesRegex(ValueError, "발음이 서로 다릅니다"):
            render_prompt_candidate(source("{e0}와 {e1}입니다.", expression("1", unit="X", spoken_unit="엑스"), expression("2", unit="X", spoken_unit="와이")))

    def test_unit_suffix_before_korean_particles_is_not_duplicated(self):
        """실제 모델의 ℃에·℃로 표기를 처리해 단위가 두 번 생성되지 않게 한다."""
        for particle in ("에", "로", "에서", "와", "입니다"):
            pair = render_prompt_candidate(source("기온은 {e0}℃" + particle + " 안내합니다.", expression("28.5", unit="℃", spoken_unit="도씨")))
            self.assertEqual(pair["normalized_text"].count("℃"), 1)
            self.assertNotIn("℃", pair["spoken_text"])
            self.assertIsNone(numeric_alignment_issue(pair["spoken_text"], pair["normalized_text"], pair["reading_aliases"]))
        with self.assertRaisesRegex(ValueError, "같은 단위가 반복"):
            render_prompt_candidate(source("기온은 {e0}℃℃입니다.", expression("28.5", unit="℃", spoken_unit="도씨")))

    def test_reported_refund_amount_cannot_be_repeated_inside_the_unit_reading(self):
        """제보된 오만 오천 중복은 단위 입력 단계에서 거부하고 정상 단위만 출력한다."""
        template = "고객님, 취소하신 예약 건에 대해 환불 금액 {e0}원과 결제 수단을 확인해 주시겠어요?"
        bad = source(template, expression("55,000", unit="원", spoken_unit="오만 오천 원"))
        with self.assertRaisesRegex(ValueError, "단위 발음에 수치"):
            render_prompt_candidate(bad)
        good = source(template, expression("55,000", unit="원", spoken_unit="원"))
        pair = render_prompt_candidate(good)
        self.assertEqual(pair["spoken_text"], "고객님, 취소하신 예약 건에 대해 환불 금액 오만 오천 원 과 결제 수단을 확인해 주시겠어요?")
        self.assertEqual(pair["normalized_text"], "고객님, 취소하신 예약 건에 대해 환불 금액 55,000원과 결제 수단을 확인해 주시겠어요?")
        self.assertFalse(pair["review_note"])

    def test_unit_value_injection_is_rejected_for_money_counts_and_measurements(self):
        """금액 외 수량·측정값에서도 숫자를 단위 발음에 섞어 검증을 우회할 수 없다."""
        cases = [expression("55,000", unit="원", spoken_unit="오만오천원"),
                 expression("55,000", unit="원", spoken_unit="삼만 원"),
                 expression("3", "native", "건", "세 건"),
                 expression("3", "native", "건", "세건"),
                 expression("5.5", unit="kg", spoken_unit="오점오킬로그램"),
                 expression("10,000", unit="kPa", spoken_unit="만 킬로파스칼")]
        for value in cases:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "단위 발음에 수치"):
                render_prompt_candidate(source("값은 {e0}입니다.", value))

    def test_persisted_aliases_cannot_teach_the_validator_to_accept_repeated_values(self):
        """잘못된 모델 별칭을 받은 검증기도 반복 수치를 정상 대응으로 승인하지 않는다."""
        bad_spoken = "환불 금액은 오만 오천 오만 오천 원입니다."
        normalized = "환불 금액은 55,000원입니다."
        for context, written in (("unit", "원"), ("literal", "원"), ("literal", "55,000원")):
            aliases = [{"written": written, "spoken": "오만 오천 원" if written == "원" else "오만 오천 오만 오천 원", "context": context}]
            self.assertIsNotNone(numeric_alignment_issue(bad_spoken, normalized, aliases))
        self.assertIsNotNone(numeric_alignment_issue("환불 금액은 오만 오천 55000원입니다.", normalized,
                                                    [{"written": "원", "spoken": "55000원", "context": "unit"}]))

    def test_normal_unit_names_and_encoded_dimensions_are_not_mistaken_for_quantity_injection(self):
        """숫자 음절로 시작하는 정상 단위 이름과 제곱·세제곱 표기를 보존한다."""
        for unit, spoken_unit in (("ion", "이온"), ("eV", "일렉트론볼트"), ("㎥", "세제곱미터"),
                                  ("㎥", "삼 제곱미터"), ("kPa", "킬로파스칼"), ("μSv/h", "마이크로시버트 매시간")):
            with self.subTest(unit=unit, spoken_unit=spoken_unit):
                pair = render_prompt_candidate(source("측정값은 {e0}입니다.", expression("2.5", unit=unit, spoken_unit=spoken_unit)))
                self.assertIsNone(numeric_alignment_issue(pair["spoken_text"], pair["normalized_text"], pair["reading_aliases"]))


class PromptWorkflowServiceTest(unittest.TestCase):
    def setUp(self):
        """로컬 모델·서버를 사용하지 않는 자동 정리 작업 폴더를 준비한다."""
        self.temporary = tempfile.TemporaryDirectory(prefix="asr-prompt-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        model = self.root / "model.gguf"
        model.write_bytes(b"GGUFtest")
        self.config = LlamaCppConfig(model, executable=sys.executable)

    def service(self, candidates):
        """LLM 출력만 고정하고 실제 생성·검토·저장 서비스를 사용한다."""
        client = SimpleNamespace(generate=lambda prompt: json.dumps({"examples": candidates}))
        service = DatasetJobService(self.config, self.root / "jobs", lambda config, cancel: client)
        self.addCleanup(service.close)
        return service

    def finish(self, service, count=1):
        """공통 규칙과 주제를 스냅샷에 저장한 프롬프트 모드 작업을 완료한다."""
        job = service.start_job({"prompt": "온도 관련 발화를 만들어줘.", "common_prompt": "숫자는 정확히 유지하세요.",
                                 "count": count, "batch_size": 3, "prompt_mode": True})
        thread = service._threads.get(job["id"])
        if thread:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        return service.get_job(job["id"])

    def test_new_unit_reviews_can_be_confirmed_once_and_restored(self):
        """낯선 단위는 텍스트에서 한 번 확인하면 음성 대상으로 사용할 수 있다."""
        service = self.service([source("기온은 {e0}입니다.", expression("-3.5", unit="℃", spoken_unit="도씨", negative="영하"))])
        job = self.finish(service)
        row = job["examples"][0]
        self.assertEqual(job["status"], "completed")
        self.assertTrue(row["needs_review"])
        self.assertIsNone(row["quality_issue"])
        self.assertEqual(row["reason"], "reading_review")
        reviewed = service.update_example(job["id"], row["id"], {"accepted": True})
        self.assertFalse(reviewed["examples"][0]["needs_review"])
        restored = DatasetJobService(self.config, service.data_dir)
        self.addCleanup(restored.close)
        self.assertFalse(restored.get_job(job["id"])["examples"][0]["needs_review"])
        self.assertTrue(restored.get_job(job["id"])["examples"][0]["accepted"])

    def test_duplicated_literal_money_is_preserved_as_a_failure_and_never_auto_selected(self):
        """미변환 금액을 포함한 모델 원문은 실패 이유와 함께 보존하고 자동 선택하지 않는다."""
        candidate = {"parts": [{"text": "환불된 금액은 칠만 오천 원입니다."},
                               {"value": "75000", "unit": "원"}]}
        service = self.service([])
        accumulator = DatasetAccumulator(validate_settings({"prompt": "금액 발화", "prompt_mode": True}))
        row = accumulator.add(service._render_prompt_candidate(candidate))
        self.assertFalse(row["accepted"])
        self.assertIn("미변환 한글 수치", row["reason"])
        self.assertEqual(row["candidate"]["parts"], candidate["parts"])
        self.assertEqual(row["spoken_text"], "")

    def test_common_preferences_persist_but_do_not_rewrite_existing_job_settings(self):
        """공통 규칙을 바꿔도 이미 생성한 작업의 조건은 실행 당시대로 보존한다."""
        service = self.service([source("안녕하세요.")])
        self.assertEqual(service.get_preferences()["common_prompt"], DEFAULT_COMMON_PROMPT)
        job = self.finish(service)
        app = DatasetWebApplication(service)
        response = app.dispatch("PATCH", "/api/preferences", {"common_prompt": "새로운 공통 규칙", "output_root": "/tmp/custom-root"})
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(app.dispatch("GET", "/api/preferences").body)["common_prompt"], DEFAULT_COMMON_PROMPT)
        self.assertEqual(service.get_job(job["id"])["settings"]["common_prompt"], DEFAULT_COMMON_PROMPT)
        restored = DatasetJobService(self.config, service.data_dir)
        self.addCleanup(restored.close)
        self.assertEqual(restored.get_preferences()["output_root"], "/tmp/custom-root")
        self.assertEqual(app.dispatch("PATCH", "/api/preferences", {"output_root": "relative"}).status, 400)

    def test_bulk_selection_is_atomic_if_any_requested_row_is_invalid(self):
        """한 항목이라도 숫자 대응이 틀리면 일부 선택만 저장하지 않는다."""
        service = self.service([source("사과 {e0}입니다.", expression("3", "native", "개", "개")), source("배 {e0}입니다.", expression("2", "native", "개", "개"))])
        job = self.finish(service, count=2)
        identifiers = [row["id"] for row in job["examples"]]
        service.update_selection(job["id"], {"example_ids": identifiers, "accepted": False})
        with service._lock:
            service._jobs[job["id"]]["examples"][1]["normalized_text"] = "배 5개입니다."
        with self.assertRaises(ValueError):
            service.update_selection(job["id"], {"example_ids": identifiers, "accepted": True})
        self.assertTrue(all(not row["accepted"] for row in service.get_job(job["id"])["examples"]))

    def test_prompt_includes_common_rules_and_the_requested_domain_without_numeric_domain_whitelist(self):
        """이번 요청과 공통 규칙을 같은 생성 요청에 넣고 단위는 열거하지 않는다."""
        prompt = build_pair_prompt("신규 물리량 단위를 추가해줘.", 2, [], prompt_mode=True, common_prompt="존댓말을 사용하세요.")
        context = json.loads(prompt.splitlines()[-1])
        self.assertEqual(context["common_prompt"], "존댓말을 사용하세요.")
        self.assertEqual(context["user_prompt"], "신규 물리량 단위를 추가해줘.")
        self.assertIn("도메인과 단위는 제한하지 않습니다", prompt)

    def test_selected_text_export_excludes_unconfirmed_readings(self):
        """새 UI의 텍스트 내보내기도 사용 선택한 검토 완료 문장만 포함한다."""
        service = self.service([source("기온은 {e0}입니다.", expression("-3.5", unit="℃", spoken_unit="도씨", negative="영하"))])
        job = self.finish(service)
        app = DatasetWebApplication(service)
        response = app.dispatch("GET", f"/api/jobs/{job['id']}/selected-export")
        self.assertEqual(response.body, b"")
        service.update_selection(job["id"], {"example_ids": [job["examples"][0]["id"]], "accepted": True})
        response = app.dispatch("GET", f"/api/jobs/{job['id']}/selected-export")
        self.assertEqual(json.loads(response.body)["normalized_text"], "기온은 -3.5℃입니다.")

    def test_malformed_prompt_output_retries_with_the_fixed_batch_size(self):
        """응답 형식 오류 뒤에도 고정된 3개 배치로 재시도한다."""
        requested = []
        next_value = [1]

        def generate(prompt):
            """첫 요청은 형식 오류를 내고 이후에는 요청 개수만큼 응답한다."""
            context = json.loads(prompt.splitlines()[-1])
            requested.append(context["count"])
            if len(requested) == 1:
                return "잘린 JSON"
            candidates = []
            for _ in range(context["count"]):
                template = ("수량은 {e0}입니다.", "창고에 남은 재고 {e0}를 보내 주세요.", "이 상품은 {e0} 단위로 묶어 판매하나요?")[next_value[0] - 1]
                candidates.append(source(template, expression(str(next_value[0]), "native", "개", "개")))
                next_value[0] += 1
            return json.dumps({"examples": candidates})

        service = DatasetJobService(self.config, self.root / "jobs", lambda config, cancel: SimpleNamespace(generate=generate))
        self.addCleanup(service.close)
        job = service.start_job({"prompt": "수량 발화를 생성하세요.", "prompt_mode": True, "count": 3, "batch_size": 3})
        thread = service._threads.get(job["id"])
        if thread:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(service.get_job(job["id"])["counts"]["accepted"], 3)
        self.assertEqual(requested, [3, 3])

    def test_unreviewed_readings_are_not_reused_as_previous_accepted_examples(self):
        """미검토 단위 대응은 다음 생성의 채택 예시로 전달하지 않는다."""
        responses = iter([[source("기온은 {e0}입니다.", expression("-3.5", unit="℃", spoken_unit="도씨", negative="영하"))],
                          [source("배송 주소를 확인해 주세요.")], [source("예약 변경을 도와드리겠습니다.")]])
        contexts = []

        def generate(prompt):
            """생성에 전달한 이전 예시를 기록하고 고정된 응답을 반환한다."""
            contexts.append(json.loads(prompt.splitlines()[-1]))
            return json.dumps({"examples": next(responses)})

        service = DatasetJobService(self.config, self.root / "jobs", lambda config, cancel: SimpleNamespace(generate=generate))
        self.addCleanup(service.close)
        job = service.start_job({"prompt": "여러 주제의 안내 발화", "prompt_mode": True, "count": 3, "batch_size": 1})
        thread = service._threads.get(job["id"])
        if thread:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(service.get_job(job["id"])["status"], "completed")
        self.assertEqual(contexts[1]["previous_examples"], [])
        self.assertEqual(contexts[2]["previous_examples"], [])
        self.assertIn("배송 주소를 확인해 주세요.", contexts[2]["used_patterns"])

    def test_legacy_bad_alias_is_demoted_on_load_and_can_be_fixed_without_changing_the_id(self):
        """저장된 중복 수치는 원본 파일 보존 상태로 차단하고 수동 수정 시 잘못된 대응만 제거한다."""
        service = self.service([source("환불 금액은 {e0}입니다.", expression("55,000", unit="원", spoken_unit="원"))])
        job = self.finish(service)
        path = service.data_dir / job["id"] / "job.json"
        saved = json.loads(path.read_text())
        row = saved["examples"][0]
        row.update(spoken_text="환불 금액은 오만 오천 오만 오천 원입니다.", reading_aliases=[{"written": "원", "spoken": "오만 오천 원", "context": "unit"}],
                   accepted=True, needs_review=True, reason="reading_review", review_note="단위 발음 확인: 원 → 오만 오천 원")
        path.write_text(json.dumps(saved, ensure_ascii=False))
        original = path.read_bytes()
        restored = DatasetJobService(self.config, service.data_dir)
        self.addCleanup(restored.close)
        loaded = restored.get_job(job["id"])
        self.assertFalse(loaded["examples"][0]["accepted"])
        self.assertTrue(loaded["examples"][0]["needs_review"])
        self.assertIsNotNone(loaded["examples"][0]["quality_issue"])
        self.assertEqual(restored.export_jsonl(job["id"]), "")
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(ValueError):
            restored.update_example(job["id"], row["id"], {"accepted": True})
        updated = restored.update_example(job["id"], row["id"], {"accepted": True, "spoken_text": "환불 금액은 오만 오천 원입니다."})
        self.assertEqual(updated["examples"][0]["id"], row["id"])
        self.assertEqual(updated["examples"][0]["reading_aliases"], [])
        self.assertEqual(updated["examples"][0]["review_note"], "")
        self.assertIsNone(updated["examples"][0]["quality_issue"])

    def test_bad_unit_response_is_rejected_then_replaced_and_not_counted_as_an_accepted_reference(self):
        """금액을 넣은 단위 응답은 실패로 남기고 다음의 정상 예제만 채택한다."""
        responses = iter([[source("환불 금액은 {e0}입니다.", expression("55,000", unit="원", spoken_unit="오만 오천 원"))],
                          [source("환불 금액은 {e0}입니다.", expression("55,000", unit="원", spoken_unit="원"))]])
        contexts = []

        def generate(prompt):
            """실패 뒤 다음 요청에 잘못된 결과가 성공 예시로 들어가는지 확인한다."""
            contexts.append(json.loads(prompt.splitlines()[-1]))
            return json.dumps({"examples": next(responses)})

        service = DatasetJobService(self.config, self.root / "jobs", lambda config, cancel: SimpleNamespace(generate=generate))
        self.addCleanup(service.close)
        job = self.finish(service)
        self.assertEqual(job["counts"], {"accepted": 1, "review": 0, "rejected": 1, "total": 2})
        self.assertEqual(contexts[1]["previous_examples"], [])
        self.assertIn("단위 발음에 수치", contexts[1]["failed_examples"][0]["detail"])
        self.assertEqual(job["examples"][1]["spoken_text"], "환불 금액은 오만 오천 원 입니다.")


if __name__ == "__main__":
    unittest.main()
