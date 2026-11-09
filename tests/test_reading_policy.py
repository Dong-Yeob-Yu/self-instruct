"""날짜·수량·문맥별 읽기를 생성과 검토에 일관되게 강제하는 회귀 검증."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from asr_corpus.local import LlamaCppConfig
from asr_corpus.number_validation import numeric_alignment_issue
from asr_corpus.prompt_generation import PROMPT_SCHEMA, render_prompt_candidate
from asr_corpus.reading_policy import validate_reading_rules
from asr_corpus.web_dataset import DatasetJobService, build_pair_prompt


def expression(value, unit, reading="native", role="unspecified", spoken_unit=None):
    """모델이 읽기 방식을 잘못 고른 입력도 재현할 수 있는 표현을 만든다."""
    return {"value": value, "unit": unit, "spoken_unit": unit if spoken_unit is None else spoken_unit,
            "reading": reading, "role": role, "negative": "", "spoken": ""}


def candidate(template, *expressions):
    """단위와 의미를 포함하는 하나의 문장 틀을 만든다."""
    return {"template": template, "expressions": list(expressions), "protected_terms": []}


def alignment(pair, spoken=None, transcript=None):
    """별도 수치 대응과 읽기 정책을 실제 저장 검증과 동일하게 적용한다."""
    return numeric_alignment_issue(spoken or pair["spoken_text"], transcript or pair["normalized_text"],
                                   pair["reading_aliases"], pair["reading_policy"])


class ReadingPolicyRegressionTest(unittest.TestCase):
    def test_dates_override_native_reading_without_adding_an_extra_day_suffix(self):
        """모델이 고유어를 선택해도 날짜는 십오일처럼 읽고 일 단위를 중복하지 않는다."""
        for value, expected in (("1", "일일"), ("3", "삼일"), ("11", "십일일"), ("15", "십오일"), ("21", "이십일일"), ("31", "삼십일일")):
            with self.subTest(value=value):
                pair = render_prompt_candidate(candidate("배송일은 {e0}일입니다.", expression(value, "일", role="calendar")))
                self.assertEqual(pair["spoken_text"], f"배송일은 {expected}입니다.")
                self.assertFalse(pair["review_note"])
                self.assertIsNone(alignment(pair))

    def test_missing_pronunciation_is_filled_only_for_known_hangul_units(self):
        """등록된 한글 단위의 빈 발음은 같은 단위로 보완하고 기호 발음은 추측하지 않는다."""
        pair = render_prompt_candidate(candidate("{e0}입니다.", expression("15", "일", spoken_unit="")))
        self.assertEqual(pair["spoken_text"], "십오일입니다.")
        self.assertIsNone(alignment(pair))
        with self.assertRaises(ValueError):
            render_prompt_candidate(candidate("{e0}입니다.", expression("15", "℃", spoken_unit="")))

    def test_unitless_numbers_require_context_review_even_if_a_role_was_supplied(self):
        """단위 없는 숫자는 모델의 의미 분류만 믿고 학습 대상으로 삼지 않는다."""
        for reading, role in (("sino", "unspecified"), ("digits", "identifier"), ("sino", "measure")):
            pair = render_prompt_candidate(candidate("현재 {e0}에 도착했습니다.", expression("2", "", reading, role)))
            self.assertIn("수치 문맥 확인", pair["review_note"])

    def test_actual_model_grouped_date_is_connected_for_explicit_human_review(self):
        """실제 모델의 날짜 묶음 누락을 값 변경 없이 연결하고 검토 대상으로 남긴다."""
        expressions = [expression("2024", "년", "year", spoken_unit=""), expression("10", "월", "month", spoken_unit=""),
                       expression("25", "일", "day", spoken_unit="")]
        pair = render_prompt_candidate(candidate("원래 예약하신 날짜가 {e0}인데 변경이 가능할까요?", *expressions))
        self.assertEqual(pair["normalized_text"], "원래 예약하신 날짜가 2024년 10월 25일인데 변경이 가능할까요?")
        self.assertIn("이천이십사년 시월 이십오일", pair["spoken_text"])
        self.assertIn("표현 묶음 확인", pair["review_note"])
        self.assertIsNone(alignment(pair))
        with self.assertRaises(ValueError):
            render_prompt_candidate(candidate("예약 연도는 {e0}년입니다.", *expressions))
        expressions[1]["value"] = "2"
        expressions[2]["value"] = "30"
        with self.assertRaises(ValueError):
            render_prompt_candidate(candidate("날짜는 {e0}입니다.", *expressions))

    def test_day_durations_are_not_limited_to_calendar_day_ranges(self):
        """90일 기간에도 한자어 읽기를 적용하되 날짜의 31일 제한을 적용하지 않는다."""
        pair = render_prompt_candidate(candidate("처리에 {e0}이 걸립니다.", expression("90", "일", role="duration")))
        self.assertEqual(pair["spoken_text"], "처리에 구십일이 걸립니다.")
        self.assertIsNone(alignment(pair))

    def test_people_hours_months_and_money_follow_unit_rules_instead_of_model_choice(self):
        """인원·시간·개월·금액을 서로 다른 수사 규칙으로 생성한다."""
        cases = [("15", "명", "sino", "열다섯 명"), ("20", "명", "sino", "스무 명"),
                 ("21", "개", "sino", "스물한 개"), ("15", "시간", "sino", "열다섯 시간"),
                 ("15", "개월", "native", "십오 개월"), ("1250", "원", "native", "천이백오십 원"),
                 ("100", "명", "native", "백 명"), ("6", "월", "native", "유월"), ("10", "월", "native", "시월")]
        for value, unit, proposed, expected in cases:
            with self.subTest(unit=unit, value=value):
                pair = render_prompt_candidate(candidate("{e0}입니다.", expression(value, unit, proposed)))
                self.assertEqual(pair["spoken_text"], expected + (" 입니다." if unit == "원" else "입니다."))
                self.assertIsNone(alignment(pair))

    def test_minutes_and_honorific_people_use_distinct_roles_even_in_the_same_sentence(self):
        """같은 분 단위도 인원은 세 분이고 대기 시간은 십오 분이다."""
        pair = render_prompt_candidate(candidate("{e0}께서 {e1} 동안 기다리셨습니다.",
                                                 expression("3", "분", "sino", "people"), expression("15", "분", "native", "duration")))
        self.assertEqual(pair["spoken_text"], "세 분께서 십오 분 동안 기다리셨습니다.")
        self.assertIn("문맥 읽기 확인", pair["review_note"])
        self.assertIsNone(alignment(pair))
        self.assertIsNotNone(alignment(pair, spoken="삼 분께서 열다섯 분 동안 기다리셨습니다."))

    def test_frequency_and_ordinal_numbers_use_separate_rules(self):
        """반복 세 번과 순서 삼 번을 의미에 따라 구분한다."""
        for role, expected in (("count", "세 번"), ("ordinal", "삼 번"), ("identifier", "삼 번")):
            pair = render_prompt_candidate(candidate("{e0}입니다.", expression("3", "번", "native", role)))
            self.assertEqual(pair["spoken_text"], expected + "입니다.")
            self.assertIsNone(alignment(pair))

    def test_unspecified_context_and_unregistered_units_are_explicit_review_items(self):
        """분·번의 의미를 모르면 검토로 남기고 새로운 단위 읽기도 자동 확정하지 않는다."""
        for unit, note in (("분", "의미 미지정"), ("번", "의미 미지정"), ("묶음", "공통 규칙 미등록")):
            pair = render_prompt_candidate(candidate("{e0}입니다.", expression("3", unit)))
            self.assertIn(note, pair["review_note"])

    def test_actual_model_honorific_suffix_does_not_repeat_the_people_unit(self):
        """실제 모델의 분께서는 틀에서 세 분분 대신 세 분으로 출력한다."""
        pair = render_prompt_candidate(candidate("고객님 {e0}분께서는 {e1} 동안 기다려 주세요.",
                                                 expression("3", "분", "sino", "people"), expression("15", "분", "native", "duration")))
        self.assertEqual(pair["spoken_text"], "고객님 세 분께서는 십오 분 동안 기다려 주세요.")
        self.assertEqual(pair["normalized_text"], "고객님 3분께서는 15분 동안 기다려 주세요.")
        self.assertIsNone(alignment(pair))
        for unit, role in (("분", "people"), ("일", "calendar")):
            with self.assertRaisesRegex(ValueError, "단위가 반복"):
                render_prompt_candidate(candidate("{e0}" + unit + unit + "께 알립니다.", expression("3", unit, role=role)))

    def test_validation_rejects_equal_values_with_the_wrong_numeral_family(self):
        """값이 15로 같아도 열다섯 일과 십오 명을 정상 학습 데이터로 통과시키지 않는다."""
        cases = [("일", "열다섯 일"), ("개월", "열다섯 개월"), ("명", "십오 명"), ("시간", "십오 시간"),
                 ("원", "일천이백오십 원")]
        for unit, wrong in cases:
            pair = render_prompt_candidate(candidate("{e0}입니다.", expression("1250" if unit == "원" else "15", unit)))
            self.assertIsNotNone(alignment(pair, spoken=wrong + "입니다."))

    def test_policy_still_checks_value_sign_content_and_repeated_slots(self):
        """읽기 강제 후에도 값·부호·문장 변조와 반복 슬롯 검증을 유지한다."""
        pair = render_prompt_candidate(candidate("{e0}부터 {e0}까지입니다.", expression("15", "일")))
        self.assertIsNone(alignment(pair))
        self.assertIsNotNone(alignment(pair, transcript="16일부터 15일까지입니다."))
        self.assertIsNotNone(alignment(pair, transcript="15일부터 15일까지가 아닙니다."))
        self.assertIsNone(alignment(pair, spoken="이십일부터 이십일까지입니다.", transcript="20일부터 20일까지입니다."))
        phone = render_prompt_candidate(candidate("연락처는 {e0}입니다.", expression("010-1234-5678", "", "phone")))
        self.assertIsNone(alignment(phone))
        custom = {"value": "API", "unit": "", "spoken_unit": "", "reading": "custom", "role": "unspecified", "negative": "", "spoken": "에이피아이"}
        pair = render_prompt_candidate(candidate("{e0}로 {e1} 신청합니다.", custom, expression("15", "명", "sino")))
        self.assertIsNone(alignment(pair))

    def test_user_rules_can_cover_new_units_and_override_fallback_by_role(self):
        """새 단위와 의미별 우선 규칙을 소스 변경 없이 전달한다."""
        rules = [{"unit": "묶음", "role": "*", "reading": "sino"}, {"unit": "묶음", "role": "count", "reading": "native"}]
        pair = render_prompt_candidate(candidate("{e0}입니다.", expression("15", "묶음", "sino", "count")), rules)
        self.assertEqual(pair["spoken_text"], "열다섯 묶음입니다.")
        self.assertFalse(pair["review_note"])
        self.assertIsNotNone(alignment(pair, spoken="십오 묶음입니다."))
        pair = render_prompt_candidate(candidate("{e0}입니다.", expression("3", "kPa", "sino", "measure", "킬로파스칼")),
                                       [{"unit": "kPa", "role": "*", "reading": "native"}])
        self.assertEqual(pair["spoken_text"], "세 킬로파스칼입니다.")
        self.assertIsNone(alignment(pair))
        self.assertIsNotNone(alignment(pair, spoken="삼 킬로파스칼입니다."))

    def test_invalid_and_duplicate_rules_are_rejected_and_defaults_are_copied(self):
        """중복·잘못된 단위 규칙을 저장하지 않으며 기본 규칙이 외부 변경으로 오염되지 않는다."""
        for rules in ({}, [{"unit": "", "role": "*", "reading": "sino"}],
                      [{"unit": "3개", "role": "*", "reading": "native"}],
                      [{"unit": "일", "role": "unknown", "reading": "sino"}],
                      [{"unit": "일", "role": "*", "reading": "custom"}],
                      [{"unit": "일", "role": "*", "reading": "sino"}] * 2):
            with self.assertRaises(ValueError):
                validate_reading_rules(rules)
        rules = validate_reading_rules()
        rules[0]["reading"] = "native"
        self.assertEqual(validate_reading_rules()[0]["reading"], "sino")

    def test_schema_and_model_request_include_semantic_roles_and_effective_rules(self):
        """LLM에 의미 분류를 요구하고 실제 적용할 규칙을 함께 전달한다."""
        for branch in PROMPT_SCHEMA["properties"]["examples"]["items"]["properties"]["parts"]["items"]["oneOf"][1:]:
            self.assertIn("role", branch["properties"])
        prompt = build_pair_prompt("날짜와 인원", 2, [], prompt_mode=True)
        self.assertEqual(json.loads(prompt.splitlines()[-1])["reading_rules"], validate_reading_rules())
        self.assertIn("15일은 십오일", prompt)


class ReadingPolicyServiceRegressionTest(unittest.TestCase):
    def setUp(self):
        """실제 서버나 사용자 데이터를 사용하지 않는 저장·검토 환경을 구성한다."""
        temporary = tempfile.TemporaryDirectory(prefix="asr-reading-policy-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        model = self.root / "model.gguf"
        model.write_bytes(b"GGUFtest")
        self.config = LlamaCppConfig(model, executable=sys.executable)
        self.candidates = [candidate("예약일은 {e0}입니다.", expression("15", "일"))]
        self.service = DatasetJobService(self.config, self.root / "jobs", lambda config, cancel: SimpleNamespace(
            generate=lambda prompt: json.dumps({"examples": self.candidates})))
        self.addCleanup(self.service.close)

    def finish(self):
        """새 작업에 저장된 공통 규칙을 적용하고 생성 스레드의 종료를 확인한다."""
        job = self.service.start_job({"prompt": "날짜와 인원 발화", "prompt_mode": True, "count": 1, "batch_size": 1})
        thread = self.service._threads.get(job["id"])
        if thread:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        return self.service.get_job(job["id"])

    def test_manually_edited_reading_can_be_saved_selected_exported_and_restored(self):
        """직접 수정한 읽기는 대응 검사 없이 저장·선택·내보내기·재로딩을 허용한다."""
        job = self.finish()
        row = job["examples"][0]
        self.assertEqual(row["spoken_text"], "예약일은 십오일입니다.")
        updated = self.service.update_example(job["id"], row["id"], {"accepted": True, "spoken_text": "예약일은 열다섯 일입니다."})
        self.assertTrue(updated["examples"][0]["accepted"])
        self.assertIsNone(updated["examples"][0]["quality_issue"])
        self.service.update_selection(job["id"], {"accepted": False, "example_ids": [row["id"]]})
        self.service.update_selection(job["id"], {"accepted": True, "example_ids": [row["id"]]})
        self.assertTrue(self.service.export_jsonl(job["id"], reviewed_only=True))
        restored = DatasetJobService(self.config, self.service.data_dir)
        self.addCleanup(restored.close)
        result = restored.get_job(job["id"])["examples"][0]
        self.assertTrue(result["accepted"])
        self.assertIsNone(result["quality_issue"])
        self.assertEqual(result["spoken_text"], "예약일은 열다섯 일입니다.")

    def test_unitless_location_like_the_actual_model_output_is_excluded_from_reviewed_export(self):
        """실제 모델의 현재 2에 도착 같은 출력은 검토 전 WAV 대상 내보내기에 포함하지 않는다."""
        self.candidates = [candidate("현재 {e0}에 도착했습니다.", expression("2", "", "digits", "identifier"))]
        job = self.finish()
        self.assertTrue(job["examples"][0]["needs_review"])
        self.assertIn("수치 문맥 확인", job["examples"][0]["review_note"])
        self.assertEqual(self.service.export_jsonl(job["id"], reviewed_only=True), "")

    def test_saved_user_rules_cannot_override_internal_rules_for_new_jobs(self):
        """이전 사용자 읽기 설정은 새 작업의 내부 규칙을 덮어쓰지 않는다."""
        rules = [{"unit": "묶음", "role": "*", "reading": "native"}]
        self.service.save_preferences({"reading_rules": rules})
        self.service.save_preferences({"common_prompt": "새로운 말투 조건"})
        self.assertEqual(self.service.get_preferences()["reading_rules"], validate_reading_rules())
        self.candidates = [candidate("{e0}입니다.", expression("15", "묶음", "sino"))]
        job = self.finish()
        self.assertEqual(job["examples"][0]["spoken_text"], "십오 묶음입니다.")
        self.assertEqual(job["settings"]["reading_rules"], validate_reading_rules())
        self.service.save_preferences({"reading_rules": [{"unit": "묶음", "role": "*", "reading": "sino"}]})
        restored = DatasetJobService(self.config, self.service.data_dir)
        self.addCleanup(restored.close)
        self.assertTrue(restored.get_job(job["id"])["examples"][0]["accepted"])
        self.assertEqual(restored.get_job(job["id"])["settings"]["reading_rules"], validate_reading_rules())
        new = self.finish()
        self.assertEqual(new["examples"][0]["spoken_text"], "십오 묶음입니다.")

    def test_old_prompt_records_with_wrong_readings_are_flagged_without_rewriting_disk(self):
        """이전 결과의 열다섯 일을 검토 대상으로 분리하되 원본 파일은 수정하지 않는다."""
        job = self.finish()
        path = self.service.data_dir / job["id"] / "job.json"
        stored = json.loads(path.read_text())
        stored["settings"].pop("reading_rules", None)
        stored["examples"][0].pop("reading_policy", None)
        stored["examples"][0]["spoken_text"] = "예약일은 열다섯 일입니다."
        path.write_text(json.dumps(stored, ensure_ascii=False))
        original = path.read_bytes()
        restored = DatasetJobService(self.config, self.service.data_dir)
        self.addCleanup(restored.close)
        row = restored.get_job(job["id"])["examples"][0]
        self.assertFalse(row["accepted"])
        self.assertTrue(row["needs_review"])
        self.assertTrue(row["quality_issue"])
        self.assertEqual(path.read_bytes(), original)
        fixed = restored.update_example(job["id"], row["id"], {"accepted": True, "spoken_text": "예약일은 십오일입니다."})
        self.assertTrue(fixed["examples"][0]["accepted"])

    def test_ambiguous_unit_remains_for_human_review_even_when_the_model_supplies_a_role(self):
        """의미 분류 오류가 자동 채택되지 않게 분 단위의 문맥을 검토한다."""
        self.candidates = [candidate("{e0}께서 기다립니다.", expression("3", "분", "sino", "people"))]
        job = self.finish()
        row = job["examples"][0]
        self.assertEqual(row["spoken_text"], "세 분께서 기다립니다.")
        self.assertTrue(row["needs_review"])
        self.assertIsNone(row["quality_issue"])
        result = self.service.update_example(job["id"], row["id"], {"accepted": True})
        self.assertFalse(result["examples"][0]["needs_review"])
