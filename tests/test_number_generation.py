"""숫자 종류별 자연스러운 읽기와 발화·전사문 값 보존을 검증한다."""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from asr_corpus.local import LlamaCppConfig
from asr_corpus.number_generation import NUMBER_SCHEMA, read_number, render_numeric_candidate
from asr_corpus.number_validation import compact_text, numeric_alignment_issue, spoken_protected_terms
from asr_corpus.web_dataset import DatasetAccumulator, DatasetJobService, PairedLlamaClient, build_pair_prompt, validate_settings


def slot(kind, value, unit):
    """숫자 종류·값·단위를 가진 고정 테스트 입력을 만든다."""
    return {"kind": kind, "value": value, "unit": unit}


def candidate(template, *numbers):
    """같은 문장 틀을 공유하는 숫자 생성 응답을 만든다."""
    return {"template": template, "numbers": list(numbers), "protected_terms": []}


class NumberGenerationRegressionTest(unittest.TestCase):
    def test_money_omits_unnecessary_il_without_losing_place_values(self):
        """천·백·십 앞의 일을 생략하고 금액 자릿값은 그대로 유지한다."""
        expected = {"0": "영 원", "1": "일 원", "10": "십 원", "100": "백 원", "1000": "천 원",
                    "1250": "천이백오십 원", "10000": "만 원", "10001": "만 일 원",
                    "123450": "십이만 삼천사백오십 원", "1234500": "백이십삼만 사천오백 원",
                    "100000000": "일억 원", "100010000": "일억 일만 원"}
        for value, spoken in expected.items():
            with self.subTest(value=value):
                normalized, actual = read_number(slot("money", value, "원"))
                self.assertEqual(actual, spoken)
                self.assertEqual(normalized, f"{int(value):,}원")
                self.assertIsNone(numeric_alignment_issue(actual, normalized))

    def test_people_and_counts_use_native_bound_forms(self):
        """인원·수량은 한·두·세·네·열·스무·스물한 형태로 읽는다."""
        values = {"1": "한", "2": "두", "3": "세", "4": "네", "10": "열", "11": "열한", "15": "열다섯",
                  "20": "스무", "21": "스물한", "25": "스물다섯", "30": "서른", "42": "마흔두", "99": "아흔아홉", "100": "백"}
        for unit in ("명", "분", "개", "권"):
            for value, prefix in values.items():
                with self.subTest(value=value, unit=unit):
                    normalized, spoken = read_number(slot("count", value, unit))
                    self.assertEqual(spoken, prefix + " " + unit)
                    self.assertIsNone(numeric_alignment_issue(spoken, normalized))

    def test_unit_context_distinguishes_people_minutes_hours_and_months(self):
        """동일 숫자라도 사람·시각·소요 시간·개월의 의미에 맞게 읽는다."""
        cases = [("count", "2", "분", "두 분"), ("minute", "2", "분", "이 분"),
                 ("hour", "2", "시", "두 시"), ("duration", "5", "시간", "다섯 시간"),
                 ("duration", "3", "개월", "삼 개월"), ("duration", "90", "분", "구십 분")]
        for kind, value, unit, spoken in cases:
            self.assertEqual(read_number(slot(kind, value, unit))[1], spoken)

    def test_dates_use_yuwol_and_siwol_without_duplicate_day_units(self):
        """월 특례와 날짜 일 단위를 정확히 읽고 조사를 보존한다."""
        for month, reading in ((6, "유월"), (10, "시월")):
            for day, day_reading in ((1, "일일"), (3, "삼일"), (10, "십일"), (11, "십일일"), (21, "이십일일"), (25, "이십오일")):
                generated = render_numeric_candidate(candidate("{n0} {n1}에 만나요.", slot("month", str(month), "월"), slot("day", str(day), "일")))
                self.assertEqual(generated["spoken_text"], f"{reading} {day_reading}에 만나요.")
                self.assertIsNone(numeric_alignment_issue(generated["spoken_text"], generated["normalized_text"]))

    def test_clock_range_and_duration_share_one_template_without_reinterpreting_values(self):
        """오후 두 시부터 다섯 시까지와 다섯 시간 동안을 혼동하지 않는다."""
        ranged = render_numeric_candidate(candidate("오후 {n0}부터 {n1}까지 진행합니다.", slot("hour", "2", "시"), slot("hour", "5", "시")))
        duration = render_numeric_candidate(candidate("오후 {n0}부터 {n1} 동안 진행합니다.", slot("hour", "2", "시"), slot("duration", "5", "시간")))
        self.assertEqual(ranged["spoken_text"], "오후 두 시부터 다섯 시까지 진행합니다.")
        self.assertEqual(ranged["normalized_text"], "오후 2시부터 5시까지 진행합니다.")
        self.assertEqual(duration["spoken_text"], "오후 두 시부터 다섯 시간 동안 진행합니다.")
        self.assertEqual(duration["normalized_text"], "오후 2시부터 5시간 동안 진행합니다.")

    def test_phone_numbers_preserve_every_digit_and_only_transcripts_keep_hyphens(self):
        """전화번호의 영·공과 반복 숫자를 자릿수 변경 없이 생성한다."""
        cases = [("070-5555-1234", "공칠공 오오오오 일이삼사"),
                 ("01098765432", "공일공 구팔칠육 오사삼이"), ("02-123-4567", "공이 일이삼 사오육칠")]
        for value, spoken in cases:
            normalized, actual = read_number(slot("phone", value, ""))
            self.assertEqual(actual, spoken)
            self.assertIsNone(numeric_alignment_issue(actual, normalized))
        self.assertEqual(read_number(slot("identifier", "001025", "")), ("001025", "공공일공이오"))

    def test_decimals_and_signs_preserve_fractional_zeroes_and_punctuation(self):
        """소수의 0과 부호를 보존하고 문장 구두점 쉼표를 숫자와 혼동하지 않는다."""
        cases = [(slot("decimal", "-0.05", "도"), "-0.05도", "마이너스 영점영오 도"),
                 (slot("money", "+1250", "원"), "+1,250원", "플러스 천이백오십 원"),
                 (slot("decimal", "3.140", ""), "3.140", "삼점일사영")]
        for number, normalized, spoken in cases:
            self.assertEqual(read_number(number), (normalized, spoken))
            self.assertIsNone(numeric_alignment_issue(spoken, normalized))
        generated = render_numeric_candidate(candidate("값은 {n0}, 차이는 없습니다.", slot("number", "1250", "")))
        self.assertIsNone(numeric_alignment_issue(generated["spoken_text"], generated["normalized_text"]))

    def test_invalid_types_values_ranges_and_slots_are_rejected_instead_of_guessed(self):
        """값·자리표시자 오류는 임의 보정하지 않고 생성 실패로 분리한다."""
        invalid = [slot("money", "12만", "원"), slot("money", "12,50", "원"), slot("money", "0.5", "원"),
                   slot("count", "010", "명"), slot("count", "-1", "명"), slot("count", "1.5", "명"),
                   slot("month", "13", "월"), slot("day", "32", "일"), slot("hour", "25", "시"),
                   slot("minute", "60", "분"), slot("phone", "010-111-222", ""), slot("unknown", "10", "명")]
        for number in invalid:
            with self.subTest(number=number), self.assertRaises(ValueError):
                read_number(number)
        for generated in (candidate("{n1}입니다.", slot("count", "10", "명")),
                          candidate("{n0}분입니다.", slot("count", "10", "명")),
                          candidate("숫자 10과 {n0}입니다.", slot("count", "10", "명")),
                          candidate("{n0}입니다.", slot("count", "10", "명"), slot("money", "1250", "원"))):
            with self.assertRaises(ValueError):
                render_numeric_candidate(generated)

    def test_calendar_dates_reject_february_thirtieth_but_allow_leap_day(self):
        """월·일 개별 범위뿐 아니라 실제 달력의 날짜 조합도 검사한다."""
        numbers = [slot("year", "2024", "년"), slot("month", "2", "월"), slot("day", "29", "일")]
        self.assertIn("2024년 2월 29일", render_numeric_candidate(candidate("{n0} {n1} {n2}입니다.", *numbers))["normalized_text"])
        numbers[0]["value"] = "2023"
        with self.assertRaises(ValueError):
            render_numeric_candidate(candidate("{n0} {n1} {n2}입니다.", *numbers))

    def test_hangul_phone_hyphens_pass_alignment_and_protection_without_hiding_wrong_digits(self):
        """제보된 전화번호 오탐을 없애되 실제 숫자 오류는 계속 검출한다."""
        spoken = "담당자님 연락처는 영칠공-오오오오-일이삼사이며, 총액은 구만 구천 원입니다."
        transcript = "담당자님 연락처는 070-5555-1234이며, 총액은 99,000원입니다."
        self.assertIsNone(numeric_alignment_issue(spoken, transcript))
        self.assertIn("영칠공-오오오오-일이삼사", spoken_protected_terms(spoken, transcript, ["070-5555-1234"]))
        self.assertIsNotNone(numeric_alignment_issue(spoken, transcript.replace("1234", "1235")))
        self.assertIsNotNone(numeric_alignment_issue("공일공-일일일-이이이이", "010-1111-2222"))
        self.assertEqual(compact_text("기온 -3.5도와 1-2-3입니다."), "기온-3.5도와1-2-3입니다.")

    def test_legacy_bad_pairs_are_not_silently_rewritten(self):
        """기존 발화·전사문은 자동 수정하지 않고 잘못된 값은 검토 대상으로 유지한다."""
        accumulator = DatasetAccumulator({"similarity_threshold": 0.95, "near_duplicate_policy": "review"})
        row = accumulator.add({"spoken_text": "백이십삼만 사천오백 원입니다.", "normalized_text": "123,450원입니다.", "protected_terms": []})
        self.assertTrue(row["needs_review"])
        self.assertFalse(row["accepted"])
        self.assertEqual(row["normalized_text"], "123,450원입니다.")


class NumberGenerationServiceTest(unittest.TestCase):
    def setUp(self):
        """실제 모델·서버 없이 테스트할 자동 정리 작업 폴더를 만든다."""
        self.temporary = tempfile.TemporaryDirectory(prefix="asr-number-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"GGUFtest")
        self.config = LlamaCppConfig(self.model, executable=sys.executable)

    def service(self, responses):
        """템플릿 응답만 반환하고 나머지는 실제 생성 서비스를 사용한다."""
        values = iter(responses)
        client = SimpleNamespace(generate=lambda prompt: json.dumps({"examples": next(values)}))
        service = DatasetJobService(self.config, self.root / "jobs", lambda config, cancellation: client)
        self.addCleanup(service.close)
        return service

    def finish(self, service):
        """숫자 모드 한 항목을 생성하고 저장된 최종 결과를 반환한다."""
        job = service.start_job({"prompt": "한국어 숫자 정규화", "count": 1, "batch_size": 1, "number_mode": True})
        thread = service._threads.get(job["id"])
        if thread:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        return service.get_job(job["id"])

    def test_numeric_mode_renders_persists_and_exports_natural_readings_from_shared_values(self):
        """실제 작업 경로에서 천이백오십 원·열 명을 생성하고 원문 응답도 보존한다."""
        source = candidate("참석자는 {n0}이고 참가비는 {n1}입니다.", slot("count", "10", "명"), slot("money", "1250", "원"))
        service = self.service([[source]])
        job = self.finish(service)
        self.assertEqual(job["status"], "completed")
        row = json.loads(service.export_jsonl(job["id"]))
        self.assertEqual(row["spoken_text"], "참석자는 열 명이고 참가비는 천이백오십 원입니다.")
        self.assertEqual(row["normalized_text"], "참석자는 10명이고 참가비는 1,250원입니다.")
        request = json.loads((self.root / "jobs" / job["id"] / "requests.jsonl").read_text())
        self.assertEqual(json.loads(request["raw_response"])["examples"], [source])
        self.assertIn("고유어 수 관형사", request["prompt"])
        self.assertTrue(job["settings"]["number_mode"])

    def test_invalid_templates_do_not_get_accepted_and_next_batch_can_recover(self):
        """잘못된 숫자 원문은 남기고 다음 배치의 올바른 문장 틀을 채택한다."""
        bad = candidate("참석자는 {n0}분입니다.", slot("count", "10", "명"))
        good = candidate("참석자는 {n0}입니다.", slot("count", "10", "명"))
        service = self.service([[bad], [good]])
        job = self.finish(service)
        self.assertEqual(job["counts"], {"accepted": 1, "review": 0, "rejected": 1, "total": 2})
        self.assertEqual(job["examples"][0]["candidate"]["template"], bad["template"])
        requests = [json.loads(line) for line in (self.root / "jobs" / job["id"] / "requests.jsonl").read_text().splitlines()]
        context = json.loads(requests[1]["prompt"].splitlines()[-1])
        self.assertEqual(context["failed_examples"][0]["template"], bad["template"])

    def test_schema_and_prompt_switch_only_when_numeric_mode_is_explicitly_enabled(self):
        """일반 생성 호환성을 유지하며 선택한 모드의 스키마만 전달한다."""
        client = PairedLlamaClient(self.config, threading.Event())
        command = client._build_command("test")
        self.assertIn("spoken_text", json.loads(command[command.index("--json-schema") + 1])["properties"]["examples"]["items"]["properties"])
        client.number_mode = True
        command = client._build_command("test")
        self.assertEqual(json.loads(command[command.index("--json-schema") + 1]), NUMBER_SCHEMA)
        self.assertIn("template", build_pair_prompt("숫자", 1, [], number_mode=True))
        self.assertFalse(validate_settings({"prompt": "일반 생성"})["number_mode"])
        with self.assertRaises(ValueError):
            validate_settings({"prompt": "숫자", "number_mode": "true"})

    def test_template_units_are_rendered_once_and_conflicting_units_still_fail(self):
        """실제 모델처럼 문장 틀에 단위를 써도 명·원을 한 번만 붙인다."""
        source = candidate("참석자는 {n0}명이고 등록비는 {n1}원입니다.", slot("count", "10", "명"), slot("money", "1250", "원"))
        generated = render_numeric_candidate(source)
        self.assertEqual(generated["spoken_text"], "참석자는 열 명이고 등록비는 천이백오십 원입니다.")
        self.assertEqual(generated["normalized_text"], "참석자는 10명이고 등록비는 1,250원입니다.")
        self.assertIsNone(numeric_alignment_issue(generated["spoken_text"], generated["normalized_text"]))
        source["numbers"][0]["unit"] = "분"
        with self.assertRaisesRegex(ValueError, "unit이 다릅니다"):
            render_numeric_candidate(source)

    def test_previous_phone_false_positive_can_be_reviewed_without_rewriting_saved_data_on_load(self):
        """이전 전화번호 오탐은 수정된 검사로 해제하되 사용자의 채택 결정은 유지한다."""
        source = candidate("연락처는 {n0}입니다.", slot("phone", "070-5555-1234", ""))
        service = self.service([[source]])
        job = self.finish(service)
        path = self.root / "jobs" / job["id"] / "job.json"
        saved = json.loads(path.read_text())
        saved["examples"][0].update(accepted=False, needs_review=True, reason="numeric_alignment", quality_issue="이전 오탐")
        path.write_text(json.dumps(saved, ensure_ascii=False))
        original = path.read_bytes()
        restored = DatasetJobService(self.config, self.root / "jobs")
        self.addCleanup(restored.close)
        row = restored.get_job(job["id"])["examples"][0]
        self.assertIsNone(row["quality_issue"])
        self.assertFalse(row["accepted"])
        self.assertEqual(path.read_bytes(), original)
        accepted = restored.update_example(job["id"], row["id"], {"accepted": True})
        self.assertTrue(accepted["examples"][0]["accepted"])


if __name__ == "__main__":
    unittest.main()
