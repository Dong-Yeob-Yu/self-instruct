"""한글 숫자 발화와 전사문 대응 검사의 고정 기대값 회귀 테스트."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_corpus.local import LlamaCppConfig
from asr_corpus.number_validation import numeric_alignment_issue, spoken_protected_terms
from asr_corpus.web_dataset import DatasetAccumulator, DatasetJobService


BAD_SPOKEN = "이 상품은 삼만 오천 원이고, 만 이십 일일에 오후 두 시부터 다섯 시간 동안 진행됩니다."
GOOD_SPOKEN = "이 상품은 삼만 오천 원이고, 십이월 이십일일에 오후 두 시부터 다섯 시간 동안 진행됩니다."
TRANSCRIPT = "이 상품은 35,000원이고, 12월 21일에 오후 2시부터 5시간 동안 진행됩니다."


class NumberAlignmentRegressionTest(unittest.TestCase):
    def test_date_units_distinguish_valid_double_il_and_preserve_particles(self):
        """날짜의 올바른 일일은 허용하고 단위 중복과 조사 추가·누락은 거부한다."""
        examples = [(1, "일일", "일일일"), (3, "삼일", "삼일일"), (10, "십일", "십일일"),
                    (11, "십일일", "십일일일"), (15, "십오일", "십오일일"),
                    (21, "이십일일", "이십일일일"), (25, "이십오일", "이십오일일"),
                    (31, "삼십일일", "삼십일일일")]
        for day, spoken_day, duplicated_day in examples:
            with self.subTest(day=day):
                self.assertIsNone(numeric_alignment_issue(f"십일월 {spoken_day} 오전입니다.", f"11월 {day}일 오전입니다."))
                self.assertIsNone(numeric_alignment_issue(f"십일월 {spoken_day}에 만나요.", f"11월 {day}일에 만나요."))
                self.assertIsNotNone(numeric_alignment_issue(f"십일월 {duplicated_day}에 만나요.", f"11월 {day}일에 만나요."))
                self.assertIsNotNone(numeric_alignment_issue(f"십일월 {spoken_day}에 만나요.", f"11월 {day}일 만나요."))
                self.assertIsNotNone(numeric_alignment_issue(f"십일월 {spoken_day} 만나요.", f"11월 {day}일에 만나요."))

    def test_reported_missing_month_is_not_accepted_as_matching_transcript(self):
        """사용자가 제보한 월 누락 예제를 숫자 대응 불일치로 판정한다."""
        self.assertIsNotNone(numeric_alignment_issue(BAD_SPOKEN, TRANSCRIPT))
        self.assertIsNone(numeric_alignment_issue(GOOD_SPOKEN, TRANSCRIPT))

    def test_supported_number_readings_keep_expected_values(self):
        """금액·날짜·시간·수량·소수·전화번호의 올바른 대응을 허용한다."""
        examples = [
            ("가격은 삼만 오천 원입니다.", "가격은 35,000원입니다."),
            ("일억 이천삼백사십오만 육천칠백팔십구 원입니다.", "123,456,789원입니다."),
            ("수수료는 일백 원입니다.", "수수료는 100원입니다."),
            ("십이월 이십일일에 만나요.", "12월 21일에 만나요."),
            ("유월 육일과 시월 십일일입니다.", "6월 6일과 10월 11일입니다."),
            ("오후 두 시부터 다섯 시간입니다.", "오후 2시부터 5시간입니다."),
            ("사과 마흔두 개와 배 스무 개입니다.", "사과 42개와 배 20개입니다."),
            ("열두 시에 와주세요.", "12시에 와주세요."),
            ("삼 점 일사입니다.", "3.14입니다."),
            ("영 점 영오입니다.", "0.05입니다."),
            ("기온은 마이너스 삼 점 오 도입니다.", "기온은 -3.5도입니다."),
            ("전화번호는 공일공 일이삼사 오육칠팔입니다.", "전화번호는 010-1234-5678입니다."),
            ("삼성 일산 지점에 세 개 보내세요.", "삼성 일산 지점에 3개 보내세요."),
            ("대표님의 전화번호는 02-777-1234이고, 개설일은 천구백이십사년 이월 오일입니다.", "대표님의 전화번호는 02-777-1234이고, 개설일은 1924년 2월 5일입니다."),
            ("010-5555-1234로 연락 주시고 예산은 이백오십만 원입니다. 회의실은 5층입니다.", "010-5555-1234로 연락 주시고 예산은 2,500,000원입니다. 회의실은 5층입니다."),
            ("가격은 칠백오십만 원이며 031-9999-8888로 문의하세요. 재고는 삼십 개입니다.", "가격은 7,500,000원이며 031-9999-8888로 문의하세요. 재고는 30개입니다."),
        ]
        for spoken, transcript in examples:
            with self.subTest(spoken=spoken):
                self.assertIsNone(numeric_alignment_issue(spoken, transcript))

    def test_changed_numeric_values_and_missing_units_require_review(self):
        """값·순서·자릿수·날짜 단위가 달라지면 검토 대상으로 분리한다."""
        examples = [
            ("사과 세 개 주세요.", "사과 4개 주세요."),
            ("십이 이십일일입니다.", "12월 21일입니다."),
            ("오후 두 시부터 다섯 시간입니다.", "오후 5시부터 2시간입니다."),
            ("삼 점 일사입니다.", "3.41입니다."),
            ("전화번호는 일공 일이삼사 오육칠팔입니다.", "전화번호는 010-1234-5678입니다."),
            ("배송은 세 개입니다.", "택배는 3개입니다."),
        ]
        for spoken, transcript in examples:
            with self.subTest(spoken=spoken):
                self.assertIsNotNone(numeric_alignment_issue(spoken, transcript))

    def test_unchanged_general_text_is_allowed(self):
        """숫자 변환이 없는 일반 발화는 그대로 허용한다."""
        self.assertIsNone(numeric_alignment_issue("오늘 날씨가 좋네요.", "오늘 날씨가 좋네요."))
        self.assertIsNone(numeric_alignment_issue("There are three apples.", "There are 3 apples."))

    def test_bad_pair_is_preserved_for_review_and_not_counted_as_accepted(self):
        """불일치 예제의 원문은 보존하고 자동 채택에서는 제외한다."""
        accumulator = DatasetAccumulator({"similarity_threshold": 0.95, "near_duplicate_policy": "drop"})
        row = accumulator.add({"spoken_text": BAD_SPOKEN, "normalized_text": TRANSCRIPT, "protected_terms": []})
        self.assertFalse(row["accepted"])
        self.assertTrue(row["needs_review"])
        self.assertEqual(row["reason"], "numeric_alignment")
        self.assertEqual(row["spoken_text"], BAD_SPOKEN)
        self.assertIsNotNone(row["quality_issue"])


class SpokenProtectedTermsRegressionTest(unittest.TestCase):
    def test_transcript_terms_are_mapped_to_original_spoken_amounts(self):
        """제보된 아라비아 숫자 보호어를 한글 발화의 띄어쓰기 그대로 보정한다."""
        spoken = "총 삼백오십 개 품목과 오천 오백 원짜리 물건이 있습니다."
        transcript = "총 350개 품목과 5,500원짜리 물건이 있습니다."
        self.assertEqual(spoken_protected_terms(spoken, transcript, ["350개", "5,500원"]), ["삼백오십 개", "오천 오백 원"])

    def test_spoken_names_and_whitespace_variants_are_preserved(self):
        """고유명사를 유지하고 보호어의 띄어쓰기만 발화 원문에 맞춘다."""
        spoken = "서울에서 삼백오십 개를 주문했습니다."
        self.assertEqual(spoken_protected_terms(spoken, "서울에서 350개를 주문했습니다.", ["서울", "삼백오십개"]), ["서울", "삼백오십 개"])

    def test_dates_native_numbers_decimals_and_phones_map_without_losing_values(self):
        """날짜 특례·고유어·소수·전화번호도 해당 발화 구간으로 보정한다."""
        cases = [
            ("유월 육일과 시월 십일일입니다.", "6월 6일과 10월 11일입니다.", ["6월 6일", "10월 11일"], ["유월 육일", "시월 십일일"]),
            ("유월입니다.", "6월입니다.", ["6"], ["유", "유월"]),
            ("오후 두 시부터 다섯 시간입니다.", "오후 2시부터 5시간입니다.", ["2시", "5시간"], ["두 시", "다섯 시간"]),
            ("기온은 마이너스 삼 점 오 도입니다.", "기온은 -3.5도입니다.", ["-3.5도"], ["마이너스 삼 점 오 도"]),
            ("공일공 일이삼사 오육칠팔로 전화하세요.", "010-1234-5678로 전화하세요.", ["010-1234-5678"], ["공일공 일이삼사 오육칠팔"]),
            ("010-1234-5678로 연락하고 세 개 주문하세요.", "010-1234-5678로 연락하고 3개 주문하세요.", ["010-1234-5678", "3개"], ["010-1234-5678", "세 개"]),
        ]
        for spoken, transcript, terms, expected in cases:
            with self.subTest(terms=terms):
                self.assertEqual(spoken_protected_terms(spoken, transcript, terms), expected)

    def test_repeated_numeric_value_keeps_each_actual_spoken_variant(self):
        """동일 숫자가 다른 읽기로 반복되면 각각의 발화 표현을 보호한다."""
        terms = spoken_protected_terms("세 개와 삼 개입니다.", "3개와 3개입니다.", ["3개"])
        self.assertEqual(terms, ["세 개", "삼 개"])

    def test_wrong_values_and_partial_digits_are_not_guessed(self):
        """숫자가 틀리거나 숫자의 일부만 지정된 경우 임의로 대응시키지 않는다."""
        self.assertEqual(spoken_protected_terms("배 두 개입니다.", "배 3개입니다.", ["3개"]), [])
        self.assertEqual(spoken_protected_terms("오천오백 원입니다.", "5,500원입니다.", ["500원"]), ["오천오백 원"])

    def test_empty_or_hallucinated_metadata_still_extracts_spoken_numeric_terms(self):
        """보호어가 누락되거나 잘못된 한글로 생성돼도 실제 숫자 발화를 직접 보호한다."""
        for terms in ([], ["일십오 개"], ["발화에 없는 장소"]):
            with self.subTest(terms=terms):
                self.assertEqual(spoken_protected_terms("십오 개입 세트입니다.", "15개입 세트입니다.", terms), ["십오 개입"])

    def test_actual_local_model_metadata_error_does_not_discard_valid_pair(self):
        """실제 로컬 모델의 일십오 개 보호어 오류가 정상 쌍을 탈락시키지 않는다."""
        accumulator = DatasetAccumulator({"similarity_threshold": 0.95, "near_duplicate_policy": "drop"})
        row = accumulator.add({"spoken_text": "이 상품은 오십만 원이고, 이천 이십사년 일월 일일에 십오 개입 세트입니다.",
                               "normalized_text": "이 상품은 500,000원이고, 2024년 1월 1일에 15개입 세트입니다.",
                               "protected_terms": ["오십만 원", "일십오 개"]})
        self.assertTrue(row["accepted"])
        self.assertEqual(row["protected_terms"], ["오십만 원", "이천 이십사년", "일월", "일일", "십오 개입"])

    def test_reported_pair_is_accepted_without_mutating_model_output(self):
        """보호어만 잘못된 정상 쌍은 원본 후보를 보존하면서 채택한다."""
        candidate = {"spoken_text": "총 삼백오십 개 품목과 오천 오백 원짜리 물건이 있습니다.",
                     "normalized_text": "총 350개 품목과 5,500원짜리 물건이 있습니다.", "protected_terms": ["350개", "5,500원"]}
        original = json.dumps(candidate, ensure_ascii=False)
        accumulator = DatasetAccumulator({"similarity_threshold": 0.95, "near_duplicate_policy": "drop"})
        row = accumulator.add(candidate)
        self.assertTrue(row["accepted"])
        self.assertEqual(row["protected_terms"], ["삼백오십 개", "오천 오백 원"])
        self.assertEqual(json.dumps(candidate, ensure_ascii=False), original)
        self.assertFalse(accumulator.add(candidate)["accepted"])

    def test_wrong_transcript_is_not_accepted_after_protected_term_correction(self):
        """보호어 보정이 실제 숫자·단위 불일치를 자동 채택하게 만들지 않는다."""
        accumulator = DatasetAccumulator({"similarity_threshold": 0.95, "near_duplicate_policy": "drop"})
        row = accumulator.add({"spoken_text": "배 두 개입니다.", "normalized_text": "배 3개입니다.", "protected_terms": ["3개"]})
        self.assertFalse(row["accepted"])
        self.assertTrue(row["needs_review"])
        self.assertEqual(row["reason"], "numeric_alignment")

    def test_unrelated_unit_error_is_reported_instead_of_protected_term_error(self):
        """단위가 실제로 다른 예제는 보호어 오류로 숨기지 않고 숫자 대응 검토로 표시한다."""
        accumulator = DatasetAccumulator({"similarity_threshold": 0.95, "near_duplicate_policy": "drop"})
        row = accumulator.add({"spoken_text": "가격은 만 오천 원이고 일 시간 동안 이용합니다.",
                               "normalized_text": "가격은 15,000원이고 1일 동안 이용합니다.", "protected_terms": ["15,000원"]})
        self.assertFalse(row["accepted"])
        self.assertTrue(row["needs_review"])
        self.assertEqual(row["reason"], "numeric_alignment")
        self.assertEqual(row["spoken_text"], "가격은 만 오천 원이고 일 시간 동안 이용합니다.")


class NumberAlignmentStorageRegressionTest(unittest.TestCase):
    def setUp(self):
        """기존 결과를 모사할 임시 폴더와 모의 로컬 클라이언트를 준비한다."""
        self.directory = tempfile.TemporaryDirectory(prefix="asr-number-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "test.gguf"
        self.model.write_bytes(b"GGUFtest")
        self.config = LlamaCppConfig(self.model, executable=sys.executable)
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("네트워크 금지")).start()
        self.addCleanup(patch.stopall)

    def create_old_job(self, reason="unique", accepted=True):
        """기존에 자동 채택됐던 불일치 예제를 저장한 파일을 만든다."""
        identifier = "a" * 32
        row = {"id": "b" * 32, "spoken_text": BAD_SPOKEN, "normalized_text": TRANSCRIPT, "protected_terms": [],
               "accepted": accepted, "needs_review": False, "reason": reason, "matched_id": None, "similarity": None}
        job = {"id": identifier, "title": "기존 결과", "settings": {"count": 1}, "status": "completed", "message": "완료",
               "created_at": "2026-10-27T00:00:00+00:00", "updated_at": "2026-10-27T00:00:00+00:00", "request_count": 1,
               "counts": {"accepted": 1, "review": 0, "rejected": 0, "total": 1}, "examples": [row]}
        target = self.root / "jobs" / identifier / "job.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        return target, identifier, row["id"]

    def test_existing_bad_pair_is_removed_from_export_without_rewriting_source_file(self):
        """기존 불일치 데이터도 내보내기에서 제외하되 저장 원본은 변경하지 않는다."""
        path, identifier, _ = self.create_old_job()
        original = path.read_bytes()
        service = DatasetJobService(self.config, self.root / "jobs")
        self.addCleanup(service.close)
        job = service.get_job(identifier)
        self.assertEqual(job["counts"], {"accepted": 0, "review": 1, "rejected": 0, "total": 1})
        self.assertEqual(service.export_jsonl(identifier), "")
        self.assertEqual(path.read_bytes(), original)

    def test_reviewer_can_fix_spoken_text_and_keep_mapping_id(self):
        """검토자가 발화문을 수정하면 ID를 유지하면서 올바른 쌍을 채택한다."""
        _, identifier, row_id = self.create_old_job()
        service = DatasetJobService(self.config, self.root / "jobs")
        self.addCleanup(service.close)
        job = service.update_example(identifier, row_id, {"accepted": True, "spoken_text": GOOD_SPOKEN, "normalized_text": TRANSCRIPT})
        self.assertEqual(job["counts"]["accepted"], 1)
        self.assertEqual(job["examples"][0]["id"], row_id)
        self.assertIsNone(job["examples"][0]["quality_issue"])
        self.assertEqual(json.loads(service.export_jsonl(identifier))["spoken_text"], GOOD_SPOKEN)

    def test_reviewer_cannot_accept_a_pair_that_still_has_missing_month(self):
        """수정하지 않은 잘못된 발화를 체크만 해서 채택할 수 없게 한다."""
        path, identifier, row_id = self.create_old_job()
        original = path.read_bytes()
        service = DatasetJobService(self.config, self.root / "jobs")
        self.addCleanup(service.close)
        with self.assertRaises(ValueError):
            service.update_example(identifier, row_id, {"accepted": True, "normalized_text": TRANSCRIPT})
        self.assertEqual(service.get_job(identifier)["counts"]["accepted"], 0)
        self.assertEqual(path.read_bytes(), original)

    def test_manually_excluded_pair_stays_excluded_after_restart(self):
        """사용자가 직접 제외한 예제는 다시 검토 상태로 되돌리지 않는다."""
        _, identifier, _ = self.create_old_job(reason="manual_review", accepted=False)
        service = DatasetJobService(self.config, self.root / "jobs")
        self.addCleanup(service.close)
        row = service.get_job(identifier)["examples"][0]
        self.assertFalse(row["accepted"])
        self.assertFalse(row["needs_review"])
        self.assertEqual(row["reason"], "manual_review")


if __name__ == "__main__":
    unittest.main()
