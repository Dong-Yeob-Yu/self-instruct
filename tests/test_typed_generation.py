"""유형별 읽기·할당량과 두 단계 LLM 생성의 회귀를 검증한다."""

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from asr_corpus.local import LlamaCppConfig
from asr_corpus.number_generation import read_number
from asr_corpus.typed_generation import (
    NUMBER_TYPES, normalize_number_template, normalize_money_template, validate_money_digit_weights, select_money_digits, generate_number, indexed_response, make_slots, reading_prompt,
    render_pair, select_types, type_quotas, validate_templates, validate_type_weights,
)
from asr_corpus.web_dataset import DatasetAccumulator, DatasetJobService, PairedLlamaClient, validate_settings


def scalar_number(kind, value):
    """값과 단위를 유지하는 네 유형의 검증용 숫자를 만든다."""
    unit = {"money": "원", "decimal": "", "interest_rate": "%", "ratio": "%"}[kind]
    written, expected = read_number({"kind": "money" if kind == "money" else "decimal", "value": value, "unit": "원" if kind == "money" else ""})
    aliases = []
    if unit == "%":
        written += "%"
        expected += " 퍼센트"
        aliases = [{"written": "%", "spoken": "퍼센트", "context": "unit"}]
    return {"id": "0", "type": kind, "context": kind, "unit": unit, "value": written, "expected": expected, "reading_aliases": aliases}


class TypedGenerationTest(unittest.TestCase):
    def test_weighted_quotas_preserve_target_and_retry_only_missing_types(self):
        """유형 비율은 목표 합계를 보존하고 재시도는 미달 유형만 채운다."""
        self.assertEqual(type_quotas({"money": 3, "count": 1}, 10), {"money": 8, "count": 2})
        rows = [{"type": "money", "accepted": True}] * 8
        rows += [{"type": "count", "accepted": False}]
        self.assertEqual(select_types({"money": 3, "count": 1}, 10, rows, 3), ["count", "count"])
        self.assertEqual(type_quotas({"money": 1, "count": 1}, 1), {"money": 1, "count": 0})

    def test_invalid_weights_and_template_numbers_are_rejected(self):
        """선택 누락과 잘못된 비율 및 본문에 직접 쓴 숫자는 거부한다."""
        for weights in ({}, {"unknown": 1}, {"money": True}, {"money": 0}, {"money": 1.5}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                validate_type_weights(weights)
        for text in ("금액은 123원 {number} 입니다.", "값은 {number}원입니다.", "{number} {number}", "숫자 없음"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_templates({"0": text})

    def test_same_number_is_read_according_to_type_and_context(self):
        """같은 삼도 수량은 세 개이고 순서는 세 번째이며 등수는 삼 등이다."""
        rng = Mock()
        rng.randint.return_value = 3
        cases = [("count", "quantity", "개", "3개", "세 개"),
                 ("ordinal", "order", "번째", "3번째", "세 번째"),
                 ("ordinal", "rank", "등", "3등", "삼 등")]
        for kind, context, unit, value, spoken in cases:
            number = generate_number({"id": "0", "type": kind, "context": context, "unit": unit}, rng)
            self.assertEqual((number["value"], number["expected"]), (value, spoken))
            candidate = render_pair(number, "안내 값은 {number} 입니다.", spoken)
            self.assertEqual(candidate["normalized_text"], f"안내 값은 {value} 입니다.")
            with self.assertRaises(ValueError):
                render_pair(number, "값은 {number} 입니다.", "삼")

    def test_money_digit_settings_validate_zero_exclusion_and_balanced_defaults(self):
        """금액 자릿수는 기본 균등 비율이고 0·잘못된 입력을 구분한다."""
        self.assertEqual(validate_money_digit_weights(None), {str(digits): 1 for digits in range(1, 10)})
        settings = validate_settings({"prompt": "금액", "type_weights": {"money": 1}, "money_digit_weights": {"1": 0, "2": 1, "9": 3}})
        self.assertEqual(settings["money_digit_weights"], {"2": 1, "9": 3})
        for invalid in ({}, {"1": 0}, {"0": 1}, {"10": 1}, {"2": -1}, {"2": True}, {"2": 1.5}, {"2": 10001}, []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_settings({"prompt": "금액", "type_weights": {"money": 1}, "money_digit_weights": invalid})

    def test_money_digit_quotas_fill_missing_accepted_digits_on_retry(self):
        """자릿수 배분은 목표를 보존하고 실패·검토 항목 대신 미달 자릿수를 재시도한다."""
        rows = [{"type": "money", "accepted": True, "generated_number": {"value": "12원"}},
                {"type": "money", "accepted": False, "generated_number": {"value": "15원"}},
                {"type": "money", "accepted": True, "needs_review": True, "generated_number": {"value": "16원"}},
                {"type": "money", "accepted": True, "generated_number": {"value": "200,054,023원"}}]
        digits = select_money_digits({"2": 1, "9": 3}, 8, rows, 6)
        self.assertEqual(digits.count(2), 1)
        self.assertEqual(digits.count(9), 5)
        self.assertEqual(select_money_digits({"1": 1, "9": 1}, 1, [], 1), [1])
        self.assertEqual(select_money_digits({"1": 1, "9": 1}, 0, [], 1), [])

    def test_money_digit_ranges_never_round_values_or_generate_leading_zeros(self):
        """각 자릿수 범위는 선행 영 없이 원 단위 정수를 생성한다."""
        rng = Mock()
        for digits in range(1, 10):
            low, high = 10 ** (digits - 1), 10 ** digits - 1
            for value in (low, high):
                rng.randint.return_value = value
                number = generate_number({"id": "0", "type": "money", "context": "amount", "unit": "원", "digits": digits}, rng)
                rng.randint.assert_called_with(low, high)
                self.assertEqual(number["value"], f"{value:,}원")
                pair = render_pair(number, "금액은 {number} 입니다.", number["expected"])
                self.assertEqual(pair["generated_number"]["digits"], digits)

    def test_money_preserves_internal_zeros_and_one_won_digits_up_to_nine_digits(self):
        """원 단위 난수의 중간 영과 억 단위 값을 반올림 없이 발화·정답에 보존한다."""
        slot = {"id": "0", "type": "money", "context": "amount", "unit": "원"}
        rng = Mock()
        for value, written, reading in ((200054023, "200,054,023원", "이억 오만 사천이십삼 원"),
                                        (100000001, "100,000,001원", "일억 일 원"),
                                        (55001, "55,001원", "오만 오천일 원"),
                                        (1, "1원", "일 원"),
                                        (999999999, "999,999,999원", "구억 구천구백구십구만 구천구백구십구 원")):
            with self.subTest(value=value):
                rng.randint.return_value = value
                number = generate_number(slot, rng)
                rng.randint.assert_called_with(1, 999999999)
                self.assertEqual(number["value"], written)
                self.assertEqual(number["expected"], reading)
                pair = render_pair(number, "안내 금액은 {number} 입니다.", reading)
                self.assertEqual(pair["normalized_text"], f"안내 금액은 {written} 입니다.")

    def test_new_money_jobs_do_not_repeat_the_fixed_model_seed_amount(self):
        """같은 모델 시드로 새 작업을 시작해도 동일한 금액 배열을 반복하지 않는다."""
        from asr_corpus.number_generation import read_number
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "test.gguf"
            model.write_bytes(b"GGUFtest")
            client = Mock()

            def generate(prompt):
                """실제 모델 대신 전달된 금액의 값과 읽기만 되돌려 준다."""
                context = json.loads(prompt.splitlines()[-1])
                if "slots" in context:
                    return json.dumps({"examples": [{"id": slot["id"], "template": "결제 후 {number} 이(가) 부족합니다."} for slot in context["slots"]]})
                self.assertTrue(all(number["template"] == "결제 후 {number} 이 부족합니다." for number in context["numbers"]))
                return json.dumps({"examples": [{"id": number["id"], "spoken": read_number({"kind": "money", "value": number["value"].removesuffix("원"), "unit": "원"})[1]} for number in context["numbers"]]})

            client.generate.side_effect = generate
            service = DatasetJobService(LlamaCppConfig(model, executable=sys.executable), root / "jobs", lambda config, cancellation: client)
            try:
                values = []
                with patch("subprocess.Popen", side_effect=AssertionError("실제 모델 실행 금지")):
                    for _ in range(2):
                        job = service.start_job({"prompt": "금액 안내", "count": 10, "batch_size": 1, "type_weights": {"money": 1}, "money_digit_weights": {"2": 1, "9": 1}})
                        service._threads[job["id"]].join(timeout=3)
                        job = service.get_job(job["id"])
                        self.assertEqual(job["status"], "completed", job["message"])
                        for row in job["examples"]:
                            if row["accepted"]:
                                self.assertTrue(row["spoken_text"].endswith("원이 부족합니다."), row)
                                self.assertTrue(row["normalized_text"].endswith("원이 부족합니다."), row)
                                self.assertNotIn("(가)", row["template"])
                        generated = [row["generated_number"] for row in job["examples"] if row["accepted"]]
                        self.assertEqual([number["digits"] for number in generated].count(2), 5)
                        self.assertEqual([number["digits"] for number in generated].count(9), 5)
                        self.assertEqual(job["settings"]["money_digit_weights"], {"2": 1, "9": 1})
                        values.append([number["value"] for number in generated])
                self.assertNotEqual(values[0], values[1])
            finally:
                service.close()

    def test_money_accepts_the_same_template_with_different_amounts_but_rejects_exact_duplicates(self):
        """금액만 다른 동일 문장 틀은 채택하고 발화·정답이 모두 같은 중복은 제외한다."""
        accumulator = DatasetAccumulator(validate_settings({"prompt": "금액 안내", "prompt_mode": True, "type_weights": {"money": 1}}))
        rng = Mock()
        rows = []
        for value in (200054023, 200054024, 55001):
            rng.randint.return_value = value
            number = generate_number({"id": "0", "type": "money", "context": "amount", "unit": "원"}, rng)
            candidate = render_pair(number, "결제 금액은 {number} 입니다.", number["expected"])
            rows.append(accumulator.add(candidate))
        self.assertTrue(all(row["accepted"] and not row["needs_review"] for row in rows), rows)
        duplicate = accumulator.add(candidate)
        self.assertFalse(duplicate["accepted"])
        self.assertEqual(duplicate["reason"], "exact_duplicate")
        invalid = accumulator.add({**candidate, "normalized_text": "결제 금액은 55,002원 입니다."})
        self.assertFalse(invalid["accepted"])
        self.assertTrue(invalid["needs_review"])
        self.assertEqual(invalid["reason"], "transcript_conflict")

    def test_small_money_values_can_repeat_in_distinct_sentences(self):
        """작은 자릿수의 값이 같아도 서로 다른 발화이면 채택할 수 있다."""
        accumulator = DatasetAccumulator(validate_settings({"prompt": "한 자리 금액", "prompt_mode": True, "type_weights": {"money": 1}}))
        number = {"id": "0", "type": "money", "context": "amount", "unit": "원", "digits": 1, "value": "5원", "expected": "오 원", "reading_aliases": []}
        for template in ("남아 있는 잔액은 {number} 입니다.", "추가 요금 {number} 결제를 진행해 주세요."):
            row = accumulator.add(render_pair(number, template, "오 원"))
            self.assertTrue(row["accepted"] and not row["needs_review"], row)

    def test_money_particles_are_corrected_in_both_spoken_text_and_answer(self):
        """원 뒤의 잘못된 조사·괄호 선택형을 양쪽 문장에 동일하게 교정한다."""
        number = {"id": "0", "type": "money", "context": "amount", "unit": "원", "value": "53원", "expected": "오십삼 원", "reading_aliases": []}
        cases = [("이(가)", "이"), ("가(이)", "이"), ("이 / 가", "이"), ("가", "이"),
                 ("을(를)", "을"), ("를", "을"), ("은(는)", "은"), ("는", "은"),
                 ("와(과)", "과"), ("와", "과"), ("로", "으로"), ("(으)로", "으로"),
                 ("도", "도"), ("만큼", "만큼")]
        for original, correct in cases:
            with self.subTest(particle=original):
                template = normalize_money_template("잔액 {number} " + original + " 부족합니다.")
                validate_templates({"0": template})
                self.assertEqual(normalize_money_template(template), template)
                row = render_pair(number, template, "오십삼 원")
                self.assertEqual(row["spoken_text"], f"잔액 오십삼 원{correct} 부족합니다.")
                self.assertEqual(row["normalized_text"], f"잔액 53원{correct} 부족합니다.")
                self.assertEqual(row["template"], f"잔액 {{number}}{correct} 부족합니다.")
                accumulator = DatasetAccumulator(validate_settings({"prompt": "조사 검증", "prompt_mode": True, "type_weights": {"money": 1}}))
                accepted = accumulator.add(row)
                self.assertTrue(accepted["accepted"] and not accepted["needs_review"], accepted)
        self.assertEqual(normalize_money_template("{number}가 부족합니다."), "{number} 이 부족합니다.")
        for suffix in ("이상 필요합니다.", "이하입니다.", "가량 필요합니다.", "입니다.", "가게에서 사용했습니다."):
            template = "{number} " + suffix
            self.assertEqual(normalize_money_template(template, attached=True), template)

    def test_temperature_reads_celsius_as_do_and_keeps_numeric_precision_and_unit(self):
        """온도는 도로 읽고 영하·소수 자릿수·℃ 정답과 단위 대응을 유지한다."""
        slot = {"id": "0", "type": "temperature", "context": "celsius", "unit": "℃"}
        for values, written, spoken in (([36, 50], "36.50℃", "삼십육점오영 도"),
                                        ([-3, 50], "-3.50℃", "영하 삼점오영 도"),
                                        ([0, 0], "0.00℃", "영점영영 도")):
            with self.subTest(values=values):
                rng = Mock()
                rng.randint.side_effect = values
                number = generate_number(slot, rng)
                self.assertEqual((number["value"], number["expected"]), (written, spoken))
                self.assertEqual(number["reading_aliases"], [{"written": "℃", "spoken": "도", "context": "unit"}])
                pair = render_pair(number, "측정한 온도는 {number} 입니다.", spoken)
                self.assertEqual(pair["normalized_text"], f"측정한 온도는 {written} 입니다.")
                accumulator = DatasetAccumulator(validate_settings({"prompt": "온도 안내", "prompt_mode": True, "type_weights": {"temperature": 1}}))
                row = accumulator.add(pair)
                self.assertTrue(row["accepted"] and not row["needs_review"], row)
                with self.assertRaises(ValueError):
                    render_pair(number, "측정한 온도는 {number} 입니다.", spoken + "씨")
        prompt = reading_prompt([number], {"0": "측정한 온도는 {number} 입니다."})
        self.assertIn("℃는 도로 읽습니다", prompt)

    def test_phone_hyphen_separators_preserve_all_digits_and_leading_zeros(self):
        """전화번호 한글 읽기의 하이픈은 공백으로 바꾸고 숫자 누락·변조는 거부한다."""
        rng = Mock()
        rng.randint.side_effect = [409, 4506]
        number = generate_number({"id": "0", "type": "phone", "context": "phone", "unit": ""}, rng)
        for reading in ("공일공-공사공구-사오공육", "공일공 - 공사공구 - 사오공육", "영일영-영사영구-사오영육"):
            pair = render_pair(number, "연락처는 {number} 입니다.", reading)
            self.assertEqual(pair["normalized_text"], "연락처는 010-0409-4506 입니다.")
            self.assertNotIn("-", pair["spoken_text"])
            row = DatasetAccumulator(validate_settings({"prompt": "전화번호", "prompt_mode": True, "type_weights": {"phone": 1}})).add(pair)
            self.assertTrue(row["accepted"] and not row["needs_review"], row)
        for reading in ("공일공-사공구-사오공육", "공일공-공사공구-사오공칠", "010-0409-4506", "공일공\n공사공구-사오공육"):
            with self.subTest(reading=reading), self.assertRaises(ValueError):
                render_pair(number, "연락처는 {number} 입니다.", reading)

    def test_adverb_hanbeon_is_not_mistaken_for_an_extra_numeric_slot(self):
        """한번 확인 같은 부사는 허용하고 명시적 횟수·수량은 계속 차단한다."""
        for text in ("{number} 로 한번 연락해 주세요.", "연락처는 {number} 입니다. 한번 확인해 주세요."):
            validate_templates({"0": text})
        for text in ("{number} 로 한 번 연락해 주세요.", "{number} 로 두 번 연락해 주세요.", "{number} 로 한번만 연락해 주세요.", "한 명이 {number} 로 연락했습니다."):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "미변환 한글 수치"):
                validate_templates({"0": text})

    def test_numeric_readings_follow_canonical_zero_and_sino_rules(self):
        """소수 영과 십·백·천 앞의 일 생략을 적용한 표준 읽기를 채택한다."""
        cases = [("decimal", "45.09", "사십오 점 영구"), ("decimal", "0.00", "영점영영"),
                 ("interest_rate", "12.20", "십이 점 이영 퍼센트"),
                 ("ratio", "79.20", "칠십구 점 이영 퍼센트"),
                 ("money", "51767", "오만 천칠백육십칠 원"),
                 ("money", "10010", "만 십 원"), ("money", "110", "백십 원")]
        for kind, value, reading in cases:
            with self.subTest(kind=kind, value=value):
                number = scalar_number(kind, value)
                pair = render_pair(number, "안내 값은 {number} 입니다.", reading)
                accumulator = DatasetAccumulator(validate_settings({"prompt": "숫자", "prompt_mode": True, "type_weights": {kind: 1}}))
                row = accumulator.add(pair)
                self.assertTrue(row["accepted"] and not row["needs_review"], row)
                self.assertEqual(pair["normalized_text"], "안내 값은 " + number["value"] + " 입니다.")

    def test_numeric_reading_variants_are_rejected_by_canonical_dataset_policy(self):
        """값이 같아도 소수 공·일천·일백처럼 생성 정책과 다른 표현을 거부한다."""
        cases = [("decimal", "45.09", "사십오 점 공구"), ("decimal", "0.00", "공점공영"),
                 ("interest_rate", "12.20", "십이 점 이공 퍼센트"),
                 ("ratio", "79.20", "칠십구 점 이공 퍼센트"),
                 ("money", "51767", "오만 일천칠백육십칠 원"),
                 ("money", "110", "일백일십 원")]
        for kind, value, reading in cases:
            with self.subTest(kind=kind, value=value), self.assertRaises(ValueError):
                render_pair(scalar_number(kind, value), "안내 값은 {number} 입니다.", reading)
        prompt = reading_prompt([scalar_number("decimal", "45.09")], {"0": "값은 {number} 입니다."})
        self.assertIn("0은 반드시 영으로", prompt)
        self.assertIn("십·백·천 앞에서는 일을 생략", prompt)

    def test_numeric_reading_variants_do_not_allow_wrong_values_or_digitwise_integers(self):
        """정수부 자릿수 읽기·잘못된 금액·소수 영 누락·단위 누락을 거부한다."""
        cases = [("decimal", "60.27", "육공 점 이칠"), ("decimal", "69.18", "육구 점 일팔"),
                 ("ratio", "54.08", "오사 점 공팔"), ("money", "80176", "팔만 천칠십육 원"),
                 ("money", "50", "쉰 원"), ("interest_rate", "12.20", "십이 점 이 퍼센트"),
                 ("decimal", "45.09", "사십오 점 구"), ("ratio", "12.20", "십이 점 이공")]
        for kind, value, reading in cases:
            with self.subTest(kind=kind, value=value), self.assertRaises(ValueError):
                render_pair(scalar_number(kind, value), "안내 값은 {number} 입니다.", reading)

    def test_four_numeric_types_accept_changed_values_in_the_same_template(self):
        """네 유형은 숫자만 다른 문장을 채택하고 완전히 같은 중복은 제외한다."""
        for kind in ("money", "decimal", "interest_rate", "ratio"):
            with self.subTest(kind=kind):
                accumulator = DatasetAccumulator(validate_settings({"prompt": "숫자", "prompt_mode": True, "type_weights": {kind: 1}}))
                values = ("51767", "51768") if kind == "money" else ("12.20", "12.21")
                for value in values:
                    number = scalar_number(kind, value)
                    candidate = render_pair(number, "안내 값은 {number} 입니다.", number["expected"])
                    row = accumulator.add(candidate)
                    self.assertTrue(row["accepted"] and not row["needs_review"], row)
                self.assertEqual(accumulator.add(candidate)["reason"], "exact_duplicate")

    def test_numeric_templates_correct_attached_particles_using_actual_unit_and_rieul_exception(self):
        """퍼센트와 소수의 조사는 마지막 발음과 ㄹ 예외에 맞춰 양쪽에 붙인다."""
        cases = [("interest_rate", "12.20", "{number}로 결정되었습니다.", "12.20%로 결정되었습니다."),
                 ("ratio", "79.20", "{number}을 달성했습니다.", "79.20%를 달성했습니다."),
                 ("decimal", "1.27", "{number} 으로 확인했습니다.", "1.27로 확인했습니다."),
                 ("decimal", "1.26", "{number} 로 확인했습니다.", "1.26으로 확인했습니다."),
                 ("decimal", "1.25", "{number} 이(가) 나왔습니다.", "1.25가 나왔습니다."),
                 ("money", "53", "{number}를 지불했습니다.", "53원을 지불했습니다.")]
        for kind, value, template, expected in cases:
            with self.subTest(kind=kind, value=value):
                number = scalar_number(kind, value)
                normalized = normalize_number_template(template, number)
                validate_templates({"0": normalized})
                pair = render_pair(number, normalized, number["expected"])
                self.assertEqual(pair["normalized_text"], expected)
                row = DatasetAccumulator(validate_settings({"prompt": "조사", "prompt_mode": True, "type_weights": {kind: 1}})).add(pair)
                self.assertTrue(row["accepted"] and not row["needs_review"], row)

    def test_all_types_and_first_ordinal_pass_pair_alignment(self):
        """열 유형과 첫 번째의 발화·정답 대응을 검증한다."""
        for slot in make_slots(list(NUMBER_TYPES), random.Random(42)):
            number = generate_number(slot, random.Random(42))
            candidate = render_pair(number, "안내 값은 {number} 입니다.", number["expected"])
            row = DatasetAccumulator(validate_settings({"prompt": "테스트"})).add(candidate)
            self.assertTrue(row["accepted"], row)
        rng = Mock()
        rng.randint.return_value = 1
        number = generate_number({"id": "0", "type": "ordinal", "context": "order", "unit": "번째"}, rng)
        self.assertEqual(number["expected"], "첫 번째")
        row = DatasetAccumulator(validate_settings({"prompt": "테스트"})).add(render_pair(number, "순서는 {number} 입니다.", "첫 번째"))
        self.assertTrue(row["accepted"], row)

    def test_calendar_and_time_context_boundaries(self):
        """윤년 날짜와 시각·소요 시간은 유효한 값과 서로 다른 단위를 사용한다."""
        rng = Mock()
        rng.randint.side_effect = [2024, 2, 29]
        number = generate_number({"id": "0", "type": "date", "context": "calendar", "unit": "년 월 일"}, rng)
        self.assertEqual(number["value"], "2024년 2월 29일")
        self.assertEqual(rng.randint.call_args.args, (1, 29))
        for context, unit, expected in (("clock", "시 분", "3시 0분"), ("duration", "시간 분", "3시간 0분")):
            rng.randint.side_effect = [3, 0]
            number = generate_number({"id": "0", "type": "time", "context": context, "unit": unit}, rng)
            self.assertEqual(number["value"], expected)

    def test_response_ids_and_changed_readings_are_rejected(self):
        """중복·누락 id와 값이 바뀐 한글 읽기는 채택하지 않는다."""
        slots = [{"id": "0"}, {"id": "1"}]
        for rows in ([], [{"id": "0", "spoken": "삼"}] * 2, [{"id": "0", "spoken": "삼"}, {"id": "2", "spoken": "사"}]):
            with self.assertRaises(ValueError):
                indexed_response(json.dumps({"examples": rows}), slots, "spoken")

    def test_new_jobs_do_not_repeat_numeric_sequences_for_non_money_types(self):
        """모델 시드와 유형 설정이 같아도 새 작업의 아홉 유형 숫자 배열은 달라진다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "test.gguf"
            model.write_bytes(b"GGUFtest")
            client = Mock()
            client.generate.return_value = '{"examples":[]}'
            config = LlamaCppConfig(model, executable=sys.executable)
            service = DatasetJobService(config, root / "jobs", lambda config, cancellation: client)
            weights = {kind: 1 for kind in NUMBER_TYPES if kind != "money"}
            self.addCleanup(service.close)
            sequences = []
            with patch("subprocess.Popen", side_effect=AssertionError("실제 모델 실행 금지")):
                with patch("asr_corpus.web_dataset.uuid.uuid4", side_effect=[Mock(hex="1" * 32), Mock(hex="2" * 32)]):
                    for _ in range(2):
                        job = service.start_job({"prompt": "난수 확인", "count": 9, "batch_size": 10, "type_weights": weights})
                        service._threads[job["id"]].join(timeout=3)
                        job = service.get_job(job["id"])
                        self.assertEqual(job["status"], "partial")
                        values = {kind: [] for kind in weights}
                        for batch in job["generation_batches"]:
                            for item in batch["examples"]:
                                values[item["type"]].append(item["value"])
                        sequences.append(values)
                        requests = [json.loads(line) for line in (root / "jobs" / job["id"] / "requests.jsonl").read_text().splitlines()]
                        self.assertEqual([record["seed"] for record in requests], [42, 43, 44, 45, 46])
            for kind in weights:
                with self.subTest(kind=kind):
                    self.assertEqual(len(sequences[0][kind]), 5)
                    self.assertNotEqual(sequences[0][kind], sequences[1][kind])
            self.assertEqual(config.seed, 42)

    def test_phone_pipeline_preserves_generated_value_type_and_export(self):
        """템플릿 다음에 만든 전화번호를 LLM 읽기와 매핑하고 유형까지 저장한다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "test.gguf"
            model.write_bytes(b"GGUFtest")
            client = Mock()
            def generate(prompt):
                """한글 읽기 요청 시점에 중간 결과가 이미 저장됐는지 확인한다."""
                context = json.loads(prompt.splitlines()[-1])
                if "slots" in context:
                    return json.dumps({"examples": [{"id": "0", "template": "연락처는 {number} 입니다. 한번 확인해 주세요."}]})
                snapshot = next(iter(service._jobs.values()))
                batch = service.get_job(snapshot["id"])["generation_batches"][0]
                self.assertEqual(batch["phase"], "spoken")
                self.assertEqual(batch["examples"][0]["template"], "연락처는 {number} 입니다. 한번 확인해 주세요.")
                self.assertEqual(batch["examples"][0]["value"], context["numbers"][0]["value"])
                self.assertNotIn("expected", batch["examples"][0])
                self.assertNotIn("spoken", batch["examples"][0])
                spoken = read_number({"kind": "phone", "value": context["numbers"][0]["value"], "unit": ""})[1]
                return json.dumps({"examples": [{"id": "0", "spoken": spoken.replace(" ", "-")}]})
            client.generate.side_effect = generate
            config = LlamaCppConfig(model, executable=sys.executable)
            service = DatasetJobService(config, root / "jobs", lambda config, cancellation: client)
            try:
                with patch("subprocess.Popen", side_effect=AssertionError("실제 모델 실행 금지")):
                    job = service.start_job({"prompt": "연락처 안내", "count": 1, "prompt_mode": True, "type_weights": {"phone": 1}})
                    service._threads[job["id"]].join(timeout=3)
                job = service.get_job(job["id"])
                self.assertEqual(job["status"], "completed", job["message"])
                batch = job["generation_batches"][0]
                self.assertEqual(batch["phase"], "completed")
                phone = batch["examples"][0]["value"]
                spoken = read_number({"kind": "phone", "value": phone, "unit": ""})[1]
                self.assertEqual(batch["examples"][0]["spoken"], spoken.replace(" ", "-"))
                self.assertRegex(phone, r"^010-[0-9]{4}-[0-9]{4}$")
                self.assertTrue(batch["examples"][0]["accepted"])
                row = job["examples"][0]
                self.assertEqual(row["normalized_text"], f"연락처는 {phone} 입니다. 한번 확인해 주세요.")
                self.assertEqual(row["type"], "phone")
                exported = json.loads(service.export_jsonl(job["id"]))
                self.assertEqual(exported["generated_number"]["value"], phone)
                self.assertEqual(exported["type"], "phone")
                requests = [json.loads(line) for line in (root / "jobs" / job["id"] / "requests.jsonl").read_text().splitlines()]
                self.assertEqual([request["phase"] for request in requests], ["template", "spoken"])
                self.assertIn(phone, requests[0]["prompt"])
                self.assertIn(phone, requests[1]["prompt"])
                self.assertNotIn('"expected"', requests[1]["prompt"])
            finally:
                service.close()

    def test_failed_template_batches_preserve_error_and_stage_history(self):
        """잘못된 템플릿이 반복돼도 배치별 오류와 실패 단계를 남긴다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "test.gguf"
            model.write_bytes(b"GGUFtest")
            client = Mock()
            client.generate.return_value = json.dumps({"examples": [{"id": "0", "template": "자리표시자가 없습니다."}]})
            service = DatasetJobService(LlamaCppConfig(model, executable=sys.executable), root / "jobs", lambda config, cancellation: client)
            try:
                with patch("subprocess.Popen", side_effect=AssertionError("실제 모델 실행 금지")):
                    job = service.start_job({"prompt": "연락처 안내", "count": 1, "type_weights": {"phone": 1}})
                    service._threads[job["id"]].join(timeout=3)
                job = service.get_job(job["id"])
                self.assertEqual(job["status"], "partial")
                self.assertEqual(len(job["generation_batches"]), 5)
                for batch in job["generation_batches"]:
                    self.assertEqual(batch["phase"], "failed")
                    self.assertEqual(batch["failed_phase"], "template")
                    self.assertIn("{number}", batch["error"])
                persisted = json.loads((root / "jobs" / job["id"] / "job.json").read_text())
                self.assertEqual(persisted["generation_batches"], job["generation_batches"])
            finally:
                service.close()

    def test_cancelled_reading_preserves_generated_template_and_number(self):
        """읽기 단계에서 중지해도 생성된 템플릿과 숫자가 작업에 남는다."""
        from asr_corpus.web_dataset import GenerationCancelledError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "test.gguf"
            model.write_bytes(b"GGUFtest")
            client = Mock()
            client.generate.side_effect = [json.dumps({"examples": [{"id": "0", "template": "연락처는 {number} 입니다."}]}), GenerationCancelledError()]
            service = DatasetJobService(LlamaCppConfig(model, executable=sys.executable), root / "jobs", lambda config, cancellation: client)
            try:
                with patch("subprocess.Popen", side_effect=AssertionError("실제 모델 실행 금지")):
                    job = service.start_job({"prompt": "연락처 안내", "count": 1, "type_weights": {"phone": 1}})
                    service._threads[job["id"]].join(timeout=3)
                job = service.get_job(job["id"])
                self.assertEqual(job["status"], "cancelled")
                batch = job["generation_batches"][0]
                self.assertEqual((batch["phase"], batch["status"]), ("spoken", "cancelled"))
                self.assertRegex(batch["examples"][0]["value"], r"^010-[0-9]{4}-[0-9]{4}$")
                self.assertEqual(batch["examples"][0]["template"], "연락처는 {number} 입니다.")
            finally:
                service.close()

    def test_wrong_llm_reading_is_retained_for_review_and_does_not_fill_quota(self):
        """LLM이 세 개를 삼 개로 읽으면 원문은 보존하고 수량 할당량은 채우지 않는다."""
        accumulator = DatasetAccumulator(validate_settings({"prompt": "수량 안내"}))
        number = {"id": "0", "type": "count", "context": "quantity", "unit": "개",
                  "value": "3개", "expected": "세 개", "reading_aliases": []}
        service = object.__new__(DatasetJobService)
        service._add_typed_candidates(accumulator, [number], {"0": "수량은 {number} 입니다."}, {"0": "삼 개"})
        row = accumulator.rows[0]
        self.assertFalse(row["accepted"])
        self.assertIn("일치하지 않습니다", accumulator.last_batch_issue)
        self.assertIn("삼 개", row["candidate"]["spoken_text"])
        self.assertEqual(select_types({"count": 1}, 1, accumulator.rows, 1), ["count"])

    def test_typed_schema_is_separate_from_legacy_modes(self):
        """유형 생성 단계에는 레거시 parts 대신 템플릿·읽기 스키마를 적용한다."""
        client = object.__new__(PairedLlamaClient)
        for phase in ("template", "spoken"):
            client.typed_phase = phase
            schema = client._response_schema("테스트")
            self.assertEqual(schema["properties"]["examples"]["items"]["required"], ["id", phase])
            schema = client._response_schema(json.dumps({"slots": [{"id": "0"}, {"id": "1"}]}))
            self.assertEqual(schema["properties"]["examples"]["maxItems"], 2)
            self.assertEqual(schema["properties"]["examples"]["items"]["properties"]["id"]["enum"], ["0", "1"])
