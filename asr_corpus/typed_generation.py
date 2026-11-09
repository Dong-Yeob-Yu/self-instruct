"""유형별 할당량과 코드 생성 숫자를 LLM 템플릿·한글 읽기에 연결한다."""

import calendar
import json
import re
from fractions import Fraction

from .composition import _literal, _validate_literal_quantities
from .number_generation import read_number, sino_reading, _native_reading


NUMBER_TYPES = {
    "money": "금액", "interest_rate": "금리", "temperature": "온도",
    "date": "날짜", "time": "시간", "phone": "전화번호", "decimal": "소수",
    "count": "수량", "ordinal": "순서", "ratio": "비율",
}


NUMERIC_VARIATION_TYPES = frozenset({"money", "decimal", "interest_rate", "ratio"})

def validate_type_weights(weights):
    """선택 유형과 양의 정수 가중치를 검증한다."""
    if not isinstance(weights, dict) or not weights or set(weights) - NUMBER_TYPES.keys():
        raise ValueError("숫자 유형을 하나 이상 선택하세요.")
    if any(type(value) is not int or not 1 <= value <= 10000 for value in weights.values()):
        raise ValueError("유형별 비율은 1~10,000 사이의 정수로 입력하세요.")
    return dict(weights)


def validate_money_digit_weights(weights):
    """금액의 1~9자리 가중치를 검증하고 0인 자릿수는 제외한다."""
    if weights is None:
        return {str(digits): 1 for digits in range(1, 10)}
    if not isinstance(weights, dict) or not weights or set(weights) - {str(digits) for digits in range(1, 10)}:
        raise ValueError("금액 자릿수는 1~9자리 중 선택하세요.")
    if any(type(value) is not int or not 0 <= value <= 10000 for value in weights.values()):
        raise ValueError("금액 자릿수 비율은 0~10,000 사이의 정수여야 합니다.")
    selected = {str(digits): weights[str(digits)] for digits in range(1, 10) if weights.get(str(digits), 0) > 0}
    if not selected:
        raise ValueError("금액 자릿수 비율을 하나 이상 1 이상으로 입력하세요.")
    return selected


def select_money_digits(weights, target, rows, count):
    """금액 채택 목표를 자릿수 비율로 배분하고 부족한 자릿수를 채운다."""
    previous = []
    for row in rows:
        if row.get("type") != "money" or not row.get("accepted") or row.get("needs_review") or row.get("quality_issue"):
            continue
        value = str(row.get("generated_number", {}).get("value", "")).replace(",", "").removesuffix("원")
        if re.fullmatch(r"[1-9][0-9]{0,8}", value):
            previous.append({"type": str(len(value)), "accepted": True})
    return [int(digits) for digits in select_types(weights, target, previous, count)]


def type_quotas(weights, count):
    """최대 나머지 방식으로 합계가 목표 개수인 유형별 할당량을 만든다."""
    total = sum(weights.values())
    shares = {kind: Fraction(count * weight, total) for kind, weight in weights.items()}
    quotas = {kind: int(share) for kind, share in shares.items()}
    ordered = sorted(shares, key=lambda kind: shares[kind] - quotas[kind], reverse=True)
    for kind in ordered[:count - sum(quotas.values())]:
        quotas[kind] += 1
    return quotas


def select_types(weights, target, rows, count):
    """채택된 유형별 개수를 제외한 부족분에서 이번 배치를 선택한다."""
    quotas = type_quotas(weights, target)
    for row in rows:
        if row["accepted"] and row.get("type") in quotas:
            quotas[row["type"]] -= 1
    selected = []
    while len(selected) < count and max(quotas.values(), default=0) > 0:
        kind = max(quotas, key=quotas.get)
        selected.append(kind)
        quotas[kind] -= 1
    return selected


def make_slots(types, rng):
    """숫자 생성 전에 템플릿에 필요한 유형·세부 문맥·단위를 정한다."""
    choices = {
        "money": [("amount", "원")], "interest_rate": [("interest", "%")],
        "temperature": [("celsius", "℃")], "date": [("calendar", "년 월 일")],
        "time": [("clock", "시 분"), ("duration", "시간 분")],
        "phone": [("phone", "")], "decimal": [("measure", "")],
        "count": [("quantity", unit) for unit in ("개", "명", "권", "잔")],
        "ordinal": [("order", "번째"), ("rank", "등")], "ratio": [("percentage", "%")],
    }
    slots = []
    for index, kind in enumerate(types):
        context, unit = rng.choice(choices[kind])
        slots.append({"id": str(index), "type": kind, "context": context, "unit": unit})
    return slots


def response_schema(field):
    """단계별로 id와 템플릿 또는 한글 읽기만 반환하게 제한한다."""
    properties = {"id": {"type": "string"}, field: {"type": "string", "minLength": 1, "maxLength": 1000}}
    return {"type": "object", "properties": {"examples": {"type": "array", "items": {
        "type": "object", "properties": properties, "required": list(properties), "additionalProperties": False,
    }}}, "required": ["examples"], "additionalProperties": False}


def template_prompt(settings, slots, previous, feedback):
    """숫자나 읽기를 생성하지 않는 유형별 문장 틀을 요청한다."""
    context = {"user_prompt": settings["prompt"], "slots": slots, "previous_examples": previous, "feedback": feedback}
    return (
        "한국어 ASR 문장 템플릿을 slots마다 하나씩 작성하세요. id를 그대로 반환하세요. "
        "숫자와 단위를 포함한 전체 표현을 {number} 하나로 치환하세요. 단위와 수치·한글 숫자는 본문에 쓰지 마세요. "
        "type과 context에 맞는 상황을 쓰세요. clock은 시각, duration은 소요 시간, order는 순번, rank는 등수입니다. "
        "자리표시자 뒤에는 공백이나 문장부호를 쓰세요. money의 단위 원은 받침이 있으므로 조사 이·은·을·과·으로를 쓰세요. "
        "이(가), 을(를), 이/가처럼 조사를 선택형으로 쓰지 말고 하나로 확정하세요. 다른 유형에서 조사가 필요하면 문장을 바꾸세요. "
        "money, decimal, interest_rate, ratio는 값이 다르면 같은 문장 틀을 사용해도 됩니다. 다른 유형은 이전 문장의 숫자만 바꾸지 말고 다른 상황과 목적을 쓰세요. "
        "한 개, 두 번처럼 자리표시자 밖에 추가 수량을 넣지 마세요. "
        "JSON examples에 id, template만 반환하세요.\n"
    ) + json.dumps(context, ensure_ascii=False)


def indexed_response(response, slots, field):
    """누락·중복·추가 id 및 잘못된 응답 자료형을 거부한다."""
    value = json.loads(response)
    rows = value.get("examples") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != len(slots):
        raise ValueError("요청한 모든 id에 대해 응답해야 합니다.")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get(field), str):
            raise ValueError("id와 결과는 문자열이어야 합니다.")
        if row["id"] in result:
            raise ValueError("응답 id가 중복되었습니다.")
        result[row["id"]] = row[field]
    if set(result) != {slot["id"] for slot in slots}:
        raise ValueError("응답 id가 요청과 다릅니다.")
    return result


def normalize_money_template(template, attached=False):
    """원 단위 금액 뒤의 조사를 보정하며 기존 호출 형식을 유지한다."""
    return _normalize_particle_template(template, "원", attached)


def normalize_number_template(template, number, attached=False):
    """네 숫자 유형의 마지막 발음에 맞춰 조사와 자리표시자 경계를 보정한다."""
    if number["type"] not in NUMERIC_VARIATION_TYPES:
        return template
    return _normalize_particle_template(template, number["expected"].rstrip()[-1], attached)


def _normalize_particle_template(template, last, attached):
    """한글 종성과 ㄹ 예외를 기준으로 조사 선택형과 공백을 정한다."""
    ending = (ord(last) - 0xAC00) % 28
    pairs = {key: key for key in ("도", "만", "까지", "부터", "보다", "씩", "만큼", "의", "에", "에서", "에게")}
    for first, second in (("이", "가"), ("은", "는"), ("을", "를"), ("과", "와"), ("으로", "로"), ("이랑", "랑"), ("이나", "나")):
        correct = first if ending and not (first == "으로" and ending == 8) else second
        pairs[first] = pairs[second] = correct
        for original in (f"{first}({second})", f"{second}({first})", f"{first}/{second}", f"{second}/{first}"):
            pairs[original] = correct
    pairs["(으)로"] = pairs["으로"]
    alternatives = [r"\s*".join(re.escape(character) for character in token) for token in sorted(pairs, key=len, reverse=True)]
    pattern = r"\{number\}\s*(" + "|".join(alternatives) + r")(?=\s|[.,!?]|$)"
    return re.sub(pattern, lambda match: "{number}" + ("" if attached else " ") + pairs[re.sub(r"\s+", "", match.group(1))], template, count=1)


def validate_templates(templates):
    """코드 생성 표현 외의 수치와 잘못된 자리표시자를 차단한다."""
    for template in templates.values():
        if template.count("{number}") != 1 or not template.strip():
            raise ValueError("템플릿에는 {number}가 정확히 한 번 필요합니다.")
        _literal(template.replace("{number}", ""), 0)
        _validate_literal_quantities(template.replace("{number}", ""), [])
        if re.search(r"\{number\}[^\s.,!?]", template):
            raise ValueError("자리표시자에 단위나 조사를 붙이지 마세요.")


def _scalar(value, kind="number", unit=""):
    """기존 공통 규칙으로 읽기 검증 기준을 만든다."""
    return read_number({"kind": kind, "value": str(value), "unit": unit})


def generate_number(slot, rng):
    """선택 유형에 맞는 값을 코드로 만들고 유형별 검증 기준을 보관한다."""
    kind, context, unit = slot["type"], slot["context"], slot["unit"]
    aliases = []
    if kind == "date":
        year, month = rng.randint(2000, 2035), rng.randint(1, 12)
        day = rng.randint(1, calendar.monthrange(year, month)[1])
        parts = [_scalar(year, "year", "년"), _scalar(month, "month", "월"), _scalar(day, "day", "일")]
        written, expected = (" ".join(part[index] for part in parts) for index in (0, 1))
    elif kind == "time":
        hour, minute = rng.randint(1, 12), rng.randint(0, 59)
        first = _scalar(hour, "hour", "시") if context == "clock" else _scalar(hour, "duration", "시간")
        second = _scalar(minute, "minute", "분")
        written, expected = first[0] + " " + second[0], first[1] + " " + second[1]
    elif kind == "phone":
        written, expected = _scalar(f"010-{rng.randint(0, 9999):04d}-{rng.randint(0, 9999):04d}", "phone")
    elif kind == "ordinal":
        value = rng.randint(1, 99)
        written = f"{value}{unit}"
        expected = ("첫" if value == 1 else _native_reading(value)) + " 번째" if context == "order" else sino_reading(value) + " 등"
    elif kind in {"money", "count"}:
        digits = slot.get("digits")
        if kind == "money" and digits is not None:
            value = rng.randint(10 ** (digits - 1), 10 ** digits - 1)
        else:
            value = rng.randint(1, 999999999) if kind == "money" else rng.randint(1, 99)
        written, expected = _scalar(value, kind, unit)
    else:
        value = f"{rng.randint(-30, 40) if kind == 'temperature' else rng.randint(0, 99)}.{rng.randint(0, 99):02d}"
        written, expected = _scalar(value, "decimal")
        pronunciation = {"temperature": "도", "interest_rate": "퍼센트", "ratio": "퍼센트", "decimal": ""}[kind]
        if kind == "temperature":
            expected = expected.replace("마이너스", "영하")
        written += unit
        expected += " " + pronunciation if pronunciation else ""
        if unit:
            aliases = [{"written": unit, "spoken": pronunciation, "context": "unit"}]
    return {**slot, "value": written, "expected": expected, "reading_aliases": aliases}


def reading_prompt(numbers, templates):
    """코드 생성 값과 유형·문맥을 바탕으로 LLM에 한글 읽기를 요청한다."""
    inputs = [{key: value for key, value in number.items() if key not in {"expected", "reading_aliases"}}
              | {"template": templates[number["id"]]} for number in numbers]
    return (
        "각 숫자를 type과 context에 맞춰 한글로 읽으세요. 값·부호·소수 자릿수·전화번호 선행 영을 보존하세요. "
        "phone은 공일공 일이삼사 오육칠팔처럼 각 자릿수를 읽고 그룹은 공백으로 구분하세요. 선행 공·영을 생략하지 마세요. "
        "money는 한자어, count는 고유어 수 관형사, date는 연월일 읽기, clock의 시와 duration의 시간은 고유어입니다. "
        "ordinal의 order는 첫 번째·두 번째, rank는 일 등·이 등입니다. 음수 온도는 영하, %는 퍼센트, ℃는 도로 읽습니다. 도씨라고 읽지 마세요. "
        "decimal, interest_rate, ratio의 정수부는 육십·팔십처럼 한자어로 읽고 육공·팔공처럼 자릿수로 읽지 마세요. "
        "소수부는 한 자리씩 읽으며 0은 반드시 영으로 읽고 공으로 읽지 마세요. 소수부 끝의 0도 생략하지 마세요. "
        "money의 십·백·천 앞에서는 일을 생략하세요. 51,767원은 오만 천칠백육십칠 원이며 일천이라고 읽지 마세요. "
        "단위를 포함한 숫자 표현만 읽고 전체 문장을 쓰지 마세요. JSON examples에 id, spoken만 반환하세요.\n"
    ) + json.dumps({"numbers": inputs}, ensure_ascii=False)


def render_pair(number, template, spoken):
    """유형별 기대 읽기를 검증한 LLM 결과와 원래 숫자를 같은 틀에 매핑한다."""
    if number["type"] == "phone":
        spoken = re.sub(r"[ \t]*-[ \t]*", " ", spoken).strip()
    compact = lambda text: re.sub(r"\s+", "", text)
    expected = compact(number["expected"])
    actual = compact(spoken)
    if number["type"] == "phone":
        expected, actual = expected.replace("영", "공"), actual.replace("영", "공")
    if not spoken.strip() or len(spoken) > 128 or actual != expected or re.search(r"[0-9\r\n]", spoken):
        raise ValueError(f"{number['type']} 유형의 한글 읽기가 생성 값과 일치하지 않습니다.")
    template = normalize_number_template(template, number, attached=True)
    return {"spoken_text": template.replace("{number}", spoken),
            "normalized_text": template.replace("{number}", number["value"]),
            "protected_terms": [spoken], "reading_aliases": number["reading_aliases"],
            "type": number["type"], "number_context": number["context"], "template": template,
            "generated_number": {key: number[key] for key in ("type", "context", "unit", "value", "digits") if key in number}}
