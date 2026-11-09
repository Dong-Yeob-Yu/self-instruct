"""종류가 명시된 숫자에서 값이 같은 한국어 발화와 정답을 만든다."""

import calendar
import re


DIGITS = "영일이삼사오육칠팔구"
PHONE_DIGITS = "공일이삼사오육칠팔구"
NATIVE_ONES = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
NATIVE_TENS = ("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")
COUNT_UNITS = ("명", "분", "개", "권", "장", "잔", "마리", "살", "대", "건", "회", "벌", "켤레", "그루", "송이", "자루", "통", "병", "봉지", "포", "상자")
KIND_UNITS = {"money": ("원",), "count": COUNT_UNITS, "year": ("년",), "month": ("월",), "day": ("일",),
              "hour": ("시",), "duration": ("시간", "개월", "일", "년", "분", "초"), "minute": ("분",), "second": ("초",),
              "decimal": ("", "도", "킬로그램", "그램", "미터", "리터", "퍼센트"),
              "phone": ("",), "identifier": ("",), "number": ("",)}
SLOT_PATTERN = re.compile(r"\{n([0-9]{1,2})\}")
NUMBER_SCHEMA = {
    "type": "object", "properties": {"examples": {"type": "array", "items": {
        "type": "object", "properties": {
            "template": {"type": "string"},
            "numbers": {"type": "array", "minItems": 1, "maxItems": 20, "items": {
                "type": "object", "properties": {
                    "kind": {"type": "string", "enum": list(KIND_UNITS)},
                    "value": {"type": "string"}, "unit": {"type": "string", "enum": sorted({unit for values in KIND_UNITS.values() for unit in values})},
                }, "required": ["kind", "value", "unit"], "additionalProperties": False,
            }},
            "protected_terms": {"type": "array", "items": {"type": "string"}},
        }, "required": ["template", "numbers", "protected_terms"], "additionalProperties": False,
    }}}, "required": ["examples"], "additionalProperties": False,
}


def numeric_prompt_instructions():
    """LLM에는 문장 틀과 숫자 의미만 생성하도록 지시한다."""
    return (
        "한국어 숫자 정규화용 ASR 학습 예제를 생성하세요. 사용자 요구사항의 상황·문체·숫자 종류를 따르세요. "
        "발화문과 정답을 각각 만들지 말고 template과 numbers만 만드세요. 실제 숫자 읽기와 전사문은 앱이 생성합니다. "
        "template에는 자연스러운 한국어 문장을 쓰고 숫자를 {n0}, {n1} 순서로 치환하세요. 단위는 template과 numbers.unit에 동일하게 쓰세요. "
        "numbers의 각 원소에는 kind, value(아라비아 숫자 문자열), unit을 넣으세요. "
        "모든 숫자 표현은 자리표시자로 넣으세요. 예를 들어 template의 {n0}명과 numbers의 unit 명이 일치해야 합니다. "
        "금액 money/원, 인원·수량 count/명·분·개·권·장·잔·마리·살·대·건·회·벌·켤레·그루·송이·자루·통·병·봉지·포·상자, "
        "연도 year/년, 월 month/월, 날짜 day/일, 시각 hour/시, 기간 duration/시간·개월·일·년·분·초, "
        "분 minute/분, 초 second/초, 소수 decimal/빈문자열·도·킬로그램·그램·미터·리터·퍼센트, "
        "전화 phone/빈문자열, 식별번호 identifier/빈문자열, 일반 수 number/빈문자열을 사용하세요. "
        "1250원은 '천이백오십 원'으로 읽으며 '일천이백오십 원'으로 만들지 않습니다. "
        "금액은 한자어 수사, 99 이하의 인원·수량은 고유어 수 관형사로 읽습니다. "
        "10명→열 명, 1명→한 명, 2명→두 명, 20명→스무 명, 21명→스물한 명입니다. "
        "count와 minute의 '분'은 뜻이 다릅니다. 사람 두 분은 count, 이 분 뒤는 minute입니다. "
        "시각과 기간을 구분하세요. '오후 2시부터 5시까지'는 hour 두 개이며 '오후 2시부터 5시간 동안'은 hour와 duration입니다. "
        "오전·오후와 조사·요일·문장 내용은 template에 그대로 남기세요. 12시간제와 24시간제를 임의로 바꾸지 마세요. "
        "6월은 유월, 10월은 시월입니다. 날짜는 3일→삼일, 10일→십일, 11일→십일일, "
        "15일→십오일, 21일→이십일일, 25일→이십오일, 31일→삼십일일로 앱이 읽습니다. "
        "날짜의 '에'는 template의 자리표시자 뒤에 쓰세요. year는 양의 정수, month는 1~12, day는 1~31, hour는 0~24입니다. "
        "전화번호는 010-1234-5678처럼 정확한 자릿수로 쓰고 선행 0을 유지하세요. 식별번호는 identifier로 넣으세요. "
        "값에 한글 자릿수 단위를 섞지 마세요. 123,450원은 value '123450', unit '원'입니다. "
        "금액·수량의 천 단위 쉼표는 앱이 넣습니다. numbers 배열의 모든 항목은 template에서 사용해야 합니다. "
        "protected_terms에는 template에 그대로 있는 고유명사만 넣고 숫자 보호어는 앱에서 추출합니다. "
        "previous_examples와 failed_examples를 복제하지 말고, 실패 이유를 참고해 상황·어순·길이가 다른 count개를 만드세요. "
        "문장 하나에 줄바꿈을 넣지 말고 JSON만 출력하세요. "
        '예: {"examples":[{"template":"참석자는 {n0}명이고 참가비는 {n1}원입니다.",'
        '"numbers":[{"kind":"count","value":"10","unit":"명"},{"kind":"money","value":"1250","unit":"원"}],'
        '"protected_terms":[]}]}\n'
    )


def _small_sino(value):
    """천·백·십 앞의 불필요한 일을 생략하여 만 미만 정수를 읽는다."""
    parts = []
    for power, unit in ((3, "천"), (2, "백"), (1, "십"), (0, "")):
        digit = value // (10 ** power) % 10
        if digit:
            parts.append(("" if digit == 1 and unit else DIGITS[digit]) + unit)
    return "".join(parts)


def sino_reading(value):
    """경 단위까지 정수를 읽고 만·억·조·경의 자릿값을 보존한다."""
    if value == 0:
        return "영"
    groups = []
    for unit in ("", "만", "억", "조", "경"):
        group, value = value % 10000, value // 10000
        if group:
            groups.append(("" if group == 1 and unit == "만" and value == 0 else _small_sino(group)) + unit)
    if value:
        raise ValueError("지원하는 숫자 자릿수를 초과했습니다.")
    return " ".join(reversed(groups))


def _native_reading(value):
    """단위 앞에서는 한·두·세·네·스무 형태의 고유어를 사용한다."""
    if not 1 <= value <= 99:
        return sino_reading(value)
    if value == 20:
        return "스무"
    tens, ones = divmod(value, 10)
    return NATIVE_TENS[tens] + NATIVE_ONES[ones]


def _parse_value(value):
    """정확한 자릿수를 가진 아라비아 숫자만 허용하고 혼합 표기를 거부한다."""
    if not isinstance(value, str) or not re.fullmatch(r"[+-]?(?:[0-9]{1,20}|[0-9]{1,3}(?:,[0-9]{3}){1,6})(?:\.[0-9]{1,8})?", value):
        raise ValueError("숫자 값은 20자리 이내의 아라비아 숫자여야 합니다.")
    cleaned = value.replace(",", "")
    unsigned = cleaned.lstrip("+-")
    integer, separator, fraction = unsigned.partition(".")
    if len(integer) > 20 or (len(integer) > 1 and integer.startswith("0")):
        raise ValueError("선행 0이 있는 번호는 phone 또는 identifier로 지정하세요.")
    return cleaned, int(integer), separator, fraction


def _read_phone(value):
    """전화번호는 숫자를 한 자리씩 읽고 전사문에만 구분자를 유지한다."""
    if not isinstance(value, str):
        raise ValueError("전화번호는 문자열이어야 합니다.")
    if re.fullmatch(r"0[0-9]{8,10}", value):
        prefix = 2 if value.startswith("02") else 3
        value = f"{value[:prefix]}-{value[prefix:-4]}-{value[-4:]}"
    if not re.fullmatch(r"0[0-9]{1,2}-[0-9]{3,4}-[0-9]{4}", value):
        raise ValueError("전화번호 자릿수를 확인하세요. 예: 010-1234-5678")
    spoken = " ".join("".join(PHONE_DIGITS[int(digit)] for digit in group) for group in value.split("-"))
    return value, spoken


def _read_identifier(value):
    """식별번호의 선행 0과 전체 자릿수를 보존한다."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,32}", value):
        raise ValueError("식별번호는 32자리 이내 숫자 문자열이어야 합니다.")
    return value, "".join(PHONE_DIGITS[int(digit)] for digit in value)


def _read_integer(kind, value, unit):
    """숫자 종류에 따라 금액·인원·날짜·시간의 읽기를 결정한다."""
    bounds = {"year": (1, 9999), "month": (1, 12), "day": (1, 31), "hour": (0, 24), "minute": (0, 59), "second": (0, 59)}
    if kind in bounds and not bounds[kind][0] <= value <= bounds[kind][1]:
        raise ValueError(f"{kind} 값의 범위를 확인하세요.")
    if kind == "month" and value in {6, 10}:
        return "유" if value == 6 else "시"
    if kind in {"count", "hour"} or (kind == "duration" and unit == "시간"):
        return _native_reading(value)
    return sino_reading(value)


def read_number(slot):
    """한 숫자 슬롯에서 같은 값을 가진 정답 표기와 발화를 생성한다."""
    if not isinstance(slot, dict):
        raise ValueError("숫자 항목은 객체여야 합니다.")
    kind, value, unit = slot.get("kind"), slot.get("value"), slot.get("unit")
    if not isinstance(kind, str) or kind not in KIND_UNITS or unit not in KIND_UNITS[kind]:
        raise ValueError("지원하는 숫자 종류와 단위 조합을 지정하세요.")
    if kind == "phone":
        return _read_phone(value)
    if kind == "identifier":
        return _read_identifier(value)
    cleaned, integer, separator, fraction = _parse_value(value)
    if kind not in {"decimal", "number", "money"} and (separator or cleaned.startswith(("+", "-"))):
        raise ValueError("인원·날짜·시각·기간에는 부호 없는 정수를 사용하세요.")
    if kind == "money" and separator:
        raise ValueError("원 단위 금액은 정수로 입력하세요.")
    spoken = _read_integer(kind, integer, unit)
    if separator:
        spoken += "점" + "".join(DIGITS[int(digit)] for digit in fraction)
    if cleaned.startswith(("+", "-")):
        spoken = ("마이너스 " if cleaned.startswith("-") else "플러스 ") + spoken
    normalized = (f"{int(cleaned):+,}" if cleaned.startswith("+") else f"{int(cleaned):,}") if kind in {"money", "count"} else cleaned
    if unit:
        spoken += ("" if kind in {"year", "month", "day"} else " ") + unit
    return normalized + unit, spoken


def _validate_calendar_dates(text):
    """생성된 월·일 조합이 실제 달력에 존재하는지 확인한다."""
    for match in re.finditer(r"(?:(\d{1,4})년\s*)?(\d{1,2})월\s*(\d{1,2})일", text):
        year, month, day = int(match.group(1) or "2000"), int(match.group(2)), int(match.group(3))
        if not 1 <= month <= 12 or not 1 <= day <= calendar.monthrange(year, month)[1]:
            raise ValueError("존재하지 않는 날짜 조합입니다.")


def _normalize_template_units(template, numbers):
    """문장 틀의 단위를 슬롯과 대조하고 일치하는 단위만 한 번 출력한다."""
    units = sorted({unit for values in KIND_UNITS.values() for unit in values if unit}, key=len, reverse=True)
    pattern = re.compile(r"^(\s*)(" + "|".join(map(re.escape, units)) + r")(?=$|[\s.,!?·]|입니다|이고|이며|이라|이나|은|는|이|가|을|를|의|과|와|부터|까지|동안|간|당|씩|중|으로|로|에서|에게|도)")
    result = template
    for match in reversed(list(SLOT_PATTERN.finditer(template))):
        unit = numbers[int(match.group(1))]["unit"]
        following = pattern.match(template[match.end():])
        if not unit or not following:
            continue
        if following.group(2) != unit:
            raise ValueError("template의 단위와 숫자 항목의 unit이 다릅니다.")
        result = result[:match.end()] + result[match.end() + following.end():]
    return result


def render_numeric_candidate(candidate):
    """공유 문장 틀에 숫자를 치환하여 숫자 외 문장 내용까지 동일하게 유지한다."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("template"), str):
        raise ValueError("숫자 생성 응답에는 template 문자열이 필요합니다.")
    template, numbers = candidate["template"], candidate.get("numbers")
    if not template.strip() or len(template) > 1000 or not isinstance(numbers, list) or not 1 <= len(numbers) <= 20:
        raise ValueError("문장 틀과 1~20개의 숫자 항목을 입력하세요.")
    matches = list(SLOT_PATTERN.finditer(template))
    if {int(match.group(1)) for match in matches} != set(range(len(numbers))):
        raise ValueError("문장 틀의 숫자 자리표시자와 numbers 순서가 맞지 않습니다.")
    remainder = SLOT_PATTERN.sub("", template)
    if re.search(r"[0-9{}]", remainder):
        raise ValueError("모든 숫자를 {n0} 형태의 자리표시자로 지정하세요.")
    readings = [read_number(slot) for slot in numbers]
    template = _normalize_template_units(template, numbers)
    normalized = SLOT_PATTERN.sub(lambda match: readings[int(match.group(1))][0], template)
    spoken = SLOT_PATTERN.sub(lambda match: readings[int(match.group(1))][1], template)
    _validate_calendar_dates(normalized)
    if len(spoken) > 1000 or len(normalized) > 1000:
        raise ValueError("완성된 발화·전사문이 1,000자를 초과했습니다.")
    return {"spoken_text": spoken, "normalized_text": normalized, "protected_terms": candidate.get("protected_terms", [])}
