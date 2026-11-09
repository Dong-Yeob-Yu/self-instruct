"""도메인 목록 없이 프롬프트의 표현과 공통 읽기 방식으로 데이터 쌍을 만든다."""

import re
import unicodedata

from .number_generation import _parse_value, _native_reading, _validate_calendar_dates, read_number, sino_reading
from .reading_policy import ROLES, apply_reading_policy, validate_reading_rules
from .composition import EXPRESSION_PATTERN, compile_candidate, parts_schema


DEFAULT_COMMON_PROMPT = (
    "자연스러운 한국어 발화를 다양하게 만드세요. 숫자와 단위의 값, 고유명사와 문장 내용을 발화와 정답에서 유지하세요.\n"
    "응답은 내부 출력 지시에 따른 parts로 작성하세요. 발화와 정답은 앱이 같은 parts에서 생성합니다.\n"
    "수치는 아라비아 숫자 value와 단위 unit으로 분리하세요. 한국어 숫자 읽기는 앱의 공통 읽기 규칙을 사용합니다.\n"
    "시각과 소요 시간을 구분하고 오전·오후, 조사, 전화번호의 선행 0을 보존하세요.\n"
    "주제에 필요한 단위와 표기 규칙은 이번 요청을 따르세요. 한 문장에 불필요하게 여러 주제를 섞지 마세요."
)
READINGS = ("sino", "native", "digits", "year", "month", "day", "hour", "phone", "custom")
NUMERAL_READING = re.compile(
    r"(?:마이너스|영하|음수|플러스|양수)?(?:"
    r"[영공일이삼사오육칠팔구십백천만억조경]+(?:점[영공일이삼사오육칠팔구]+)?|"
    r"(?:스물|서른|마흔|예순|일흔|여든|아흔|열|쉰)(?:다섯|여섯|일곱|여덟|아홉|하나|한|둘|두|셋|세|넷|네)?|"
    r"다섯|여섯|일곱|여덟|아홉|스무|하나|한|둘|두|셋|세|넷|네)"
)
EXPRESSION_PROPERTIES = {
    "value": {"type": "string"}, "reading": {"type": "string", "enum": list(READINGS)},
    "unit": {"type": "string", "maxLength": 128}, "spoken_unit": {"type": "string", "maxLength": 128},
    "negative": {"type": "string", "enum": ["", "마이너스", "영하", "음수"]},
    "spoken": {"type": "string"}, "role": {"type": "string", "enum": list(ROLES)},
}
EXPRESSION_KEYS = list(EXPRESSION_PROPERTIES)
EXPRESSION_SCHEMA = {"oneOf": [
    {"type": "object", "properties": {**EXPRESSION_PROPERTIES,
        "value": {"type": "string", "maxLength": 128, "pattern": "^[+-]?[0-9][0-9,]*(\\.[0-9]+)?$"},
        "reading": {"type": "string", "enum": [reading for reading in READINGS if reading not in {"custom", "phone", "digits"}]},
        "spoken": {"type": "string", "enum": [""]},
    }, "required": EXPRESSION_KEYS, "additionalProperties": False},
    {"type": "object", "properties": {**EXPRESSION_PROPERTIES,
        "value": {"type": "string", "maxLength": 128, "pattern": "^[^0-9+-].*$"},
        "reading": {"type": "string", "enum": ["custom"]},
        "spoken": {"type": "string", "minLength": 1, "maxLength": 128},
    }, "required": EXPRESSION_KEYS, "additionalProperties": False},
    {"type": "object", "properties": {**EXPRESSION_PROPERTIES,
        "value": {"type": "string", "pattern": "^[0-9]{1,32}$"},
        "reading": {"type": "string", "enum": ["digits"]},
        "spoken": {"type": "string", "enum": [""]},
    }, "required": EXPRESSION_KEYS, "additionalProperties": False},
    {"type": "object", "properties": {**EXPRESSION_PROPERTIES,
        "value": {"type": "string", "pattern": "^(0[0-9]{8,10}|0[0-9]{1,2}-[0-9]{3,4}-[0-9]{4})$"},
        "reading": {"type": "string", "enum": ["phone"]},
        "spoken": {"type": "string", "enum": [""]},
    }, "required": EXPRESSION_KEYS, "additionalProperties": False},
]}
PROMPT_SCHEMA = {
    "type": "object", "properties": {"examples": {"type": "array", "items": {
        "type": "object", "properties": {
            "plan_id": {"type": "string", "maxLength": 32},
            "parts": parts_schema(EXPRESSION_SCHEMA),
            "protected_terms": {"type": "array", "items": {"type": "string"}},
        }, "required": ["plan_id", "parts", "protected_terms"], "additionalProperties": False,
    }}}, "required": ["examples"], "additionalProperties": False,
}


def prompt_instructions():
    """어떤 주제든 문장 순서 그대로 텍스트와 변환 표현을 생성하도록 지시한다."""
    return (
        "user_prompt와 common_prompt에 맞는 한국어 ASR 예제를 정확히 count개 만드세요. 도메인과 단위는 제한하지 않습니다. "
        "[요청과 출력 형식의 역할] user_prompt와 common_prompt는 주제·상황·필수 포함 내용·문체와 최종 표기 요구입니다. "
        "응답 구조는 이 내부 지시의 examples와 parts 형식으로 통일하세요. "
        "요청에 '발화는 한글로, 정답은 아라비아 숫자로'라고 적혀 있어도 두 문장을 직접 작성하지 마세요. "
        "같은 parts에서 앱이 발화와 정답을 생성합니다. 요청의 읽기·표기 예시는 표현 객체로 옮기세요. "
        "planned_slots가 있으면 그 상황·목적·전개를 따르고 id를 plan_id로 쓰세요. 없으면 plan_id는 빈 문자열입니다. "
        "used_patterns는 복제할 예시가 아닌 금지 형식입니다. 값·이름만 바꾸지 말고 다른 사건과 발화 목적을 쓰세요. "
        "used_amounts는 이 작업에서 이미 사용한 금액입니다. 금액을 만들 때 이 목록과 이번 배치 안의 금액을 재사용하지 마세요. "
        "각 예제는 plan_id, parts, protected_terms입니다. parts에 문장을 읽는 순서대로 작성하세요. "
        "일반 문장은 {\"text\":\"문장 내용\"}으로, 숫자·단위나 발음이 다른 약어는 표현 객체로 넣으세요. "
        "별도의 문장 틀이나 연결 번호를 만들지 마세요. 일반 문장과 표현 사이의 공백·조사도 text에 그대로 넣으세요. "
        "요청이 수치를 요구하지 않고 실제로 수치가 없는 문장만 text 객체 하나로 작성할 수 있습니다. "
        "요청에서 수치를 필수로 지정하면 각 예제에 해당 표현 객체를 포함하세요. "
        "본문에 없는 표현을 덧붙이지 마세요. "
        "수치는 한글 숫자나 다른 언어의 숫자로 text에 쓰지 말고 숫자 객체로 분리하세요. "
        "한글로 읽은 수치를 text에 남긴 뒤 같은 값이나 다른 값을 표현 객체로 덧붙이지 마세요. "
        "숫자 객체는 value와 unit만 필수입니다. 읽기·역할은 필요한 경우에만 reading, role로 추가하세요. "
        "일반적인 수치와 한글 단위에는 value·unit만 쓰고 앱의 공통 읽기 규칙을 사용하세요. "
        "value는 단위 없는 숫자입니다. unit은 정답 단위, spoken_unit은 단위 발음만 쓰고, 숫자 읽기를 단위에 중복하지 마세요. "
        "숫자 발음은 앱이 생성하므로 spoken을 쓰지 마세요. 한글 단위의 spoken_unit은 생략할 수 있습니다. "
        "기호·외국어 단위는 spoken_unit에 한글 발음을 넣으세요. 단위가 없으면 unit은 빈 문자열입니다. "
        "여러 값으로 된 표현은 필요한 숫자 객체와 text를 순서대로 이어 쓰세요. 같은 문장에 여러 묶음이 있어도 같습니다. "
        "reading은 sino, native, year, month, day, hour, digits(식별번호 숫자열), phone 중 선택하세요. "
        "role은 calendar, duration, clock, people, count, identifier, ordinal, measure, unspecified 중 선택하세요. "
        "reading_rules는 앱의 공통 규칙입니다. 15일은 십오일, 15명은 열다섯 명입니다. 분은 시간과 인원을 구분하세요. "
        "negative는 영하·마이너스·음수 또는 빈 문자열입니다. "
        "새 단위는 unit 표기와 spoken_unit 발음을 선언하면 됩니다. 도메인 이름이나 단위 종류를 새로 정의하지 마세요. "
        "발음이 다른 약어·고유 표현은 reading custom, value 원문, spoken 발음, role unspecified로 쓰세요. "
        "그대로 읽는 일반 용어는 text에 남기세요. protected_terms에는 고유명사만 넣거나 []로 두세요. "
        "failed_examples의 오류를 고치되 문장을 복제하지 마세요. JSON만 출력하세요. "
        '형식: {"examples":[{"plan_id":"","parts":[{"text":"값은 "},{"value":"3.5","unit":"%",'
        '"spoken_unit":"퍼센트"},{"text":"입니다."}],"protected_terms":[]}]}'
        ' 변환 예: 십만 오천 원이라는 최종 발화는 [{"value":"105000","unit":"원"}]으로 작성하며 '
        'text에 "십만 오천 원"을 중복 작성하지 않습니다.\n'
    )

def _read_scalar(expression):
    """단위와 독립적으로 값과 읽기 방식의 대응을 생성한다."""
    value, reading = expression["value"], expression["reading"]
    if reading == "custom":
        spoken = expression["spoken"]
        if not value.strip() or not spoken.strip() or re.fullmatch(r"[+\-\d,.]+", value):
            raise ValueError("숫자는 공통 숫자 읽기를 사용하고 별도 발음에는 원문과 발음을 입력하세요.")
        return value, spoken
    if expression["spoken"]:
        raise ValueError("숫자 발음은 앱에서 생성합니다. spoken은 빈 문자열로 두세요.")
    if reading in {"phone", "digits"}:
        return read_number({"kind": "phone" if reading == "phone" else "identifier", "value": value, "unit": ""})
    if reading in {"year", "month", "day", "hour"}:
        unit = {"year": "년", "month": "월", "day": "일", "hour": "시"}[reading]
        normalized, spoken = read_number({"kind": reading, "value": value, "unit": unit})
        return normalized[:-len(unit)], spoken[:-len(unit)].rstrip()
    cleaned, integer, separator, fraction = _parse_value(value)
    if reading == "native" and (separator or cleaned.startswith(("+", "-"))):
        raise ValueError("고유어 수 관형사에는 부호 없는 정수를 사용하세요.")
    spoken = _native_reading(integer) if reading == "native" else sino_reading(integer)
    if separator:
        spoken += "점" + "".join("영일이삼사오육칠팔구"[int(digit)] for digit in fraction)
    if cleaned.startswith("-"):
        spoken = (expression["negative"] or "마이너스") + " " + spoken
    elif cleaned.startswith("+"):
        spoken = "플러스 " + spoken
    return value, spoken


def read_expression(expression):
    """임의의 단위 표기와 발음을 같은 숫자에 결합한다."""
    keys = ("value", "reading", "unit", "spoken_unit", "negative", "spoken")
    if not isinstance(expression, dict) or any(not isinstance(expression.get(key), str) for key in keys):
        raise ValueError("표현의 값·읽기·단위·발음은 문자열이어야 합니다.")
    if expression["reading"] not in READINGS or expression["negative"] not in {"", "마이너스", "영하", "음수"}:
        raise ValueError("지원하는 공통 읽기 방식과 음수 표현을 선택하세요.")
    if any(len(expression[key]) > 128 for key in keys) or bool(expression["unit"]) != bool(expression["spoken_unit"]):
        raise ValueError("단위 표기와 발음을 함께 입력하고 각 표현은 128자 이내로 작성하세요.")
    if re.search(r"\d", expression["spoken_unit"]):
        raise ValueError("단위 발음의 숫자도 한글로 풀어 쓰세요.")
    normalized, spoken = _read_scalar(expression)
    unit, spoken_unit = expression["unit"], expression["spoken_unit"]
    if unit == "원" and expression["reading"] != "custom":
        cleaned, integer, separator, fraction = _parse_value(normalized)
        sign = cleaned[0] if cleaned.startswith(("+", "-")) else ""
        normalized = sign + f"{integer:,}" + (separator + fraction if separator else "")
    validate_reading_alias({"written": unit, "spoken": spoken_unit, "context": "unit"}, allow_empty=True)
    space = "" if expression["reading"] in {"year", "month", "day"} or (expression["reading"] == "sino" and unit in {"년", "월", "일"}) else " "
    return normalized + unit, spoken + (space + spoken_unit if spoken_unit else "")


def _compact_reading(text):
    """단위 검증에서 유니코드와 공백 표기의 차이만 정리한다."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def _unit_encodes_number(written, reading):
    """세제곱 단위처럼 표기 자체에 명시된 수치는 추가 수량과 구분한다."""
    digits = re.findall(r"[0-9]{1,4}", unicodedata.normalize("NFKC", written))
    return any(reading in {sino_reading(int(value)).replace(" ", ""), _native_reading(int(value))} for value in digits)


def _has_quantity_in_unit(written, spoken):
    """단위 발음 앞에 별도 수치가 덧붙은 명확한 경우를 독립적으로 검출한다."""
    source, target = _compact_reading(written), _compact_reading(spoken)
    if source == target:
        return False
    if target.endswith(source) and NUMERAL_READING.fullmatch(target[:-len(source)]):
        return True
    first = spoken.strip().split()[0]
    if len(spoken.strip().split()) > 1 and NUMERAL_READING.fullmatch(first) and not _unit_encodes_number(written, first):
        return True
    prefix = NUMERAL_READING.match(target)
    return bool(prefix and len(prefix.group()) > 1 and not _unit_encodes_number(written, prefix.group()))


def validate_reading_alias(alias, allow_empty=False):
    """모델의 대응 선언이 단위에 수치를 추가하거나 숫자 검증을 우회하지 못하게 한다."""
    if not isinstance(alias, dict):
        raise ValueError("표기·발음 대응 정보가 올바르지 않습니다.")
    written, spoken, context = alias.get("written"), alias.get("spoken"), alias.get("context")
    if not isinstance(written, str) or not isinstance(spoken, str) or context not in {"unit", "literal"}:
        raise ValueError("표기·발음 대응 정보가 올바르지 않습니다.")
    if allow_empty and not written and not spoken:
        return
    if not written.strip() or not spoken.strip():
        raise ValueError("표기와 발음을 함께 입력하세요.")
    if context == "unit" and re.search(r"\d", spoken):
        raise ValueError("단위 발음에 수치를 넣을 수 없습니다.")
    if _has_quantity_in_unit(written, spoken):
        raise ValueError("단위 발음에 수치가 포함됐습니다. spoken_unit에는 원·건 같은 단위만 입력하세요.")
    if context == "literal" and re.match(r"^[+\-]?\d", written.strip()):
        raise ValueError("숫자 표기는 별도 발음 대응으로 우회하지 말고 공통 숫자 읽기를 사용하세요.")


def valid_reading_aliases(aliases):
    """수동 수정 시 잘못된 기존 대응을 제거하되 정상 단위·약어 대응은 보존한다."""
    valid = []
    for alias in aliases or []:
        try:
            validate_reading_alias(alias)
            valid.append(alias)
        except ValueError:
            continue
    return valid


def _template_units(template, expressions):
    """슬롯과 같은 단위가 문장 틀에도 있으면 한 번만 붙인다."""
    result = template
    for match in reversed(list(EXPRESSION_PATTERN.finditer(template))):
        unit = expressions[int(match.group(1))]["unit"]
        if not unit:
            continue
        boundary = r"(?=$|[^A-Za-z0-9])" if not "가" <= unit[-1] <= "힣" else r"(?=$|[\s.,!?]|입니다|이고|이며|이니|이라|은|는|이|가|을|를|의|과|와|부터|까지|동안|간|당|씩|중|으로|로|에|도|께)"
        if re.match(r"\s*" + re.escape(unit) + r"\s*" + re.escape(unit) + boundary, template[match.end():]):
            raise ValueError("문장 틀에 같은 단위가 반복되었습니다.")
        following = re.match(r"\s*" + re.escape(unit) + boundary, template[match.end():])
        if following:
            if re.match(r"\s*" + re.escape(unit) + boundary, template[match.end() + following.end():]):
                raise ValueError("문장 틀에 같은 단위가 반복되었습니다.")
            result = result[:match.end()] + result[match.end() + following.end():]
    return result


def _spoken_particles(template, readings):
    """표현의 실제 한글 발음에 맞춰 공통 문장 틀의 조사 짝을 정한다."""
    result = template
    pairs = {"은": ("은", "는"), "는": ("은", "는"), "이": ("이", "가"), "가": ("이", "가"),
             "을": ("을", "를"), "를": ("을", "를"), "과": ("과", "와"), "와": ("과", "와")}
    for match in reversed(list(EXPRESSION_PATTERN.finditer(template))):
        last = readings[int(match.group(1))][1].rstrip()[-1]
        particle = re.match(r"(으로|로|은|는|이|가|을|를|과|와)(?=\s|[.,!?]|$)", template[match.end():])
        if not particle or not "가" <= last <= "힣":
            continue
        ending = (ord(last) - 0xAC00) % 28
        original = particle.group()
        replacement = ("으로" if ending not in {0, 8} else "로") if original in {"으로", "로"} else pairs[original][0 if ending else 1]
        result = result[:match.end()] + replacement + result[match.end() + particle.end():]
    return result


def _spoken_expression(match, template, expressions, readings):
    """TTS에서 원 뒤의 조사·어미를 구분하도록 표현 경계에 공백을 넣는다."""
    index = int(match.group(1))
    spoken = readings[index][1]
    if expressions[index]["unit"] == "원" and re.match(r"[가-힣]", template[match.end():]):
        return spoken + " "
    return spoken


def render_prompt_candidate(candidate, reading_rules=None):
    """문장 조각을 한 번 연결해 도메인에 관계없이 발화·전사문의 값을 보존한다."""
    template, expressions, format_notes = compile_candidate(candidate)
    rules = validate_reading_rules(reading_rules)
    expressions, policy_notes = apply_reading_policy(expressions, rules)
    readings = [read_expression(expression) for expression in expressions]
    template = _template_units(template, expressions)
    template = _spoken_particles(template, readings)
    normalized = EXPRESSION_PATTERN.sub(lambda match: readings[int(match.group(1))][0], template)
    spoken = EXPRESSION_PATTERN.sub(lambda match: _spoken_expression(match, template, expressions, readings), template)
    if not normalized.strip() or not spoken.strip():
        raise ValueError("생성한 문장의 내용이 비어 있습니다.")
    if len(normalized) > 1000 or len(spoken) > 1000:
        raise ValueError("완성된 발화·전사문이 1,000자를 초과했습니다.")
    _validate_calendar_dates(normalized)
    aliases, notes = _reading_aliases(expressions)
    terms = candidate.get("protected_terms", [])
    if not isinstance(terms, list):
        raise ValueError("protected_terms는 배열이어야 합니다.")
    return {"spoken_text": spoken, "normalized_text": normalized, "protected_terms": terms + [reading[1] for reading in readings],
            "plan_id": candidate.get("plan_id", ""),
            "reading_aliases": aliases, "review_note": " · ".join(notes + policy_notes + format_notes),
            "reading_policy": {"rules": rules, "expressions": [expressions[int(match.group(1))]
                               for match in EXPRESSION_PATTERN.finditer(template)]}}


def _reading_aliases(expressions):
    """작업에서 선언한 표기·발음 대응을 검증과 사람 검토에 연결한다."""
    aliases, notes = [], []
    for expression in expressions:
        if expression["unit"] and expression["unit"] != expression["spoken_unit"]:
            aliases.append({"written": expression["unit"], "spoken": expression["spoken_unit"], "context": "unit"})
            notes.append(f"단위 발음 확인: {expression['unit']} → {expression['spoken_unit']}")
        if expression["reading"] == "custom":
            validate_reading_alias({"written": expression["value"], "spoken": expression["spoken"], "context": "literal"})
            aliases.append({"written": expression["value"], "spoken": expression["spoken"], "context": "literal"})
            notes.append(f"표현 확인: {expression['value']} → {expression['spoken']}")
    meanings = {}
    for alias in aliases:
        key = (alias["written"], alias["context"])
        if key in meanings and meanings[key] != alias["spoken"]:
            raise ValueError("같은 예제에서 동일한 표기의 발음이 서로 다릅니다.")
        meanings[key] = alias["spoken"]
    return aliases, list(dict.fromkeys(notes))


def apply_reading_aliases(text, aliases):
    """선언된 단위·별도 발음만 검증용 전사문에 적용한다."""
    result = text
    if not isinstance(aliases, list):
        raise ValueError("표기·발음 대응은 배열이어야 합니다.")
    for alias in aliases:
        validate_reading_alias(alias)
    for alias in sorted(aliases or [], key=lambda item: len(item.get("written", "")), reverse=True):
        written, spoken = alias.get("written"), alias.get("spoken")
        pattern = (r"(?<=[0-9])" if alias.get("context") == "unit" else "") + re.escape(written.replace(" ", ""))
        result = re.sub(pattern, lambda match: spoken.replace(" ", ""), result)
    return result
