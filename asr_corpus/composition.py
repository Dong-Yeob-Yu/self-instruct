"""주제와 단위 목록 없이 순서가 있는 문장 조각을 하나의 발화·정답에 연결한다."""

import re
import copy

from .reading_policy import DEFAULT_READING_RULES


EXPRESSION_PATTERN = re.compile(r"\{e([0-9]{1,2})\}")
TEXT_PATTERN = r"^[^0-9{}<>\r\n]*$"
KOREAN_QUANTITY = (
    r"(?:마이너스|영하|음수|플러스|양수)?\s*(?:"
    r"[영공일이삼사오육칠팔구십백천만억조경]+(?:\s+[영공일이삼사오육칠팔구십백천만억조경]+)*"
    r"(?:\s*점\s*[영공일이삼사오육칠팔구]+)?|"
    r"(?:스물|서른|마흔|예순|일흔|여든|아흔|열|쉰)(?:다섯|여섯|일곱|여덟|아홉|한|두|세|네)?|"
    r"다섯|여섯|일곱|여덟|아홉|스무|하나|한|둘|두|셋|세|넷|네)"
)


def parts_schema(expression_schema):
    """일반 문장과 변환 표현을 같은 순서로 받는 공통 출력 형식을 만든다."""
    text = {"type": "object", "properties": {"text": {"type": "string", "maxLength": 1000, "pattern": TEXT_PATTERN}},
            "required": ["text"], "additionalProperties": False}
    numeric = copy.deepcopy(expression_schema["oneOf"][0])
    numeric["required"] = ["value", "unit"]
    numeric["properties"]["value"]["pattern"] = r"^([+-]?[0-9][0-9,]*(\.[0-9]+)?|0[0-9]{1,2}-[0-9]{3,4}-[0-9]{4})$"
    numeric["properties"]["reading"]["enum"] = list(dict.fromkeys(
        reading for branch in expression_schema["oneOf"] for reading in branch["properties"]["reading"]["enum"] if reading != "custom"))
    custom = copy.deepcopy(expression_schema["oneOf"][1])
    custom["required"] = ["value", "reading", "spoken"]
    return {"type": "array", "minItems": 1, "maxItems": 60, "items": {"oneOf": [text, numeric, custom]}}


def compile_candidate(candidate):
    """새 순서형 문장과 이전 번호형 문장을 같은 내부 표현으로 변환한다."""
    if not isinstance(candidate, dict):
        raise ValueError("생성 결과는 JSON 객체여야 합니다.")
    if "parts" in candidate:
        template, expressions = _compile_parts(candidate["parts"])
        return template, expressions, []
    parts, notes = _legacy_parts(candidate)
    template, expressions = _compile_parts(parts)
    return template, expressions, notes


def _compile_parts(parts):
    """각 표현의 연결 위치를 코드에서 정해 모델이 번호를 맞출 필요를 없앤다."""
    if not isinstance(parts, list) or not 1 <= len(parts) <= 60:
        raise ValueError("문장은 1~60개의 텍스트·표현 조각으로 작성하세요.")
    template, expressions = [], []
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ValueError(f"{index + 1}번째 조각은 텍스트 또는 표현 객체여야 합니다.")
        if set(part) == {"text"}:
            template.append(_literal(part["text"], index))
            continue
        if "text" in part or "value" not in part:
            raise ValueError(f"{index + 1}번째 조각에서 일반 문장과 변환 표현을 섞지 마세요.")
        template.append("{e" + str(len(expressions)) + "}")
        expressions.append(_complete_expression(part))
    if len(expressions) > 20:
        raise ValueError("한 문장에는 변환 표현을 최대 20개까지 사용할 수 있습니다.")
    compiled = "".join(template)
    _validate_literal_quantities(compiled, expressions)
    return compiled, expressions


def _validate_literal_quantities(template, expressions):
    """일반 문장에 남은 명확한 한글 수치·단위를 거부하되 모호한 일반어는 보존한다."""
    units = {rule["unit"] for rule in DEFAULT_READING_RULES}
    units.update(item[key] for item in expressions for key in ("unit", "spoken_unit")
                 if isinstance(item.get(key), str))
    choices = "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True) if unit)
    pattern = re.compile(r"(?<![가-힣A-Za-z0-9])(?P<number>" + KOREAN_QUANTITY + r")\s*(?P<unit>" + choices
                         + r")(?=$|[^가-힣A-Za-z0-9]|입니다|이고|이며|으로|에서|에게|까지|부터|은|는|이|가|을|를|에|과|와|도|만|씩)")
    for match in pattern.finditer(template):
        number = match["number"].strip()
        if number in {"이", "일"}:
            continue
        if match.group().strip() == "한번" and (match.end() == len(template) or not template[match.end()].isalpha()):
            continue
        raise ValueError(f"일반 문장에 미변환 한글 수치 '{match.group().strip()}'가 있습니다. text에서 제거하고 value·unit 표현 객체로 분리하세요.")


def _literal(text, index):
    """그대로 읽는 문장에는 연결 번호나 변환하지 않은 숫자가 섞이지 않게 한다."""
    if not isinstance(text, str) or len(text) > 1000 or re.fullmatch(TEXT_PATTERN, text) is None:
        raise ValueError(f"{index + 1}번째 일반 문장의 숫자·변환 표현은 별도 표현 조각으로 분리하세요.")
    if any(character.isdecimal() or ord(character) < 32 or ord(character) == 127 for character in text):
        raise ValueError(f"{index + 1}번째 일반 문장에 변환되지 않은 숫자 또는 제어 문자가 있습니다.")
    return text


def _complete_expression(part):
    """일반 숫자는 값·단위만 받아 공통 읽기 정책에 필요한 기본 속성을 채운다."""
    unit = part.get("unit", "")
    reading = part.get("reading", _default_reading(part))
    pronunciation = unit if isinstance(unit, str) and re.fullmatch(r"[가-힣]+", unit) else ""
    return {"reading": reading, "unit": unit, "spoken_unit": pronunciation, "negative": "", "spoken": "",
            "role": "unspecified", **part}


def _default_reading(part):
    """전화번호·선행 영·식별자처럼 형태나 역할로 명확한 공통 읽기만 선택한다."""
    value = part.get("value", "")
    if not isinstance(value, str):
        return "sino"
    if re.fullmatch(r"0[0-9]{1,2}-[0-9]{3,4}-[0-9]{4}", value):
        return "phone"
    if part.get("role") == "identifier" or (len(value) > 1 and value.startswith("0") and value.isascii() and value.isdigit()):
        return "digits"
    return "sino"


def _legacy_parts(candidate):
    """이전 자료의 명확한 연결만 복구하고 추측한 값은 생성하지 않는다."""
    template, expressions = candidate.get("template"), candidate.get("expressions")
    if not isinstance(template, str) or not template.strip() or len(template) > 1000:
        raise ValueError("문장 틀 template은 1~1,000자로 입력하세요.")
    if not isinstance(expressions, list) or len(expressions) > 20:
        raise ValueError("이전 표현 배열은 최대 20개여야 합니다.")
    used = {int(match.group(1)) for match in EXPRESSION_PATTERN.finditer(template)}
    if used == set(range(len(expressions))):
        return _split_template(template, expressions), []
    if not used:
        _literal(template, 0)
        return [{"text": template}], ["형식 복구 확인: 본문에 사용하지 않은 표현을 제외했습니다."]
    expanded = _expand_uniform_groups(template, expressions, used)
    if expanded is None:
        raise ValueError("자리표시자와 표현 배열의 연결이 불명확합니다. 누락된 값을 추측하지 않고 다시 생성합니다.")
    return _split_template(expanded, expressions), ["표현 묶음 확인: 이전 자료의 묶음을 문장 순서에 맞춰 연결했습니다."]


def _expand_uniform_groups(template, expressions, used):
    """단위·읽기·역할의 반복 구조가 같은 기존 묶음만 일반적인 방식으로 연결한다."""
    count = len(used)
    if used != set(range(count)) or len(expressions) % count or len(expressions) <= count:
        return None
    width = len(expressions) // count
    signatures = [_group_signature(expressions[start:start + width]) for start in range(0, len(expressions), width)]
    if signatures[0] is None or any(signature != signatures[0] for signature in signatures):
        return None
    if _has_explicit_scalar_unit(template, expressions, width):
        return None
    return EXPRESSION_PATTERN.sub(lambda match: " ".join("{e" + str(int(match.group(1)) * width + offset) + "}"
                                                        for offset in range(width)), template)


def _group_signature(expressions):
    """도메인 이름 대신 표현 자체의 단위·읽기·역할로 묶음 구조를 비교한다."""
    if any(not isinstance(item, dict) for item in expressions):
        return None
    units = [item.get("unit") for item in expressions]
    if any(not isinstance(unit, str) or not unit for unit in units) or len(set(units)) != len(units):
        return None
    return [(item["unit"], item.get("reading"), item.get("role", "unspecified")) for item in expressions]


def _has_explicit_scalar_unit(template, expressions, width):
    """문장에 단일 값의 단위가 이미 명시됐으면 묶음으로 확대하지 않는다."""
    for match in EXPRESSION_PATTERN.finditer(template):
        unit = expressions[int(match.group(1)) * width]["unit"]
        if re.match(r"\s*" + re.escape(unit), template[match.end():]):
            return True
    return False


def _split_template(template, expressions):
    """기존 문장에 실제 등장한 표현만 순서대로 연결한다."""
    parts, position = [], 0
    for match in EXPRESSION_PATTERN.finditer(template):
        if match.start() > position:
            parts.append({"text": template[position:match.start()]})
        parts.append(expressions[int(match.group(1))])
        position = match.end()
    if position < len(template):
        parts.append({"text": template[position:]})
    return parts or [{"text": ""}]
