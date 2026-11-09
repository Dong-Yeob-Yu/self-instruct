"""단위와 의미에 따른 숫자 읽기 정책을 생성·검토에 공통 적용한다."""

import re

ROLES = ("unspecified", "calendar", "duration", "clock", "people", "count", "identifier", "ordinal", "measure")
POLICY_READINGS = ("sino", "native", "digits", "year", "month", "day", "hour")
DEFAULT_READING_RULES = [
    *({"unit": unit, "role": "*", "reading": "sino"} for unit in ("일", "개월", "원", "초", "회", "층")),
    {"unit": "년", "role": "*", "reading": "year"},
    {"unit": "월", "role": "*", "reading": "month"},
    {"unit": "시", "role": "*", "reading": "hour"},
    *({"unit": unit, "role": "*", "reading": "native"} for unit in ("명", "개", "시간", "살", "권", "장", "대", "건", "마리", "잔", "병")),
    {"unit": "분", "role": "duration", "reading": "sino"},
    {"unit": "분", "role": "clock", "reading": "sino"},
    {"unit": "분", "role": "people", "reading": "native"},
    {"unit": "번", "role": "count", "reading": "native"},
    {"unit": "번", "role": "ordinal", "reading": "sino"},
    {"unit": "번", "role": "identifier", "reading": "digits"},
]


def validate_reading_rules(rules=None):
    """사용자 규칙을 복사하고 중복·지원하지 않는 읽기 설정을 거부한다."""
    source = DEFAULT_READING_RULES if rules is None else rules
    if not isinstance(source, list) or len(source) > 100:
        raise ValueError("숫자 읽기 규칙은 최대 100개의 배열로 입력하세요.")
    result, seen = [], set()
    for rule in source:
        if not isinstance(rule, dict):
            raise ValueError("숫자 읽기 규칙의 단위·의미·읽기를 확인하세요.")
        unit, role, reading = rule.get("unit"), rule.get("role"), rule.get("reading")
        if not isinstance(unit, str) or not unit.strip() or len(unit) > 32 or any(character.isdigit() or character.isspace() for character in unit):
            raise ValueError("규칙 단위는 숫자·공백 없이 1~32자로 입력하세요.")
        if role not in ("*", *ROLES) or reading not in POLICY_READINGS:
            raise ValueError("지원하는 의미와 숫자 읽기 방식을 선택하세요.")
        if (unit, role) in seen:
            raise ValueError("같은 단위와 의미의 읽기 규칙이 중복되었습니다.")
        seen.add((unit, role))
        result.append({"unit": unit, "role": role, "reading": reading})
    return result


def resolve_reading(unit, role, rules):
    """의미가 일치하는 규칙을 우선하고 없으면 단위의 공통 규칙을 찾는다."""
    matching = [rule for rule in rules if rule["unit"] == unit]
    exact = next((rule for rule in matching if rule["role"] == role), None)
    fallback = next((rule for rule in matching if rule["role"] == "*"), None)
    return (exact or fallback or {}).get("reading"), bool(matching and fallback is None)


def apply_reading_policy(expressions, rules):
    """모델의 읽기를 확정 규칙으로 교정하고 문맥 구분이 필요한 표현은 검토 표시한다."""
    resolved, notes = [], []
    for expression in expressions:
        if not isinstance(expression, dict):
            raise ValueError("표현은 JSON 객체여야 합니다.")
        item = dict(expression)
        role = item.get("role", "unspecified")
        if role not in ROLES:
            raise ValueError("표현의 의미 role을 확인하세요.")
        reading, contextual = resolve_reading(item.get("unit"), role, rules)
        if reading:
            item["reading"] = reading
        if reading and not item.get("spoken_unit") and re.fullmatch(r"[가-힣]+", item["unit"]):
            item["spoken_unit"] = item["unit"]
        if contextual:
            meaning = role if role != "unspecified" else "의미 미지정"
            notes.append(f"문맥 읽기 확인: {item.get('value', '')}{item.get('unit', '')} ({meaning})")
        if item.get("unit") and not reading and not contextual and item.get("reading") != "custom":
            notes.append(f"읽기 규칙 확인: {item['unit']} (공통 규칙 미등록)")
        if not item.get("unit") and item.get("reading") not in {"custom", "phone"}:
            notes.append("수치 문맥 확인: 단위 없는 숫자이므로 문장 의미를 확인하세요.")
        item["role"] = role
        resolved.append(item)
    return resolved, list(dict.fromkeys(notes))
