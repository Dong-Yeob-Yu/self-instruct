"""한글 숫자 발화와 숫자 전사문의 대응을 보수적인 규칙으로 검증한다."""

import re
import unicodedata


NUMBER_PATTERN = re.compile(r"[+-]?\d(?:\d|,(?=\d))*(?:\.\d+)?")
DIGIT_WORDS = ("[영공]", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
NATIVE_UNITS = ("", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉")
NATIVE_BOUND_UNITS = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
NATIVE_TENS = ("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")
NUMERIC_ISSUE = "발화문과 전사문의 숫자·날짜·단위 또는 문장 내용이 대응하지 않습니다."
PHONE_PATTERN = re.compile(r"(?<![0-9])([0영공][0-9영공일이삼사오육칠팔구]{1,2})-([0-9영공일이삼사오육칠팔구]{3,4})-([0-9영공일이삼사오육칠팔구]{4})(?![0-9])")


def compact_text(text):
    """숫자 검증에서만 유니코드와 띄어쓰기를 정리한다."""
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFC", text))
    return PHONE_PATTERN.sub(r"\1\2\3", compact)


def _four_digit_pattern(value):
    """만 미만의 한자어 수사를 나타내는 패턴을 만든다."""
    parts = []
    for power, unit in ((3, "천"), (2, "백"), (1, "십"), (0, "")):
        digit = (value // (10 ** power)) % 10
        if digit:
            parts.append(("(?:일)?" if digit == 1 and unit else DIGIT_WORDS[digit]) + unit)
    return "".join(parts)


def _sino_pattern(digits):
    """정수의 한자어 읽기를 만·억·조·경 단위로 만든다."""
    if len(digits) > 20:
        return None
    value = int(digits)
    if value == 0:
        return "[영공]"
    groups = []
    units = ("", "만", "억", "조", "경")
    while value:
        groups.append(value % 10000)
        value //= 10000
    return "".join(_group_pattern(groups[index], units[index]) for index in range(len(groups) - 1, -1, -1))


def _group_pattern(value, unit):
    """큰 수 단위 앞에서 생략 가능한 일과 빈 자릿수를 처리한다."""
    if value == 0:
        return ""
    if value == 1 and unit:
        return "(?:일)?" + unit
    return _four_digit_pattern(value) + unit


def _native_patterns(digits):
    """아흔아홉 이하의 고유어 수사와 수 관형사 읽기를 만든다."""
    if len(digits) > 2 or not 1 <= int(digits) <= 99:
        return []
    value = int(digits)
    tens, ones = divmod(value, 10)
    prefix = NATIVE_TENS[tens]
    variants = [prefix + NATIVE_UNITS[ones], prefix + NATIVE_BOUND_UNITS[ones]]
    if value == 20:
        variants.append("스무")
    return variants


def _integer_patterns(digits, following):
    """자릿수를 보존하면서 정수에 대응하는 읽기 후보를 만든다."""
    variants = [re.escape(digits), "".join(DIGIT_WORDS[int(character)] for character in digits)]
    if len(digits) > 1 and digits.startswith("0"):
        return variants
    sino = _sino_pattern(digits)
    if sino:
        variants.append(sino)
    variants.extend(_native_patterns(digits))
    if digits == "1" and following.startswith("번째"):
        variants.append("첫")
    if following.startswith("월") and digits in {"6", "10"}:
        variants.append("유" if digits == "6" else "시")
    return variants


def _number_pattern(token, following):
    """정수·소수·부호와 천 단위 쉼표를 대응 가능한 읽기로 바꾼다."""
    sign = token[0] if token.startswith(("+", "-")) else ""
    unsigned = token[1:] if sign else token
    integer, separator, fraction = unsigned.partition(".")
    if "," in integer and not re.fullmatch(r"\d{1,3}(?:,\d{3})+", integer):
        return None
    digits = integer.replace(",", "")
    if len(digits) > 64:
        return None
    variants = _integer_patterns(digits, following)
    if separator:
        fraction_pattern = "".join(DIGIT_WORDS[int(character)] for character in fraction)
        variants = [f"(?:{'|'.join(variants)})점{fraction_pattern}", re.escape(unsigned)]
    pattern = "(?:" + "|".join(dict.fromkeys(variants)) + ")"
    if sign:
        pattern = ("(?:마이너스|영하|음수|-)" if sign == "-" else "(?:플러스|양수|\\+)") + pattern
    return pattern


def _transcript_pattern(transcript, following="", reading_policy=None):
    """전사문에 있는 숫자만 읽기 패턴으로 치환하고 나머지 내용은 고정한다."""
    normalized = transcript
    parts = []
    position = 0
    for index, match in enumerate(NUMBER_PATTERN.finditer(normalized)):
        suffix = normalized[match.end():] + following
        pattern = _policy_number_pattern(match.group(), suffix, reading_policy, index) if reading_policy is not None else _number_pattern(match.group(), suffix)
        if pattern is None:
            return None
        parts.extend((re.escape(normalized[position:match.start()]), pattern))
        position = match.end()
    parts.append(re.escape(normalized[position:]))
    return "".join(parts)


def _policy_number_pattern(token, following, policy, index):
    """수치가 같아도 확정된 단위·의미의 읽기와 다른 발음은 통과시키지 않는다."""
    from .prompt_generation import _read_scalar
    from .reading_policy import resolve_reading, validate_reading_rules
    rules = validate_reading_rules(policy.get("rules"))
    expressions = [item for item in policy.get("expressions", []) if item.get("reading") != "custom"]
    binding = expressions[index] if index < len(expressions) else None
    unit = binding.get("unit", "") if binding else ""
    if binding and not following.startswith(compact_text(binding.get("spoken_unit", unit))):
        binding = None
    if binding is None:
        unit = next((rule["unit"] for rule in sorted(rules, key=lambda rule: len(rule["unit"]), reverse=True)
                     if following.startswith(rule["unit"])), "")
    role = binding.get("role", "unspecified") if binding else "unspecified"
    reading, contextual = resolve_reading(unit, role, rules)
    if not reading and binding:
        reading = binding.get("reading")
    if not reading:
        if contextual:
            raise ValueError("문맥에 따른 숫자 읽기를 확인하세요.")
        return _number_pattern(token, following)
    expression = {"value": token, "reading": reading, "negative": binding.get("negative", "") if binding else "", "spoken": ""}
    return re.escape(compact_text(_read_scalar(expression)[1]))


def _spoken_source_offsets(text):
    """공백과 전화번호 구분자를 제거한 비교 위치를 원문 위치에 대응시킨다."""
    source = unicodedata.normalize("NFC", text)
    positions = [index for index, character in enumerate(source) if not character.isspace()]
    compact = "".join(source[index] for index in positions)
    separators = set()
    for match in PHONE_PATTERN.finditer(compact):
        separators.update(index for index in range(match.start(), match.end()) if compact[index] == "-")
    offsets = [position for index, position in enumerate(positions) if index not in separators]
    return source, "".join(source[index] for index in offsets), offsets


def _protected_span_pattern(transcript, start, end):
    """숫자 토큰을 자르지 않는 보호어 구간을 전체 문장의 캡처 패턴으로 만든다."""
    if any(match.start() < boundary < match.end() for match in NUMBER_PATTERN.finditer(transcript) for boundary in (start, end)):
        return None
    prefix = _transcript_pattern(transcript[:start], transcript[start:])
    protected = _transcript_pattern(transcript[start:end], transcript[end:])
    suffix = _transcript_pattern(transcript[end:])
    if any(part is None for part in (prefix, protected, suffix)):
        return None
    return prefix + "(?P<protected>" + protected + ")" + suffix


def _map_transcript_term(spoken, transcript, term):
    """전사문 보호어를 숫자 대응이 확인된 발화 구간으로 매핑한다."""
    spans = []
    for occurrence in re.finditer(re.escape(term), transcript):
        pattern = _protected_span_pattern(transcript, occurrence.start(), occurrence.end())
        match = re.fullmatch(pattern, spoken) if pattern is not None else None
        if match:
            spans.append(match.span("protected"))
    return spans


def spoken_protected_terms(spoken_text, normalized_text, terms):
    """발화에서 확인한 보호어와 숫자 대응으로 직접 추출한 보호어만 반환한다."""
    if not isinstance(terms, list) or any(not isinstance(term, str) or not term.strip() for term in terms):
        return terms
    source, spoken, offsets = _spoken_source_offsets(spoken_text)
    transcript = compact_text(normalized_text)
    corrected = []
    for term in terms:
        corrected.extend(_resolve_spoken_term(source, spoken, offsets, transcript, term))
    for term in _numeric_transcript_terms(transcript):
        resolved = _resolve_spoken_term(source, spoken, offsets, transcript, term)
        corrected.extend(value for value in resolved if not any(compact_text(value) in compact_text(existing) for existing in corrected))
    return list(dict.fromkeys(corrected))


def _numeric_transcript_terms(transcript):
    """전사문 숫자와 바로 뒤의 수량·날짜·시간 단위를 보호어 후보로 추출한다."""
    terms = []
    for match in NUMBER_PATTERN.finditer(transcript):
        unit = re.match(r"(?:킬로그램|퍼센트|개입(?!니)|명분|시간|개월|년|월|일|시|분|초|원|개|명|권|번|층|호|살|대|장|건|회|미터|그램|리터|도)", transcript[match.end():])
        terms.append(match.group() + (unit.group() if unit else ""))
    return terms


def _resolve_spoken_term(source, spoken, offsets, transcript, term):
    """직접 일치하거나 전사문에서 대응하는 보호어를 발화의 원래 띄어쓰기로 추출한다."""
    compact_term = compact_text(term)
    if not compact_term:
        return []
    spans = [match.span() for match in re.finditer(re.escape(compact_term), spoken)]
    if not spans:
        spans = _map_transcript_term(spoken, transcript, compact_term)
    if not spans:
        return []
    return [source[offsets[start]:offsets[end - 1] + 1] for start, end in spans]


def numeric_alignment_issue(spoken_text, normalized_text, reading_aliases=None, reading_policy=None):
    """한글 숫자 대응을 입증하지 못한 예제를 자동 채택에서 분리한다."""
    spoken = compact_text(spoken_text)
    transcript = compact_text(normalized_text)
    if reading_aliases is not None:
        from .prompt_generation import apply_reading_aliases
        try:
            transcript = apply_reading_aliases(transcript, reading_aliases)
        except ValueError:
            return "수치가 중복되었거나 단위·표현 대응이 잘못되었습니다. 발화문과 정답을 확인하세요."
    if spoken == transcript and reading_policy is None:
        return None
    if reading_aliases is None and (not re.search(r"[가-힣]", spoken) or not NUMBER_PATTERN.search(transcript)):
        return None
    try:
        pattern = _transcript_pattern(transcript, reading_policy=reading_policy)
    except (ValueError, TypeError, KeyError):
        return "숫자 읽기 규칙과 문맥이 맞지 않습니다. 날짜·기간·인원·번호의 읽기를 확인하세요."
    if pattern is None or re.fullmatch(pattern, spoken) is None:
        return NUMERIC_ISSUE
    return None
