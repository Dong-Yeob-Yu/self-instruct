"""도메인을 고정하지 않고 상황별 계획과 수치에 독립적인 문장 형식을 관리한다."""

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from .number_validation import NUMBER_PATTERN, PHONE_PATTERN


MAX_PLAN_SLOTS = 16
PLAN_SCHEMA = {
    "type": "object", "properties": {"plans": {"type": "array", "minItems": 1, "maxItems": MAX_PLAN_SLOTS,
        "items": {"type": "object", "properties": {
            "situation": {"type": "string", "minLength": 1, "maxLength": 160},
            "purpose": {"type": "string", "minLength": 1, "maxLength": 160},
            "form": {"type": "string", "minLength": 1, "maxLength": 160},
        }, "required": ["situation", "purpose", "form"], "additionalProperties": False},
    }}, "required": ["plans"], "additionalProperties": False,
}


def planning_instructions():
    """요청 범위 안에서 서로 다른 상황·목적·전개를 먼저 설계하도록 지시한다."""
    return (
        "ASR 학습 문장을 생성하기 전에 다양성 계획만 만드세요. JSON의 user_prompt와 common_prompt 범위만 따르세요. "
        "정확히 plan_count개의 계획을 plans 배열에 작성하세요. 실제 문장·숫자 예시·전화번호는 생성하지 마세요. "
        "각 계획은 situation(구체적인 상황), purpose(발화로 전달하거나 확인할 정보), form(문장 전개 방식)입니다. "
        "모든 계획의 상황과 정보 목적이 달라야 합니다. 대상명·숫자·동의어만 바꾸는 것은 다른 상황이 아닙니다. "
        "여러 주제를 요청했다면 주제별로 고르게 나누고, 한 주제라도 서로 다른 사건·조건·관점을 찾으세요. "
        "연락처로 문의하라는 안내처럼 같은 해결 방법을 반복하지 마세요. "
        "질문·답변·경험 보고·문제 제기·조건 설명·비교 등 요청이 허용하는 발화 목적을 분산하세요. "
        "전개도 결론 먼저·이유 먼저·조건 먼저·짧은 문장 등 다양하게 하되 사용자 지정 문체·형식을 지키세요. "
        "도메인·주제를 새로 추가하지 마세요. 제공된 피드백이 있으면 중복 계획을 새 상황으로 교체하세요. "
        '설명 없이 {"plans":[{"situation":"상황","purpose":"정보 목적","form":"문장 전개"}]} 형식의 JSON만 출력하세요.\n'
    )


def validate_generation_plan(response, target):
    """계획의 개수·문자열·중복을 검사하고 안정적인 슬롯과 할당 개수를 만든다."""
    plans = response.get("plans") if isinstance(response, dict) else None
    count = min(target, MAX_PLAN_SLOTS)
    if not isinstance(plans, list) or len(plans) != count:
        raise ValueError(f"서로 다른 상황 계획 {count}개가 필요합니다.")
    result, seen = [], set()
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict) or any(not isinstance(plan.get(key), str) or not 1 <= len(plan[key].strip()) <= 160
                                             for key in ("situation", "purpose", "form")):
            raise ValueError("계획의 상황·목적·전개를 각각 1~160자로 작성하세요.")
        key = tuple(re.sub(r"\W+", "", plan[field]) for field in ("situation", "purpose"))
        if key in seen:
            raise ValueError("상황과 발화 목적이 같은 계획이 반복되었습니다.")
        seen.add(key)
        result.append({**{field: plan[field].strip() for field in ("situation", "purpose", "form")},
                       "id": f"p{index + 1}", "quota": target // count + (index < target % count)})
    return result


def select_plan_slots(plans, rows, attempts, count):
    """사용하지 않은 계획부터 순환하고 실패한 슬롯 하나가 나머지를 막지 않게 한다."""
    used = Counter(row.get("plan_id") for row in rows if row["accepted"])
    available = [plan for plan in plans if used[plan["id"]] < plan["quota"]]
    return sorted(available, key=lambda plan: (attempts.get(plan["id"], 0), used[plan["id"]] / plan["quota"]))[:count]


def structure_pattern(text, protected_terms=()):
    """숫자·연락처·명시된 고유명사를 가려 값만 다른 문장을 같은 형식으로 비교한다."""
    result = unicodedata.normalize("NFC", text).strip()
    result = re.sub(r"^(?:(?:고객님|안녕하세요)(?:[\s,.!?]+|$))+", "", result)
    for term in sorted(set(protected_terms), key=len, reverse=True):
        if term and term in result:
            result = result.replace(term, "〈보호값〉")
    result = PHONE_PATTERN.sub("〈값〉", result)
    result = NUMBER_PATTERN.sub("〈값〉", result)
    return re.sub(r"\s+", " ", result).strip()


def _structure_key(pattern):
    """공백·문장부호만 다른 형식을 동일하게 비교한다."""
    return re.sub(r"[\W_]+", "", pattern).lower()


def _meaning_markers(pattern):
    """부정·허용·비교 조건이 달라진 문장은 높은 문자 유사도만으로 합치지 않는다."""
    return tuple(re.findall(r"이상|이하|초과|미만|이내|이외|이전|이후|가능|불가|않|못|없|아니", pattern))


class StructureIndex:
    def __init__(self):
        """값이 아닌 문장 형식으로 채택 결과를 비교할 인덱스를 준비한다."""
        self._exact = {}
        self._accepted = []

    def find(self, pattern):
        """동일 형식 또는 긴 문장에서 표현만 조금 바꾼 반복을 찾는다."""
        key = _structure_key(pattern)
        if key in self._exact:
            return self._exact[key], 1.0
        markers = _meaning_markers(pattern)
        for previous, identifier, previous_markers in self._accepted:
            if markers != previous_markers:
                continue
            score = self._similarity(key, previous)
            if score >= 0.9:
                return identifier, score
        return None, None

    def _similarity(self, key, previous):
        """짧은 문장의 우연한 겹침을 피하고 긴 문장의 높은 유사도만 비교한다."""
        if min(len(key), len(previous)) < 24:
            return 0.0
        matcher = SequenceMatcher(None, key, previous, autojunk=False)
        if matcher.real_quick_ratio() < 0.9 or matcher.quick_ratio() < 0.9:
            return 0.0
        return matcher.ratio()

    def add(self, pattern, identifier):
        """실제 채택된 형식만 후속 후보의 비교 대상으로 등록한다."""
        key = _structure_key(pattern)
        self._exact[key] = identifier
        self._accepted.append((key, identifier, _meaning_markers(pattern)))

    def discard(self, identifier):
        """정답 충돌로 채택이 취소된 행은 형식 비교 기준에서도 제거한다."""
        self._exact = {key: value for key, value in self._exact.items() if value != identifier}
        self._accepted = [entry for entry in self._accepted if entry[1] != identifier]
