"""발화 원문을 보존하면서 중복과 검토 대상을 판정한다."""

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple


def normalize_text(text: str) -> str:
    """비교용으로 유니코드와 연속 공백만 정규화한다."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def parse_generated_examples(response_text: str) -> List[object]:
    """JSON 배열 또는 examples 배열을 가진 객체에서 후보를 읽는다."""
    content = response_text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", content, flags=re.DOTALL)
    if fenced:
        content = fenced.group(1)
    data = json.loads(content)
    examples = data.get("examples") if isinstance(data, dict) else data
    if not isinstance(examples, list):
        raise ValueError("생성 응답은 JSON 배열 또는 examples 배열이어야 합니다.")
    return examples


@dataclass(frozen=True)
class CandidateExample:
    id: str
    text: str
    protected_terms: Tuple[str, ...]

    def to_record(self) -> Dict[str, object]:
        """원문과 안정적인 ID를 JSON 저장용 레코드로 반환한다."""
        return asdict(self)


@dataclass(frozen=True)
class FilteringDecision:
    candidate: object
    example: Optional[CandidateExample]
    accepted: bool
    reason: str
    matched_id: Optional[str] = None
    similarity: Optional[float] = None
    needs_review: bool = False

    def to_record(self) -> Dict[str, object]:
        """제외 또는 검토 판정의 근거와 원래 후보를 반환한다."""
        return {
            "candidate": self.candidate,
            "example_id": self.example.id if self.example else None,
            "accepted": self.accepted,
            "reason": self.reason,
            "matched_id": self.matched_id,
            "similarity": self.similarity,
            "needs_review": self.needs_review,
        }


def create_example(candidate: object, max_characters: int) -> CandidateExample:
    """후보의 형식과 보호어를 검증하고 원문 기반 ID를 만든다."""
    record = {"text": candidate} if isinstance(candidate, str) else candidate
    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
        raise ValueError("text는 문자열이어야 합니다.")
    text = record["text"]
    if not text.strip() or len(text) > max_characters:
        raise ValueError("발화가 비어 있거나 최대 글자 수를 초과했습니다.")
    if any(unicodedata.category(character) == "Cc" for character in text):
        raise ValueError("발화에는 줄바꿈 등의 제어 문자를 넣을 수 없습니다.")
    terms = record.get("protected_terms", [])
    if not isinstance(terms, list) or any(not isinstance(term, str) or not term.strip() for term in terms):
        raise ValueError("protected_terms는 비어 있지 않은 문자열의 배열이어야 합니다.")
    if any(normalize_text(term) not in normalize_text(text) for term in terms):
        raise ValueError("보호어는 발화 원문에 포함되어야 합니다.")
    identifier = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return CandidateExample(identifier, text, tuple(sorted(set(map(normalize_text, terms)))))


def protected_signature(example: CandidateExample, terms: Tuple[str, ...]) -> Tuple[str, ...]:
    """명시된 보호어와 숫자·영문·한글 수량 표현의 차이를 감지한다."""
    text = normalize_text(example.text)
    lexical_values = re.findall(r"[0-9]+(?:[.,:/-][0-9]+)*|[A-Za-z]+[A-Za-z0-9_-]*", text)
    korean_values = re.findall(
        r"(?:영|공|일|이|삼|사|오|육|칠|팔|구|십|백|천|만|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|스무)"
        r"(?:[영공일이삼사오육칠팔구십백천만억조한두세네다섯여일곱덟아홉열스무]*)(?:\s*)(?:시|분|초|개|명|원|번|층|호|월|년|살)",
        text,
    )
    occurrences = sorted((match.start(), term) for term in terms for match in re.finditer(re.escape(term), text))
    known_terms = tuple(term for _, term in occurrences)
    return tuple(lexical_values) + tuple(korean_values) + known_terms


class ExampleDeduplicator:
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        near_duplicate_policy: str = "review",
        protected_terms: Iterable[str] = (),
        max_characters: int = 500,
    ) -> None:
        """유사도 기준과 보존 정책을 설정하고 채택 예제 인덱스를 만든다."""
        if not 0 < similarity_threshold <= 1:
            raise ValueError("유사도 기준은 0 초과 1 이하여야 합니다.")
        if near_duplicate_policy not in {"review", "drop"}:
            raise ValueError("유사 예제 정책은 review 또는 drop이어야 합니다.")
        if max_characters < 1:
            raise ValueError("최대 글자 수는 양수여야 합니다.")
        self.similarity_threshold = similarity_threshold
        self.near_duplicate_policy = near_duplicate_policy
        self.protected_terms = tuple(sorted({normalize_text(term) for term in protected_terms if term.strip()}))
        self.max_characters = max_characters
        self.accepted_examples: List[CandidateExample] = []
        self._exact_index: Dict[str, CandidateExample] = {}

    def consider(self, candidate: object) -> FilteringDecision:
        """기존 채택 예제와 비교해 후보의 채택·제외·검토 여부를 결정한다."""
        try:
            example = create_example(candidate, self.max_characters)
        except ValueError as error:
            return FilteringDecision(candidate, None, False, str(error))
        key = normalize_text(example.text)
        existing = self._exact_index.get(key)
        if existing is not None:
            return FilteringDecision(candidate, example, False, "exact_duplicate", existing.id, 1.0)
        match, score = self._find_similar(example)
        if match is not None:
            return self._decide_similar(candidate, example, match, score)
        self._accept(example)
        return FilteringDecision(candidate, example, True, "unique")

    def _find_similar(self, example: CandidateExample) -> Tuple[Optional[CandidateExample], float]:
        """보호 대상 값이 같은 예제 중 기준 이상의 가장 유사한 예제를 찾는다."""
        best_match = None
        best_score = 0.0
        for existing in self.accepted_examples:
            score = self._similarity(example, existing)
            if score >= self.similarity_threshold and score > best_score:
                best_match, best_score = existing, score
        return best_match, best_score

    def _similarity(self, example: CandidateExample, existing: CandidateExample) -> float:
        """보호어가 달라지면 유사 중복에서 제외하고 문자 유사도를 계산한다."""
        terms = tuple(sorted(set(self.protected_terms + example.protected_terms + existing.protected_terms)))
        if protected_signature(example, terms) != protected_signature(existing, terms):
            return 0.0
        return SequenceMatcher(None, normalize_text(example.text), normalize_text(existing.text), autojunk=False).ratio()

    def _decide_similar(
        self, candidate: object, example: CandidateExample, match: CandidateExample, score: float
    ) -> FilteringDecision:
        """설정에 따라 유사 예제를 제외하거나 채택 후 검토 대상으로 남긴다."""
        accepted = self.near_duplicate_policy == "review"
        if accepted:
            self._accept(example)
        return FilteringDecision(candidate, example, accepted, "near_duplicate", match.id, score, accepted)

    def _accept(self, example: CandidateExample) -> None:
        """채택 예제를 다음 후보와도 비교할 수 있도록 즉시 등록한다."""
        self.accepted_examples.append(example)
        self._exact_index[normalize_text(example.text)] = example
