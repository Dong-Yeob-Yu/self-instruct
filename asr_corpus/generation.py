"""로컬 LLM에서 발화 후보를 생성하고 배치별 필터링 결과를 반환한다."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from .examples import ExampleDeduplicator, FilteringDecision, parse_generated_examples


@dataclass(frozen=True)
class LocalLlmConfig:
    endpoint: str
    model: str
    backend: str = "openai-compatible"
    temperature: float = 0.8
    max_tokens: int = 2048
    timeout: float = 120.0
    api_key_env: Optional[str] = None

    def __post_init__(self) -> None:
        """통신 전에 서버 주소와 생성 설정을 검증한다."""
        address = urllib.parse.urlsplit(self.endpoint)
        if address.scheme not in {"http", "https"} or not address.hostname:
            raise ValueError("endpoint에는 http 또는 https 전체 주소를 지정해야 합니다.")
        if address.username or address.password or address.query or address.fragment:
            raise ValueError("endpoint에는 인증 정보, 쿼리 또는 fragment를 넣을 수 없습니다.")
        if not self.model.strip() or self.backend not in {"openai-compatible", "ollama"}:
            raise ValueError("모델 이름과 지원하는 backend를 지정해야 합니다.")
        if self.max_tokens < 1 or self.timeout <= 0 or not 0 <= self.temperature <= 2:
            raise ValueError("토큰 수·제한 시간은 양수이고 temperature는 0~2여야 합니다.")


class LocalLlmClient:
    def __init__(self, config: LocalLlmConfig) -> None:
        """모델을 로드하거나 서버에 연결하지 않고 클라이언트 설정만 보관한다."""
        self.config = config

    def generate(self, prompt: str) -> str:
        """명시된 로컬 API로 비스트리밍 생성 요청을 한 번 보낸다."""
        request = self._build_request(prompt)
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"로컬 LLM 요청이 실패했습니다. HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RuntimeError("로컬 LLM에 연결하지 못했습니다. 주소와 실행 상태를 확인하세요.") from error
        return self._extract_text(data)

    def _build_request(self, prompt: str) -> urllib.request.Request:
        """선택한 API 규격에 맞는 JSON 요청과 선택적 인증 헤더를 만든다."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            headers["Authorization"] = "Bearer " + self._read_api_key()
        payload = self._build_payload(prompt)
        return urllib.request.Request(
            self.config.endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST"
        )

    def _read_api_key(self) -> str:
        """사용자가 지정한 환경 변수에서만 인증 값을 읽는다."""
        key = os.environ.get(self.config.api_key_env or "")
        if not key:
            raise ValueError("지정한 인증 환경 변수에 값이 없습니다.")
        return key

    def _build_payload(self, prompt: str) -> Dict[str, object]:
        """Ollama 또는 OpenAI 호환 로컬 서버용 생성 파라미터를 만든다."""
        if self.config.backend == "ollama":
            return {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": self.config.temperature, "num_predict": self.config.max_tokens},
            }
        return {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

    def _extract_text(self, data: object) -> str:
        """API 응답에서 생성 문자열을 읽고 잘린 응답과 잘못된 형식을 거부한다."""
        try:
            text, finish_reason = self._response_fields(data)
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("로컬 LLM 응답 형식이 선택한 backend와 맞지 않습니다.") from error
        if finish_reason in {"length", "max_tokens"}:
            raise ValueError("LLM 응답이 길이 제한으로 잘렸습니다. batch-size를 줄이거나 max-tokens를 늘리세요.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("로컬 LLM이 비어 있거나 문자열이 아닌 응답을 반환했습니다.")
        return text

    def _response_fields(self, data: object) -> tuple:
        """백엔드별 생성 본문과 종료 사유를 추출한다."""
        if self.config.backend == "ollama":
            return data["response"], data.get("done_reason")
        choice = data["choices"][0]
        return choice["message"]["content"], choice.get("finish_reason")


def build_prompt(requirement: str, count: int, previous_examples: List[str], protected_terms: tuple = ()) -> str:
    """요구사항과 기존 발화를 바탕으로 구조화된 새 발화 생성 프롬프트를 만든다."""
    context = json.dumps(
        {"requirement": requirement, "count": count, "previous_examples": previous_examples, "protected_terms": protected_terms},
        ensure_ascii=False,
    )
    return (
        "ASR 학습용 발화 예제를 생성하세요. 요구사항에 지정된 언어와 주제, 말투를 따르세요. "
        "설명, 정답, 지시문 목록이 아니라 사람이 실제 말할 수 있는 독립적인 발화를 만드세요. "
        "단순 반복을 피하고 다양한 표현을 사용하세요. previous_examples를 그대로 복제하지 마세요. "
        "숫자, 시간, 장소, 인명, 제품명 등 학습에 중요한 표현을 protected_terms에 원문 그대로 기록하세요. "
        "발화 하나에 줄바꿈을 넣지 마세요. JSON 이외의 설명은 출력하지 마세요. "
        '반환 형식은 {"examples": [{"text": "발화 원문", "protected_terms": []}]}입니다.\n'
        + context
    )


@dataclass(frozen=True)
class GenerationBatch:
    request_index: int
    prompt: str
    raw_response: str
    decisions: List[FilteringDecision]
    error: Optional[str] = None


class CorpusGenerator:
    def __init__(self, client: LocalLlmClient, deduplicator: ExampleDeduplicator) -> None:
        """교체 가능한 LLM 클라이언트와 중복 판정기를 연결한다."""
        self.client = client
        self.deduplicator = deduplicator

    def generate(self, requirement: str, count: int, batch_size: int, max_requests: int) -> Iterator[GenerationBatch]:
        """목표 개수 또는 요청 상한까지 생성해 원문과 판정 결과를 배치별로 반환한다."""
        if not requirement.strip():
            raise ValueError("실제 생성 시에는 학습 요구사항을 입력해야 합니다.")
        if min(count, batch_size, max_requests) < 1:
            raise ValueError("목표 개수, 배치 크기, 최대 요청 횟수는 양수여야 합니다.")
        for request_index in range(1, max_requests + 1):
            remaining = count - len(self.deduplicator.accepted_examples)
            if remaining <= 0:
                return
            yield self._generate_batch(requirement, min(batch_size, remaining), count, request_index)

    def _generate_batch(self, requirement: str, batch_count: int, target_count: int, index: int) -> GenerationBatch:
        """한 배치의 응답을 파싱하고 형식 오류도 원문과 함께 반환한다."""
        previous = [example.text for example in self.deduplicator.accepted_examples[-8:]]
        prompt = build_prompt(requirement, batch_count, previous, self.deduplicator.protected_terms)
        raw_response = self.client.generate(prompt)
        try:
            candidates = parse_generated_examples(raw_response)
        except ValueError as error:
            return GenerationBatch(index, prompt, raw_response, [], str(error))
        decisions = self._filter_candidates(candidates, target_count)
        return GenerationBatch(index, prompt, raw_response, decisions)

    def _filter_candidates(self, candidates: List[object], target_count: int) -> List[FilteringDecision]:
        """배치 내 중복까지 판정하고 목표 개수를 넘는 후보는 채택하지 않는다."""
        decisions = []
        for candidate in candidates:
            if len(self.deduplicator.accepted_examples) >= target_count:
                decisions.append(FilteringDecision(candidate, None, False, "target_reached"))
                continue
            decisions.append(self.deduplicator.consider(candidate))
        return decisions
