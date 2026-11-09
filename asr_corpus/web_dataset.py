"""웹 작업의 발화·전사문 쌍과 생성 실행 상태를 관리한다."""

import copy
import json
import re
import random
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .typed_generation import (NUMERIC_VARIATION_TYPES, validate_money_digit_weights, select_money_digits, type_quotas, validate_type_weights, select_types, make_slots, response_schema, template_prompt,
                               indexed_response, normalize_number_template, validate_templates, generate_number, reading_prompt, render_pair)
from .model_runtime import ModelRuntime
from .excel_import import read_excel_examples
from .examples import ExampleDeduplicator, create_example, normalize_text, parse_generated_examples
from .local import LlamaCppClient
from .number_validation import numeric_alignment_issue, spoken_protected_terms
from .number_generation import NUMBER_SCHEMA, numeric_prompt_instructions, render_numeric_candidate
from .prompt_generation import DEFAULT_COMMON_PROMPT, PROMPT_SCHEMA, prompt_instructions, render_prompt_candidate, valid_reading_aliases
from .reading_policy import validate_reading_rules
from .diversity import MAX_PLAN_SLOTS, PLAN_SCHEMA, StructureIndex, planning_instructions, select_plan_slots, structure_pattern, validate_generation_plan


PAIR_SCHEMA = {
    "type": "object",
    "properties": {"examples": {
        "type": "array", "items": {
            "type": "object",
            "properties": {
                "normalized_text": {"type": "string"}, "spoken_text": {"type": "string"},
                "protected_terms": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["spoken_text", "normalized_text", "protected_terms"], "additionalProperties": False,
        },
    }},
    "required": ["examples"], "additionalProperties": False,
}
ACTIVE_STATUSES = {"queued", "running", "cancelling"}
MAX_STALLED_BATCHES = 5
BATCH_DIVERSITY_HINTS = (
    "이전 예제와 다른 주어·대상·상황을 사용하세요.",
    "이전 예제와 다른 문장 길이·어순·정보 배치를 사용하세요.",
    "숫자가 필요한 요청이면 이전과 다른 수치·단위 조합을 사용하세요.",
    "사용자가 허용한 범위에서 문장 종결과 표현 방식을 바꾸세요.",
)


def current_timestamp():
    """작업 시각을 애플리케이션에서 생성한다."""
    return datetime.now(timezone.utc).isoformat()


class GenerationCancelledError(Exception):
    """사용자가 요청한 로컬 추론 중지를 나타낸다."""


class PairedLlamaClient(LlamaCppClient):
    supports_planning = True

    def __init__(self, config, cancellation):
        """기존 로컬 추론 설정에 중지 신호를 연결한다."""
        super().__init__(config)
        self.cancellation = cancellation
        self.number_mode = False
        self.prompt_mode = False
        self.planning_mode = False
        self.typed_phase = None
        self._uses_cli_output = False

    def _build_command(self, prompt):
        """기존 오프라인 실행 옵션에서 출력 스키마만 발화·전사문 쌍으로 바꾼다."""
        command = super()._build_command(prompt)
        command[0] = self._structured_executable()
        self._uses_cli_output = Path(command[0]).name == "llama-cli"
        if self._uses_cli_output:
            command.extend(["--reasoning-budget", "0"])
        schema = self._response_schema(prompt)
        command[command.index("--json-schema") + 1] = json.dumps(schema)
        return command

    def _response_schema(self, prompt):
        """이번 요청의 개수와 계획 id까지 출력 제약에 반영한다."""
        if self.typed_phase:
            return self._typed_response_schema(prompt)
        schema = copy.deepcopy(PLAN_SCHEMA if self.planning_mode else PROMPT_SCHEMA if self.prompt_mode else NUMBER_SCHEMA if self.number_mode else PAIR_SCHEMA)
        try:
            context = json.loads(prompt.splitlines()[-1])
        except (ValueError, IndexError):
            return schema
        if not isinstance(context, dict) or not (self.planning_mode or self.prompt_mode):
            return schema
        key = "plans" if self.planning_mode else "examples"
        count = context.get("plan_count" if self.planning_mode else "count")
        if type(count) is int and 1 <= count <= MAX_PLAN_SLOTS:
            schema["properties"][key].update(minItems=count, maxItems=count)
        slots = context.get("planned_slots", [])
        if self.prompt_mode and not self.planning_mode and isinstance(slots, list) and slots:
            identifiers = [slot["id"] for slot in slots if isinstance(slot, dict) and isinstance(slot.get("id"), str)]
            schema["properties"]["examples"]["items"]["properties"]["plan_id"]["enum"] = identifiers
        return schema

    def _typed_response_schema(self, prompt):
        """유형별 단계의 요청 개수와 id를 JSON 출력 스키마에도 고정한다."""
        schema = response_schema(self.typed_phase)
        try:
            context = json.loads(prompt.splitlines()[-1])
        except (ValueError, IndexError):
            return schema
        slots = context.get("slots", context.get("numbers", [])) if isinstance(context, dict) else []
        if not isinstance(slots, list) or not slots:
            return schema
        identifiers = [slot["id"] for slot in slots if isinstance(slot, dict) and isinstance(slot.get("id"), str)]
        items = schema["properties"]["examples"]
        items.update(minItems=len(slots), maxItems=len(slots))
        items["items"]["properties"]["id"]["enum"] = identifiers
        return schema

    def _structured_executable(self):
        """완료 도구와 같은 설치 위치의 구조화 응답 지원 CLI를 우선 사용한다."""
        executable = Path(self.config.executable)
        if executable.name != "llama-completion":
            return str(executable)
        resolved = shutil.which(str(executable))
        if not resolved:
            return str(executable)
        candidate = Path(resolved).resolve().with_name("llama-cli")
        return str(candidate) if candidate.is_file() else str(executable)

    def _extract_response(self, output, validate_examples=True):
        """CLI 화면 출력에서 마지막 JSON 응답과 알려진 종료 표시만 분리한다."""
        if not self._uses_cli_output:
            return super()._extract_response(output, validate_examples)
        for match in reversed(list(re.finditer(rb"(?m)^[ \t]*(?=\{)", output))):
            try:
                text = output[match.end():].decode("utf-8")
            except UnicodeDecodeError:
                continue
            response = self._decode_cli_json(text, validate_examples)
            if response is not None:
                return response
        raise ValueError("로컬 CLI의 완전한 JSON 응답을 찾지 못했습니다. 작은 배치로 다시 생성합니다.")

    def _decode_cli_json(self, text, validate_examples):
        """프롬프트 예시를 답으로 오인하지 않도록 응답 뒤의 CLI 종료 형식까지 확인한다."""
        try:
            value, end = json.JSONDecoder().raw_decode(text)
        except ValueError:
            return None
        if not re.fullmatch(r"\s*(?:\[ Prompt:[^\]\r\n]+\]\s*)?Exiting\.\.\.\s*", text[end:]):
            return None
        if not isinstance(value, dict) or ("examples" if validate_examples else "plans") not in value:
            return None
        response = text[:end]
        if validate_examples:
            parse_generated_examples(response)
        return response

    def generate_plan(self, prompt):
        """문장 생성과 분리된 계획 스키마로 상황·발화 목적을 먼저 생성한다."""
        self.planning_mode = True
        try:
            return self.generate(prompt)
        finally:
            self.planning_mode = False

    def generate(self, prompt):
        """로컬 추론을 실행하고 사용자 중지와 제한 시간을 처리한다."""
        if self.cancellation.is_set():
            raise GenerationCancelledError()
        with subprocess.Popen(self._build_command(prompt), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            output, error = self._wait_for_process(process)
            if process.returncode != 0:
                raise RuntimeError("로컬 모델 실행 실패: " + error[-1200:].decode("utf-8", errors="replace"))
        return self._extract_response(output, validate_examples=not self.planning_mode)

    def _wait_for_process(self, process):
        """추론 종료를 기다리면서 중지 요청을 짧은 간격으로 확인한다."""
        started = time.monotonic()
        while True:
            self._check_interruption(process, started)
            result = self._try_communicate(process)
            if result is not None:
                return result

    def _check_interruption(self, process, started):
        """작업 소유 프로세스만 종료하고 중지 또는 시간 초과를 알린다."""
        if self.cancellation.is_set():
            self._stop_process(process)
            raise GenerationCancelledError()
        if time.monotonic() - started > self.config.timeout:
            self._stop_process(process)
            raise RuntimeError("생성 제한 시간을 초과했습니다. 배치 크기를 줄여 다시 시도하세요.")

    def _try_communicate(self, process):
        """출력 수집을 잠시 기다리고 추론 중이면 제어를 돌려준다."""
        try:
            return process.communicate(timeout=0.3)
        except subprocess.TimeoutExpired:
            return None

    def _stop_process(self, process):
        """사용자가 중지한 추론 프로세스를 종료하고 출력을 정리한다."""
        process.terminate()
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()


def build_pair_prompt(prompt, count, previous, feedback=None, batch_index=1, number_mode=False, prompt_mode=False, common_prompt="", reading_rules=None, planned_slots=None, used_patterns=None, used_amounts=None):
    """사용자 조건에 이전 예제·실패 이유와 배치별 다양화 지침을 붙인다."""
    context = {"user_prompt": prompt, "count": count, "previous_examples": previous,
               "failed_examples": feedback or [], "batch_index": batch_index,
               "diversity_hint": BATCH_DIVERSITY_HINTS[(batch_index - 1) % len(BATCH_DIVERSITY_HINTS)]}
    if prompt_mode:
        context["previous_examples"] = []
        context["common_prompt"] = common_prompt
        context["reading_rules"] = validate_reading_rules(reading_rules)
        context["planned_slots"] = planned_slots or []
        context["used_amounts"] = used_amounts or []
        context["used_patterns"] = used_patterns if used_patterns is not None else [
            structure_pattern(row.get("normalized_text", "")) for row in previous if row.get("normalized_text")]
        context["diversity_hint"] = "계획마다 다른 사건·발화 목적·문장 전개를 쓰고 used_patterns의 형식을 반복하지 마세요."
        return prompt_instructions() + json.dumps(context, ensure_ascii=False)
    if number_mode:
        return numeric_prompt_instructions() + json.dumps(context, ensure_ascii=False)
    return (
        "사용자의 요구사항에 맞는 ASR 학습용 발화 예제를 생성하세요. user_prompt의 언어와 변환 규칙을 따르세요. "
        "spoken_text에는 TTS가 읽을 자연스러운 발화문을, normalized_text에는 ASR이 출력해야 할 정답 전사문을 넣으세요. "
        "normalized_text를 먼저 정한 뒤, 모든 숫자와 단위의 값을 그대로 유지하며 spoken_text를 작성하세요. "
        "한국어 숫자 정규화를 요청하면 발화문의 숫자를 한글로 쓰고 전사문의 해당 숫자만 아라비아 숫자로 변환하세요. "
        "금액은 자연스러운 한자어 수사로 읽으세요. 1250원→천이백오십 원이며 일천이백오십 원으로 쓰지 마세요. "
        "인원·수량은 고유어 수 관형사를 우선하세요. 10명→열 명, 1명→한 명, 20명→스무 명, 21명→스물한 명입니다. "
        "날짜의 월과 일을 생략하거나 임의로 바꾸지 마세요. 6월은 유월, 10월은 시월입니다. "
        "날짜는 숫자 읽기 뒤에 날짜 단위 '일'을 한 번만 붙이세요. "
        "1일→일일, 3일→삼일, 10일→십일, 11일→십일일, 15일→십오일, "
        "21일→이십일일, 25일→이십오일, 31일→삼십일일입니다. "
        "3일을 '삼일일', 15일을 '십오일일', 25일을 '이십오일일'로 쓰지 마세요. "
        "11일·21일·31일은 숫자 읽기 자체가 '일'로 끝나므로 날짜 단위와 합쳐 '일일'로 끝나는 것이 맞습니다. "
        "조사 '에'는 임의로 추가하거나 삭제하지 마세요. "
        "정답이 '3일에'이면 발화는 '삼일에', 정답이 '3일'이면 발화는 '삼일'로 작성하세요. "
        "오후 2시부터 5시간은 오후 두 시부터 다섯 시간으로 읽고, 35,000원은 삼만 오천 원으로 읽습니다. "
        "예: normalized_text가 '이 상품은 35,000원이고, 12월 15일에 오후 2시부터 5시간 동안 진행됩니다.'이면 "
        "spoken_text는 '이 상품은 삼만 오천 원이고, 십이월 십오일에 오후 두 시부터 다섯 시간 동안 진행됩니다.'입니다. "
        "별도 변환 요청이 없으면 두 필드를 동일하게 작성하세요. 고유명사와 숫자 외 문장 내용은 유지하세요. "
        "전화번호의 선행 0을 보존하세요. 문장 하나에 줄바꿈을 넣지 마세요. 설명과 코드 블록은 출력하지 마세요. "
        "사용자 조건을 우선하고 그 범위 안에서 diversity_hint에 따라 요청한 count만큼 다양하게 생성하세요. "
        "previous_examples와 failed_examples는 이전 결과의 일부이며 복제할 정답 예시가 아닙니다. "
        "실패 이유를 참고해 동일한 오류를 피하고, 숫자만 바꾸지 말고 상황과 문장 구조도 바꾸세요. "
        "숫자 외 조사와 문장 내용은 두 필드에서 그대로 유지하세요. "
        "protected_terms에는 숫자·장소·인명 등 중요한 표현을 spoken_text에 등장하는 원문 그대로 적으세요. "
        "전사문의 아라비아 숫자를 보호어로 복사하지 마세요. 발화가 '삼백오십 개와 오천 오백 원'이면 "
        "protected_terms는 ['삼백오십 개', '오천 오백 원']입니다. 정확한 발화 표현을 고를 수 없으면 빈 배열을 쓰세요. "
        'JSON 형식은 {"examples":[{"spoken_text":"발화문","normalized_text":"정답 전사문","protected_terms":[]}]}입니다.\n'
        + json.dumps(context, ensure_ascii=False)
    )


def validate_settings(payload):
    """웹 요청의 생성 범위와 프롬프트를 검증한다."""
    if not isinstance(payload, dict):
        raise ValueError("생성 설정은 JSON 객체여야 합니다.")
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 12000:
        raise ValueError("프롬프트를 1~12,000자로 입력하세요.")
    count = payload.get("count", 20)
    batch_size = payload.get("batch_size", 3 if payload.get("prompt_mode") is True else 1)
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("목표 개수는 1~1,000 사이의 정수여야 합니다.")
    if type(batch_size) is not int or not 1 <= batch_size <= 10:
        raise ValueError("배치 크기는 1~10 사이의 정수여야 합니다.")
    threshold = payload.get("similarity_threshold", 0.95)
    if type(threshold) not in {int, float} or not 0 < threshold <= 1:
        raise ValueError("유사도 기준은 0 초과 1 이하여야 합니다.")
    policy = payload.get("near_duplicate_policy", "review")
    if policy not in {"review", "drop"}:
        raise ValueError("유사 예제 정책을 확인하세요.")
    number_mode = payload.get("number_mode", False)
    if type(number_mode) is not bool:
        raise ValueError("숫자 읽기 자동 생성 여부는 boolean으로 전달하세요.")
    prompt_mode = payload.get("prompt_mode", False)
    common_prompt = payload.get("common_prompt", "")
    output_root = payload.get("output_root", "/tmp/ai-dataset/")
    if type(prompt_mode) is not bool or not isinstance(common_prompt, str) or len(common_prompt) > 6000:
        raise ValueError("공통 규칙은 6,000자 이내 문자열이고 프롬프트 모드는 boolean이어야 합니다.")
    if prompt_mode:
        if batch_size not in {1, 3, 5, 10}:
            raise ValueError("배치 크기는 1·3·5·10개 중 선택하세요.")
        threshold, policy = 0.95, "review"
        common_prompt = DEFAULT_COMMON_PROMPT
    if not isinstance(output_root, str) or not output_root.strip() or not Path(output_root).expanduser().is_absolute():
        raise ValueError("저장 위치는 절대 경로로 입력하세요.")
    auto_audio = payload.get("auto_audio", False)
    if type(auto_audio) is not bool:
        raise ValueError("자동 음성 생성 여부는 boolean으로 전달하세요.")
    type_weights = validate_type_weights(payload["type_weights"]) if "type_weights" in payload else None
    money_digit_weights = validate_money_digit_weights(payload.get("money_digit_weights")) if "money" in (type_weights or {}) or "money_digit_weights" in payload else None
    return {"auto_audio": auto_audio, "type_weights": type_weights, "money_digit_weights": money_digit_weights, "prompt": prompt.strip(), "count": count, "batch_size": batch_size, "similarity_threshold": threshold,
            "near_duplicate_policy": policy, "number_mode": number_mode, "prompt_mode": prompt_mode,
            "common_prompt": common_prompt, "output_root": output_root.strip(),
            "reading_rules": validate_reading_rules(None if prompt_mode else payload.get("reading_rules"))}


def job_counts(job):
    """채택·검토·제외 수와 처리한 후보 수를 계산한다."""
    rows = job["examples"]
    return {
        "accepted": sum(row["accepted"] for row in rows),
        "review": sum(row["needs_review"] for row in rows),
        "rejected": sum(not row["accepted"] and not row["needs_review"] for row in rows),
        "total": len(rows),
    }


class DatasetAccumulator:
    def __init__(self, settings):
        """발화 기준 중복 판정기와 대응 전사문을 추적할 인덱스를 준비한다."""
        self.deduplicator = ExampleDeduplicator(settings["similarity_threshold"], settings["near_duplicate_policy"], max_characters=1000)
        self.rows = []
        self._spoken_rows = {}
        self._source_ids = {}
        self._pair_rows = {}
        self._invalid_rows = {}
        self.used_amounts = set()
        self.last_batch_issue = None
        self.structure_index = StructureIndex() if settings.get("prompt_mode") else None

    def add(self, candidate):
        """발화·전사문 쌍을 검증하고 중복과 정답 충돌을 함께 판정한다."""
        row = {"id": uuid.uuid4().hex, "spoken_text": "", "normalized_text": "", "protected_terms": [],
               "accepted": False, "needs_review": False, "reason": "invalid", "matched_id": None, "similarity": None, "quality_issue": None}
        if isinstance(candidate, dict) and isinstance(candidate.get("plan_id"), str):
            row["plan_id"] = candidate["plan_id"]
        try:
            candidate = self._validate_candidate(candidate)
        except ValueError as error:
            return self._add_invalid(row, candidate, str(error))
        row.update(spoken_text=candidate["spoken_text"], normalized_text=candidate["normalized_text"], protected_terms=candidate.get("protected_terms", []))
        for key in ("type", "number_context", "template", "generated_number"):
            if key in candidate:
                row[key] = copy.deepcopy(candidate[key])
        if "reading_aliases" in candidate:
            row.update(reading_aliases=candidate["reading_aliases"], review_note=candidate.get("review_note", ""))
        if "reading_policy" in candidate:
            row["reading_policy"] = candidate["reading_policy"]
        row["quality_issue"] = numeric_alignment_issue(row["spoken_text"], row["normalized_text"], row.get("reading_aliases"), row.get("reading_policy"))
        key = normalize_text(row["spoken_text"])
        pair_key = (key, normalize_text(row["normalized_text"]))
        previous = self._pair_rows.get(pair_key)
        if previous is not None:
            return self._add_duplicate(row, previous)
        self._pair_rows[pair_key] = row
        existing = self._spoken_rows.get(key, [])
        if self._has_conflicting_transcript(row, existing):
            return self._add_conflict(row, existing, key)
        if row["quality_issue"]:
            return self._add_alignment_review(row, key)
        if self._repeated_structure(row):
            self.rows.append(row)
            return row
        amounts = {value.replace(",", "") for value in re.findall(r"(?<![\d,.])\d[\d,]*(?:\.\d+)?(?=원)", row["normalized_text"])} if self.structure_index is not None else set()
        if row.get("type") != "money" and amounts & self.used_amounts:
            row.update(needs_review=True, reason="repeated_amount", review_note="이 작업에서 이미 사용한 금액입니다. 다른 금액으로 수정하거나 필요한 반복인지 확인하세요.")
            self.rows.append(row)
            return row
        decision = self.deduplicator.consider({"text": row["spoken_text"], "protected_terms": row["protected_terms"]})
        row.update(accepted=decision.accepted, needs_review=decision.needs_review, reason=decision.reason,
                   matched_id=self._source_ids.get(decision.matched_id), similarity=decision.similarity)
        if row.get("review_note") and decision.accepted:
            row.update(needs_review=True, reason="reading_review")
        self.rows.append(row)
        self._spoken_rows.setdefault(key, []).append(row)
        if decision.accepted:
            self.used_amounts.update(amounts)
            self._source_ids[decision.example.id] = row["id"]
            if self.structure_index is not None and row.get("type") not in NUMERIC_VARIATION_TYPES:
                self.structure_index.add(row["structure_pattern"], row["id"])
        return row

    def _repeated_structure(self, row):
        """숫자·고유명사를 바꾼 반복은 기존 값 보호 판정과 별도로 제외한다."""
        if self.structure_index is None or row.get("type") in NUMERIC_VARIATION_TYPES:
            return False
        pattern = structure_pattern(row["normalized_text"], row["protected_terms"])
        row["structure_pattern"] = pattern
        matched_id, score = self.structure_index.find(pattern)
        if matched_id is None:
            return False
        row.update(accepted=False, needs_review=False, reason="structure_duplicate", matched_id=matched_id, similarity=score)
        return True

    def _add_duplicate(self, row, previous):
        """채택·검토·제외 여부와 관계없이 같은 후보의 재등장을 제외한다."""
        row.update(accepted=False, needs_review=False, reason="exact_duplicate", matched_id=previous["id"])
        self.rows.append(row)
        return row

    def _add_invalid(self, row, candidate, reason):
        """형식 오류 원문을 보존하고 동일한 잘못된 후보의 반복도 제외한다."""
        row.update(reason=reason, candidate=candidate)
        key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        previous = self._invalid_rows.get(key)
        if previous is not None:
            return self._add_duplicate(row, previous)
        self._invalid_rows[key] = row
        self.rows.append(row)
        return row

    def _add_alignment_review(self, row, key):
        """숫자 대응이 맞지 않는 원문을 보존하되 자동 채택과 내보내기에서 제외한다."""
        row.update(accepted=False, needs_review=True, reason="numeric_alignment")
        self.rows.append(row)
        self._spoken_rows.setdefault(key, []).append(row)
        return row

    def _validate_candidate(self, candidate):
        """두 텍스트를 검증하고 전사문 기준 보호어를 발화 표현으로 보정한다."""
        if not isinstance(candidate, dict):
            raise ValueError("발화·전사문 객체가 필요합니다.")
        if candidate.get("number_generation_issue"):
            raise ValueError(candidate["number_generation_issue"])
        create_example({"text": candidate.get("spoken_text")}, 1000)
        create_example({"text": candidate.get("normalized_text")}, 1000)
        terms = spoken_protected_terms(candidate["spoken_text"], candidate["normalized_text"], candidate.get("protected_terms", []))
        create_example({"text": candidate["spoken_text"], "protected_terms": terms}, 1000)
        return {**candidate, "protected_terms": terms}

    def _has_conflicting_transcript(self, row, existing):
        """동일 발화에 다른 정답이 있는지 확인한다."""
        return any(normalize_text(previous["normalized_text"]) != normalize_text(row["normalized_text"]) for previous in existing)

    def _add_conflict(self, row, existing, key):
        """정답이 충돌하는 발화는 양쪽 모두 채택에서 제외하고 검토 대상으로 남긴다."""
        for previous in existing:
            previous.update(accepted=False, needs_review=True, reason="transcript_conflict")
            if self.structure_index is not None:
                self.structure_index.discard(previous["id"])
        row.update(needs_review=True, reason="transcript_conflict", matched_id=existing[0]["id"])
        self.rows.append(row)
        self._spoken_rows.setdefault(key, []).append(row)
        return row


class DatasetJobService:
    def __init__(self, model_config, data_dir, client_factory=PairedLlamaClient):
        """기존 실행 기록을 읽고 한 번에 하나의 로컬 생성 작업만 허용한다."""
        self.model_config = model_config
        self.data_dir = Path(data_dir)
        self.client_factory = client_factory
        self.model_runtime = ModelRuntime()
        self.on_generation_completed = None
        self._closing = False
        self._lock = threading.RLock()
        self._jobs = {}
        self._cancellations = {}
        self._threads = {}
        self._load_jobs()

    def get_preferences(self):
        """새 작업에 공통으로 사용할 규칙과 저장 루트를 읽는다."""
        defaults = {"common_prompt": DEFAULT_COMMON_PROMPT, "output_root": "/tmp/ai-dataset/", "reading_rules": validate_reading_rules()}
        try:
            stored = json.loads((self.data_dir / "preferences.json").read_text(encoding="utf-8"))
            return {**self._validate_preferences(stored), "common_prompt": DEFAULT_COMMON_PROMPT,
                    "reading_rules": validate_reading_rules()}
        except (OSError, ValueError, TypeError):
            return defaults

    def _validate_preferences(self, payload):
        """공통 규칙과 출력 경로만 저장하도록 설정 입력을 제한한다."""
        if not isinstance(payload, dict):
            raise ValueError("공통 설정은 JSON 객체여야 합니다.")
        checked = validate_settings({"prompt": "설정 검증", "prompt_mode": True,
                                     "output_root": payload.get("output_root", "/tmp/ai-dataset/")})
        return {key: checked[key] for key in ("common_prompt", "output_root", "reading_rules")}

    def save_preferences(self, payload):
        """새 작업용 공통 설정을 저장하고 기존 작업의 규칙은 유지한다."""
        if not isinstance(payload, dict):
            raise ValueError("공통 설정은 JSON 객체여야 합니다.")
        preferences = self._validate_preferences({**self.get_preferences(), **payload})
        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            target = self.data_dir / "preferences.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(preferences, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(target)
        return preferences

    def _load_jobs(self):
        """이전 실행 결과를 읽고 중간에 끊긴 작업은 중단 상태로 복구한다."""
        for path in sorted(self.data_dir.glob("*/job.json"))[-100:]:
            self._load_job_file(path)

    def _load_job_file(self, path):
        """손상된 기록은 건너뛰고 유효한 작업만 작업 목록에 복원한다."""
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] in ACTIVE_STATUSES:
                job.update(status="interrupted", message="앱 종료로 작업이 중단되었습니다. 저장된 결과를 확인하세요.")
            self._audit_loaded_examples(job)
            self._jobs[job["id"]] = job
        except (OSError, ValueError, KeyError, TypeError):
            return

    def _audit_loaded_examples(self, job):
        """기존 저장 파일은 수정하지 않고 불일치한 예제를 메모리에서 검토 대상으로 분리한다."""
        for row in job["examples"]:
            if row.get("manually_edited"):
                continue
            if row.get("reason") in {"manual_review", "superseded_by_review", "exact_duplicate"} and not row["accepted"]:
                continue
            self._recover_loaded_example(job, row)
            policy = self._row_reading_policy(job, row)
            issue = numeric_alignment_issue(row["spoken_text"], row["normalized_text"], row.get("reading_aliases"), policy)
            if issue:
                row.update(accepted=False, needs_review=True, reason="numeric_alignment", quality_issue=issue)
            elif row.get("reason") == "numeric_alignment":
                row["quality_issue"] = None
        job["counts"] = job_counts(job)

    def _recover_loaded_example(self, job, row):
        """복구 가능한 이전 원문만 같은 ID의 검토 대상으로 되살리고 파일은 자동 수정하지 않는다."""
        if not job["settings"].get("prompt_mode") or row.get("spoken_text") or row.get("normalized_text"):
            return
        if not isinstance(row.get("candidate"), dict):
            return
        try:
            pair = render_prompt_candidate(row["candidate"], job["settings"].get("reading_rules"))
        except ValueError:
            return
        row.update(pair)
        row.update(accepted=False, needs_review=True, reason="format_recovery", quality_issue=None)
        row["review_note"] = "형식 복구 확인: 원문에서 복구한 발화·정답을 확인한 뒤 선택하세요. " + pair.get("review_note", "")

    def _row_reading_policy(self, job, row):
        """기존 프롬프트 결과도 검사하되 실행 당시 저장된 읽기 규칙을 우선한다."""
        if not job["settings"].get("prompt_mode"):
            return None
        return row.get("reading_policy", {"rules": job["settings"].get("reading_rules"), "expressions": []})

    def list_jobs(self):
        """최근 작업을 본문 예제 없이 요약해 반환한다."""
        with self._lock:
            return [self._summary(job) for job in sorted(self._jobs.values(), key=lambda item: item["created_at"], reverse=True)]

    def _summary(self, job):
        """작업 선택 목록에 필요한 제목과 진행 수를 반환한다."""
        return {key: copy.deepcopy(value) for key, value in job.items() if key != "examples"}

    def get_job(self, job_id):
        """실행 스레드가 변경 중인 상태를 독립된 스냅샷으로 반환한다."""
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError("작업을 찾지 못했습니다.")
            return copy.deepcopy(self._jobs[job_id])

    def start_job(self, payload):
        """설정을 검증해 새 디렉터리를 만들고 백그라운드 생성을 시작한다."""
        if isinstance(payload, dict) and "reading_rules" not in payload:
            payload = {**payload, "reading_rules": self.get_preferences()["reading_rules"]}
        settings = validate_settings(payload)
        with self._lock:
            if any(job["status"] in ACTIVE_STATUSES for job in self._jobs.values()):
                raise RuntimeError("이미 생성 중인 작업이 있습니다. 완료하거나 중지한 후 다시 시작하세요.")
            job = self._create_job(settings)
            thread = threading.Thread(target=self._run_job, args=(job["id"],), daemon=True)
            self._threads[job["id"]] = thread
            thread.start()
            return copy.deepcopy(job)

    def _create_job(self, settings):
        """작업 고유 ID와 저장 디렉터리를 생성한다."""
        identifier = uuid.uuid4().hex
        timestamp = current_timestamp()
        job = {"id": identifier, "title": settings["prompt"].splitlines()[0][:48], "settings": settings,
               "status": "queued", "message": "로컬 모델 생성을 준비하고 있습니다.", "created_at": timestamp,
               "updated_at": timestamp, "request_count": 0, "examples": [], "counts": {"accepted": 0, "review": 0, "rejected": 0, "total": 0}}
        (self.data_dir / identifier).mkdir(parents=True, exist_ok=False)
        self._jobs[identifier] = job
        self._cancellations[identifier] = threading.Event()
        self._save(job)
        return job

    def cancel_job(self, job_id):
        """실행 중인 작업에 중지 신호를 전달한다."""
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] not in ACTIVE_STATUSES:
                return copy.deepcopy(job)
            self._cancellations[job_id].set()
            job.update(status="cancelling", message="현재 추론을 중지하고 결과를 보존하고 있습니다.")
            self._save(job)
            return copy.deepcopy(job)

    def _run_job(self, job_id):
        """생성 작업의 완료·오류·중지 상태를 일관되게 저장한다."""
        try:
            with self.model_runtime.use("llm", self._cancellations[job_id], GenerationCancelledError):
                self._generate_batches(job_id)
            self._notify_generation_completed(job_id)
        except GenerationCancelledError:
            self._finish(job_id, "cancelled", "생성을 중지했습니다. 지금까지의 결과는 저장되었습니다.")
        except Exception as error:
            self._finish(job_id, "failed", str(error)[:1600])

    def _notify_generation_completed(self, job_id):
        """LLM 실행 구간이 끝난 뒤에만 후속 음성 작업을 통지한다."""
        job = self.get_job(job_id)
        if job.get("auto_audio_status") != "pending" or self._closing:
            return
        try:
            if self.on_generation_completed is None:
                raise RuntimeError("자동 음성 생성 서비스가 연결되지 않았습니다.")
            self.on_generation_completed(job)
            status, error = "started", ""
        except Exception as exception:
            status, error = "failed", str(exception)[:1000]
        with self._lock:
            self._jobs[job_id].update(auto_audio_status=status, auto_audio_error=error)
            self._save(self._jobs[job_id])

    def _generate_batches(self, job_id):
        """채택 목표를 채울 때까지 부족한 수를 추가 생성하고 배치마다 저장한다."""
        job = self.get_job(job_id)
        settings = job["settings"]
        cancellation = self._cancellations[job_id]
        client = self.client_factory(replace(self.model_config), cancellation)
        if settings.get("prompt_mode") and not settings.get("type_weights") and getattr(client, "supports_planning", False) is True:
            self._prepare_generation_plan(job_id, client)
        accumulator = DatasetAccumulator(settings)
        index = 0
        stalled_batches = 0
        batch_limit = settings["batch_size"]
        while True:
            if cancellation.is_set():
                raise GenerationCancelledError()
            accepted_before = sum(row["accepted"] for row in accumulator.rows)
            remaining = settings["count"] - accepted_before
            if remaining <= 0:
                break
            index += 1
            self._generate_batch(job_id, client, accumulator, index, min(batch_limit, remaining))
            accepted_after = sum(row["accepted"] for row in accumulator.rows)
            stalled_batches = 0 if accepted_after > accepted_before else stalled_batches + 1
            if stalled_batches >= MAX_STALLED_BATCHES:
                self._finish(job_id, "partial", f"{MAX_STALLED_BATCHES}개 배치 연속으로 채택 수가 늘지 않아 생성을 중단했습니다. "
                             f"채택 {accepted_after}/{settings['count']}개와 검토 결과는 저장되었습니다. 프롬프트와 검토 예제를 확인하세요.")
                return
        self._complete_generation(job_id)

    def _prepare_generation_plan(self, job_id, client):
        """도메인을 요청에서 받아 상황별 계획을 먼저 검증하고 작업에 저장한다."""
        settings = self.get_job(job_id)["settings"]
        feedback = ""
        for attempt in range(2):
            with self._lock:
                self._jobs[job_id]["message"] = "서로 다른 상황과 발화 목적을 계획하고 있습니다."
                self._save(self._jobs[job_id])
            context = {"user_prompt": settings["prompt"], "common_prompt": settings.get("common_prompt", ""),
                       "plan_count": min(settings["count"], MAX_PLAN_SLOTS), "feedback": feedback}
            prompt = planning_instructions() + json.dumps(context, ensure_ascii=False)
            seed = (self.model_config.seed + 104729 + attempt) % (2 ** 32)
            limit = min(self.model_config.context_size // 2, max(self.model_config.max_tokens, context["plan_count"] * 160 + 200))
            client.config = replace(self.model_config, seed=seed, max_tokens=limit)
            try:
                response = client.generate_plan(prompt)
            except ValueError as error:
                feedback = str(error)
                continue
            self._record_request(job_id, 0, prompt, response, seed, phase="planning")
            try:
                plans = validate_generation_plan(json.loads(response), settings["count"])
            except ValueError as error:
                feedback = str(error)
                continue
            with self._lock:
                self._jobs[job_id].update(generation_plan=plans, plan_attempts={})
                self._save(self._jobs[job_id])
            return
        raise ValueError("서로 다른 상황의 생성 계획을 만들지 못했습니다. " + feedback)

    def _batch_plan_slots(self, job, accumulator, count):
        """배치 크기 안에서 할당량과 시도 횟수로 상황별 계획을 선택한다."""
        plans = job.get("generation_plan", [])
        if not plans:
            return []
        slots = select_plan_slots(plans, accumulator.rows, job.get("plan_attempts", {}), count)
        if not slots:
            raise ValueError("생성 계획의 남은 할당량을 확인할 수 없습니다.")
        with self._lock:
            attempts = self._jobs[job["id"]].setdefault("plan_attempts", {})
            for slot in slots:
                attempts[slot["id"]] = attempts.get(slot["id"], 0) + 1
            self._save(self._jobs[job["id"]])
        return slots

    def _generate_batch(self, job_id, client, accumulator, index, count):
        """한 배치의 원문을 먼저 보존한 뒤 유효한 발화 쌍을 채택한다."""
        job = self.get_job(job_id)
        self._set_progress(job_id, index)
        if job["settings"].get("type_weights"):
            self._generate_typed_batch(job_id, client, accumulator, index, count)
            return
        slots = self._batch_plan_slots(job, accumulator, count)
        count = len(slots) if slots else count
        previous = [self._prompt_example(row) for row in accumulator.rows
                    if row["accepted"] and not row["needs_review"] and not row.get("quality_issue")][-4:]
        prompt = build_pair_prompt(job["settings"]["prompt"], count, previous,
                                   self._generation_feedback(accumulator), index, job["settings"].get("number_mode", False),
                                   job["settings"].get("prompt_mode", False), job["settings"].get("common_prompt", ""),
                                   job["settings"].get("reading_rules"), slots,
                                   self._used_structure_patterns(accumulator), sorted(accumulator.used_amounts)[-128:])
        seed = (self.model_config.seed + index - 1) % (2 ** 32)
        output_limit = self.model_config.max_tokens
        if job["settings"].get("prompt_mode"):
            output_limit = min(self.model_config.context_size // 2, max(output_limit, count * 1100))
        client.config = replace(self.model_config, seed=seed, max_tokens=output_limit)
        client.number_mode = job["settings"].get("number_mode", False)
        client.prompt_mode = job["settings"].get("prompt_mode", False)
        try:
            response = client.generate(prompt)
        except ValueError as error:
            accumulator.last_batch_issue = str(error)[:160]
            return
        self._record_request(job_id, index, prompt, response, seed)
        try:
            candidates = parse_generated_examples(response)
        except ValueError:
            accumulator.last_batch_issue = "응답이 완전한 JSON 객체와 examples 배열 형식이 아닙니다."
            return
        accumulator.last_batch_issue = None if candidates else "응답의 examples 배열이 비어 있습니다."
        if job["settings"].get("prompt_mode", False):
            candidates = [self._render_prompt_candidate(candidate, job["settings"].get("reading_rules"), slots) for candidate in candidates]
        elif job["settings"].get("number_mode", False):
            candidates = [self._render_number_candidate(candidate) for candidate in candidates]
        self._add_candidates(accumulator, candidates, job["settings"]["count"],
                             {plan["id"]: plan["quota"] for plan in job.get("generation_plan", [])})
        with self._lock:
            self._jobs[job_id]["examples"] = copy.deepcopy(accumulator.rows)
            self._save(self._jobs[job_id])

    def _typed_request(self, job_id, client, index, phase, prompt):
        """유형 생성 단계별 스키마를 적용하고 원본 요청·응답을 보존한다."""
        client.typed_phase = phase
        try:
            response = client.generate(prompt)
        finally:
            client.typed_phase = None
        self._record_request(job_id, index, prompt, response, client.config.seed, phase=phase)
        return response

    def _generate_typed_batch(self, job_id, client, accumulator, index, count):
        """코드 생성 숫자를 유형별 LLM 템플릿·읽기와 같은 틀에 연결한다."""
        settings = self.get_job(job_id)["settings"]
        seed = (self.model_config.seed + index - 1) % (2 ** 32)
        rng = random.Random(f"{job_id}:{seed}:numbers")
        types = select_types(settings["type_weights"], settings["count"], accumulator.rows, count)
        slots = make_slots(types, rng)
        if "money" in types:
            digit_weights = validate_money_digit_weights(settings.get("money_digit_weights"))
            money_target = type_quotas(settings["type_weights"], settings["count"])["money"]
            digits = iter(select_money_digits(digit_weights, money_target, accumulator.rows, types.count("money")))
            for slot in slots:
                if slot["type"] == "money":
                    slot["digits"] = next(digits)
        client.config = replace(self.model_config, seed=seed)
        previous = [row["template"] for row in accumulator.rows if row.get("template")][-12:]
        money_rng = random.Random(f"{job_id}:{seed}:money")
        numbers = [generate_number(slot, money_rng if slot["type"] == "money" else rng) for slot in slots]
        trace = [{**slot, "value": number["value"]} for slot, number in zip(slots, numbers)]
        self._publish_generation_batch(job_id, index, "template", trace)
        try:
            prompt = template_prompt(settings, trace, previous, accumulator.last_batch_issue)
            response = self._typed_request(job_id, client, index, "template", prompt)
            templates = indexed_response(response, slots, "template")
            templates = {number["id"]: normalize_number_template(templates[number["id"]], number) for number in numbers}
            validate_templates(templates)
            trace = [{**item, "template": templates[item["id"]]} for item in trace]
            self._publish_generation_batch(job_id, index, "spoken", trace)
            prompt = reading_prompt(numbers, templates)
            response = self._typed_request(job_id, client, index, "spoken", prompt)
            readings = indexed_response(response, slots, "spoken")
            trace = [{**item, "spoken": readings[item["id"]]} for item in trace]
            self._publish_generation_batch(job_id, index, "mapping", trace)
            rows = self._add_typed_candidates(accumulator, numbers, templates, readings)
            trace = [{**item, **{key: row[key] for key in ("spoken_text", "normalized_text", "accepted", "needs_review", "reason", "quality_issue")},
                      "result_id": row["id"]} for item, row in zip(trace, rows)]
            self._publish_generation_batch(job_id, index, "completed", trace)
        except ValueError as error:
            accumulator.last_batch_issue = str(error)[:240]
            self._publish_generation_batch(job_id, index, "failed", trace, str(error))
        with self._lock:
            self._jobs[job_id]["examples"] = copy.deepcopy(accumulator.rows)
            self._save(self._jobs[job_id])

    def _publish_generation_batch(self, job_id, index, phase, examples, error=None):
        """현재 생성 단계와 중간 산출물을 저장해 UI 조회와 재열기를 지원한다."""
        labels = {"template": "LLM이 템플릿을 생성하고 있습니다.", "number": "유형에 맞는 숫자를 코드로 생성하고 있습니다.",
                  "spoken": "생성한 숫자를 LLM이 한글로 읽고 있습니다.", "mapping": "발화와 정답을 매핑하고 검증하고 있습니다.",
                  "completed": "배치 매핑을 마쳤습니다.", "failed": "이번 배치의 오류를 기록했습니다. 미달 유형을 다시 시도합니다."}
        record = {"index": index, "phase": phase, "status": phase if phase in {"completed", "failed"} else "running",
                  "examples": copy.deepcopy(examples), "updated_at": current_timestamp()}
        if error:
            record["error"] = error[:1000]
        with self._lock:
            job = self._jobs[job_id]
            batches = job.setdefault("generation_batches", [])
            if batches and batches[-1]["index"] == index:
                if phase == "failed":
                    record["failed_phase"] = batches[-1]["phase"]
                batches[-1] = record
            else:
                batches.append(record)
            job["message"] = f"배치 {index} · {labels[phase]}"
            self._save(job)

    def _add_typed_candidates(self, accumulator, numbers, templates, readings):
        """정합성을 통과한 항목을 채택하고 잘못된 읽기는 검토 원문으로 남긴다."""
        accumulator.last_batch_issue = None
        rows = []
        for number in numbers:
            template, spoken = templates[number["id"]], readings[number["id"]]
            try:
                candidate = render_pair(number, template, spoken)
            except ValueError as error:
                accumulator.last_batch_issue = str(error)
                candidate = {"type": number["type"], "template": template, "generated_number": number,
                             "spoken_text": template.replace("{number}", spoken),
                             "normalized_text": template.replace("{number}", number["value"]),
                             "number_generation_issue": str(error)}
            rows.append(accumulator.add(candidate))
        return rows

    def _used_structure_patterns(self, accumulator):
        """금지 형식은 최근의 제한된 분량만 전달해 문맥을 과도하게 채우지 않는다."""
        patterns = list(dict.fromkeys(row["structure_pattern"] for row in accumulator.rows
                                     if row["accepted"] and row.get("structure_pattern")))
        result, remaining = [], 1200
        for pattern in reversed(patterns):
            if len(result) >= 12 or remaining <= 0:
                break
            excerpt = pattern[:min(160, remaining)]
            result.append(excerpt)
            remaining -= len(excerpt)
        return list(reversed(result))

    def _render_number_candidate(self, candidate):
        """숫자 종류별 읽기를 적용하고 실패 원문은 검토 기록에 보존한다."""
        try:
            return render_numeric_candidate(candidate)
        except ValueError as error:
            source = candidate if isinstance(candidate, dict) else {"source": candidate}
            return {**source, "number_generation_issue": str(error)}

    def _render_prompt_candidate(self, candidate, reading_rules=None, slots=None):
        """프롬프트의 표현을 렌더링하고 지원 밖의 입력은 원문과 함께 남긴다."""
        try:
            if slots and (not isinstance(candidate, dict) or candidate.get("plan_id") not in {slot["id"] for slot in slots}):
                raise ValueError("이번 배치의 계획 id를 plan_id로 지정하고 각 계획에 맞는 문장을 만드세요.")
            return render_prompt_candidate(candidate, reading_rules)
        except ValueError as error:
            source = candidate if isinstance(candidate, dict) else {"source": candidate}
            return {**source, "number_generation_issue": str(error)}

    def _prompt_example(self, row):
        """이전 후보의 텍스트를 제한된 길이로 전달해 문맥이 과도하게 커지지 않게 한다."""
        candidate = row.get("candidate")
        source = candidate if isinstance(candidate, dict) else row
        example = {key: source.get(key, "")[:160] if isinstance(source.get(key, ""), str) else ""
                   for key in ("spoken_text", "normalized_text")}
        if isinstance(source.get("template"), str):
            example.update(template=source["template"][:160], numbers=source.get("numbers", [])[:20] if isinstance(source.get("numbers"), list) else [])
            if isinstance(source.get("expressions"), list):
                example["expressions"] = source["expressions"][:20]
        if isinstance(source.get("parts"), list):
            example["parts_preview"] = json.dumps(source["parts"], ensure_ascii=False)[:600]
        return example

    def _generation_feedback(self, accumulator):
        """최근 실패 예제와 이유를 중복 없이 최대 네 개까지 다음 요청에 전달한다."""
        feedback = []
        seen = set()
        for row in reversed(accumulator.rows):
            if row["accepted"]:
                continue
            example = ({"pattern": row.get("structure_pattern", "")[:160]} if row["reason"] == "structure_duplicate"
                       else self._prompt_example(row))
            key = json.dumps(example, ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            feedback.append({**example, "reason": row["reason"],
                             "detail": ("숫자·이름만 바꾼 형식입니다. 다른 사건·정보 목적·전개로 다시 생성하세요."
                                        if row["reason"] == "structure_duplicate" else (row.get("quality_issue") or row["reason"])[:160])})
            if len(feedback) >= 4:
                break
        if accumulator.last_batch_issue:
            feedback.insert(0, {"reason": "invalid_response", "detail": accumulator.last_batch_issue})
        return feedback[:4]

    def _add_candidates(self, accumulator, candidates, target, plan_quotas=None):
        """목표 개수를 넘지 않게 배치 안의 후보까지 순서대로 판정한다."""
        for candidate in candidates:
            if sum(row["accepted"] for row in accumulator.rows) >= target:
                break
            accumulator.add(self._check_plan_quota(accumulator, candidate, plan_quotas))

    def _check_plan_quota(self, accumulator, candidate, plan_quotas):
        """같은 상황의 초과 제출이 전체 생성 목표를 채우지 못하게 한다."""
        if not plan_quotas or not isinstance(candidate, dict) or candidate.get("plan_id") not in plan_quotas:
            return candidate
        identifier = candidate["plan_id"]
        used = sum(row["accepted"] and row.get("plan_id") == identifier for row in accumulator.rows)
        if used >= plan_quotas[identifier]:
            return {**candidate, "number_generation_issue": "이 상황의 할당량을 채웠습니다. 이번 배치의 다른 계획을 사용하세요."}
        return candidate

    def _record_request(self, job_id, index, prompt, response, seed, phase=None):
        """추론 원문과 배치 seed를 파일에 추가해 반복 원인을 재현할 수 있게 한다."""
        path = self.data_dir / job_id / "requests.jsonl"
        record = {"request_index": index, "seed": seed, "prompt": prompt, "raw_response": response}
        if phase:
            record["phase"] = phase
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _set_progress(self, job_id, index):
        """현재 배치 번호와 진행 메시지를 갱신한다."""
        with self._lock:
            job = self._jobs[job_id]
            accepted = job["counts"]["accepted"]
            target = job["settings"]["count"]
            job.update(status="running", request_count=index, message=f"채택 {accepted}/{target}개 · 배치 {index} 생성 중입니다. 목표까지 추가 생성합니다.")
            self._save(job)

    def _complete_generation(self, job_id):
        """채택 목표를 채운 작업을 완료 상태로 기록한다."""
        job = self.get_job(job_id)
        self._finish(job_id, "completed", f"채택 목표 {job['settings']['count']}개를 채웠습니다. 전사문을 검토한 뒤 내보내세요.")

    def _finish(self, job_id, status, message):
        """종료 상태를 저장하고 작업 스레드 참조를 해제한다."""
        with self._lock:
            job = self._jobs[job_id]
            job.update(status=status, message=message)
            if status in {"completed", "partial"} and job["settings"].get("auto_audio") and any(row["accepted"] and not row["needs_review"] and not row.get("quality_issue") for row in job["examples"]):
                job["auto_audio_status"] = "pending"
            batches = job.get("generation_batches", [])
            if batches and batches[-1]["status"] == "running":
                batches[-1].update(status=status, error=message)
            self._save(job)
            self._threads.pop(job_id, None)

    def add_manual_examples(self, payload, job_id=None):
        """직접 입력한 발화·정답을 모델 실행 없이 새 작업 또는 기존 목록에 저장한다."""
        rows = self._manual_rows(payload)
        with self._lock:
            if any(job["status"] in ACTIVE_STATUSES for job in self._jobs.values()):
                raise RuntimeError("텍스트 생성이 끝난 뒤 직접 입력 발화를 추가하세요.")
            job = self._jobs[job_id] if job_id is not None else None
            existing = {normalize_text(row["spoken_text"]) for row in job["examples"]} if job else set()
            self._assert_manual_unique(rows, existing)
            if job is None:
                settings = validate_settings({"prompt": "직접 입력 발화", "count": len(rows), "batch_size": 1,
                                              "output_root": payload.get("output_root", "/tmp/ai-dataset/")})
                job = self._create_job(settings)
                job.update(status="completed", manual_only=True)
            working = copy.deepcopy(job)
            working["examples"].extend(rows)
            working["message"] = f"직접 입력 발화 {len(rows)}개를 추가했습니다."
            if working.get("manual_only"):
                working["settings"]["count"] = len(working["examples"])
            self._save(working)
            self._jobs[working["id"]] = working
            return copy.deepcopy(working)

    def _manual_rows(self, payload):
        """직접 입력의 행별 형식·길이를 검증하고 사용 가능한 수동 항목을 만든다."""
        if not isinstance(payload, dict):
            raise ValueError("직접 입력 발화와 정답을 JSON 객체로 전달하세요.")
        examples = read_excel_examples(payload) if "excel_base64" in payload else payload.get("examples")
        if not isinstance(examples, list) or not 1 <= len(examples) <= 1000:
            raise ValueError("직접 입력 발화는 한 번에 1~1,000개를 추가하세요.")
        return [self._manual_row(example, index) for index, example in enumerate(examples, 1)]

    def _manual_row(self, example, index):
        """직접 작성한 내용을 기존 수동 수정과 동일한 검토 완료 상태로 기록한다."""
        if not isinstance(example, dict):
            raise ValueError(f"{index}번째 발화와 정답을 확인하세요.")
        try:
            spoken = create_example({"text": example.get("spoken_text")}, 1000).text.strip()
            transcript = create_example({"text": example.get("normalized_text")}, 1000).text.strip()
        except ValueError as error:
            raise ValueError(f"{index}번째 항목: {error}") from error
        return {"id": uuid.uuid4().hex, "spoken_text": spoken, "normalized_text": transcript,
                "protected_terms": [], "accepted": True, "needs_review": False,
                "reason": "manual_review", "quality_issue": None, "matched_id": None, "similarity": None,
                "manually_edited": True, "source": "manual"}

    def _assert_manual_unique(self, rows, existing):
        """기존 발화나 같은 입력 안의 중복·정답 충돌은 부분 저장 전에 거부한다."""
        seen = set(existing)
        for index, row in enumerate(rows, 1):
            key = normalize_text(row["spoken_text"])
            if key in seen:
                raise ValueError(f"{index}번째 발화가 이미 있습니다. 기존 항목을 수정하거나 중복 문장을 제거하세요.")
            seen.add(key)

    def update_example(self, job_id, example_id, payload):
        """생성이 끝난 행의 전사문과 채택 상태를 검토 결과로 저장한다."""
        if not isinstance(payload, dict) or type(payload.get("accepted")) is not bool:
            raise ValueError("채택 여부를 boolean으로 전달하세요.")
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] in ACTIVE_STATUSES:
                raise RuntimeError("생성이 끝난 후 예제를 검토하세요.")
            row = next((item for item in job["examples"] if item["id"] == example_id), None)
            if row is None:
                raise KeyError("예제를 찾지 못했습니다.")
            self._apply_review(job, row, payload)
            self._save(job)
            return copy.deepcopy(job)

    def update_selection(self, job_id, payload):
        """텍스트 검토의 선택을 한 번에 검증·저장하고 다음 단계에 그대로 사용한다."""
        if not isinstance(payload, dict) or type(payload.get("accepted")) is not bool:
            raise ValueError("선택 여부를 boolean으로 전달하세요.")
        identifiers = payload.get("example_ids")
        if not isinstance(identifiers, list) or not identifiers or any(not isinstance(value, str) for value in identifiers):
            raise ValueError("선택할 예제 ID 배열이 필요합니다.")
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] in ACTIVE_STATUSES:
                raise RuntimeError("텍스트 생성이 끝난 뒤 선택하세요.")
            working = copy.deepcopy(job)
            rows = {row["id"]: row for row in working["examples"]}
            if len(set(identifiers)) != len(identifiers) or any(value not in rows for value in identifiers):
                raise ValueError("중복되지 않는 현재 작업의 ID만 선택하세요.")
            for identifier in identifiers:
                self._apply_review(working, rows[identifier], {"accepted": payload["accepted"]})
            self._save(working)
            self._jobs[job_id] = working
            return copy.deepcopy(working)

    def _apply_review(self, job, row, payload):
        """직접 수정한 내용은 검토자의 판단으로 저장하고 자동 생성 결과만 대응 검사한다."""
        spoken = payload.get("spoken_text", row["spoken_text"])
        transcript = payload.get("normalized_text", row["normalized_text"])
        create_example({"text": spoken}, 1000)
        create_example({"text": transcript}, 1000)
        aliases = valid_reading_aliases(row["reading_aliases"]) if "reading_aliases" in row else None
        manually_edited = row.get("manually_edited", False) or "spoken_text" in payload or "normalized_text" in payload
        issue = None if manually_edited else numeric_alignment_issue(spoken, transcript, aliases, self._row_reading_policy(job, row))
        if payload["accepted"] and issue:
            raise ValueError(issue + " 발화문과 전사문을 수정한 뒤 채택하세요.")
        row.update(spoken_text=spoken, normalized_text=transcript)
        if aliases is not None:
            if aliases != row["reading_aliases"]:
                row["review_note"] = " · ".join(("단위 발음 확인: " if alias["context"] == "unit" else "표현 확인: ")
                                                 + alias["written"] + " → " + alias["spoken"] for alias in aliases)
            row["reading_aliases"] = aliases
        if payload["accepted"]:
            self._exclude_conflicts(job, row)
        row.update(accepted=payload["accepted"], needs_review=False, reason="manual_review", quality_issue=issue,
                   manually_edited=manually_edited)
        row["protected_terms"] = [term for term in row["protected_terms"] if normalize_text(term) in normalize_text(spoken)]

    def _exclude_conflicts(self, job, selected):
        """검토자가 선택한 행 외의 동일 발화는 중복 채택에서 제외한다."""
        for row in job["examples"]:
            if row["id"] != selected["id"] and normalize_text(row["spoken_text"]) == normalize_text(selected["spoken_text"]):
                row.update(accepted=False, needs_review=False, reason="superseded_by_review")

    def export_jsonl(self, job_id, reviewed_only=False):
        """채택된 발화·전사문과 안정적인 ID를 다음 단계용 JSONL로 내보낸다."""
        job = self.get_job(job_id)
        rows = [{"id": row["id"], "spoken_text": row["spoken_text"], "normalized_text": row["normalized_text"],
                 "protected_terms": row["protected_terms"], "needs_review": row["needs_review"]} for row in job["examples"]
                if row["accepted"] and (not reviewed_only or (not row["needs_review"] and not row.get("quality_issue")))]
        source_rows = {row["id"]: row for row in job["examples"]}
        for row in rows:
            source = source_rows[row["id"]]
            row.update({key: source[key] for key in ("type", "number_context", "generated_number", "template") if key in source})
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    def _save(self, job):
        """하나의 작업 파일을 원자적으로 교체해 부분 저장을 방지한다."""
        job["updated_at"] = current_timestamp()
        job["counts"] = job_counts(job)
        target = self.data_dir / job["id"] / "job.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def close(self):
        """앱 종료 시 이 서비스가 실행한 추론 작업에만 중지 신호를 보낸다."""
        with self._lock:
            self._closing = True
            events = list(self._cancellations.values())
            threads = list(self._threads.values())
        for event in events:
            event.set()
        for thread in threads:
            thread.join(timeout=5)
