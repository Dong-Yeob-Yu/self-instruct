"""python -m asr_corpus 명령으로 생성·중복 제거를 실행한다."""

import argparse
import json
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Tuple

from .examples import ExampleDeduplicator, FilteringDecision
from .generation import CorpusGenerator, GenerationBatch, LocalLlmClient, LocalLlmConfig, build_prompt


class CorpusOutputWriter:
    def __init__(self, output_dir: Path, configuration: Dict[str, object]) -> None:
        """새 실행 디렉터리에 기록할 설정과 통계를 준비한다."""
        self.output_dir = output_dir
        self.configuration = configuration
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.counts = {"accepted": 0, "rejected": 0, "review": 0, "requests": 0}
        self.status = "completed"
        self._stack = ExitStack()
        self._streams = {}

    def __enter__(self) -> "CorpusOutputWriter":
        """기존 결과를 덮어쓰지 않도록 새 디렉터리와 파일만 생성한다."""
        self.output_dir.mkdir(parents=True, exist_ok=False)
        try:
            for name in ("accepted", "rejected", "review", "requests"):
                self._streams[name] = self._stack.enter_context((self.output_dir / f"{name}.jsonl").open("x", encoding="utf-8"))
        except OSError:
            self._stack.close()
            raise
        return self

    def __exit__(self, error_type: object, error: object, traceback: object) -> None:
        """부분 결과를 닫고 성공·실패 상태를 실행 요약에 남긴다."""
        self._stack.close()
        if error_type is not None:
            self.status = "failed"
        summary = {
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "configuration": self.configuration,
            "counts": self.counts,
        }
        with (self.output_dir / "run.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    def write(self, name: str, record: object) -> None:
        """각 레코드를 즉시 기록해 완료된 배치 결과를 보존한다."""
        self._streams[name].write(json.dumps(record, ensure_ascii=False) + "\n")
        self._streams[name].flush()
        self.counts[name] += 1

    def write_decision(self, decision: FilteringDecision, metadata: Dict[str, object]) -> None:
        """채택 원문과 제외·검토 근거를 각각의 파일에 기록한다."""
        if not decision.accepted:
            self.write("rejected", {**decision.to_record(), **metadata})
            return
        self.write("accepted", {**decision.example.to_record(), "needs_review": decision.needs_review, **metadata})
        if decision.needs_review:
            self.write("review", {**decision.to_record(), **metadata})


def add_filter_options(parser: argparse.ArgumentParser) -> None:
    """생성과 독립 필터 명령에 동일한 중복 판정 옵션을 추가한다."""
    parser.add_argument("--similarity-threshold", type=float, default=0.95, help="문자 유사도 기준, 기본 0.95")
    parser.add_argument(
        "--near-duplicate-policy", choices=("review", "drop"), default="review",
        help="review: 유사 문장을 보존하고 검토 기록, drop: 유사 문장도 제외",
    )
    parser.add_argument("--protected-term", action="append", default=[], help="다른 값이면 보존할 보호어, 여러 번 지정 가능")
    parser.add_argument("--max-characters", type=int, default=500, help="발화당 최대 글자 수, 기본 500")
    parser.add_argument("--output-dir", type=Path, required=True, help="아직 존재하지 않는 새 실행 디렉터리")


def create_parser() -> argparse.ArgumentParser:
    """학습 주제나 특정 모델을 기본값으로 고정하지 않는 CLI를 만든다."""
    parser = argparse.ArgumentParser(description="로컬 LLM 기반 ASR 발화 예제 생성 및 중복 제거")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="로컬 LLM으로 발화를 생성하고 중복 제거")
    generate.add_argument("--requirement", required=True, help="실행할 때 지정하는 자연어 학습 요구사항")
    generate.add_argument("--endpoint", required=True, help="전체 API 주소: /v1/chat/completions 또는 /api/generate 포함")
    generate.add_argument("--model", required=True, help="로컬 서버에 등록된 모델 이름")
    generate.add_argument("--backend", choices=("openai-compatible", "ollama"), default="openai-compatible")
    generate.add_argument("--count", type=int, default=100, help="채택할 발화 수, 기본 100")
    generate.add_argument("--batch-size", type=int, default=10, help="요청당 발화 수, 기본 10")
    generate.add_argument("--max-requests", type=int, default=30, help="반복 요청 상한, 기본 30")
    generate.add_argument("--max-tokens", type=int, default=2048)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--timeout", type=float, default=120.0)
    generate.add_argument("--api-key-env", help="인증이 필요한 경우 키를 읽을 환경 변수 이름")
    add_filter_options(generate)
    deduplicate = commands.add_parser("deduplicate", help="기존 JSONL 파일만 대상으로 중복 제거, LLM 연결 없음")
    deduplicate.add_argument("--input-file", type=Path, required=True, help="각 줄이 text 객체 또는 문자열인 UTF-8 JSONL")
    add_filter_options(deduplicate)
    prompt = commands.add_parser("prompt", help="네트워크 호출 없이 생성 프롬프트 출력")
    prompt.add_argument("--requirement", required=True)
    prompt.add_argument("--count", type=int, default=10)
    prompt.add_argument("--protected-term", action="append", default=[])
    return parser


def create_deduplicator(args: argparse.Namespace) -> ExampleDeduplicator:
    """CLI 옵션에서 중복 판정기 설정을 구성한다."""
    return ExampleDeduplicator(
        args.similarity_threshold, args.near_duplicate_policy, args.protected_term, args.max_characters
    )


def parse_input_line(line: str) -> object:
    """입력 행을 JSON으로 읽고 문법 오류를 발화 형식 검증 대상으로 넘긴다."""
    try:
        return json.loads(line)
    except ValueError:
        return {"invalid_json": line.rstrip("\n")}


def read_candidates(input_file: Path) -> Iterator[Tuple[int, object]]:
    """행 번호를 보존하면서 JSONL 후보를 순서대로 읽는다."""
    with input_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            yield line_number, parse_input_line(line)


def run_deduplication(args: argparse.Namespace) -> int:
    """LLM 없이 기존 파일의 후보를 판정하고 별도의 실행 결과를 저장한다."""
    deduplicator = create_deduplicator(args)
    if not args.input_file.is_file():
        raise ValueError("입력 파일이 존재하지 않습니다.")
    with CorpusOutputWriter(args.output_dir, configuration_record(args)) as writer:
        for line_number, candidate in read_candidates(args.input_file):
            writer.write_decision(deduplicator.consider(candidate), {"source_line": line_number})
    print(json.dumps(writer.counts, ensure_ascii=False))
    return 0


def write_generation_batch(writer: CorpusOutputWriter, batch: GenerationBatch) -> None:
    """생성 원문을 먼저 보존하고 배치의 필터링 결과를 기록한다."""
    writer.write("requests", {
        "request_index": batch.request_index,
        "prompt": batch.prompt,
        "raw_response": batch.raw_response,
        "error": batch.error,
    })
    if batch.error:
        writer.write("rejected", {"request_index": batch.request_index, "reason": "invalid_response", "error": batch.error})
    for decision in batch.decisions:
        writer.write_decision(decision, {"request_index": batch.request_index})


def configuration_record(args: argparse.Namespace) -> Dict[str, object]:
    """인증 값 없이 실행 옵션과 경로를 요약용으로 직렬화한다."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def run_generation(args: argparse.Namespace) -> int:
    """요청 상한 내에서 발화를 생성하고 미달성 시에도 부분 결과를 남긴다."""
    if not args.requirement.strip() or min(args.count, args.batch_size, args.max_requests) < 1:
        raise ValueError("요구사항과 양수인 목표 개수·배치 크기·요청 상한을 지정해야 합니다.")
    config = LocalLlmConfig(
        endpoint=args.endpoint, model=args.model, backend=args.backend, temperature=args.temperature,
        max_tokens=args.max_tokens, timeout=args.timeout, api_key_env=args.api_key_env,
    )
    generator = CorpusGenerator(LocalLlmClient(config), create_deduplicator(args))
    with CorpusOutputWriter(args.output_dir, configuration_record(args)) as writer:
        for batch in generator.generate(args.requirement, args.count, args.batch_size, args.max_requests):
            write_generation_batch(writer, batch)
        if writer.counts["accepted"] < args.count:
            writer.status = "target_not_reached"
    print(json.dumps({"status": writer.status, **writer.counts}, ensure_ascii=False))
    return 0 if writer.status == "completed" else 2


def main(argv: object = None) -> int:
    """인자를 검증해 선택한 명령을 실행하고 오류는 간결한 한글 메시지로 알린다."""
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1


def dispatch(args: argparse.Namespace) -> int:
    """프롬프트 확인과 오프라인 필터링을 실제 생성 경로와 분리해 실행한다."""
    if args.command == "prompt":
        if not args.requirement.strip() or args.count < 1:
            raise ValueError("요구사항과 양수인 예제 수를 지정해야 합니다.")
        print(build_prompt(args.requirement, args.count, [], tuple(args.protected_term)))
        return 0
    if args.command == "deduplicate":
        return run_deduplication(args)
    return run_generation(args)


if __name__ == "__main__":
    sys.exit(main())
