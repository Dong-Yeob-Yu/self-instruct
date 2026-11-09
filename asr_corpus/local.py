"""설치된 Gemma 4 GGUF를 서버와 다운로드 없이 직접 실행한다."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .__main__ import CorpusOutputWriter, add_filter_options, configuration_record, create_deduplicator, write_generation_batch
from .examples import parse_generated_examples
from .generation import CorpusGenerator


MODEL_REPOSITORY = "ggml-org/gemma-4-E4B-it-GGUF"
MODEL_FILENAME = "gemma-4-e4b-it-Q4_K_M.gguf"
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "examples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "protected_terms": {"type": "array", "items": {"type": "string"}}},
                "required": ["text", "protected_terms"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["examples"],
    "additionalProperties": False,
}


def find_default_model(cache_dir: Optional[Path] = None) -> Path:
    """현재 캐시에서 완전한 Gemma 4 E4B Q4_K_M 파일만 찾아 반환한다."""
    cache = cache_dir or Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    repository = cache.expanduser() / ("models--" + MODEL_REPOSITORY.replace("/", "--"))
    candidates = sorted((repository / "snapshots").glob("*/" + MODEL_FILENAME))
    available = [path for path in candidates if path.is_file()]
    if not available:
        raise ValueError("완전한 Gemma 4 E4B 모델 파일을 찾지 못했습니다. --model-file로 로컬 GGUF를 지정하세요.")
    return available[-1].resolve()


@dataclass(frozen=True)
class LlamaCppConfig:
    model_file: Path
    executable: str = "llama-completion"
    context_size: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.6
    seed: int = 42
    timeout: float = 180.0

    def __post_init__(self) -> None:
        """다운로드나 실행 전에 로컬 가중치와 생성 범위를 검증한다."""
        if not self.model_file.is_file():
            raise ValueError("지정한 로컬 모델 파일이 존재하지 않습니다.")
        with self.model_file.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError("모델 파일은 유효한 GGUF 형식이어야 합니다.")
        if self.max_tokens < 1 or self.context_size <= self.max_tokens:
            raise ValueError("생성 토큰 수는 양수이고 context-size보다 작아야 합니다.")
        if not math.isfinite(self.timeout) or self.timeout <= 0 or not 0 <= self.temperature <= 2:
            raise ValueError("제한 시간은 유한한 양수이고 temperature는 0~2여야 합니다.")


class LlamaCppClient:
    def __init__(self, config: LlamaCppConfig) -> None:
        """설정만 보관하고 기존 llama-completion 실행 파일을 확인한다."""
        self.config = config
        self.executable = shutil.which(config.executable)
        if self.executable is None:
            raise ValueError("llama-completion을 찾지 못했습니다. --llama-executable로 실행 파일을 지정하세요.")

    def generate(self, prompt: str) -> str:
        """서버를 열지 않고 한 배치만 추론한 뒤 자식 프로세스를 종료한다."""
        try:
            result = subprocess.run(
                self._build_command(prompt), capture_output=True, stdin=subprocess.DEVNULL, timeout=self.config.timeout
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("로컬 모델 추론 제한 시간을 초과했습니다. 배치 크기나 토큰 수를 줄이세요.") from error
        if result.returncode != 0:
            detail = result.stderr[-1600:].decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp 실행 실패 ({result.returncode}): {detail}")
        return self._extract_response(result.stdout)

    def _build_command(self, prompt: str) -> List[str]:
        """오프라인·단일 응답·JSON 제약을 명시한 인자 배열을 만든다."""
        return [
            self.executable, "--model", str(self.config.model_file),
            "--ctx-size", str(self.config.context_size), "--n-predict", str(self.config.max_tokens),
            "--temp", str(self.config.temperature), "--seed", str(self.config.seed),
            "--single-turn", "--conversation", "--jinja", "--reasoning", "off",
            "--offline", "--no-display-prompt", "--simple-io", "--color", "off", "--no-warmup", "--no-escape",
            "--json-schema", json.dumps(RESPONSE_SCHEMA), "--prompt", prompt,
        ]

    def _extract_response(self, output: bytes, validate_examples: bool = True) -> str:
        """CLI의 종료 표식만 제거하고 잘린 JSON과 깨진 한글은 거부한다."""
        try:
            text = output.decode("utf-8").lstrip()
            _, end = json.JSONDecoder().raw_decode(text)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("로컬 모델 출력이 완전한 UTF-8 JSON이 아닙니다. batch-size나 max-tokens를 조정하세요.") from error
        if text[end:].strip() not in {"", "[end of text]"}:
            raise ValueError("JSON 뒤에 예상하지 못한 CLI 출력이 있습니다.")
        response = text[:end]
        if validate_examples:
            parse_generated_examples(response)
        return response


def create_parser() -> argparse.ArgumentParser:
    """기본 모델을 자동으로 찾되 학습 요구사항은 실행 시 입력받는다."""
    parser = argparse.ArgumentParser(description="캐시된 Gemma 4 E4B로 ASR 발화 생성 및 중복 제거, 서버 연결 없음")
    parser.set_defaults(command="generate-local", backend="llama-completion")
    parser.add_argument("--requirement", required=True, help="실행 시 지정하는 자연어 학습 요구사항")
    parser.add_argument("--model-file", type=Path, help="생략하면 캐시의 Gemma 4 E4B Q4_K_M 자동 선택")
    parser.add_argument("--llama-executable", default="llama-completion", help="설치된 llama-completion 실행 파일")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-requests", type=int, default=30)
    parser.add_argument("--context-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0)
    add_filter_options(parser)
    return parser


def run_generation(args: argparse.Namespace) -> int:
    """기존 생성·필터·저장 파이프라인을 직접 추론 클라이언트와 연결한다."""
    if not args.requirement.strip() or min(args.count, args.batch_size, args.max_requests) < 1:
        raise ValueError("요구사항과 양수인 목표 개수·배치 크기·요청 상한을 지정해야 합니다.")
    args.model_file = args.model_file.expanduser().resolve() if args.model_file else find_default_model()
    config = LlamaCppConfig(
        model_file=args.model_file, executable=args.llama_executable, context_size=args.context_size,
        max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed, timeout=args.timeout,
    )
    generator = CorpusGenerator(LlamaCppClient(config), create_deduplicator(args))
    with CorpusOutputWriter(args.output_dir, configuration_record(args)) as writer:
        for batch in generator.generate(args.requirement, args.count, args.batch_size, args.max_requests):
            write_generation_batch(writer, batch)
        if writer.counts["accepted"] < args.count:
            writer.status = "target_not_reached"
    print(json.dumps({"status": writer.status, **writer.counts}, ensure_ascii=False))
    return 0 if writer.status == "completed" else 2


def main(argv: Optional[List[str]] = None) -> int:
    """로컬 추론 명령을 실행하고 실패를 종료 코드로 반환한다."""
    args = create_parser().parse_args(argv)
    try:
        return run_generation(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
