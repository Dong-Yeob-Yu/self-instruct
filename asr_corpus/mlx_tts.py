"""MLX TTS를 작업 전용 자식 프로세스에서 한 번 로딩하고 재사용한다."""

import hashlib
import json
import math
import os
import selectors
import subprocess
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .tts import AudioCancelledError, inspect_wav


MODEL_ROOT = Path.home() / "Desktop/personal/dev/ai-models"
DEFAULT_MLX_TTS_DIR = MODEL_ROOT / "MOSS-TTS-8B-BF16"
DEFAULT_MLX_CODEC_DIR = MODEL_ROOT / "MOSS-Audio-Tokenizer-BF16"
DEFAULT_MLX_PYTHON = MODEL_ROOT / ".voxtral-mlx-tools/bin/python"


@dataclass(frozen=True)
class MlxTtsConfig:
    model_dir: Path
    codec_dir: Path
    executable: str
    timeout: int = 600
    load_timeout: int = 1800
    max_tokens: int = 2048
    memory_limit_gb: int = 20
    audio_temperature: float = 1.5

    @property
    def model_file(self):
        """기존 내보내기와 호환되는 모델 식별 경로를 반환한다."""
        return self.model_dir / "config.json"

    @property
    def display_name(self):
        """화면에 실제 모델 디렉터리 이름과 실행 방식을 표시한다."""
        return self.model_dir.name + " · MLX"

    @property
    def model_name(self):
        """학습 자료에 기록할 모델 식별자를 반환한다."""
        return self.model_dir.name

    def validate(self):
        """모델을 로딩하거나 다운로드하지 않고 로컬 자산만 검사한다."""
        if not Path(self.executable).is_file():
            raise ValueError("MLX TTS Python 실행 파일이 없습니다.")
        if type(self.audio_temperature) not in {int, float} or not math.isfinite(self.audio_temperature) or self.audio_temperature <= 0:
            raise ValueError("TTS 음성 샘플링 온도는 유한한 양수여야 합니다.")
        if type(self.memory_limit_gb) is not int or not 1 <= self.memory_limit_gb <= 128:
            raise ValueError("TTS 메모리 한도는 1~128GiB의 정수여야 합니다.")
        for directory in (self.model_dir, self.codec_dir):
            self._validate_assets(directory)
        if not (self.model_dir / "tokenizer.json").is_file():
            raise ValueError("MOSS-TTS 텍스트 토크나이저가 없습니다.")

    def _validate_assets(self, directory):
        """설정과 분할 가중치 누락 및 양자화 모델 혼입을 확인한다."""
        if not (directory / "config.json").is_file() or not any(directory.glob("*.safetensors")):
            raise ValueError(f"MLX TTS 모델 파일이 준비되지 않았습니다: {directory}")
        config = json.loads((directory / "config.json").read_text())
        if config.get("quantization") or config.get("quantization_config"):
            raise ValueError("BF16 모델 경로에 양자화된 모델이 지정되었습니다.")
        dtype = config.get("dtype", config.get("torch_dtype"))
        if dtype not in {"bfloat16", "bf16"}:
            raise ValueError(f"BF16 모델·코덱이 필요합니다. 현재 정밀도: {dtype}")
        index = directory / "model.safetensors.index.json"
        if index.is_file():
            files = set(json.loads(index.read_text())["weight_map"].values())
            if any(not (directory / name).is_file() for name in files):
                raise ValueError(f"분할 가중치 다운로드가 완료되지 않았습니다: {directory}")

    def fingerprint(self, text):
        """모델·코덱 변경과 생성 설정을 WAV 재사용 식별자에 반영한다."""
        files = [(str(path), path.stat().st_size, path.stat().st_mtime_ns)
                 for directory in (self.model_dir, self.codec_dir)
                 for path in sorted(directory.glob("*")) if path.is_file()]
        settings = [text, files, self.max_tokens, self.audio_temperature, "Korean", 16000, "PCM16", 1, "bf16-sequential-codec-v2"]
        return hashlib.sha256(json.dumps(settings, ensure_ascii=False).encode("utf-8")).hexdigest()


class MlxTtsClient:
    def __init__(self, config, cancellation):
        """실제 모델 로딩은 첫 음성 요청까지 미루고 작업별 프로세스를 준비한다."""
        self.config = config
        self.cancellation = cancellation
        self._process = None
        self._log = None
        self.last_metrics = {}

    def generate(self, text, destination):
        """단일 발화도 본체 종료 후 코덱을 로딩하는 순서로 변환한다."""
        results = self.generate_batch([(text, destination)])
        return results[0]

    def generate_batch(self, requests, on_event=None):
        """전체 음성 코드를 먼저 만든 뒤 본체를 내리고 코덱으로 전체 WAV를 만든다."""
        if not requests:
            return []
        self.config.validate()
        results = []
        with tempfile.TemporaryDirectory(prefix="moss-audio-codes-") as directory:
            artifacts = [str(Path(directory) / f"{index}.safetensors") for index in range(len(requests))]
            try:
                self._ensure_worker("encode")
                encoded = self._encode_requests(requests, artifacts, on_event)
                self.close()
                self._ensure_worker("decode")
                results = self._decode_requests(requests, artifacts, encoded, on_event)
            finally:
                self.close()
        return results

    def _encode_requests(self, requests, artifacts, on_event):
        """본체 하나로 모든 발화의 음성 코드를 순차 생성한다."""
        results = []
        for index, (text, destination) in enumerate(requests):
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = {"text": text, "codes": artifacts[index]}
            response = self._request(payload, "encoded")
            results.append(response)
            if on_event:
                on_event(index, "encoded", response)
        return results

    def _decode_requests(self, requests, artifacts, encoded, on_event):
        """코덱 하나로 전체 음성 코드를 WAV로 변환하고 항목별 최대 메모리를 보존한다."""
        results = []
        for index, (_, destination) in enumerate(requests):
            response = self._request({"codes": artifacts[index], "destination": str(destination.resolve())}, "generated")
            metrics = {"peak_memory_gib": max(encoded[index].get("peak_memory_gib") or 0, response.get("peak_memory_gib") or 0),
                       "encode_worker_pid": encoded[index]["worker_pid"], "decode_worker_pid": response["worker_pid"]}
            self.last_metrics = metrics
            result = {**inspect_wav(destination), **metrics}
            results.append(result)
            if on_event:
                on_event(index, "generated", result)
        return results

    def _request(self, value, expected):
        """현재 단계 워커에 요청하고 취소와 오류를 확인한다."""
        if self.cancellation.is_set():
            raise AudioCancelledError()
        self._process.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        response = self._read_response(self.config.timeout)
        if response.get("event") != expected:
            raise RuntimeError(response.get("error", "MLX TTS 응답이 올바르지 않습니다."))
        return {**response, "worker_pid": self._process.pid}

    def _ensure_worker(self, phase="encode"):
        """모델은 자식 프로세스에서만 로딩하고 준비 완료 후 발화를 전달한다."""
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        self.config.validate()
        self._log = tempfile.TemporaryFile(mode="w+b")
        command = [self.config.executable, "-u", "-m", "asr_corpus.tts_worker",
                   "--model-dir", str(self.config.model_dir), "--codec-dir", str(self.config.codec_dir),
                   "--phase", phase, "--max-tokens", str(self.config.max_tokens), "--memory-limit-gb", str(self.config.memory_limit_gb),
                   "--audio-temperature", str(self.config.audio_temperature)]
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1"}
        self._process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parent.parent),
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                         text=True, encoding="utf-8", env=environment)
        response = self._read_response(self.config.load_timeout)
        if response.get("event") != "ready":
            raise RuntimeError(response.get("error", "MLX TTS 모델 로딩에 실패했습니다."))

    def _read_response(self, timeout):
        """워커 응답을 기다리면서 취소·시간 초과·프로세스 종료를 처리한다."""
        started = time.monotonic()
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while True:
                if self.cancellation.is_set():
                    raise AudioCancelledError()
                if time.monotonic() - started > timeout:
                    raise RuntimeError("MLX TTS 처리 제한 시간을 초과했습니다.")
                if selector.select(timeout=0.1):
                    line = self._process.stdout.readline()
                    if not line:
                        raise RuntimeError("MLX TTS 워커가 종료되었습니다. " + self._error_tail())
                    return json.loads(line)
                if self._process.poll() is not None:
                    raise RuntimeError("MLX TTS 워커가 종료되었습니다. " + self._error_tail())

    def _error_tail(self):
        """워커 실패 원인을 제한된 길이로 반환한다."""
        self._log.seek(0, 2)
        size = self._log.tell()
        self._log.seek(max(0, size - 2000))
        return self._log.read().decode("utf-8", errors="replace")

    def close(self):
        """자신의 워커 종료를 기다려 모델 메모리 해제 이후에만 반환한다."""
        process, self._process = self._process, None
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
