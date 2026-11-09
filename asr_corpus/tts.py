"""로컬 Magpie TTS 실행과 Qwen ASR용 WAV 검증을 담당한다."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TTS_DIR = Path.home() / "Desktop/personal/dev/ai-models/magpie-tts-multilingual-v2607"


@dataclass(frozen=True)
class MagpieTtsConfig:
    model_dir: Path = DEFAULT_TTS_DIR
    executable: str = str(Path.home() / ".local/bin/nemo-speech")
    speaker: int = 0
    device: str = "metal"
    timeout: int = 300

    @property
    def model_file(self):
        """다운로드한 F16 모델의 로컬 경로를 반환한다."""
        return self.model_dir / "magpie_tts_multilingual_357m.v2607.f16.gguf"

    @property
    def codec_file(self):
        """다운로드한 NanoCodec 디코더의 로컬 경로를 반환한다."""
        return self.model_dir / "nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"

    @property
    def tokenizer_dir(self):
        """로컬 원본에서 추출한 토크나이저 디렉터리를 반환한다."""
        return self.model_dir / "tokenizer"

    def validate(self):
        """네트워크 다운로드 없이 실행에 필요한 로컬 파일만 검사한다."""
        if shutil.which(self.executable) is None:
            raise ValueError("Magpie 실행 도구를 찾지 못했습니다. nemo-speech 설치 경로를 확인하세요.")
        for path in (self.model_file, self.codec_file, self.tokenizer_dir / "model_config.yaml"):
            if not path.is_file():
                raise ValueError(f"Magpie 로컬 파일이 없습니다: {path}")
        if type(self.speaker) is not int or not 0 <= self.speaker <= 4:
            raise ValueError("Magpie 화자 번호는 0~4여야 합니다.")

    def fingerprint(self, text):
        """발화와 모델·생성 설정이 같은 WAV만 재사용하도록 식별자를 만든다."""
        files = [(str(path), path.stat().st_size, path.stat().st_mtime_ns)
                 for path in (self.model_file, self.codec_file, self.tokenizer_dir / "model_config.yaml") if path.is_file()]
        settings = [text, files, self.speaker, self.device, "ko-KR", 16000, 1]
        return hashlib.sha256(json.dumps(settings, ensure_ascii=False).encode("utf-8")).hexdigest()


class AudioCancelledError(Exception):
    """사용자가 요청한 음성 생성 중지를 나타낸다."""


def inspect_wav(path):
    """실제 샘플까지 읽어 60초 이내의 16kHz 모노 PCM16 WAV인지 검사한다."""
    try:
        with wave.open(str(path), "rb") as handle:
            if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getcomptype()) != (1, 2, 16000, "NONE"):
                raise ValueError("16kHz 모노 PCM16 WAV가 필요합니다.")
            frames = handle.getnframes()
            if not 0 < frames <= 16000 * 60 or len(handle.readframes(frames)) != frames * 2:
                raise ValueError("WAV가 비어 있거나 손상되었거나 60초를 초과했습니다.")
    except (wave.Error, EOFError) as error:
        raise ValueError("올바른 WAV 파일이 아닙니다.") from error
    return {"sampling_rate": 16000, "num_frames": frames, "duration_seconds": frames / 16000}


class MagpieTtsClient:
    def __init__(self, config, cancellation):
        """로컬 파일 설정과 작업 전용 중지 신호를 보관한다."""
        self.config = config
        self.cancellation = cancellation

    def generate(self, text, destination):
        """발화문을 명령행 옵션과 분리해 전달하고 검증된 WAV만 게시한다."""
        self.config.validate()
        if self.cancellation.is_set():
            raise AudioCancelledError()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tts-", dir=destination.parent) as temporary:
            source, output = Path(temporary) / "spoken.txt", Path(temporary) / "speech.wav"
            source.write_text(text, encoding="utf-8")
            command = self._command(source, output)
            self._run(command)
            information = inspect_wav(output)
            if self.cancellation.is_set():
                raise AudioCancelledError()
            os.replace(output, destination)
        return information

    def _command(self, source, output):
        """모델·코덱·토크나이저를 절대 로컬 경로로 지정한다."""
        return [self.config.executable, "synthesize", "--input", str(source), "--output", str(output),
                "--magpie-model", str(self.config.model_file), "--codec-model", str(self.config.codec_file),
                "--tokenizer-dir", str(self.config.tokenizer_dir), "--language", "ko-KR",
                "--speaker", str(self.config.speaker), "--sample-rate", "16000", "--device", self.config.device, "--seed", "1"]

    def _run(self, command):
        """소유한 추론 프로세스만 중지하고 오류를 한국어로 전달한다."""
        started = time.monotonic()
        with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            output, error = self._wait(process, started)
            if process.returncode != 0:
                detail = (error or output)[-1800:].decode("utf-8", errors="replace")
                raise RuntimeError("Magpie 음성 생성 실패: " + detail)

    def _wait(self, process, started):
        """시간 제한과 중지 이벤트를 확인하며 추론 출력을 수집한다."""
        while True:
            self._check_interruption(process, started)
            result = self._communicate(process)
            if result is not None:
                return result

    def _check_interruption(self, process, started):
        """취소 또는 제한 시간 초과 시 해당 프로세스를 종료한다."""
        cancelled = self.cancellation.is_set()
        if not cancelled and time.monotonic() - started <= self.config.timeout:
            return
        process.terminate()
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        if cancelled:
            raise AudioCancelledError()
        raise RuntimeError("음성 생성 제한 시간을 초과했습니다.")

    def _communicate(self, process):
        """긴 추론 중에도 중지 여부를 짧은 간격으로 확인한다."""
        try:
            return process.communicate(timeout=0.3)
        except subprocess.TimeoutExpired:
            return None
