"""BF16 TTS 본체와 코덱을 별도 프로세스에서 순차 실행한다."""

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path


def emit(value):
    """제어 응답만 표준 출력에 기록한다."""
    print(json.dumps(value, ensure_ascii=False), flush=True)


def save_audio_codes(model, request, max_tokens, audio_temperature=1.5):
    """코덱 없이 MOSS 본체로 한글 발화의 음성 코드를 생성해 보존한다."""
    import mlx.core as mx
    processor = model._processor()
    conversation = [processor.build_user_message(text=request["text"], language="Korean")]
    batch = processor([conversation], mode="generation")
    mx.random.seed(1)
    outputs = model.generate_delay_pattern_ids(
        batch["input_ids"], max_new_tokens=max_tokens,
        text_temperature=float(model._generation_config_value("temperature", 1.5)),
        text_top_p=float(model._generation_config_value("top_p", 1.0)), text_top_k=50,
        audio_temperature=audio_temperature,
        audio_top_p=float(model._generation_config_value("top_p", 0.8)),
        audio_top_k=int(model._generation_config_value("top_k", 25)),
        audio_repetition_penalty=float(model._generation_config_value("repetition_penalty", 1.0)),
    )
    segments = extract_code_segments(outputs, model.config.audio_pad_code)
    if not segments:
        raise ValueError("생성된 음성 코드가 없습니다.")
    mx.save_safetensors(request["codes"], {str(index): codes for index, codes in enumerate(segments)})
    del outputs, segments, batch
    mx.clear_cache()
    return {"peak_memory_gib": round(mx.get_peak_memory() / (1024 ** 3), 3)}


def extract_code_segments(outputs, pad_code):
    """MOSS 지연 패턴과 패딩을 제거하되 분리된 음성 구간의 순서를 보존한다."""
    import mlx.core as mx
    from mlx_audio.tts.models.moss_tts.processor import apply_de_delay_pattern
    segments = []
    for start_length, generation_ids in outputs:
        if start_length:
            raise ValueError("현재 TTS 작업은 참조 음성 없는 텍스트 합성만 지원합니다.")
        codes = apply_de_delay_pattern(generation_ids[:, 1:].astype(mx.int32))
        indices = [index for index, is_pad in enumerate(mx.all(codes == pad_code, axis=1).tolist()) if not is_pad]
        segments.extend(split_code_segments(codes, indices))
    return segments


def split_code_segments(codes, indices):
    """연속하지 않은 코드 인덱스를 독립된 디코딩 구간으로 분리한다."""
    if not indices:
        return []
    boundaries = [0] + [index for index in range(1, len(indices)) if indices[index] != indices[index - 1] + 1] + [len(indices)]
    return [codes[indices[start]:indices[end - 1] + 1] for start, end in zip(boundaries[:-1], boundaries[1:])]


def write_audio(codec, request):
    """작은 코덱만 로딩한 상태에서 음성 코드를 16kHz 모노 PCM16 WAV로 변환한다."""
    import mlx.core as mx
    import numpy as np
    weights = mx.load(request["codes"])
    chunks = [codec.decode_audio_codes(weights[key]) for key in sorted(weights, key=int)]
    audio = np.concatenate([np.asarray(chunk.astype(mx.float32)).reshape(-1) for chunk in chunks])
    rate = codec.sample_rate
    publish_wav(audio, rate, Path(request["destination"]))
    del weights, chunks, audio
    mx.clear_cache()
    return {"peak_memory_gib": round(mx.get_peak_memory() / (1024 ** 3), 3)}


def publish_wav(audio, rate, destination):
    """유효한 음성만 리샘플링해 목적지에 원자적으로 게시한다."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    if not np.isfinite(audio).all() or not 0 < len(audio) <= rate * 60:
        raise ValueError("음성이 비어 있거나 잘못되었거나 60초를 초과했습니다.")
    if rate != 16000:
        divisor = math.gcd(int(rate), 16000)
        audio = resample_poly(audio, 16000 // divisor, int(rate) // divisor)
    temporary = destination.with_suffix(".tmp.wav")
    try:
        sf.write(temporary, np.clip(audio, -1, 1), 16000, format="WAV", subtype="PCM_16")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_phase_model(args):
    """음성 코드 생성에는 본체만, WAV 변환에는 코덱만 로딩한다."""
    if args.phase == "encode":
        from mlx_audio.tts.utils import load_model
        return load_model(str(args.model_dir))
    from .moss_codec import load_synthesis_codec
    return load_synthesis_codec(args.codec_dir)


def main():
    """한 단계의 모델 하나를 재사용하며 부모의 요청을 순차 처리한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--codec-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("encode", "decode"), default="encode")
    parser.add_argument("--audio-temperature", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--memory-limit-gb", type=int, default=20)
    args = parser.parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            mx.set_memory_limit(args.memory_limit_gb * 1024 ** 3)
            mx.set_cache_limit(256 * 1024 ** 2)
            recommended = mx.device_info()["max_recommended_working_set_size"]
            mx.set_wired_limit(min(args.memory_limit_gb * 1024 ** 3, recommended))
            model = load_phase_model(args)
        emit({"event": "ready"})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                with contextlib.redirect_stdout(sys.stderr):
                    metrics = save_audio_codes(model, request, args.max_tokens, args.audio_temperature) if args.phase == "encode" else write_audio(model, request)
                emit({"event": "encoded" if args.phase == "encode" else "generated", **metrics})
            except Exception as error:
                emit({"event": "error", "error": str(error)[:1800]})
    except Exception as error:
        emit({"event": "error", "error": str(error)[:1800]})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
