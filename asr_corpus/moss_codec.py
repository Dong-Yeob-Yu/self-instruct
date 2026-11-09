"""텍스트 음성 합성에 필요한 MOSS 코덱 디코더만 BF16으로 로딩한다."""


def load_synthesis_codec(codec_dir):
    """참조 음성 인코더를 제외해 사용하지 않는 가중치의 GPU 점유를 줄인다."""
    import mlx.core as mx
    from mlx_audio.codec import MossAudioTokenizer
    from mlx_audio.codec.models.moss_audio_tokenizer.moss_audio_tokenizer import (
        AudioTokenizerConfig, _load_weights_from_dir, _sanitize_audio_tokenizer_weights,
    )
    config = AudioTokenizerConfig.from_file(codec_dir / "config.json")
    weights = _sanitize_audio_tokenizer_weights(_load_weights_from_dir(codec_dir))
    weights = {key: value for key, value in weights.items() if not key.startswith("encoder.")}
    projection_keys = {key for key in weights if key.endswith((".input_proj.weight", ".output_proj.weight"))}
    codec = MossAudioTokenizer(config, projection_keys=projection_keys)
    codec.encoder = []
    codec.load_weights(list(weights.items()), strict=True)
    mx.eval(codec.parameters())
    codec.eval()
    return codec
