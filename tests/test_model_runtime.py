"""모델 실행 직렬화와 작업 단위 TTS 프로세스의 해제를 검증한다."""

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from asr_corpus.local import LlamaCppConfig
from asr_corpus.mlx_tts import MlxTtsClient, MlxTtsConfig
from asr_corpus.model_runtime import ModelRuntime
from asr_corpus.moss_codec import load_synthesis_codec
from asr_corpus import tts_worker
from asr_corpus.tts import AudioCancelledError, inspect_wav
from asr_corpus.web import DatasetWebApplication
from asr_corpus.web_audio import AudioJobService
from asr_corpus.web_dataset import DatasetJobService


FAKE_WORKER = '''
import json,sys,wave
print(json.dumps({"event":"ready"}),flush=True)
for line in sys.stdin:
    value=json.loads(line)
    if "text" in value:
        with open(value["codes"],"wb") as output:
            output.write(b"codes")
        print(json.dumps({"event":"encoded"}),flush=True)
    else:
        with wave.open(value["destination"],"wb") as output:
            output.setparams((1,2,16000,1600,"NONE","not compressed"))
            output.writeframes(bytes(3200))
        print(json.dumps({"event":"generated"}),flush=True)
'''


class ModelRuntimeTest(unittest.TestCase):
    def test_audio_temperature_defaults_to_one_point_five_and_changes_wav_fingerprint(self):
        """음성 온도는 기본 1.5이며 다른 온도의 기존 WAV를 재사용하지 않는다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MlxTtsConfig(root, root, sys.executable)
            self.assertEqual(config.audio_temperature, 1.5)
            self.assertNotEqual(config.fingerprint("영하 삼점오 도씨입니다."),
                                replace(config, audio_temperature=1.7).fingerprint("영하 삼점오 도씨입니다."))
            for value in (0, -1, True, float("nan"), float("inf")):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "음성 샘플링 온도"):
                    replace(config, audio_temperature=value).validate()

    def test_worker_passes_audio_temperature_to_sampler_separately_from_text_temperature(self):
        """음성 온도만 1.5로 적용하고 텍스트 샘플링 설정은 덮어쓰지 않는다."""
        core = Mock()
        core.get_peak_memory.return_value = 1024
        model = Mock()
        model._generation_config_value.side_effect = lambda key, default: 1.3 if key == "temperature" else default
        model._processor.return_value.return_value = {"input_ids": "inputs"}
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=core), "mlx.core": core}), patch.object(tts_worker, "extract_code_segments", return_value=["codes"]):
            tts_worker.save_audio_codes(model, {"text": "영하 삼점오 도씨입니다.", "codes": "/test/codes.safetensors"}, 128)
        options = model.generate_delay_pattern_ids.call_args.kwargs
        self.assertEqual(options["audio_temperature"], 1.5)
        self.assertEqual(options["text_temperature"], 1.3)

    def test_worker_pins_memory_only_within_device_recommendation(self):
        """워커의 메모리 상주 범위가 장비 권장치와 지정 예산을 넘지 않는다."""
        for recommended, expected in ((18 * 1024 ** 3, 18 * 1024 ** 3), (24 * 1024 ** 3, 20 * 1024 ** 3)):
            core = Mock()
            core.device_info.return_value = {"max_recommended_working_set_size": recommended}
            model = Mock()
            modules = {"mlx": SimpleNamespace(core=core), "mlx.core": core,
                       "mlx_audio": SimpleNamespace(), "mlx_audio.tts": SimpleNamespace(),
                       "mlx_audio.tts.utils": SimpleNamespace(load_model=Mock(return_value=model))}
            arguments = ["tts_worker", "--model-dir", "/test/model", "--codec-dir", "/test/codec", "--memory-limit-gb", "20"]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", arguments), patch.object(sys, "stdin", []), patch.object(tts_worker, "emit"), patch("asr_corpus.moss_codec.load_synthesis_codec"):
                self.assertEqual(tts_worker.main(), 0)
            core.set_wired_limit.assert_called_once_with(expected)
            core.set_memory_limit.assert_called_once_with(20 * 1024 ** 3)

    def test_synthesis_codec_keeps_decoder_and_quantizer_but_does_not_load_encoder(self):
        """BF16 디코더·코드 복원 가중치는 유지하고 사용하지 않는 인코더는 로딩하지 않는다."""
        encoder, decoder, quantizer = object(), object(), object()
        weights = {"encoder.0.weight": encoder, "decoder.0.input_proj.weight": decoder, "quantizer.weight": quantizer}
        codec = Mock()
        factory = Mock(return_value=codec)
        core = Mock()
        implementation = SimpleNamespace(AudioTokenizerConfig=Mock(),
            _load_weights_from_dir=Mock(return_value=weights),
            _sanitize_audio_tokenizer_weights=lambda values: values)
        modules = {"mlx": SimpleNamespace(core=core), "mlx.core": core,
                   "mlx_audio": SimpleNamespace(), "mlx_audio.codec": SimpleNamespace(MossAudioTokenizer=factory),
                   "mlx_audio.codec.models.moss_audio_tokenizer.moss_audio_tokenizer": implementation}
        with patch.dict(sys.modules, modules):
            result = load_synthesis_codec(Path("/test/codec"))
        self.assertIs(result, codec)
        self.assertEqual(codec.encoder, [])
        codec.load_weights.assert_called_once_with([("decoder.0.input_proj.weight", decoder), ("quantizer.weight", quantizer)], strict=True)
        core.eval.assert_called_once_with(codec.parameters.return_value)

    def test_bf16_assets_reject_quantization_wrong_dtype_and_missing_shards(self):
        """BF16 경로의 양자화·FP32 혼입과 분할 가중치 누락을 거부한다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, codec = root / "model", root / "codec"
            for folder in (model, codec):
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps({"dtype": "bfloat16"}))
                (folder / "model.safetensors").write_bytes(b"test")
            (model / "tokenizer.json").write_text("{}")
            config = MlxTtsConfig(model, codec, sys.executable)
            config.validate()
            (model / "config.json").write_text(json.dumps({"dtype": "bfloat16", "quantization": {"bits": 8}}))
            with self.assertRaisesRegex(ValueError, "양자화"):
                config.validate()
            (model / "config.json").write_text(json.dumps({"dtype": "bfloat16"}))
            (codec / "config.json").write_text(json.dumps({"dtype": "float32"}))
            with self.assertRaisesRegex(ValueError, "정밀도"):
                config.validate()
            (codec / "config.json").write_text(json.dumps({"dtype": "bfloat16"}))
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "분할 가중치"):
                config.validate()

    def test_tts_waits_until_llm_execution_has_released(self):
        """LLM 점유 중에는 TTS 실행이 시작되지 않고 해제 후에만 실행한다."""
        runtime = ModelRuntime()
        entered = threading.Event()
        completed = threading.Event()
        cancellation = threading.Event()

        def tts():
            """LLM 종료를 기다린 뒤 TTS 실행 구간에 진입한다."""
            with runtime.use("tts", cancellation, AudioCancelledError):
                entered.set()
            completed.set()

        with runtime.use("llm", cancellation, AudioCancelledError):
            worker = threading.Thread(target=tts)
            worker.start()
            self.assertFalse(entered.wait(timeout=0.15))
            self.assertEqual(runtime.snapshot()["active_model"], "llm")
        self.assertTrue(completed.wait(timeout=2))
        worker.join(timeout=2)
        self.assertEqual(runtime.snapshot(), {"active_model": None, "status": "idle", "last_released": "tts"})

    def test_runtime_releases_after_failure_and_cancellation(self):
        """오류와 취소 이후에도 실행 잠금과 모델 점유를 해제한다."""
        runtime = ModelRuntime()
        cancellation = threading.Event()
        with self.assertRaises(RuntimeError):
            with runtime.use("tts", cancellation, AudioCancelledError):
                raise RuntimeError("테스트 오류")
        self.assertIsNone(runtime.snapshot()["active_model"])
        cancellation.set()
        with self.assertRaises(AudioCancelledError):
            with runtime.use("tts", cancellation, AudioCancelledError):
                self.fail("취소된 작업 실행 금지")
        self.assertIsNone(runtime.snapshot()["active_model"])

    def test_one_encoder_and_one_decoder_process_generate_whole_batch_and_are_reaped(self):
        """전체 발화는 본체·코덱을 각각 한 번만 로딩하고 두 프로세스 모두 종료한다."""
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = MlxTtsClient(MlxTtsConfig(root, root, sys.executable), threading.Event())
            captured, processes, events = [], [], []

            def popen(command, **options):
                """앞 단계 프로세스가 종료된 뒤에만 다음 단계를 실행한다."""
                if processes:
                    self.assertIsNotNone(processes[-1].poll())
                captured.append(command)
                process = real_popen([sys.executable, "-u", "-c", FAKE_WORKER], **options)
                processes.append(process)
                return process

            try:
                with patch.object(MlxTtsConfig, "validate"), patch("asr_corpus.mlx_tts.subprocess.Popen", side_effect=popen):
                    results = client.generate_batch([("삼만 원", root / "first.wav"), ("영하 삼 도씨", root / "second.wav")],
                        lambda index, phase, metrics: events.append((index, phase)))
                self.assertEqual(len(captured), 2)
                self.assertEqual([command[command.index("--phase") + 1] for command in captured], ["encode", "decode"])
                self.assertTrue(all(command[command.index("--audio-temperature") + 1] == "1.5" for command in captured))
                self.assertEqual(events, [(0, "encoded"), (1, "encoded"), (0, "generated"), (1, "generated")])
                self.assertEqual(results[0]["encode_worker_pid"], results[1]["encode_worker_pid"])
                self.assertEqual(results[0]["decode_worker_pid"], results[1]["decode_worker_pid"])
                self.assertNotEqual(results[0]["encode_worker_pid"], results[0]["decode_worker_pid"])
                self.assertTrue(all(process.poll() is not None for process in processes))
                self.assertEqual(inspect_wav(root / "first.wav")["sampling_rate"], 16000)
                self.assertEqual(inspect_wav(root / "second.wav")["num_frames"], 1600)
            finally:
                client.close()

    def test_worker_is_terminated_when_generation_is_cancelled(self):
        """작업 취소 시 살아 있는 TTS 워커도 종료하고 회수한다."""
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cancellation = threading.Event()
            client = MlxTtsClient(MlxTtsConfig(root, root, sys.executable), cancellation)
            process = None
            try:
                with patch.object(MlxTtsConfig, "validate"), patch("asr_corpus.mlx_tts.subprocess.Popen", side_effect=lambda command, **options: real_popen([sys.executable, "-u", "-c", FAKE_WORKER], **options)):
                    client._ensure_worker("encode")
                    process = client._process
                    with patch.object(client, "_read_response", side_effect=AudioCancelledError):
                        with self.assertRaises(AudioCancelledError):
                            client.generate("네 개", root / "second.wav")
                self.assertIsNotNone(process.poll())
                self.assertIsNone(client._process)
            finally:
                client.close()

    def test_code_padding_gaps_preserve_independent_segments(self):
        """패딩으로 분리된 코드 구간을 연결하지 않고 독립 구간으로 보존한다."""
        self.assertEqual(tts_worker.split_code_segments(list(range(8)), [0, 1, 4, 6, 7]), [[0, 1], [4], [6, 7]])
        self.assertEqual(tts_worker.split_code_segments(list(range(8)), []), [])

    def test_auto_audio_receives_only_spoken_text_after_llm_release(self):
        """LLM 종료 후 전체 한글 발화만 TTS에 보내고 마지막에 클라이언트를 종료한다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"GGUFtest")
            llm = Mock()
            llm.generate.return_value = json.dumps({"examples": [
                {"spoken_text": "사과 세 개 주세요.", "normalized_text": "사과 3개 주세요.", "protected_terms": []},
                {"spoken_text": "예약은 다섯 시 입니다.", "normalized_text": "예약은 5시 입니다.", "protected_terms": []}]})
            dataset = DatasetJobService(LlamaCppConfig(model, executable=sys.executable), root / "jobs", lambda config, cancellation: llm)
            config = Mock()
            config.fingerprint.side_effect = lambda text: ("a" if "사과" in text else "b") * 64
            texts, closed = [], []

            class Client:
                def generate(self, text, path):
                    """모델 점유와 한글 발화를 확인하고 WAV를 생성한다."""
                    self_case.assertEqual(dataset.model_runtime.snapshot()["active_model"], "tts")
                    texts.append(text)
                    import wave
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with wave.open(str(path), "wb") as handle:
                        handle.setparams((1, 2, 16000, 1600, "NONE", "not compressed"))
                        handle.writeframes(bytes(3200))

                def close(self):
                    """종료가 TTS 실행 구간을 벗어나기 전에 호출되는지 기록한다."""
                    self_case.assertEqual(dataset.model_runtime.snapshot()["active_model"], "tts")
                    closed.append(True)

            self_case = self
            audio = AudioJobService(dataset, config, lambda config, cancellation: Client())
            app = DatasetWebApplication(dataset, audio)
            try:
                response = app.dispatch("POST", "/api/jobs", {"prompt": "숫자 안내", "count": 2, "batch_size": 2,
                    "auto_audio": True, "output_root": str(root / "output")})
                job = json.loads(response.body)
                worker = dataset._threads.get(job["id"])
                if worker:
                    worker.join(timeout=3)
                if audio._thread:
                    audio._thread.join(timeout=3)
                final = dataset.get_job(job["id"])
                self.assertEqual(final["auto_audio_status"], "started")
                self.assertEqual(texts, ["사과 세 개 주세요.", "예약은 다섯 시 입니다."])
                self.assertEqual(closed, [True])
                self.assertEqual(dataset.model_runtime.snapshot()["last_released"], "tts")
                self.assertIsNone(dataset.model_runtime.snapshot()["active_model"])
                self.assertEqual(audio.decorate_job(final)["audio"]["counts"]["ready"], 2)
            finally:
                audio.close()
                dataset.close()
