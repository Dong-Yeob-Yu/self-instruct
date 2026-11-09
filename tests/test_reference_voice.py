"""목소리 지정 제거 후 기존 데이터와 엑셀 업로드 호환을 검증한다."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from asr_corpus import tts_worker
from asr_corpus.local import LlamaCppConfig
from asr_corpus.mlx_tts import MlxTtsConfig
from asr_corpus.web import DatasetWebApplication
from asr_corpus.web_audio import AudioJobService
from asr_corpus.web_dataset import DatasetJobService
from test_asr_audio import write_wav
from test_excel_import import excel_payload, row


class VoiceSelectionRemovalRegressionTest(unittest.TestCase):
    def setUp(self):
        """실제 모델 없이 엑셀 업로드와 음성 생성을 검증할 저장소를 만든다."""
        temporary = tempfile.TemporaryDirectory(prefix="voice-removal-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        model, codec = self.root / "model", self.root / "codec"
        for directory in (model, codec):
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps({"dtype": "bfloat16"}))
            (directory / "model.safetensors").write_bytes(b"test")
        (model / "tokenizer.json").write_text("{}")
        llm = self.root / "test.gguf"
        llm.write_bytes(b"GGUFtest")
        self.dataset = DatasetJobService(LlamaCppConfig(llm, executable=sys.executable), self.root / "jobs", Mock(side_effect=AssertionError("LLM 실행 금지")))
        self.addCleanup(self.dataset.close)
        self.config = MlxTtsConfig(model, codec, sys.executable)
        self.calls = []
        self.audio = AudioJobService(self.dataset, self.config, self._client)
        self.addCleanup(self.audio.close)
        self.app = DatasetWebApplication(self.dataset, self.audio)

    def _client(self, config, cancellation):
        """참조 음성 설정이 없는 기본 클라이언트만 생성한다."""
        self.assertFalse(hasattr(config, "reference_audio"))
        def generate(text, destination):
            """전달된 발화를 기록하고 검증용 WAV를 만든다."""
            self.calls.append(text)
            write_wav(destination)
        return SimpleNamespace(generate=generate)

    def _job(self):
        """엑셀 A·B열을 기존 추가 API로 저장한다."""
        payload = excel_payload(row(1, "발화문", "정답지") + row(2, "사과 두 개", "사과 2개"))
        response = self.app.dispatch("POST", "/api/jobs/manual", payload)
        self.assertEqual(response.status, 201, response.body)
        return json.loads(response.body)

    def _generate(self, job):
        """저장된 발화를 선택해 음성 생성을 완료한다."""
        payload = {"example_ids": [item["id"] for item in job["examples"]], "output_root": str(self.root / "output")}
        response = self.app.dispatch("POST", f"/api/jobs/{job['id']}/audio", payload)
        self.assertEqual(response.status, 202, response.body)
        if self.audio._thread:
            self.audio._thread.join(timeout=3)
            self.assertFalse(self.audio._thread.is_alive())
        return self.audio.decorate_job(self.dataset.get_job(job["id"]))

    def test_voice_registration_and_assignment_endpoints_are_removed(self):
        """목소리 등록과 설정 API를 더 이상 제공하지 않는다."""
        job = self._job()
        self.assertEqual(self.app.dispatch("POST", "/api/voices", {}).status, 404)
        self.assertEqual(self.app.dispatch("PATCH", f"/api/jobs/{job['id']}/audio", {}).status, 405)
        config = json.loads(self.app.dispatch("GET", "/api/config").body)
        self.assertNotIn("voices", config)
        self.assertNotIn("reference_voice_available", config)

    def test_excel_column_c_is_ignored_and_a_b_pairs_remain_unchanged(self):
        """엑셀 C열의 이름·수식은 무시하고 A·B열만 저장한다."""
        rows = row(1, "발화문", "정답지")
        rows += row(2, "사과 두 개", "사과 2개").replace("</row>", '<c r="C2" t="inlineStr"><is><t>등록되지 않은 목소리</t></is></c></row>')
        rows += row(3, "배 세 개", "배 3개").replace("</row>", '<c r="C3"><f>1+1</f><v>2</v></c></row>')
        response = self.app.dispatch("POST", "/api/jobs/manual", excel_payload(rows))
        self.assertEqual(response.status, 201, response.body)
        job = json.loads(response.body)
        self.assertEqual([item["normalized_text"] for item in job["examples"]], ["사과 2개", "배 3개"])
        self.assertTrue(all("voice_id" not in item for item in job["examples"]))

    def test_old_voice_metadata_does_not_condition_new_generation_or_delete_saved_files(self):
        """이전 목소리 설정은 생성에 적용하지 않고 보관 음성 파일은 유지한다."""
        job = self._job()
        identifier = job["examples"][0]["id"]
        self.dataset._jobs[job["id"]]["examples"][0]["voice_id"] = "a" * 64
        self.audio._state(job["id"])["voice_assignments"] = {identifier: "a" * 64}
        voices = self.dataset.data_dir / "voices"
        voices.mkdir()
        preserved = voices / "user-reference.wav"
        write_wav(preserved)
        before = preserved.read_bytes()
        current = self._generate(job)
        self.assertEqual(self.calls, ["사과 두 개"])
        self.assertEqual(current["examples"][0]["audio"]["status"], "ready")
        self.assertEqual(current["examples"][0]["audio"]["fingerprint"], self.config.fingerprint("사과 두 개"))
        self.assertEqual(preserved.read_bytes(), before)
        self._generate(job)
        self.assertEqual(len(self.calls), 1)

    def test_worker_does_not_load_reference_codes_from_legacy_requests(self):
        """이전 요청에 참조 경로가 남아 있어도 텍스트만 모델에 전달한다."""
        core, model = Mock(), Mock()
        core.get_peak_memory.return_value = 1024
        model._processor.return_value.return_value = {"input_ids": "inputs"}
        model._generation_config_value.side_effect = lambda key, default: default
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=core), "mlx.core": core}), patch.object(tts_worker, "extract_code_segments", return_value=["codes"]):
            tts_worker.save_audio_codes(model, {"text": "발화문 그대로", "codes": "/codes", "reference_codes": "/old-reference"}, 128)
        model._processor.return_value.build_user_message.assert_called_once_with(text="발화문 그대로", language="Korean")
        core.load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
