"""확인한 생성 데이터만 삭제하고 모델·설정·다른 파일을 보존하는지 검증한다."""

import copy
import json
import sys
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from asr_corpus.local import LlamaCppConfig
from asr_corpus.tts import MagpieTtsConfig
from asr_corpus.web import DatasetWebApplication
from asr_corpus.web_audio import AudioJobService
from asr_corpus.web_dataset import DatasetJobService


class GeneratedDataDeletionRegressionTest(unittest.TestCase):
    def setUp(self):
        """실제 사용자 경로와 모델을 사용하지 않는 자동 정리 환경을 준비한다."""
        temporary = tempfile.TemporaryDirectory(prefix="asr-deletion-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "output"
        self.model = self.root / "models" / "model.gguf"
        self.model.parent.mkdir()
        self.model.write_bytes(b"GGUFweights")
        self.config = LlamaCppConfig(self.model, executable=sys.executable)
        self.tts_config = MagpieTtsConfig(self.root / "tts-models", executable=sys.executable)
        self.tts_config.tokenizer_dir.mkdir(parents=True)
        self.tts_config.model_file.write_bytes(b"GGUFtts")
        self.tts_config.codec_file.write_bytes(b"GGUFcodec")
        (self.tts_config.tokenizer_dir / "model_config.yaml").write_text("test: true\n")
        pairs = [{"spoken_text": "사과 세 개입니다.", "normalized_text": "사과 3개입니다.", "protected_terms": []},
                 {"spoken_text": "배 두 개를 주세요.", "normalized_text": "배 2개를 주세요.", "protected_terms": []}]
        client = SimpleNamespace(generate=lambda prompt: json.dumps({"examples": pairs}))
        self.dataset = DatasetJobService(self.config, self.root / "jobs", lambda config, cancel: client)
        self.addCleanup(self.dataset.close)
        self.dataset.save_preferences({"common_prompt": "유지할 공통 규칙", "output_root": str(self.output)})
        self.audio = AudioJobService(self.dataset, self.tts_config, lambda config, cancel: SimpleNamespace(generate=self.write_wav))
        self.addCleanup(self.audio.close)
        self.app = DatasetWebApplication(self.dataset, self.audio)
        replacement = patch("asr_corpus.web_deletion.DEFAULT_OUTPUT_ROOT", str(self.output))
        replacement.start()
        self.addCleanup(replacement.stop)

    def write_wav(self, text, path):
        """실제 TTS 대신 16kHz PCM16 테스트 음성을 만든다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setparams((1, 2, 16000, 1600, "NONE", "not compressed"))
            handle.writeframes(b"\x00\x00" * 1600)

    def job(self):
        """앱이 기록한 두 문장의 생성 작업을 완료한다."""
        job = self.dataset.start_job({"prompt": "한국어 발화", "count": 2, "batch_size": 2, "output_root": str(self.output)})
        thread = self.dataset._threads.get(job["id"])
        if thread:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        return self.dataset.get_job(job["id"])

    def payload(self, job, root=None):
        """선택 항목과 테스트 소유 출력 루트를 반환한다."""
        return {"example_ids": [row["id"] for row in job["examples"]], "output_root": str(root or self.output)}

    def generate(self, job, root=None):
        """테스트 작업의 WAV가 모두 생성될 때까지 기다린다."""
        self.audio.start(job["id"], self.payload(job, root))
        self.audio._thread.join(timeout=3)
        self.assertFalse(self.audio._thread.is_alive())

    def preview(self):
        """삭제하지 않고 API가 계산한 확인 대상을 읽는다."""
        response = self.app.dispatch("POST", "/api/data/deletion-preview", {})
        self.assertEqual(response.status, 200, response.body)
        return json.loads(response.body)

    def delete(self, preview):
        """사용자가 확인한 토큰을 명시적으로 승인하여 실행한다."""
        return self.app.dispatch("POST", "/api/data/delete-all", {"token": preview["token"], "confirmed": True})

    def test_preview_is_read_only_and_delete_removes_all_generated_artifacts_but_keeps_user_files(self):
        """작업·모든 WAV·여러 매핑 결과만 삭제하고 모델·공통 설정·추가 파일은 보존한다."""
        job = self.job()
        self.generate(job)
        mappings = [self.audio.create_mapping(job["id"], self.payload(job)) for _ in range(2)]
        user_note = self.dataset.data_dir / job["id"] / "my-note.txt"
        user_note.write_text("보존할 메모")
        model_copy = self.output / "model.gguf"
        model_copy.write_bytes(b"GGUFuser")
        mapped_note = Path(mappings[0]["output_dir"]) / "notes.txt"
        mapped_note.write_text("보존할 추가 파일")
        settings = self.dataset.data_dir / "preferences.json"
        original_settings = settings.read_bytes()
        original_job = (user_note.parent / "job.json").read_bytes()
        preview = self.preview()
        self.assertEqual((preview["jobs"], preview["examples"], preview["wav_files"], preview["datasets"]), (1, 2, 6, 2))
        self.assertEqual(preview["files"], 17)
        self.assertEqual((user_note.parent / "job.json").read_bytes(), original_job)
        response = self.delete(preview)
        self.assertEqual(response.status, 200, response.body)
        self.assertEqual(json.loads(response.body)["deleted_files"], 17)
        self.assertEqual(self.dataset.list_jobs(), [])
        self.assertEqual(list(self.output.rglob("*.wav")), [])
        self.assertEqual(self.audio._states, {})
        self.assertEqual(settings.read_bytes(), original_settings)
        self.assertEqual(self.model.read_bytes(), b"GGUFweights")
        self.assertTrue(self.tts_config.model_file.exists())
        self.assertTrue(model_copy.exists())
        self.assertEqual(user_note.read_text(), "보존할 메모")
        self.assertEqual(mapped_note.read_text(), "보존할 추가 파일")
        self.assertFalse((Path(mappings[1]["output_dir"])).exists())
        self.assertEqual(self.preview()["files"], 0)

    def test_delete_removes_previous_fingerprint_and_legacy_wavs_alongside_current_audio(self):
        """현재 WAV뿐 아니라 같은 발화의 이전 버전과 레거시 저장 WAV도 삭제한다."""
        job = self.job()
        self.generate(job)
        example_id = job["examples"][0]["id"]
        previous = self.output / "wav" / job["id"] / f"{example_id}-{'0' * 16}.wav"
        legacy = self.dataset.data_dir / job["id"] / "audio" / f"{example_id}-{'1' * 16}.wav"
        self.write_wav("이전 버전", previous)
        self.write_wav("레거시", legacy)
        preview = self.preview()
        self.assertEqual(preview["wav_files"], 4)
        response = self.delete(preview)
        self.assertEqual(response.status, 200, response.body)
        self.assertEqual(json.loads(response.body)["deleted_wav_files"], 4)
        self.assertEqual(list(self.output.rglob("*.wav")), [])
        self.assertEqual(list(self.dataset.data_dir.rglob("*.wav")), [])

    def test_confirmation_is_required_and_the_token_cannot_be_reused(self):
        """확인하지 않은 삭제와 이미 사용한 확인 토큰을 거부한다."""
        job = self.job()
        for payload in ({}, {"confirmed": True, "token": "unknown"}, {"confirmed": False, "token": "unknown"}):
            self.assertIn(self.app.dispatch("POST", "/api/data/delete-all", payload).status, (400, 409))
        self.assertTrue((self.dataset.data_dir / job["id"] / "job.json").exists())
        preview = self.preview()
        self.assertEqual(self.delete(preview).status, 200)
        self.assertEqual(self.delete(preview).status, 409)

    def test_changed_targets_require_a_new_preview(self):
        """확인 창을 연 뒤 파일이 바뀌면 아무것도 삭제하지 않고 재확인을 요청한다."""
        job = self.job()
        preview = self.preview()
        path = self.dataset.data_dir / job["id"] / "job.json"
        path.write_text(path.read_text() + "\n")
        self.assertEqual(self.delete(preview).status, 409)
        self.assertTrue(path.is_file())
        self.assertEqual(self.delete(self.preview()).status, 200)

    def test_expired_confirmation_does_not_delete_any_files(self):
        """오래된 삭제 확인은 만료시켜 대상 파일을 보존한다."""
        job = self.job()
        preview = self.preview()
        with patch("asr_corpus.web_deletion.time.monotonic", return_value=float("inf")):
            self.assertEqual(self.delete(preview).status, 409)
        self.assertTrue((self.dataset.data_dir / job["id"] / "job.json").exists())

    def test_running_text_or_audio_generation_blocks_preview_and_confirmation(self):
        """텍스트·음성 생성 중에는 삭제를 강행하지 않는다."""
        job = self.job()
        preview = self.preview()
        self.dataset._jobs[job["id"]]["status"] = "running"
        self.assertEqual(self.app.dispatch("POST", "/api/data/deletion-preview", {}).status, 409)
        self.assertEqual(self.delete(preview).status, 409)
        self.dataset._jobs[job["id"]]["status"] = "completed"
        self.audio._states[job["id"]] = {"status": "running", "examples": {}}
        self.assertEqual(self.app.dispatch("POST", "/api/data/deletion-preview", {}).status, 409)
        self.assertEqual(self.delete(preview).status, 409)
        self.audio._states[job["id"]]["status"] = "completed"

    def test_jobs_older_than_the_loaded_hundred_are_also_deleted(self):
        """화면의 최근 100개 제한 밖에 있는 작업도 전체 삭제에 포함한다."""
        template = self.job()
        for index in range(1, 102):
            identifier = f"{index:032x}"
            job = copy.deepcopy(template)
            job["id"] = identifier
            directory = self.dataset.data_dir / identifier
            directory.mkdir()
            (directory / "job.json").write_text(json.dumps(job))
        restored = DatasetJobService(self.config, self.dataset.data_dir)
        self.addCleanup(restored.close)
        self.assertEqual(len(restored.list_jobs()), 100)
        self.app = DatasetWebApplication(restored)
        preview = self.preview()
        self.assertEqual(preview["jobs"], 102)
        self.assertEqual(self.delete(preview).status, 200)
        self.assertEqual(list(self.dataset.data_dir.glob("*/job.json")), [])

    def test_previous_wav_and_mapping_roots_are_all_recorded_and_deleted(self):
        """저장 루트를 바꿔도 이전 WAV·매핑 결과를 누락하지 않는다."""
        job = self.job()
        first, second = self.root / "first-wav", self.root / "second-wav"
        self.generate(job, first)
        self.generate(job, second)
        mappings = [self.audio.create_mapping(job["id"], self.payload(job, self.root / name)) for name in ("first-map", "second-map")]
        preview = self.preview()
        self.assertEqual((preview["wav_files"], preview["datasets"]), (8, 2))
        self.assertIn(str(first), preview["roots"])
        self.assertIn(str(self.root / "first-map"), preview["roots"])
        self.assertEqual(self.delete(preview).status, 200)
        self.assertFalse(list(first.rglob("*.wav")))
        self.assertFalse(list(second.rglob("*.wav")))
        self.assertTrue(all(not Path(mapping["output_dir"]).exists() for mapping in mappings))

    def test_symlinked_wav_parent_never_deletes_files_or_directories_outside_the_output_root(self):
        """WAV 상위 폴더가 링크로 바뀌어도 외부 파일과 폴더를 삭제하지 않는다."""
        job = self.job()
        self.generate(job)
        original = self.output / "wav"
        external = self.root / "external-wav"
        original.rename(external)
        original.symlink_to(external, target_is_directory=True)
        preview = self.preview()
        self.assertGreater(preview["skipped"], 0)
        self.assertEqual(preview["wav_files"], 0)
        self.assertEqual(self.delete(preview).status, 200)
        self.assertEqual(len(list(external.rglob("*.wav"))), 2)
        self.assertTrue((external / job["id"]).exists())
        self.assertTrue(original.is_symlink())

    def test_symlinked_job_folder_and_unknown_mapping_are_preserved(self):
        """외부 작업 링크와 다른 출처의 데이터셋을 삭제 대상으로 보지 않는다."""
        job = self.job()
        external = self.root / "external-job"
        external.mkdir()
        foreign_id = "f" * 32
        (external / "job.json").write_text(json.dumps({**job, "id": foreign_id}))
        (self.dataset.data_dir / foreign_id).symlink_to(external, target_is_directory=True)
        foreign_mapping = self.output / ("dataset-" + job["id"][:8] + "-" + "a" * 12)
        foreign_mapping.mkdir(parents=True)
        (foreign_mapping / "dataset_info.json").write_text(json.dumps({"source_job_id": foreign_id}))
        preview = self.preview()
        self.assertEqual(preview["jobs"], 1)
        self.assertEqual(self.delete(preview).status, 200)
        self.assertTrue((external / "job.json").exists())
        self.assertTrue((foreign_mapping / "dataset_info.json").exists())

    def test_manifest_paths_cannot_redirect_deletion_to_other_files(self):
        """변조된 전사의 상위·절대 경로는 삭제 대상으로 해석하지 않는다."""
        job = self.job()
        self.generate(job)
        mapping = self.audio.create_mapping(job["id"], self.payload(job))
        outside = self.root / "important.wav"
        outside.write_bytes(b"KEEP")
        manifest = Path(mapping["transcripts"])
        manifest.write_text(json.dumps({"id": job["examples"][0]["id"], "audio": str(outside), "text": "other"}) + "\n")
        self.assertEqual(self.delete(self.preview()).status, 200)
        self.assertEqual(outside.read_bytes(), b"KEEP")

    def test_a_model_path_is_protected_even_when_named_like_a_generated_wav(self):
        """모델이 생성 WAV와 같은 이름이어도 실제 모델 경로는 보호한다."""
        job = self.job()
        self.generate(job)
        row = job["examples"][0]
        model = self.audio.audio_file(job["id"], row["id"])
        model.write_bytes(b"GGUFspecial-model")
        self.dataset.model_config = replace(self.config, model_file=model)
        preview = self.preview()
        self.assertGreater(preview["skipped"], 0)
        self.assertEqual(self.delete(preview).status, 200)
        self.assertEqual(model.read_bytes(), b"GGUFspecial-model")

    def test_partial_file_failure_keeps_metadata_for_retry_and_does_not_report_success(self):
        """삭제 실패 시 성공으로 보고하지 않고 남은 메타데이터로 다시 확인·삭제한다."""
        job = self.job()
        self.generate(job)
        mapping = self.audio.create_mapping(job["id"], self.payload(job))
        preview = self.preview()
        original_unlink = Path.unlink

        def fail_metadata(path, *args, **kwargs):
            """매핑 메타데이터에만 일시적인 권한 오류를 발생시킨다."""
            if path.name == "dataset_info.json":
                raise PermissionError("테스트 오류")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", fail_metadata):
            response = self.delete(preview)
        self.assertEqual(response.status, 409)
        self.assertIn("일부 생성 파일", json.loads(response.body)["error"])
        self.assertTrue((self.dataset.data_dir / job["id"] / "job.json").exists())
        self.assertTrue((Path(mapping["output_dir"]) / "dataset_info.json").exists())
        self.assertEqual(self.delete(self.preview()).status, 200)
        self.assertEqual(self.dataset.list_jobs(), [])
        self.assertFalse(Path(mapping["output_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
