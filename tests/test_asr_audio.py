"""로컬 음성 생성과 Qwen ASR 학습 자료 정합성을 서버 없이 검증한다."""

import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from asr_corpus.local import LlamaCppConfig
from asr_corpus.tts import AudioCancelledError, MagpieTtsClient, MagpieTtsConfig, inspect_wav
from asr_corpus.web import DatasetRequestHandler, DatasetWebApplication
from asr_corpus.web_audio import AudioJobService
from asr_corpus.web_dataset import DatasetJobService


def write_wav(path, frames=1600):
    """고정된 길이의 PCM16 WAV를 테스트 전용 폴더에 기록한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, 16000, frames, "NONE", "not compressed"))
        handle.writeframes(b"\x00\x00" * frames)


class AsrAudioRegressionTest(unittest.TestCase):
    def setUp(self):
        """실제 모델과 네트워크를 사용하지 않는 독립 작업 폴더를 만든다."""
        self.temporary = tempfile.TemporaryDirectory(prefix="asr-audio-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config = MagpieTtsConfig(self.root / "models", executable=sys.executable, device="cpu")
        self.config.tokenizer_dir.mkdir(parents=True)
        self.config.model_file.write_bytes(b"GGUFmodel")
        self.config.codec_file.write_bytes(b"GGUFcodec")
        (self.config.tokenizer_dir / "model_config.yaml").write_text("test: true\n")
        pairs = [{"spoken_text": "사과 세 개 주세요.", "normalized_text": "사과 3개 주세요.", "protected_terms": []},
                 {"spoken_text": "배 두 개가 필요합니다.", "normalized_text": "배 2개가 필요합니다.", "protected_terms": []},
                 {"spoken_text": "오늘은 우유 한 개를 주문했습니다.", "normalized_text": "오늘은 우유 1개를 주문했습니다.", "protected_terms": []}]
        client = SimpleNamespace(generate=lambda prompt: json.dumps({"examples": pairs}))
        self.dataset = DatasetJobService(LlamaCppConfig(self.config.model_file, executable=sys.executable),
                                         self.root / "jobs", lambda config, cancellation: client)
        self.addCleanup(self.dataset.close)
        created = self.dataset.start_job({"prompt": "한국어 숫자 변환", "count": 3, "batch_size": 3})
        thread = self.dataset._threads.get(created["id"])
        if thread:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.job = self.dataset.get_job(created["id"])
        self.assertEqual(self.job["counts"]["accepted"], 3)
        self.calls = []
        self.audio = self.new_audio()
        self.app = DatasetWebApplication(self.dataset, self.audio)

    def new_audio(self, factory=None, config=None):
        """음성 출력만 대체한 실제 작업 서비스를 만든다."""
        service = AudioJobService(self.dataset, config or self.config,
                                  factory or (lambda config, cancellation: SimpleNamespace(generate=self.generate)))
        self.addCleanup(service.close)
        return service

    def generate(self, text, path):
        """전달된 TTS 원문을 기록하고 짧은 WAV를 만든다."""
        self.calls.append(text)
        write_wav(path)

    def payload(self, identifiers=None, root=None):
        """명시적으로 선택한 항목과 테스트 전용 출력 루트를 전달한다."""
        rows = self.dataset.get_job(self.job["id"])["examples"]
        selected = identifiers if identifiers is not None else [row["id"] for row in rows if row["accepted"] and not row["needs_review"]]
        return {"example_ids": selected, "output_root": str(root or self.root / "output")}

    def finish(self, service=None, identifiers=None):
        """음성 작업이 종료될 때까지 기다리고 최신 스냅샷을 읽는다."""
        service = service or self.audio
        service.start(self.job["id"], self.payload(identifiers))
        service._thread.join(timeout=3)
        self.assertFalse(service._thread.is_alive())
        return service.decorate_job(self.dataset.get_job(self.job["id"]))

    def test_manual_examples_can_generate_wavs_and_save_a_selected_dataset(self):
        """직접 입력 목록을 기존 TTS와 선택 데이터셋 저장에 그대로 연결한다."""
        response = self.app.dispatch("POST", "/api/jobs/manual", {
            "examples": [{"spoken_text": "체온은 삼십육 점 오 도입니다.", "normalized_text": "체온은 36.5℃입니다."},
                         {"spoken_text": "잔액은 오십삼 원입니다.", "normalized_text": "잔액은 53원입니다."}],
            "output_root": str(self.root / "manual-output")})
        self.assertEqual(response.status, 201, response.body)
        job = json.loads(response.body)
        selected = [job["examples"][0]["id"]]
        payload = {"example_ids": selected, "output_root": str(self.root / "manual-output")}
        response = self.app.dispatch("POST", f"/api/jobs/{job['id']}/audio", payload)
        self.assertEqual(response.status, 202, response.body)
        self.audio._thread.join(timeout=3)
        self.assertFalse(self.audio._thread.is_alive())
        self.assertEqual(self.calls, ["체온은 삼십육 점 오 도입니다."])
        response = self.app.dispatch("POST", f"/api/jobs/{job['id']}/mapping", {**payload, "include_all_ready": True})
        self.assertEqual(response.status, 201, response.body)
        result = json.loads(response.body)
        self.assertEqual(result["count"], 1)
        transcript = json.loads(Path(result["transcripts"]).read_text())
        self.assertEqual(transcript["text"], "체온은 36.5℃입니다.")
        self.assertTrue((Path(result["output_dir"]) / transcript["audio"]).is_file())

    def test_manual_addition_is_blocked_during_audio_generation(self):
        """음성 생성 중 직접 추가 요청은 거부하고 기존 목록을 보존한다."""
        state = self.audio._state(self.job["id"])
        state["status"] = "running"
        try:
            response = self.app.dispatch("POST", f"/api/jobs/{self.job['id']}/examples", {
                "examples": [{"spoken_text": "새 문장", "normalized_text": "새 문장"}]})
            self.assertEqual(response.status, 409)
            self.assertEqual(len(self.dataset.get_job(self.job["id"])["examples"]), 3)
        finally:
            state["status"] = "completed"

    def test_batch_audio_tracks_all_code_and_wav_stages_and_closes_client(self):
        """전체 한글 발화를 한 배치로 전달하고 코드 완료·WAV 완료·메모리를 저장한다."""
        captured, closed = [], []

        class BatchClient:
            def generate_batch(self, requests, on_event):
                """모든 코드를 먼저 처리한 뒤 전체 WAV를 생성한다."""
                captured.extend(text for text, path in requests)
                for index in range(len(requests)):
                    on_event(index, "encoded", {"peak_memory_gib": 15.9})
                for index, (text, path) in enumerate(requests):
                    write_wav(path)
                    on_event(index, "generated", {"peak_memory_gib": 15.9})

            def close(self):
                """배치 완료 후 자원을 해제한 호출을 기록한다."""
                closed.append(True)

        service = self.new_audio(lambda config, cancellation: BatchClient())
        job = self.finish(service)
        self.assertEqual(captured, [row["spoken_text"] for row in self.job["examples"]])
        self.assertEqual(job["audio"]["counts"], {"total": 3, "ready": 3, "failed": 0})
        self.assertTrue(all(row["audio"]["peak_memory_gib"] == 15.9 for row in job["examples"]))
        self.assertEqual(closed, [True])

    def test_batch_error_marks_unconverted_rows_failed_and_closes_client(self):
        """배치 오류 시 미변환 항목을 실패로 표시하고 워커 종료를 호출한다."""
        closed = []

        class BatchClient:
            def generate_batch(self, requests, on_event):
                """첫 항목의 코드를 만든 뒤 테스트용 메모리 오류를 발생시킨다."""
                on_event(0, "encoded", {})
                raise RuntimeError("테스트 메모리 오류")

            def close(self):
                """오류 경로에서도 자원 해제 호출을 기록한다."""
                closed.append(True)

        service = self.new_audio(lambda config, cancellation: BatchClient())
        job = self.finish(service)
        self.assertEqual(job["audio"]["status"], "failed")
        self.assertEqual(job["audio"]["counts"], {"total": 3, "ready": 0, "failed": 3})
        self.assertEqual(closed, [True])

    def test_selected_examples_generate_from_spoken_text_and_export_current_answers(self):
        """발화문으로 만든 음성과 아라비아 숫자 정답을 정확히 연결한다."""
        job = self.finish()
        self.assertEqual(self.calls, [row["spoken_text"] for row in self.job["examples"]])
        self.assertEqual(job["audio"]["counts"], {"total": 3, "ready": 3, "failed": 0})
        path = self.audio.export_training(self.job["id"])
        with zipfile.ZipFile(path) as archive:
            train = [json.loads(line) for line in archive.read("train.jsonl").decode().splitlines()]
            validation = [json.loads(line) for line in archive.read("validation.jsonl").decode().splitlines()]
            self.assertEqual((len(train), len(validation)), (2, 1))
            expected = {row["id"]: row for row in self.job["examples"]}
            for row in train + validation:
                self.assertEqual(row["text"], expected[row["id"]]["normalized_text"])
                self.assertEqual(row["sampling_rate"], 16000)
                self.assertEqual(row["num_frames"], 1600)
                self.assertEqual(row["duration_seconds"], 0.1)
                self.assertTrue(archive.read(row["audio"]).startswith(b"RIFF"))
            info = json.loads(archive.read("dataset_info.json"))
            self.assertEqual(info["split_counts"], {"train": 2, "validation": 1})

    def test_review_pending_or_rejected_examples_do_not_generate_or_export(self):
        """검토 미완료·제외 항목은 음성과 학습 자료에 포함하지 않는다."""
        identifier = self.job["id"]
        with self.dataset._lock:
            self.dataset._jobs[identifier]["examples"][0].update(needs_review=True)
            self.dataset._jobs[identifier]["examples"][1].update(accepted=False)
        job = self.finish()
        self.assertEqual(job["audio"]["counts"]["total"], 1)
        self.assertEqual(self.calls, [self.job["examples"][2]["spoken_text"]])
        with self.assertRaisesRegex(ValueError, "최소 2개"):
            self.audio.export_training(identifier)

    def test_restart_reuses_finished_wavs_and_only_missing_files_are_regenerated(self):
        """재실행 후 완료 음성은 재사용하고 사라진 파일만 다시 만든다."""
        self.finish()
        row = self.job["examples"][0]
        self.audio.audio_file(self.job["id"], row["id"]).unlink()
        restarted = self.new_audio()
        self.finish(restarted)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.calls[-1], row["spoken_text"])

    def test_spoken_edit_invalidates_wav_and_preserves_the_original_file(self):
        """발화 수정 시 이전 WAV를 숨기고 새로운 음성을 별도 파일로 만든다."""
        self.finish()
        identifier, row = self.job["id"], self.job["examples"][0]
        previous = self.audio.audio_file(identifier, row["id"])
        self.dataset.update_example(identifier, row["id"], {"accepted": True, "spoken_text": "사과 네 개 주세요.", "normalized_text": "사과 4개 주세요."})
        current = self.audio.decorate_job(self.dataset.get_job(identifier))
        self.assertEqual(current["examples"][0]["audio"]["status"], "outdated")
        with self.assertRaises(RuntimeError):
            self.audio.audio_file(identifier, row["id"])
        with self.assertRaisesRegex(ValueError, "모두 생성"):
            self.audio.export_training(identifier)
        self.finish()
        self.assertTrue(previous.is_file())
        self.assertNotEqual(previous, self.audio.audio_file(identifier, row["id"]))
        self.assertEqual(len(self.calls), 4)

    def test_transcript_only_edit_reuses_audio_but_exports_the_new_text(self):
        """발화가 같으면 음성을 재사용하며 현재 정답 전사문을 내보낸다."""
        self.finish()
        identifier, row = self.job["id"], self.job["examples"][0]
        self.dataset.update_example(identifier, row["id"], {"accepted": True, "normalized_text": "사과 3 개 주세요."})
        path = self.audio.export_training(identifier)
        with zipfile.ZipFile(path) as archive:
            rows = [json.loads(line) for split in ("train", "validation") for line in archive.read(f"{split}.jsonl").decode().splitlines()]
        exported = next(item for item in rows if item["id"] == row["id"])
        self.assertEqual(exported["text"], "사과 3 개 주세요.")
        self.assertEqual(len(self.calls), 3)

    def test_model_settings_change_invalidates_existing_audio(self):
        """화자나 모델 설정이 바뀌면 기존 WAV를 현재 설정의 음성으로 취급하지 않는다."""
        self.finish()
        changed = self.new_audio(config=replace(self.config, speaker=1))
        snapshot = changed.decorate_job(self.job)
        self.assertEqual(snapshot["audio"]["counts"]["ready"], 0)
        self.assertTrue(all(row["audio"]["status"] == "outdated" for row in snapshot["examples"]))

    def test_failures_are_visible_and_retry_skips_successful_audio(self):
        """일부 실패를 표시하고 재시도 시 성공 음성을 다시 생성하지 않는다."""
        calls = Mock(side_effect=[RuntimeError("test failure"), None, None])

        def generate(text, path):
            """첫 항목만 실패하고 나머지 WAV를 기록한다."""
            calls(text)
            write_wav(path)

        self.audio.client_factory = lambda config, cancellation: SimpleNamespace(generate=generate)
        job = self.finish()
        self.assertEqual(job["audio"]["counts"], {"total": 3, "ready": 2, "failed": 1})
        self.assertIn("test failure", job["examples"][0]["audio"]["error"])
        self.audio.client_factory = lambda config, cancellation: SimpleNamespace(generate=self.generate)
        self.finish()
        self.assertEqual(self.calls, [self.job["examples"][0]["spoken_text"]])

    def test_cancel_keeps_completed_audio_and_disallows_review_during_generation(self):
        """생성 중 편집·동시 추론을 막고 중지 후 완료 WAV를 보존한다."""
        started = threading.Event()

        def factory(config, cancellation):
            """두 번째 음성에서 중지 신호를 기다리는 클라이언트를 만든다."""
            def generate(text, path):
                """첫 음성을 저장한 뒤 다음 항목에서 중지를 기다린다."""
                if self.calls:
                    started.set()
                    cancellation.wait(timeout=3)
                    raise AudioCancelledError()
                self.generate(text, path)
            return SimpleNamespace(generate=generate)

        self.audio.client_factory = factory
        self.audio.start(self.job["id"], self.payload())
        self.assertTrue(started.wait(timeout=3))
        row = self.job["examples"][0]
        self.assertEqual(self.app.dispatch("PATCH", f"/api/jobs/{self.job['id']}/examples/{row['id']}", {"accepted": False}).status, 409)
        self.assertEqual(self.app.dispatch("POST", "/api/jobs", {"prompt": "test"}).status, 409)
        self.audio.cancel(self.job["id"])
        self.audio._thread.join(timeout=3)
        job = self.audio.decorate_job(self.job)
        self.assertEqual(job["audio"]["status"], "cancelled")
        self.assertEqual(job["audio"]["counts"]["ready"], 1)
        self.assertEqual(job["examples"][1]["audio"]["status"], "interrupted")

    def test_interrupted_audio_state_is_recovered_after_app_restart(self):
        """앱 종료 당시 생성 중이던 항목을 완료로 오인하지 않고 복구한다."""
        row = self.job["examples"][0]
        state = {"status": "running", "message": "running", "examples": {row["id"]: {"status": "generating", "fingerprint": self.config.fingerprint(row["spoken_text"])}}}
        path = self.dataset.data_dir / self.job["id"] / "audio.json"
        path.write_text(json.dumps(state))
        restarted = self.new_audio()
        job = restarted.decorate_job(self.job)
        self.assertEqual(job["audio"]["status"], "interrupted")
        self.assertEqual(job["examples"][0]["audio"]["status"], "interrupted")

    def test_training_split_keeps_identical_transcripts_together(self):
        """동일 정답을 가진 여러 음성이 train·validation 양쪽에 섞이지 않는다."""
        groups = self.audio._validation_groups([{"normalized_text": "같은 정답"}, {"normalized_text": "같은 정답"}, {"normalized_text": "다른 정답"}])
        self.assertEqual(len(groups), 1)
        with self.assertRaises(ValueError):
            self.audio._validation_groups([{"normalized_text": "같은 정답"}] * 2)

    def test_api_serves_only_known_current_audio_and_a_streamed_training_archive(self):
        """재생·다운로드 API가 알려진 WAV와 Qwen ASR ZIP만 제공한다."""
        self.finish()
        identifier, row = self.job["id"], self.job["examples"][0]
        config = json.loads(self.app.dispatch("GET", "/api/config").body)
        self.assertTrue(config["audio_available"])
        response = self.app.dispatch("GET", f"/api/jobs/{identifier}/examples/{row['id']}/audio")
        self.assertEqual(response.content_type, "audio/wav")
        self.assertEqual(inspect_wav(response.file_path)["sampling_rate"], 16000)
        missing = self.app.dispatch("GET", f"/api/jobs/{identifier}/examples/{'0' * 32}/audio")
        self.assertEqual(missing.status, 404)
        self.assertEqual(self.app.dispatch("GET", f"/api/jobs/{identifier}/examples/../../audio").status, 404)
        exported = self.app.dispatch("GET", f"/api/jobs/{identifier}/training-export")
        self.assertEqual(exported.content_type, "application/zip")
        self.assertTrue(exported.temporary)
        handler = DatasetRequestHandler.__new__(DatasetRequestHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler._write_response(exported)
        self.assertFalse(exported.file_path.exists())
        self.assertTrue(handler.wfile.getvalue().startswith(b"PK"))

    def test_symlink_outside_job_folder_is_not_served(self):
        """작업 폴더 밖을 가리키는 음성 심볼릭 링크는 제공하지 않는다."""
        self.finish()
        identifier, row = self.job["id"], self.job["examples"][0]
        path = self.audio.audio_file(identifier, row["id"])
        outside = self.root / "outside.wav"
        path.rename(outside)
        path.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.audio.audio_file(identifier, row["id"])

    def test_damaged_or_unsupported_wav_cannot_be_exported(self):
        """헤더만 있거나 잘못된 형식의 WAV를 학습 자료로 내보내지 않는다."""
        self.finish()
        row = self.job["examples"][0]
        path = self.audio.audio_file(self.job["id"], row["id"])
        path.write_bytes(path.read_bytes()[:50])
        with self.assertRaises(ValueError):
            self.audio.export_training(self.job["id"])

    def test_missing_runtime_is_reported_without_launching_or_downloading(self):
        """실행 도구가 없으면 자동 다운로드 없이 준비 오류를 보여준다."""
        missing = self.new_audio(config=replace(self.config, executable="/missing/nemo-speech"))
        with patch("subprocess.Popen") as process:
            self.assertFalse(missing.availability()["audio_available"])
            with self.assertRaises(ValueError):
                missing.start(self.job["id"], self.payload())
            process.assert_not_called()

    def test_tts_uses_local_assets_and_a_text_file_instead_of_interpreting_text_as_options(self):
        """발화문은 UTF-8 파일로 전달하며 CLI 옵션으로 해석하지 않는다."""
        destination = self.root / "outputs" / "test.wav"
        client = MagpieTtsClient(self.config, threading.Event())
        captured = []

        def run(command):
            """실행 명령의 텍스트 전달과 출력 파일을 검증한다."""
            captured.extend(command)
            self.assertEqual(Path(command[command.index("--input") + 1]).read_text(), "--help 사과 세 개")
            write_wav(Path(command[command.index("--output") + 1]))

        with patch.object(client, "_run", side_effect=run):
            client.generate("--help 사과 세 개", destination)
        self.assertNotIn("--help 사과 세 개", captured)
        self.assertEqual(captured[captured.index("--sample-rate") + 1], "16000")
        self.assertEqual(captured[captured.index("--magpie-model") + 1], str(self.config.model_file))
        self.assertTrue(destination.exists())
        self.assertEqual(list(destination.parent.glob(".tts-*")), [])

    def test_tts_cancellation_terminates_only_its_own_process(self):
        """취소 시 작업이 생성한 TTS 프로세스만 종료한다."""
        cancellation = Mock()
        cancellation.is_set.side_effect = [False, True]
        process = MagicMock()
        process.__enter__.return_value = process
        process.communicate.side_effect = [subprocess.TimeoutExpired("fake", 0.3), (b"", b"")]
        client = MagpieTtsClient(self.config, cancellation)
        with patch("subprocess.Popen", return_value=process):
            with self.assertRaises(AudioCancelledError):
                client._run(["fake"])
        process.terminate.assert_called_once()
        process.kill.assert_not_called()

    def test_only_explicitly_selected_accepted_examples_generate_wav(self):
        """채택 항목 중 사용자가 선택한 발화만 WAV로 생성한다."""
        selected = self.job["examples"][1]
        result = self.finish(identifiers=[selected["id"]])
        self.assertEqual(self.calls, [selected["spoken_text"]])
        self.assertEqual(result["audio"]["counts"]["ready"], 1)
        path = self.audio.audio_file(self.job["id"], selected["id"])
        self.assertTrue(path.is_relative_to(self.root / "output" / "wav"))

    def test_audio_selection_rejects_unselected_invalid_and_rejected_rows(self):
        """선택 누락·중복·존재하지 않는 ID·제외 항목의 음성 생성을 거부한다."""
        row = self.job["examples"][0]
        for identifiers in ([], ["unknown"], [row["id"], row["id"]]):
            with self.assertRaises(ValueError):
                self.audio.start(self.job["id"], self.payload(identifiers))
        self.dataset.update_example(self.job["id"], row["id"], {"accepted": False})
        with self.assertRaises(ValueError):
            self.audio.start(self.job["id"], self.payload([row["id"]]))

    def test_mapping_uses_only_reselected_wavs_and_copies_into_the_requested_root(self):
        """청취 후 다시 선택한 WAV만 새 루트에 정답과 연결하며 원본을 보존한다."""
        self.finish()
        row = self.job["examples"][1]
        source = self.audio.audio_file(self.job["id"], row["id"])
        root = self.root / "custom-mapping-root"
        result = self.audio.create_mapping(self.job["id"], self.payload([row["id"]], root))
        destination = Path(result["output_dir"])
        self.assertEqual(destination.parent, root)
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["count"], 1)
        record = json.loads((destination / "transcripts.jsonl").read_text())
        self.assertEqual(record["id"], row["id"])
        self.assertEqual(record["text"], row["normalized_text"])
        self.assertEqual((destination / record["audio"]).read_bytes(), source.read_bytes())
        self.assertTrue(source.is_file())
        self.assertFalse((destination / "train.jsonl").exists())

    def test_mapping_creates_qwen_splits_and_never_overwrites_previous_mapping(self):
        """여러 정답은 학습·검증 목록도 만들며 재매핑은 새 폴더에 저장한다."""
        self.finish()
        first = self.audio.create_mapping(self.job["id"], self.payload())
        second = self.audio.create_mapping(self.job["id"], self.payload())
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        destination = Path(first["output_dir"])
        self.assertTrue(first["training_ready"])
        train = (destination / "train.jsonl").read_text().splitlines()
        validation = (destination / "validation.jsonl").read_text().splitlines()
        self.assertEqual((len(train), len(validation)), (2, 1))
        self.assertEqual(len(list((destination / "audio").glob("*.wav"))), 3)
        self.assertFalse(list(destination.parent.glob(".mapping-*")))

    def test_mapping_rejects_missing_wavs_and_relative_output_roots(self):
        """WAV 생성 전 매핑과 상대 출력 경로를 거부한다."""
        with self.assertRaisesRegex(ValueError, "WAV"):
            self.audio.create_mapping(self.job["id"], self.payload())
        payload = self.payload()
        payload["output_root"] = "relative/path"
        with self.assertRaisesRegex(ValueError, "절대 경로"):
            self.audio.start(self.job["id"], payload)

    def test_mapping_api_returns_the_created_dataset_path(self):
        """매핑 버튼 API가 선택한 음성과 정답을 저장한 실제 경로를 반환한다."""
        self.finish()
        response = self.app.dispatch("POST", f"/api/jobs/{self.job['id']}/mapping", self.payload())
        self.assertEqual(response.status, 201)
        result = json.loads(response.body)
        self.assertTrue(Path(result["transcripts"]).is_file())
        self.assertEqual(result["count"], 3)

    def test_audio_http_ranges_support_playback_seeking_and_reject_invalid_offsets(self):
        """WAV 재생·탐색 요청에 정확한 바이트와 범위 헤더를 반환한다."""
        self.finish()
        row = self.job["examples"][0]
        response = self.app.dispatch("GET", f"/api/jobs/{self.job['id']}/examples/{row['id']}/audio")
        content = response.file_path.read_bytes()
        for requested, expected_status, expected in (("bytes=0-43", 206, content[:44]),
                                                     ("bytes=44-", 206, content[44:]),
                                                     ("bytes=-10", 206, content[-10:]),
                                                     ("bytes=999999-", 416, b""),
                                                     ("bytes=-0", 416, b""),
                                                     ("bytes=1-0", 416, b"")):
            handler = DatasetRequestHandler.__new__(DatasetRequestHandler)
            handler.headers = {"Range": requested}
            handler.wfile = io.BytesIO()
            handler.send_response, handler.send_header, handler.end_headers = Mock(), Mock(), Mock()
            handler._write_response(response)
            handler.send_response.assert_called_once_with(expected_status)
            self.assertEqual(handler.wfile.getvalue(), expected)
            handler.send_header.assert_any_call("Content-Length", str(len(expected)))

    def test_changing_generation_root_saves_selected_audio_in_the_new_location(self):
        """생성 루트를 바꾸면 선택한 음성을 새 경로에 만들고 이전 WAV를 보존한다."""
        self.finish()
        row = self.job["examples"][0]
        previous = self.audio.audio_file(self.job["id"], row["id"])
        root = self.root / "new-output"
        self.audio.start(self.job["id"], self.payload([row["id"]], root))
        self.audio._thread.join(timeout=3)
        current = self.audio.audio_file(self.job["id"], row["id"])
        self.assertTrue(current.is_relative_to(root))
        self.assertTrue(previous.is_file())
        self.assertTrue(current.is_file())

    def test_audio_exclusion_is_restored_and_is_respected_by_dataset_saving(self):
        """음성 제외 결정을 저장하고 새로고침 후에도 최종 자료에서 제외한다."""
        job = self.finish()
        row = job["examples"][0]
        response = self.app.dispatch("PATCH", f"/api/jobs/{job['id']}/examples/{row['id']}/audio-review",
                                     {"included": False, "fingerprint": row["audio"]["fingerprint"]})
        self.assertEqual(response.status, 200)
        restored = self.new_audio()
        current = restored.decorate_job(self.dataset.get_job(job["id"]))
        self.assertFalse(current["examples"][0]["audio"]["included"])
        with self.assertRaisesRegex(ValueError, "제외한 항목"):
            restored.create_mapping(job["id"], self.payload())
        selected = [item["id"] for item in current["examples"][1:]]
        result = restored.create_mapping(job["id"], self.payload(selected))
        self.assertEqual(result["count"], 2)
        reloaded = self.new_audio().decorate_job(self.dataset.get_job(job["id"]))
        self.assertEqual(reloaded["audio"]["last_mapping"]["output_dir"], result["output_dir"])

    def test_footer_saves_all_ready_wavs_even_if_legacy_audio_review_excluded_one(self):
        """하단 전체 저장은 이전 화면에서 제외했던 유효 WAV도 함께 저장한다."""
        job = self.finish()
        row = job["examples"][0]
        self.audio.review_audio(job["id"], row["id"], {"included": False, "fingerprint": row["audio"]["fingerprint"]})
        payload = {**self.payload(), "include_all_ready": True}
        result = self.audio.create_mapping(job["id"], payload)
        self.assertEqual(result["count"], 3)
        self.assertEqual(len(list((Path(result["output_dir"]) / "audio").glob("*.wav"))), 3)

    def test_old_audio_review_cannot_apply_to_a_new_fingerprint(self):
        """이전 음성 화면의 제외 요청을 새로 생성된 음성에 적용하지 않는다."""
        job = self.finish()
        row = job["examples"][0]
        response = self.app.dispatch("PATCH", f"/api/jobs/{job['id']}/examples/{row['id']}/audio-review",
                                     {"included": False, "fingerprint": "old"})
        self.assertEqual(response.status, 409)
        current = self.audio.decorate_job(self.job)
        self.assertIsNot(current["examples"][0]["audio"].get("included"), False)


if __name__ == "__main__":
    unittest.main()
