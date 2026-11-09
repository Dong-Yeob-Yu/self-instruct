"""모델 실행 없이 직접 입력 발화를 생성·추가·복원하는 API를 검증한다."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from asr_corpus.local import LlamaCppConfig
from asr_corpus.web import DatasetWebApplication
from asr_corpus.web_dataset import DatasetJobService


class ManualExamplesRegressionTest(unittest.TestCase):
    def setUp(self):
        """실제 모델 파일이나 서버가 없는 독립 저장소를 준비한다."""
        temporary = tempfile.TemporaryDirectory(prefix="manual-examples-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        model = self.root / "test-model.gguf"
        model.write_bytes(b"GGUFtest")
        self.config = LlamaCppConfig(model, executable=sys.executable)
        model.unlink()
        self.factory = Mock(side_effect=AssertionError("LLM 실행 금지"))
        self.service = DatasetJobService(self.config, self.root / "jobs", self.factory)
        self.addCleanup(self.service.close)
        self.app = DatasetWebApplication(self.service)

    def payload(self, examples=None):
        """한글 발화와 아라비아 정답의 테스트 입력을 반환한다."""
        return {"examples": examples or [{"spoken_text": "잔액은 오십삼 원입니다.", "normalized_text": "잔액은 53원입니다."}],
                "output_root": str(self.root / "output")}

    def create(self):
        """추론 스레드 없이 수동 작업을 생성한다."""
        with patch("threading.Thread.start", side_effect=AssertionError("추론 스레드 실행 금지")):
            response = self.app.dispatch("POST", "/api/jobs/manual", self.payload())
        self.assertEqual(response.status, 201, response.body)
        return json.loads(response.body)

    def test_manual_job_is_saved_and_restored_without_a_model_or_inference_thread(self):
        """모델 없이 새 목록을 만들고 재시작 후에도 검토 완료 상태를 복원한다."""
        job = self.create()
        self.assertEqual(job["status"], "completed")
        self.assertTrue(job["manual_only"])
        self.assertEqual(job["request_count"], 0)
        self.assertEqual(job["counts"]["accepted"], 1)
        self.assertEqual(job["examples"][0]["source"], "manual")
        self.assertTrue(job["examples"][0]["manually_edited"])
        self.factory.assert_not_called()
        restored = DatasetJobService(self.config, self.root / "jobs", self.factory)
        self.addCleanup(restored.close)
        current = restored.get_job(job["id"])
        self.assertEqual(current["examples"], job["examples"])
        exported = json.loads(restored.export_jsonl(job["id"], reviewed_only=True))
        self.assertEqual(exported["normalized_text"], "잔액은 53원입니다.")

    def test_append_preserves_existing_rows_and_accepts_multiple_manual_pairs(self):
        """기존 목록을 유지하면서 여러 발화·정답을 한 번에 추가한다."""
        job = self.create()
        examples = [{"spoken_text": "오늘은 영하 삼 점 오 도입니다.", "normalized_text": "오늘은 -3.5℃입니다."},
                    {"spoken_text": "사과 두 개 주세요.", "normalized_text": "사과 2개 주세요."}]
        response = self.app.dispatch("POST", f"/api/jobs/{job['id']}/examples", self.payload(examples))
        self.assertEqual(response.status, 201, response.body)
        updated = json.loads(response.body)
        self.assertEqual(updated["examples"][0], job["examples"][0])
        self.assertEqual(updated["counts"]["accepted"], 3)
        self.assertEqual(updated["settings"]["count"], 3)
        self.assertEqual([row["spoken_text"] for row in updated["examples"][1:]], [row["spoken_text"] for row in examples])
        self.factory.assert_not_called()

    def test_invalid_or_duplicate_batch_is_rejected_without_partial_writes(self):
        """한 행이 잘못되거나 기존 발화와 중복·충돌하면 전체 추가를 취소한다."""
        job = self.create()
        saved = self.root / "jobs" / job["id"] / "job.json"
        before = saved.read_bytes()
        valid = {"spoken_text": "새 문장입니다.", "normalized_text": "새 문장입니다."}
        invalid = [{"spoken_text": "", "normalized_text": "빈 발화"},
                   {"spoken_text": "발화", "normalized_text": ""},
                   {"spoken_text": "줄\n바꿈", "normalized_text": "줄바꿈"},
                   {"spoken_text": "가" * 1001, "normalized_text": "길이 초과"},
                   job["examples"][0],
                   {"spoken_text": job["examples"][0]["spoken_text"], "normalized_text": "다른 정답"}]
        for example in invalid:
            with self.subTest(example=example):
                response = self.app.dispatch("POST", f"/api/jobs/{job['id']}/examples", self.payload([valid, example]))
                self.assertEqual(response.status, 400, response.body)
                self.assertEqual(saved.read_bytes(), before)
                self.assertEqual(len(self.service.get_job(job["id"])["examples"]), 1)
        response = self.app.dispatch("POST", f"/api/jobs/{job['id']}/examples", self.payload([valid, valid]))
        self.assertEqual(response.status, 400)
        self.assertEqual(saved.read_bytes(), before)

    def test_invalid_new_job_does_not_leave_a_job_folder(self):
        """빈 목록·잘못된 경로·잘못된 입력은 작업 생성 전에 거부한다."""
        for payload in (None, {}, {"examples": []}, {"examples": "문장"}, {**self.payload(), "output_root": "relative"}):
            with self.subTest(payload=payload):
                self.assertEqual(self.app.dispatch("POST", "/api/jobs/manual", payload).status, 400)
                self.assertEqual(self.service.list_jobs(), [])
                self.assertEqual(list((self.root / "jobs").glob("*/job.json")), [])

    def test_active_text_generation_blocks_manual_addition(self):
        """생성 중인 목록은 직접 추가로 덮어쓰지 않는다."""
        job = self.create()
        self.service._jobs[job["id"]]["status"] = "running"
        try:
            response = self.app.dispatch("POST", f"/api/jobs/{job['id']}/examples", self.payload())
            self.assertEqual(response.status, 409)
            self.assertEqual(len(self.service._jobs[job["id"]]["examples"]), 1)
        finally:
            self.service._jobs[job["id"]]["status"] = "completed"

    def test_missing_job_returns_not_found(self):
        """없는 작업에 추가해도 새 작업이나 파일을 만들지 않는다."""
        response = self.app.dispatch("POST", "/api/jobs/" + "f" * 32 + "/examples", self.payload())
        self.assertEqual(response.status, 404)
        self.assertEqual(self.service.list_jobs(), [])


if __name__ == "__main__":
    unittest.main()
