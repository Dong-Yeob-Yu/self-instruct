"""직접 로컬 추론 연결과 CLI 출력 정규화의 회귀 테스트."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from asr_corpus.local import MODEL_FILENAME, LlamaCppClient, LlamaCppConfig, find_default_model, main


class LocalModelRegressionTest(unittest.TestCase):
    def setUp(self):
        """네트워크와 실제 모델 실행을 차단하고 자동 정리할 파일을 준비한다."""
        self.directory = tempfile.TemporaryDirectory(prefix="asr-local-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "test.gguf"
        self.model.write_bytes(b"GGUFtest")
        self.process = patch("subprocess.run", side_effect=AssertionError("실제 모델 실행 금지")).start()
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("네트워크 연결 금지")).start()
        self.addCleanup(patch.stopall)

    def make_client(self):
        """프로세스를 실행하지 않는 테스트 클라이언트를 생성한다."""
        return LlamaCppClient(LlamaCppConfig(self.model, executable=sys.executable))

    def fake_result(self, output, returncode=0):
        """모의 llama.cpp 실행 결과를 반환하도록 구성한다."""
        self.process.side_effect = None
        self.process.return_value = subprocess.CompletedProcess([], returncode, output, b"test failure")

    def test_cached_model_is_found_without_download(self):
        """다운로드가 완료된 기본 GGUF만 자동으로 찾는다."""
        snapshot = self.root / "models--ggml-org--gemma-4-E4B-it-GGUF" / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        (snapshot / MODEL_FILENAME).write_bytes(b"GGUFtest")
        self.assertTrue(find_default_model(self.root).samefile(snapshot / MODEL_FILENAME))
        self.network.assert_not_called()
        self.process.assert_not_called()

    def test_missing_or_incomplete_model_does_not_trigger_download(self):
        """캐시가 없거나 미완료 파일만 있으면 다운로드 없이 실패한다."""
        (self.root / "model.gguf.incomplete").write_bytes(b"GGUF")
        with self.assertRaises(ValueError):
            find_default_model(self.root)
        self.network.assert_not_called()

    def test_terminal_end_marker_is_removed_without_altering_utterance(self):
        """CLI 종료 표식만 제거하며 발화의 원문과 공백은 그대로 유지한다."""
        payload = {"examples": [{"text": "  원문 발화입니다.  ", "protected_terms": []}]}
        output = (json.dumps(payload, ensure_ascii=False) + " [end of text]\n\n").encode("utf-8")
        self.fake_result(output)
        self.assertEqual(json.loads(self.make_client().generate("프롬프트")), payload)

    def test_cli_uses_offline_single_turn_json_and_no_shell(self):
        """서버나 셸 없이 오프라인 단일 JSON 추론을 실행하도록 설정한다."""
        self.fake_result(b'{"examples": []}')
        self.make_client().generate("공백과 $()를 포함하는 프롬프트")
        arguments = self.process.call_args.args[0]
        for flag in ["--offline", "--single-turn", "--jinja", "--json-schema", "--no-escape"]:
            self.assertIn(flag, arguments)
        self.assertEqual(arguments[-1], "공백과 $()를 포함하는 프롬프트")
        self.assertEqual(self.process.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertNotIn("shell", self.process.call_args.kwargs)
        self.network.assert_not_called()

    def test_truncated_json_and_invalid_utf8_are_rejected(self):
        """잘린 JSON이나 깨진 한글을 자동 보정해 학습 데이터에 넣지 않는다."""
        for output in [b'{"examples": [', b'{"examples": ["\xff"]}', b'{"examples": []} unexpected']:
            self.fake_result(output)
            with self.assertRaises(ValueError):
                self.make_client().generate("프롬프트")

    def test_nonzero_exit_and_timeout_are_reported(self):
        """실행 실패와 시간 초과를 정상 응답으로 처리하지 않는다."""
        self.fake_result(b"", returncode=1)
        with self.assertRaises(RuntimeError):
            self.make_client().generate("프롬프트")
        self.process.side_effect = subprocess.TimeoutExpired("llama-completion", 1)
        with self.assertRaises(RuntimeError):
            self.make_client().generate("프롬프트")

    def test_non_gguf_file_is_rejected_before_inference(self):
        """잘못된 파일을 지정하면 모델 프로세스를 실행하지 않는다."""
        self.model.write_bytes(b"not a model")
        with self.assertRaises(ValueError):
            self.make_client()
        self.process.assert_not_called()

    def test_invalid_context_and_timeout_are_rejected(self):
        """컨텍스트를 초과하는 생성 설정과 무한 제한 시간을 거부한다."""
        with self.assertRaises(ValueError):
            LlamaCppConfig(self.model, context_size=1024, max_tokens=2048)
        with self.assertRaises(ValueError):
            LlamaCppConfig(self.model, timeout=float("nan"))

    def test_local_entry_point_reuses_filtering_and_output_format(self):
        """직접 추론도 기존 필터와 동일한 출력 형식을 사용한다."""
        self.fake_result('{"examples": ["첫 발화", "첫 발화", "다른 발화"]} [end of text]'.encode("utf-8"))
        output = self.root / "output"
        arguments = [
            "--requirement", "테스트용 요구사항", "--model-file", str(self.model),
            "--llama-executable", sys.executable, "--count", "2", "--max-requests", "1", "--output-dir", str(output),
        ]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(arguments), 0)
        summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["counts"], {"accepted": 2, "rejected": 1, "review": 0, "requests": 1})
        self.assertEqual(summary["configuration"]["backend"], "llama-completion")
        self.network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
