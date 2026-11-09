"""실제 서버 실행 없이 호스트 옵션과 요청 출처 검증을 확인한다."""

import io
import unittest
from email.message import Message
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from asr_corpus.web import DatasetRequestHandler, create_server, main, run_application


class WebHostTest(unittest.TestCase):
    def handler(self, bind_host, host, origin=None):
        """소켓 없이 바인딩 주소와 요청 헤더를 가진 처리기를 만든다."""
        handler = DatasetRequestHandler.__new__(DatasetRequestHandler)
        handler.server = SimpleNamespace(server_address=(bind_host, 8765), bind_host=bind_host)
        handler.headers = Message()
        handler.headers["Host"] = host
        if origin is not None:
            handler.headers["Origin"] = origin
        return handler

    def test_host_option_defaults_to_loopback_and_accepts_all_interfaces(self):
        """기본 로컬 바인딩을 유지하면서 전체 인터페이스 옵션을 전달한다."""
        with patch("asr_corpus.web.run_application", return_value=0) as run:
            self.assertEqual(main([]), 0)
            self.assertEqual(run.call_args.args[0].host, "127.0.0.1")
            self.assertEqual(run.call_args.args[0].tts_backend, "mlx")
            self.assertEqual(run.call_args.args[0].tts_memory_limit_gb, 20)
            self.assertEqual(run.call_args.args[0].model_file.name, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
            self.assertEqual(main(["--model-file", "/tmp/custom.gguf"]), 0)
            self.assertEqual(str(run.call_args.args[0].model_file), "/tmp/custom.gguf")
            self.assertEqual(main(["--host", "0.0.0.0", "--port", "8765"]), 0)
            self.assertEqual(run.call_args.args[0].host, "0.0.0.0")
            self.assertEqual(run.call_args.args[0].port, 8765)

    def test_server_uses_requested_host_and_keeps_loopback_default(self):
        """서버 생성자가 지정된 호스트를 사용하며 기존 호출도 호환된다."""
        application = object()
        with patch("asr_corpus.web.ThreadingHTTPServer") as server_class:
            server = create_server(application, 8765, "0.0.0.0")
            server_class.assert_called_once_with(("0.0.0.0", 8765), DatasetRequestHandler)
            self.assertIs(server.application, application)
            self.assertEqual(server.bind_host, "0.0.0.0")
            server_class.reset_mock()
            create_server(application, 8765)
            server_class.assert_called_once_with(("127.0.0.1", 8765), DatasetRequestHandler)

    def test_wildcard_accepts_lan_ip_and_hostname_with_same_origin(self):
        """전체 바인딩에서는 접속한 IP나 호스트명과 같은 출처를 허용한다."""
        for host in ("192.168.0.10:8765", "my-mac.local:8765", "localhost:8765", "127.0.0.1:8765"):
            with self.subTest(host=host):
                self.handler("0.0.0.0", host, "http://" + host)._validate_origin()
                self.handler("0.0.0.0", host)._validate_origin()

    def test_wildcard_rejects_other_origins_and_invalid_hosts(self):
        """외부 바인딩에서도 다른 출처와 잘못된 호스트 및 포트를 거부한다."""
        for host, origin in (("192.168.0.10:8765", "http://another-site.example:8765"),
                             ("192.168.0.10:8765", "http://localhost:8765"),
                             ("192.168.0.10:8765", "null"),
                             ("192.168.0.10:9999", None), ("", None),
                             ("user@192.168.0.10:8765", None),
                             ("192.168.0.10:8765/path", None)):
            with self.subTest(host=host, origin=origin), self.assertRaises(ValueError):
                self.handler("0.0.0.0", host, origin)._validate_origin()

    def test_default_binding_rejects_lan_host(self):
        """기본 로컬 바인딩은 외부 주소의 요청을 계속 거부한다."""
        with self.assertRaises(ValueError):
            self.handler("127.0.0.1", "192.168.0.10:8765")._validate_origin()
        self.handler("127.0.0.1", "localhost:8765", "http://localhost:8765")._validate_origin()

    def test_explicit_binding_accepts_its_address(self):
        """특정 IP를 지정한 경우 해당 주소와 같은 출처를 허용한다."""
        self.handler("192.168.0.10", "192.168.0.10:8765", "http://192.168.0.10:8765")._validate_origin()

    def test_application_passes_host_to_server_and_reports_it(self):
        """실행 진입점이 호스트를 서버에 전달하고 같은 주소를 표시한다."""
        with patch("asr_corpus.web.run_application", return_value=0) as run:
            main(["--host", "0.0.0.0", "--port", "8765"])
            args = run.call_args.args[0]
        server = MagicMock()
        with patch("asr_corpus.web.DatasetJobService"), patch("asr_corpus.web.AudioJobService"), \
                patch("asr_corpus.web.create_server", return_value=server) as create, \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(run_application(args), 0)
            self.assertEqual(create.call_args.args[1:], (8765, "0.0.0.0"))
            server.__enter__.return_value.serve_forever.assert_called_once()
            self.assertIn("http://0.0.0.0:8765", output.getvalue())


if __name__ == "__main__":
    unittest.main()
