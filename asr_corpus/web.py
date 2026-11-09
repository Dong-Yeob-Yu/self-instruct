"""로컬 모델과 데이터 작업 화면을 연결하는 표준 라이브러리 웹 앱."""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .local import LlamaCppConfig
from .tts import DEFAULT_TTS_DIR, MagpieTtsConfig
from .mlx_tts import MlxTtsConfig, MlxTtsClient, DEFAULT_MLX_TTS_DIR, DEFAULT_MLX_CODEC_DIR, DEFAULT_MLX_PYTHON
from .web_audio import AudioJobService
from .web_dataset import DatasetJobService
from .web_deletion import GeneratedDataDeletionService


STATIC_DIR = Path(__file__).with_name("web_static")
JOB_ROUTE = re.compile(r"^/api/jobs/([0-9a-f]{32})(?:/(cancel|export|selected-export|selection|examples|examples/([0-9a-f]{32})))?$")
AUDIO_ROUTE = re.compile(r"^/api/jobs/([0-9a-f]{32})/(audio(?:/cancel)?|mapping|training-export|examples/([0-9a-f]{32})/audio(?:-review)?)$")


@dataclass(frozen=True)
class WebResponseDto:
    status: int
    body: bytes
    content_type: str = "application/json; charset=utf-8"
    filename: str = ""
    file_path: Path = None
    temporary: bool = False


def json_response(status, payload):
    """웹 응답을 UTF-8 JSON으로 직렬화한다."""
    return WebResponseDto(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))


class DatasetWebApplication:
    def __init__(self, service, audio_service=None):
        """서버를 시작하지 않고 요청 처리기와 생성 서비스를 연결한다."""
        self.service = service
        self.audio_service = audio_service
        self.deletion_service = GeneratedDataDeletionService(service, audio_service)
        self.service.on_generation_completed = self._start_automatic_audio

    def _snapshot(self, job):
        """음성 서비스가 연결된 경우 최신 WAV 상태를 함께 반환한다."""
        result = self.audio_service.decorate_job(job) if self.audio_service else dict(job)
        result["model_runtime"] = self.service.model_runtime.snapshot()
        return result

    def _start_automatic_audio(self, job):
        """LLM 종료 후 채택된 한글 발화 전체를 한 번의 TTS 작업으로 전달한다."""
        if self.audio_service is None:
            raise RuntimeError("음성 생성 서비스가 연결되지 않았습니다.")
        identifiers = [row["id"] for row in job["examples"] if row["accepted"] and not row["needs_review"] and not row.get("quality_issue")]
        self.audio_service.start(job["id"], {"example_ids": identifiers, "output_root": job["settings"]["output_root"]})

    def dispatch(self, method, path, payload=None):
        """네트워크 없이도 검증할 수 있도록 HTTP 라우팅을 분리한다."""
        try:
            return self._route(method, path, payload)
        except KeyError:
            return json_response(404, {"error": "요청한 작업이나 예제를 찾지 못했습니다."})
        except ValueError as error:
            return json_response(400, {"error": str(error)})
        except RuntimeError as error:
            return json_response(409, {"error": str(error)})
        except OSError:
            return json_response(500, {"error": "로컬 파일을 읽거나 저장하지 못했습니다."})

    def _route(self, method, path, payload):
        """정적 화면과 작업 API를 명시적으로 허용한 경로에서만 제공한다."""
        if method == "GET" and path in {"/", "/app.js", "/styles.css"}:
            return self._static_response(path)
        if method == "GET" and path == "/api/config":
            audio = self.audio_service.availability() if self.audio_service else {"audio_available": False}
            return json_response(200, {"model_name": self.service.model_config.model_file.stem, "model_file": self.service.model_config.model_file.name,
                                       "mapping_available": self.audio_service is not None, **audio})
        if method == "GET" and path == "/api/jobs":
            return json_response(200, {"jobs": self.service.list_jobs()})
        if method == "POST" and path == "/api/data/deletion-preview":
            return json_response(200, self.deletion_service.preview())
        if method == "POST" and path == "/api/data/delete-all":
            return json_response(200, self.deletion_service.delete_all(payload))
        if method == "GET" and path == "/api/preferences":
            return json_response(200, self.service.get_preferences())
        if method == "PATCH" and path == "/api/preferences":
            return json_response(200, self.service.save_preferences(payload))
        if method == "POST" and path == "/api/jobs/manual":
            self._assert_manual_audio_idle()
            return json_response(201, self._snapshot(self.service.add_manual_examples(payload)))
        if method == "POST" and path == "/api/jobs":
            if isinstance(payload, dict) and payload.get("auto_audio") is True:
                if self.audio_service is None:
                    raise ValueError("자동 음성 생성에 필요한 TTS 서비스가 없습니다.")
                self.audio_service.config.validate()
            if self.audio_service and self.audio_service.is_active():
                raise RuntimeError("음성 생성이 끝난 뒤 텍스트 생성을 시작하세요.")
            return json_response(202, self._snapshot(self.service.start_job(payload)))
        audio_match = AUDIO_ROUTE.fullmatch(path)
        if audio_match and self.audio_service:
            return self._audio_route(method, audio_match, payload)
        match = JOB_ROUTE.fullmatch(path)
        if match:
            return self._job_route(method, match, payload)
        return json_response(404, {"error": "존재하지 않는 경로입니다."})

    def _assert_manual_audio_idle(self):
        """음성 작업 중 직접 입력이 목록에 끼어들지 않도록 확인한다."""
        if self.audio_service and self.audio_service.is_active():
            raise RuntimeError("음성 생성이 끝난 뒤 직접 입력 발화를 추가하세요.")

    def _job_route(self, method, match, payload):
        """작업 조회·중지·검토·내보내기를 처리한다."""
        job_id, action, example_id = match.groups()
        if method == "GET" and action is None:
            return json_response(200, self._snapshot(self.service.get_job(job_id)))
        if method == "POST" and action == "examples" and example_id is None:
            self._assert_manual_audio_idle()
            return json_response(201, self._snapshot(self.service.add_manual_examples(payload, job_id)))
        if method == "POST" and action == "cancel":
            return json_response(200, self.service.cancel_job(job_id))
        if method == "PATCH" and example_id:
            if self.audio_service and self.audio_service.is_active(job_id):
                raise RuntimeError("음성 생성을 완료하거나 중지한 뒤 발화·전사문을 수정하세요.")
            return json_response(200, self._snapshot(self.service.update_example(job_id, example_id, payload)))
        if method == "PATCH" and action == "selection":
            if self.audio_service and self.audio_service.is_active(job_id):
                raise RuntimeError("음성 생성이 끝난 뒤 텍스트 선택을 변경하세요.")
            return json_response(200, self._snapshot(self.service.update_selection(job_id, payload)))
        if method == "GET" and action in {"export", "selected-export"}:
            return WebResponseDto(200, self.service.export_jsonl(job_id, reviewed_only=action == "selected-export").encode("utf-8"), "application/x-ndjson; charset=utf-8", f"dataset-{job_id[:8]}.jsonl")
        return json_response(405, {"error": "허용하지 않는 요청 방식입니다."})

    def _audio_route(self, method, match, payload):
        """음성 생성·중지·재생과 Qwen ASR 학습 자료 다운로드를 처리한다."""
        job_id, action, example_id = match.groups()
        if method == "POST" and action == "audio":
            return json_response(202, self.audio_service.start(job_id, payload))
        if method == "POST" and action == "mapping":
            return json_response(201, self.audio_service.create_mapping(job_id, payload))
        if method == "POST" and action == "audio/cancel":
            return json_response(200, self.audio_service.cancel(job_id))
        if method == "PATCH" and example_id and action.endswith("audio-review"):
            return json_response(200, self.audio_service.review_audio(job_id, example_id, payload))
        if method == "GET" and example_id and action.endswith("/audio"):
            return WebResponseDto(200, b"", "audio/wav", file_path=self.audio_service.audio_file(job_id, example_id))
        if method == "GET" and action == "training-export":
            path = self.audio_service.export_training(job_id)
            return WebResponseDto(200, b"", "application/zip", f"qwen-asr-{job_id[:8]}.zip", path, True)
        return json_response(405, {"error": "허용하지 않는 요청 방식입니다."})

    def _static_response(self, path):
        """알려진 정적 파일만 읽어 임의의 로컬 파일 접근을 방지한다."""
        files = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/styles.css": ("styles.css", "text/css")}
        filename, content_type = files[path]
        return WebResponseDto(200, (STATIC_DIR / filename).read_bytes(), content_type + "; charset=utf-8")


class DatasetRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """화면과 읽기 API 요청을 처리한다."""
        self._handle_request("GET")

    def do_POST(self):
        """생성과 중지 요청을 처리한다."""
        self._handle_request("POST")

    def do_PATCH(self):
        """사용자 검토 결과를 저장한다."""
        self._handle_request("PATCH")

    def _handle_request(self, method):
        """로컬 요청을 검증하고 응답을 안전한 공통 헤더와 함께 보낸다."""
        try:
            self._validate_origin()
            payload = self._read_payload() if method in {"POST", "PATCH"} else None
            response = self.server.application.dispatch(method, urlsplit(self.path).path, payload)
        except ValueError as error:
            response = json_response(400, {"error": str(error)})
        self._write_response(response)

    def _validate_origin(self):
        """바인딩 주소에 맞는 요청과 같은 출처의 브라우저 접근만 허용한다."""
        bind_host = getattr(self.server, "bind_host", self.server.server_address[0])
        port = self.server.server_address[1]
        request_host = self.headers.get("Host", "")
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"{bind_host}:{port}"}
        if bind_host == "0.0.0.0":
            hosts = {request_host} if re.fullmatch(rf"[A-Za-z0-9][A-Za-z0-9.-]*:{port}", request_host) else set()
        origins = {"http://" + host for host in hosts}
        if request_host not in hosts or self.headers.get("Origin") not in origins | {None}:
            raise ValueError("이 앱의 접속 주소와 같은 출처에서만 사용할 수 있습니다.")

    def _read_payload(self):
        """길이 제한이 있는 JSON 요청만 읽는다."""
        if self.headers.get_content_type() != "application/json":
            raise ValueError("application/json 요청이 필요합니다.")
        length = int(self.headers.get("Content-Length", "0"))
        path = urlsplit(getattr(self, "path", "")).path
        manual_upload = path == "/api/jobs/manual" or bool(re.fullmatch(r"/api/jobs/[0-9a-f]{32}/examples", path))
        maximum = 8 * 1024 * 1024 if manual_upload else 65536
        if not 0 < length <= maximum:
            raise ValueError(f"요청 크기는 1~{maximum:,}바이트여야 합니다.")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _write_response(self, response):
        """콘텐츠 보안 정책과 다운로드 파일명을 지정해 응답한다."""
        size = response.file_path.stat().st_size if response.file_path else len(response.body)
        status, offset, length = self._response_range(response, size)
        self.send_response(status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(length))
        if response.content_type == "audio/wav":
            self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {offset}-{offset + length - 1}/{size}")
        if status == 416:
            self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; media-src 'self'; frame-ancestors 'none'")
        if response.filename:
            self.send_header("Content-Disposition", f'attachment; filename="{response.filename}"')
        self.end_headers()
        try:
            if length:
                self._write_body(response, offset, length)
        except (BrokenPipeError, ConnectionResetError):
            return
        finally:
            if response.temporary and response.file_path:
                response.file_path.unlink(missing_ok=True)

    def _response_range(self, response, size):
        """브라우저 오디오 탐색에 필요한 단일 바이트 범위를 검증한다."""
        requested = getattr(self, "headers", {}).get("Range")
        if response.content_type != "audio/wav" or not response.file_path or requested is None:
            return response.status, 0, size
        match = re.fullmatch(r"bytes=([0-9]{0,20})-([0-9]{0,20})", requested)
        if not match or not any(match.groups()):
            return 416, 0, 0
        first, last = match.groups()
        offset = int(first) if first else max(0, size - int(last))
        end = min(size - 1, int(last)) if first and last else size - 1
        if offset > end or offset >= size or (not first and int(last) == 0):
            return 416, 0, 0
        return 206, offset, end - offset + 1

    def _write_body(self, response, offset, length):
        """WAV·ZIP 파일을 제한된 크기의 버퍼로 범위에 맞춰 전송한다."""
        if response.file_path:
            with response.file_path.open("rb") as handle:
                handle.seek(offset)
                self._copy_file_range(handle, length)
            return
        self.wfile.write(response.body)

    def _copy_file_range(self, source, length):
        """전체 파일을 메모리에 올리지 않고 요청한 길이만 전송한다."""
        remaining = length
        while remaining > 0:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                return
            self.wfile.write(chunk)
            remaining -= len(chunk)

    def log_message(self, format_string, *args):
        """프롬프트나 예제 본문을 로그에 남기지 않고 요청 상태만 출력한다."""
        sys.stderr.write(f"[{self.log_date_time_string()}] {format_string % args}\n")


def create_server(application, port, host="127.0.0.1"):
    """명시적으로 실행할 때 지정한 주소에 웹 서버를 바인딩한다."""
    server = ThreadingHTTPServer((host, port), DatasetRequestHandler)
    server.application = application
    server.bind_host = host
    return server


def main(argv=None):
    """로컬 모델 경로와 저장 위치를 받아 웹 앱을 실행한다."""
    parser = argparse.ArgumentParser(description="발화 데이터 스튜디오 · 로컬 모델 웹 앱")
    parser.add_argument("--host", default="127.0.0.1", help="서버 바인딩 주소 (전체 인터페이스: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-file", type=Path, default=Path.home() / "Desktop/personal/dev/ai-models/gemma4-26b-a4b-q4/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
    parser.add_argument("--llama-executable", default="llama-completion")
    parser.add_argument("--data-dir", type=Path, default=Path("data/asr_web"))
    parser.add_argument("--tts-backend", choices=("magpie", "mlx"), default="mlx")
    parser.add_argument("--tts-codec-dir", type=Path, default=DEFAULT_MLX_CODEC_DIR)
    parser.add_argument("--tts-python", default=str(DEFAULT_MLX_PYTHON))
    parser.add_argument("--tts-model-dir", type=Path)
    parser.add_argument("--tts-memory-limit-gb", type=int, default=20)
    parser.add_argument("--tts-executable", default=str(Path.home() / ".local/bin/nemo-speech"))
    parser.add_argument("--tts-speaker", type=int, choices=range(5), default=0)
    parser.add_argument("--tts-device", choices=("metal", "cpu", "auto"), default="metal")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("포트는 1~65535 사이여야 합니다.")
    return run_application(args)


def create_audio_service(service, args):
    """선택한 백엔드의 로컬 설정과 클라이언트를 연결한다."""
    if args.tts_backend == "mlx":
        if args.tts_codec_dir is None:
            raise ValueError("MLX TTS에는 --tts-codec-dir를 지정하세요.")
        model_dir = args.tts_model_dir or DEFAULT_MLX_TTS_DIR
        config = MlxTtsConfig(model_dir.expanduser().resolve(), args.tts_codec_dir.expanduser().resolve(), args.tts_python,
                              memory_limit_gb=args.tts_memory_limit_gb)
        return AudioJobService(service, config, MlxTtsClient)
    model_dir = args.tts_model_dir or DEFAULT_TTS_DIR
    config = MagpieTtsConfig(model_dir.expanduser().resolve(), args.tts_executable, args.tts_speaker, args.tts_device)
    return AudioJobService(service, config)


def run_application(args):
    """앱 종료 시 생성 프로세스를 정리하고 기존 결과는 보존한다."""
    service = None
    audio_service = None
    try:
        config = LlamaCppConfig(args.model_file.expanduser().resolve(), executable=args.llama_executable, context_size=8192)
        service = DatasetJobService(config, args.data_dir.expanduser().resolve())
        audio_service = create_audio_service(service, args)
        with create_server(DatasetWebApplication(service, audio_service), args.port, args.host) as server:
            print(f"발화 데이터 스튜디오: http://{args.host}:{args.port}", flush=True)
            server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n웹 앱을 종료합니다.", flush=True)
    except (OSError, ValueError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    finally:
        if service is not None:
            service.close()
        if audio_service is not None:
            audio_service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
