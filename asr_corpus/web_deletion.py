"""확인한 앱 생성 파일만 삭제하고 모델·설정·사용자 파일을 보호한다."""

import hashlib
import json
import re
import secrets
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from .web_dataset import ACTIVE_STATUSES


IDENTIFIER = re.compile(r"[0-9a-f]{32}")
WAV_NAME = re.compile(r"([0-9a-f]{32})-[0-9a-f]{16}\.wav")
MAPPING_NAME = re.compile(r"dataset-([0-9a-f]{8})-[0-9a-f]{12}")
JOB_FILES = {"job.json": 5, "job.json.tmp": 5, "audio.json": 4, "audio.json.tmp": 4, "requests.jsonl": 3}
MAPPING_FILES = {"transcripts.jsonl": 1, "train.jsonl": 1, "validation.jsonl": 1, "dataset_info.json": 2}
DEFAULT_OUTPUT_ROOT = "/tmp/ai-dataset/"


@dataclass(frozen=True)
class DeletionPreviewDto:
    token: str
    jobs: int
    examples: int
    wav_files: int
    datasets: int
    files: int
    bytes: int
    roots: list
    skipped: int


class GeneratedDataDeletionService:
    def __init__(self, dataset_service, audio_service=None):
        """데이터 소유 서비스와 일회용 삭제 확인 정보를 보관한다."""
        self.dataset_service = dataset_service
        self.audio_service = audio_service
        self._pending = None

    def _locks(self):
        """음성→텍스트 순서의 기존 잠금을 사용해 생성·삭제 경합을 막는다."""
        stack = ExitStack()
        if self.audio_service is not None:
            stack.enter_context(self.audio_service._lock)
        stack.enter_context(self.dataset_service._lock)
        return stack

    def _assert_idle(self):
        """현재 앱에서 텍스트나 음성을 생성 중이면 삭제를 거부한다."""
        if any(job["status"] in ACTIVE_STATUSES for job in self.dataset_service._jobs.values()):
            raise RuntimeError("텍스트 생성을 완료하거나 중지한 뒤 삭제하세요.")
        if self.audio_service is not None and self.audio_service.is_active():
            raise RuntimeError("음성 생성을 완료하거나 중지한 뒤 삭제하세요.")

    def preview(self):
        """실제 삭제 없이 작업과 생성 파일의 개수·경로를 확인한다."""
        with self._locks():
            self._assert_idle()
            plan = self._collect()
            token = secrets.token_urlsafe(32)
            self._pending = (token, self._fingerprint(plan), time.monotonic() + 300)
            result = DeletionPreviewDto(token, len(plan["jobs"]), plan["examples"], plan["wav_files"], plan["datasets"],
                                        len(plan["files"]), sum(item["identity"][2] for item in plan["files"].values()),
                                        sorted(plan["roots"]), len(plan["skipped"]))
            return asdict(result)

    def delete_all(self, payload):
        """직전에 확인한 동일 대상만 삭제하고 메모리 작업 목록도 정리한다."""
        if not isinstance(payload, dict) or payload.get("confirmed") is not True or not isinstance(payload.get("token"), str):
            raise ValueError("삭제 대상 확인 후 전체 삭제를 승인하세요.")
        with self._locks():
            self._assert_idle()
            pending = self._pending
            if pending is None or pending[0] != payload["token"] or time.monotonic() > pending[2]:
                raise RuntimeError("삭제 확인이 만료되었습니다. 삭제 대상을 다시 확인하세요.")
            plan = self._collect()
            if self._fingerprint(plan) != pending[1]:
                self._pending = None
                raise RuntimeError("삭제 대상이 변경되었습니다. 대상을 다시 확인하세요.")
            self._pending = None
            try:
                self._delete_files(plan)
            except OSError as error:
                self._forget_deleted_jobs(plan)
                raise RuntimeError("일부 생성 파일을 삭제하지 못했습니다. 남은 대상을 다시 확인하고 시도하세요.") from error
            self._forget_deleted_jobs(plan)
            self._remove_empty_directories(plan)
            removed = [job for job in plan["jobs"].values() if not (Path(job["directory"]) / "job.json").exists()]
            return {"deleted_jobs": len(removed), "deleted_examples": sum(job["example_count"] for job in removed),
                    "deleted_wav_files": plan["wav_files"], "deleted_datasets": sum(not (Path(name) / "dataset_info.json").exists() for name in plan["mapping_dirs"]),
                    "deleted_files": len(plan["files"]), "skipped": len(plan["skipped"])}

    def _identity(self, path):
        """확인한 파일이 교체되거나 수정되었는지 비교할 정보를 반환한다."""
        status = path.lstat()
        return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)

    def _fingerprint(self, plan):
        """대상 파일과 메타데이터 변경을 삭제 전에 다시 확인한다."""
        content = {"files": plan["files"], "jobs": sorted(plan["jobs"]), "skipped": sorted(plan["skipped"])}
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()

    def _read_json(self, path):
        """심볼릭 링크가 아닌 정상 JSON 객체만 읽는다."""
        if not path.is_file() or path.is_symlink():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _protected(self, path):
        """설정 파일과 실제 모델 경로는 파일명과 무관하게 보호한다."""
        protected = [self.dataset_service.model_config.model_file.resolve(),
                     (self.dataset_service.data_dir / "preferences.json").resolve()]
        if self.audio_service is not None:
            protected.append(self.audio_service.config.model_dir.resolve())
        return any(path == value or path.is_relative_to(value) for value in protected)

    def _add_file(self, plan, path, root, priority, wav=False):
        """소유 경로 안의 일반 파일만 확인 목록에 넣고 링크·모델은 제외한다."""
        if not path.exists() and not path.is_symlink():
            return
        resolved = path.resolve()
        if resolved != path or not resolved.is_relative_to(root) or not path.is_file() or self._protected(resolved):
            plan["skipped"].add(str(path))
            return
        if str(path) not in plan["files"]:
            plan["files"][str(path)] = {"identity": self._identity(path), "root": str(root), "priority": priority}
            plan["wav_files"] += int(wav)
        plan["directories"].add(str(path.parent))

    def _add_root(self, roots, value):
        """기록에 있는 절대 출력 경로만 탐색 대상으로 사용한다."""
        if isinstance(value, str) and value.strip() and Path(value).expanduser().is_absolute():
            roots.add(Path(value).expanduser().resolve())

    def _audio_roots(self, roots, audio):
        """현재·이전 음성과 매핑의 출력 루트를 함께 복원한다."""
        self._add_root(roots, audio.get("output_root"))
        for value in audio.get("output_roots", []) if isinstance(audio.get("output_roots"), list) else []:
            self._add_root(roots, value)
        records = audio.get("examples", {})
        for record in records.values() if isinstance(records, dict) else []:
            if isinstance(record, dict):
                self._add_root(roots, record.get("output_root"))
        mappings = audio.get("mappings", []) if isinstance(audio.get("mappings"), list) else []
        for record in [audio.get("last_mapping"), *mappings]:
            self._mapping_root(roots, record)

    def _mapping_root(self, roots, record):
        """과거 형식의 매핑 기록에서도 출력 폴더의 부모 경로를 찾는다."""
        if not isinstance(record, dict):
            return
        self._add_root(roots, record.get("output_root"))
        value = record.get("output_dir")
        if isinstance(value, str) and Path(value).is_absolute():
            self._add_root(roots, str(Path(value).parent))

    def _collect(self):
        """화면의 최근 100개 제한과 관계없이 저장된 모든 정상 작업을 조사한다."""
        plan = {"jobs": {}, "examples": 0, "wav_files": 0, "datasets": 0, "mapping_dirs": set(), "files": {}, "roots": set(), "directories": set(), "skipped": set()}
        root = self.dataset_service.data_dir.resolve()
        roots = set()
        self._add_root(roots, DEFAULT_OUTPUT_ROOT)
        self._add_root(roots, self.dataset_service.get_preferences()["output_root"])
        for directory in sorted(root.iterdir()) if root.is_dir() else []:
            self._collect_job(plan, directory, root, roots)
        for output_root in sorted(roots):
            self._collect_output_root(plan, output_root)
        return plan

    def _collect_job(self, plan, directory, root, roots):
        """작업 ID와 저장 내용이 일치하는 폴더의 앱 생성 파일만 수집한다."""
        if not IDENTIFIER.fullmatch(directory.name):
            return
        if directory.is_symlink() or not directory.is_dir():
            plan["skipped"].add(str(directory))
            return
        job = self._read_json(directory / "job.json")
        if job.get("id") != directory.name or not isinstance(job.get("examples"), list):
            plan["skipped"].add(str(directory))
            return
        identifiers = {row["id"] for row in job["examples"] if isinstance(row, dict) and isinstance(row.get("id"), str) and IDENTIFIER.fullmatch(row["id"])}
        plan["jobs"][directory.name] = {"directory": str(directory), "example_ids": identifiers, "example_count": len(job["examples"])}
        plan["examples"] += len(job["examples"])
        plan["roots"].add(str(root))
        settings = job.get("settings", {})
        self._add_root(roots, settings.get("output_root") if isinstance(settings, dict) else None)
        self._audio_roots(roots, self._read_json(directory / "audio.json"))
        for name, priority in JOB_FILES.items():
            self._add_file(plan, directory / name, root, priority)
        for path in directory.glob("qwen-asr-*.zip"):
            self._add_file(plan, path, root, 0)
        self._collect_wavs(plan, directory / "audio", directory, identifiers)
        plan["directories"].add(str(directory))

    def _collect_wavs(self, plan, directory, root, identifiers):
        """해당 작업의 고유 ID와 버전 형식에 맞는 WAV만 수집한다."""
        if directory.is_symlink() or directory.resolve() != directory:
            plan["skipped"].add(str(directory))
            return
        if not directory.is_dir():
            return
        for path in directory.iterdir():
            match = WAV_NAME.fullmatch(path.name)
            if match and match.group(1) in identifiers:
                self._add_file(plan, path, root, 0, wav=True)
        plan["directories"].add(str(directory))

    def _collect_output_root(self, plan, root):
        """등록된 루트의 작업별 WAV와 출처가 확인된 매핑 결과를 찾는다."""
        if not root.is_dir():
            return
        before = len(plan["files"])
        for identifier, job in plan["jobs"].items():
            self._collect_wavs(plan, root / "wav" / identifier, root, job["example_ids"])
        for directory in root.glob("dataset-*"):
            self._collect_mapping(plan, directory, root)
        if len(plan["files"]) > before:
            plan["roots"].add(str(root))

    def _collect_mapping(self, plan, directory, root):
        """폴더명과 메타데이터가 해당 작업을 가리키는 매핑만 수집한다."""
        match = MAPPING_NAME.fullmatch(directory.name)
        if not match or directory.is_symlink() or not directory.is_dir():
            return
        information = self._read_json(directory / "dataset_info.json")
        identifier = information.get("source_job_id")
        if not isinstance(identifier, str) or identifier not in plan["jobs"] or match.group(1) != identifier[:8]:
            return
        for example_id in plan["jobs"][identifier]["example_ids"]:
            self._add_file(plan, directory / "audio" / f"{example_id}.wav", root, 0, wav=True)
        for name, priority in MAPPING_FILES.items():
            self._add_file(plan, directory / name, root, priority)
        plan["datasets"] += 1
        plan["mapping_dirs"].add(str(directory))
        plan["directories"].update({str(directory / "audio"), str(directory)})

    def _delete_files(self, plan):
        """생성 파일부터 삭제하고 복구·재확인에 필요한 작업 메타데이터는 마지막에 삭제한다."""
        targets = sorted(plan["files"].items(), key=lambda item: (item[1]["priority"], item[0]))
        for name, item in targets:
            path = Path(name)
            if path.resolve() != path or self._identity(path) != item["identity"] or self._protected(path):
                raise OSError("확인한 파일이 변경되었습니다.")
            path.unlink()

    def _forget_deleted_jobs(self, plan):
        """실제로 작업 기록이 사라진 항목만 메모리에서도 제거한다."""
        for identifier, job in plan["jobs"].items():
            if (Path(job["directory"]) / "job.json").exists():
                continue
            self.dataset_service._jobs.pop(identifier, None)
            self.dataset_service._cancellations.pop(identifier, None)
            self.dataset_service._threads.pop(identifier, None)
            if self.audio_service is not None:
                self.audio_service._states.pop(identifier, None)

    def _remove_empty_directories(self, plan):
        """사용자 파일이 남은 폴더와 저장 루트는 유지하고 빈 작업 폴더만 정리한다."""
        for name in sorted(plan["directories"], key=lambda value: len(Path(value).parts), reverse=True):
            path = Path(name)
            if path.is_symlink() or path.resolve() != path or not path.is_dir() or self._protected(path):
                continue
            try:
                path.rmdir()
            except OSError:
                continue
