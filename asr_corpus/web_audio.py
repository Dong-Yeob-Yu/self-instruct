"""채택된 발화의 로컬 WAV 생성·복구·Qwen ASR 내보내기를 관리한다."""

import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path


from .examples import normalize_text
from .tts import AudioCancelledError, MagpieTtsClient, inspect_wav
from .web_dataset import ACTIVE_STATUSES, current_timestamp


def eligible_examples(job):
    """검토를 마치고 채택한 예제만 음성·학습 데이터 대상으로 선택한다."""
    return [row for row in job["examples"] if row["accepted"] and not row["needs_review"] and not row.get("quality_issue")]


class AudioJobService:
    def __init__(self, dataset_service, config, client_factory=MagpieTtsClient):
        """기존 텍스트 작업과 독립된 음성 진행 상태를 연결한다."""
        self.dataset_service = dataset_service
        self.config = config
        self.client_factory = client_factory
        self._lock = threading.RLock()
        self._states = {}
        self._thread = None
        self._cancellation = threading.Event()

    def availability(self):
        """로컬 실행 준비 여부와 조치할 오류를 화면에 전달한다."""
        try:
            self.config.validate()
            return {"audio_available": True, "audio_error": "", "tts_model": getattr(self.config, "display_name", "NVIDIA Magpie TTS · F16"), "training_model": "Qwen ASR"}
        except ValueError as error:
            return {"audio_available": False, "audio_error": str(error), "tts_model": getattr(self.config, "display_name", "NVIDIA Magpie TTS · F16"), "training_model": "Qwen ASR"}

    def _state(self, job_id):
        """저장된 음성 상태를 복원하고 끊긴 생성은 재시도 가능하게 표시한다."""
        if job_id not in self._states:
            self._states[job_id] = self._load_state(job_id)
        return self._states[job_id]

    def _load_state(self, job_id):
        """부분·손상된 상태 파일이 텍스트 데이터 조회를 막지 않게 한다."""
        path = self.dataset_service.data_dir / job_id / "audio.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or not isinstance(state.get("examples"), dict):
                raise ValueError("잘못된 음성 상태")
            state["examples"] = {identifier: record for identifier, record in state["examples"].items() if isinstance(record, dict)}
            if state.get("status") in ACTIVE_STATUSES:
                state.update(status="interrupted", message="앱 종료로 음성 생성이 중단되었습니다. 다시 생성하면 이어서 처리합니다.")
            return state
        except (OSError, ValueError):
            return {"status": "idle", "message": "채택한 발화문을 WAV로 생성하세요.", "examples": {}}

    def is_active(self, job_id=None):
        """전체 또는 특정 작업의 음성 생성 중 여부를 확인한다."""
        with self._lock:
            return any(state.get("status") in ACTIVE_STATUSES for identifier, state in self._states.items()
                       if job_id is None or job_id == identifier)

    def decorate_job(self, job):
        """텍스트 원본을 변경하지 않고 최신 발화에 대응하는 음성 상태를 붙인다."""
        result = copy.deepcopy(job)
        with self._lock:
            state = self._state(job["id"])
            for row in result["examples"]:
                row["audio"] = self._row_audio(job["id"], row, state)
            selected = eligible_examples(result)
            counts = {"total": len(selected), "ready": sum(row["audio"]["status"] == "ready" for row in selected),
                      "failed": sum(row["audio"]["status"] == "failed" for row in selected)}
            result["audio"] = {"status": state.get("status", "idle"), "message": state.get("message", ""), "counts": counts}
            result["audio"]["last_mapping"] = copy.deepcopy(state.get("last_mapping"))
        return result

    def _row_audio(self, job_id, row, state):
        """현재 발화와 생성 설정이 달라지거나 파일이 없으면 이전 WAV를 숨긴다."""
        record = state["examples"].get(row["id"], {})
        fingerprint = self.config.fingerprint(row["spoken_text"])
        if record.get("fingerprint") != fingerprint:
            return {"status": "outdated" if record else "missing"}
        status = record.get("status", "missing")
        if status == "ready" and not self._path(job_id, row["id"], fingerprint, record.get("output_root")).is_file():
            return {"status": "missing"}
        if status == "generating" and state.get("status") not in ACTIVE_STATUSES:
            return {"status": "interrupted"}
        return {**record, "url": f"/api/jobs/{job_id}/examples/{row['id']}/audio" if status == "ready" else ""}

    def _path(self, job_id, example_id, fingerprint, output_root=None):
        """저장 기록의 경로 대신 검증한 고유 ID로 WAV 경로를 만든다."""
        if not re.fullmatch(r"[0-9a-f]{32}", job_id) or not re.fullmatch(r"[0-9a-f]{32}", example_id) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("올바르지 않은 음성 식별자입니다.")
        root = self._output_root(output_root) if output_root else (self.dataset_service.data_dir / job_id).resolve()
        directory = root / "wav" / job_id if output_root else root / "audio"
        path = directory / f"{example_id}-{fingerprint[:16]}.wav"
        if not path.resolve().is_relative_to(root):
            raise ValueError("작업 폴더 내부의 음성만 사용할 수 있습니다.")
        return path

    def _output_root(self, value):
        """사용자가 지정한 절대 저장 루트를 검증한다."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("저장 루트 경로를 입력하세요.")
        path = Path(value.strip()).expanduser()
        if not path.is_absolute():
            raise ValueError("저장 루트는 절대 경로로 입력하세요.")
        return path.resolve()

    def _selected_rows(self, job, payload, ready=False):
        """명시적으로 선택한 유효한 채택 항목만 처리 대상으로 확정한다."""
        if not isinstance(payload, dict):
            raise ValueError("선택 항목과 저장 경로를 JSON 객체로 전달하세요.")
        identifiers = payload.get("example_ids")
        if not isinstance(identifiers, list) or not identifiers or any(not isinstance(value, str) for value in identifiers):
            raise ValueError("처리할 항목을 하나 이상 선택하세요.")
        allowed = {row["id"]: row for row in eligible_examples(job)}
        if len(set(identifiers)) != len(identifiers) or any(value not in allowed for value in identifiers):
            raise ValueError("검토를 마친 채택 항목만 중복 없이 선택하세요.")
        rows = [allowed[value] for value in identifiers]
        if ready and any(row["audio"]["status"] != "ready" for row in rows):
            raise ValueError("현재 발화에 맞는 WAV가 생성된 항목만 매핑할 수 있습니다.")
        if ready and payload.get("include_all_ready") is not True and any(row["audio"].get("included") is False for row in rows):
            raise ValueError("음성 검토에서 제외한 항목은 저장할 수 없습니다.")
        return rows

    def start(self, job_id, payload):
        """선택한 채택 예제의 WAV를 지정한 루트에 생성한다."""
        self.config.validate()
        with self._lock:
            job = self.dataset_service.get_job(job_id)
            if self.is_active() or any(item["status"] in ACTIVE_STATUSES for item in self.dataset_service.list_jobs()):
                raise RuntimeError("진행 중인 텍스트·음성 생성이 끝난 뒤 시작하세요.")
            rows = self._selected_rows(job, payload)
            output_root = str(self._output_root(payload.get("output_root", "/tmp/ai-dataset/")))
            state = self._state(job_id)
            pending = [row for row in rows if self._row_audio(job_id, row, state)["status"] != "ready"
                       or state["examples"].get(row["id"], {}).get("output_root") != output_root]
            if not pending:
                return self.decorate_job(job)
            state.update(status="queued", output_root=output_root, message=f"선택한 {len(pending)}개 발화의 음성 생성을 준비합니다.")
            history = state.get("output_roots", [])
            state["output_roots"] = list(dict.fromkeys([*(history if isinstance(history, list) else []), output_root]))
            self._save(job_id)
            self._cancellation = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(job_id, pending, self._cancellation, output_root), daemon=True)
            self._thread.start()
            return self.decorate_job(job)

    def cancel(self, job_id):
        """현재 음성 생성만 중지하고 이미 완료한 WAV는 유지한다."""
        with self._lock:
            job = self.dataset_service.get_job(job_id)
            if self.is_active(job_id):
                self._cancellation.set()
                self._state(job_id).update(status="cancelling", message="음성 생성을 중지하고 완료한 WAV를 보존합니다.")
                self._save(job_id)
            return self.decorate_job(job)

    def _run(self, job_id, rows, cancellation, output_root):
        """항목별 오류를 기록하고 성공한 음성을 보존하며 나머지를 진행한다."""
        try:
            with self.dataset_service.model_runtime.use("tts", cancellation, AudioCancelledError):
                self._generate_all_rows(job_id, rows, cancellation, output_root)
            self._finish(job_id, "completed", "음성 생성이 끝났습니다. 각 발화를 재생해 확인하세요. 실패 항목은 다시 생성할 수 있습니다.")
        except AudioCancelledError:
            self._finish(job_id, "cancelled", "음성 생성을 중지했습니다. 완료한 WAV는 저장되었습니다.")
        except Exception as error:
            self._finish(job_id, "failed", f"음성 생성 실패: {error}")

    def _generate_all_rows(self, job_id, rows, cancellation, output_root):
        """하나의 TTS 클라이언트를 재사용하고 성공·취소·오류 모두 종료 후 반환한다."""
        client = self.client_factory(self.config, cancellation)
        try:
            self._generate_with_client(client, job_id, rows, cancellation, output_root)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _generate_with_client(self, client, job_id, rows, cancellation, output_root):
        """배치 지원 여부에 따라 선택한 발화를 묶거나 개별 처리한다."""
        if callable(getattr(type(client), "generate_batch", None)):
            self._generate_batch_audio(client, job_id, rows, output_root)
            return
        for index, row in enumerate(rows, 1):
            self._generate_row(client, job_id, row, index, len(rows), cancellation, output_root)

    def _generate_batch_audio(self, client, job_id, rows, output_root):
        """대형 BF16 본체와 코덱을 순차 로딩하는 배치에 전체 한글 발화를 전달한다."""
        fingerprints = [self.config.fingerprint(row["spoken_text"]) for row in rows]
        requests = [(row["spoken_text"], self._path(job_id, row["id"], fingerprint, output_root)) for row, fingerprint in zip(rows, fingerprints)]
        with self._lock:
            state = self._state(job_id)
            state["examples"].update({row["id"]: {"status": "generating", "phase": "audio_codes", "fingerprint": fingerprint}
                                      for row, fingerprint in zip(rows, fingerprints)})
            state.update(status="running", message="BF16 TTS 본체 로딩 · 전체 발화의 음성 코드를 먼저 생성합니다.")
            self._save(job_id)

        def on_event(index, phase, metrics):
            """음성 코드와 WAV 단계의 진행 및 검증 결과를 저장한다."""
            row, fingerprint = rows[index], fingerprints[index]
            if phase == "encoded":
                record = {"status": "generating", "phase": "codes_ready", "fingerprint": fingerprint}
                message = f"TTS 음성 코드 {index + 1}/{len(rows)}개 생성 · 전체 코드 생성 후 본체를 내리고 WAV로 변환합니다."
            else:
                record = {"status": "ready", "fingerprint": fingerprint, "output_root": output_root, **inspect_wav(requests[index][1]),
                          "peak_memory_gib": metrics["peak_memory_gib"]}
                message = f"TTS 본체 종료 · 코덱으로 WAV {index + 1}/{len(rows)}개 변환했습니다."
            self._set_record(job_id, row["id"], record, message)

        try:
            client.generate_batch(requests, on_event)
        except AudioCancelledError:
            raise
        except Exception as error:
            self._mark_batch_audio_failed(job_id, rows, fingerprints, str(error))
            raise

    def _mark_batch_audio_failed(self, job_id, rows, fingerprints, error):
        """성공한 WAV는 보존하고 배치 오류로 미변환된 항목만 실패로 기록한다."""
        with self._lock:
            state = self._state(job_id)
            failed = {row["id"]: {"status": "failed", "fingerprint": fingerprint, "error": error[:1800]}
                      for row, fingerprint in zip(rows, fingerprints)
                      if state["examples"].get(row["id"], {}).get("status") != "ready"}
            state["examples"].update(failed)
            self._save(job_id)

    def _generate_row(self, client, job_id, row, index, total, cancellation, output_root):
        """하나의 발화 결과를 검증한 뒤 해당 발화 버전의 음성 상태로 기록한다."""
        if cancellation.is_set():
            raise AudioCancelledError()
        fingerprint = self.config.fingerprint(row["spoken_text"])
        self._set_record(job_id, row["id"], {"status": "generating", "fingerprint": fingerprint}, f"음성 {index}/{total}개 생성 중입니다.")
        try:
            destination = self._path(job_id, row["id"], fingerprint, output_root)
            metrics = client.generate(row["spoken_text"], destination)
            record = {"status": "ready", "fingerprint": fingerprint, "output_root": output_root, **inspect_wav(destination)}
            if isinstance(metrics, dict) and type(metrics.get("peak_memory_gib")) in {int, float}:
                record["peak_memory_gib"] = metrics["peak_memory_gib"]
        except AudioCancelledError:
            raise
        except Exception as error:
            record = {"status": "failed", "fingerprint": fingerprint, "error": str(error)}
        self._set_record(job_id, row["id"], record, f"음성 {index}/{total}개 처리했습니다.")

    def _set_record(self, job_id, example_id, record, message):
        """한 항목의 상태를 원자적으로 저장해 앱 재실행 후 이어받게 한다."""
        with self._lock:
            state = self._state(job_id)
            state["examples"][example_id] = record
            state.update(status="cancelling" if self._cancellation.is_set() else "running", message=message)
            self._save(job_id)

    def _finish(self, job_id, status, message):
        """음성 작업의 최종 상태를 기록한다."""
        with self._lock:
            self._state(job_id).update(status=status, message=message)
            self._save(job_id)

    def _save(self, job_id):
        """완전한 JSON만 교체하여 기존 음성 메타데이터를 보존한다."""
        state = self._state(job_id)
        state["updated_at"] = current_timestamp()
        destination = self.dataset_service.data_dir / job_id / "audio.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def audio_file(self, job_id, example_id):
        """알려진 예제의 최신 발화에 대응하는 검증된 WAV만 제공한다."""
        with self._lock:
            job = self.dataset_service.get_job(job_id)
            row = next((row for row in job["examples"] if row["id"] == example_id), None)
            if row is None:
                raise KeyError("예제를 찾지 못했습니다.")
            audio = self._row_audio(job_id, row, self._state(job_id))
            if audio["status"] != "ready":
                raise RuntimeError("현재 발화문의 WAV를 먼저 생성하세요.")
            path = self._path(job_id, example_id, audio["fingerprint"], audio.get("output_root"))
            inspect_wav(path)
            return path

    def review_audio(self, job_id, example_id, payload):
        """완성된 음성의 포함·제외 결정을 저장하여 새로고침 후에도 유지한다."""
        if not isinstance(payload, dict) or type(payload.get("included")) is not bool:
            raise ValueError("음성 포함 여부를 boolean으로 전달하세요.")
        with self._lock:
            if self.is_active(job_id):
                raise RuntimeError("음성 생성이 끝난 뒤 검토 결과를 저장하세요.")
            self.audio_file(job_id, example_id)
            record = self._state(job_id)["examples"][example_id]
            if payload.get("fingerprint") != record["fingerprint"]:
                raise RuntimeError("음성이 변경되었습니다. 최신 음성을 다시 확인하세요.")
            record["included"] = payload["included"]
            self._save(job_id)
            return self.decorate_job(self.dataset_service.get_job(job_id))

    def create_mapping(self, job_id, payload):
        """청취 후 선택한 WAV와 현재 정답을 새 데이터셋 폴더로 확정한다."""
        with self._lock:
            job = self.decorate_job(self.dataset_service.get_job(job_id))
            if self.is_active(job_id) or job["status"] in ACTIVE_STATUSES:
                raise RuntimeError("텍스트·음성 생성이 끝난 뒤 매핑하세요.")
            rows = self._selected_rows(job, payload, ready=True)
            root = self._output_root(payload.get("output_root", "/tmp/ai-dataset/"))
            root.mkdir(parents=True, exist_ok=True)
            name = f"dataset-{job_id[:8]}-{uuid.uuid4().hex[:12]}"
            destination = root / name
            with tempfile.TemporaryDirectory(prefix=".mapping-", dir=root) as temporary:
                staging = Path(temporary) / name
                self._write_mapping(staging, job_id, rows)
                staging.rename(destination)
            result = {"output_dir": str(destination), "output_root": str(root), "count": len(rows), "transcripts": str(destination / "transcripts.jsonl"),
                      "training_ready": len({normalize_text(row["normalized_text"]) for row in rows}) >= 2}
            self._state(job_id)["last_mapping"] = {**result, "created_at": current_timestamp()}
            history = self._state(job_id).get("mappings", [])
            self._state(job_id)["mappings"] = [*(history if isinstance(history, list) else []), result]
            self._save(job_id)
            return result

    def _write_mapping(self, destination, job_id, rows):
        """선택 음성만 복사하고 원문 정답을 상대 경로로 연결한다."""
        (destination / "audio").mkdir(parents=True)
        records = [self._copy_mapping_audio(destination, job_id, row) for row in rows]
        self._write_jsonl(destination / "transcripts.jsonl", records)
        groups = {normalize_text(row["normalized_text"]) for row in rows}
        splits = {}
        if len(groups) >= 2:
            validation = self._validation_groups(rows)
            splits = {"train": [], "validation": []}
            for record in records:
                split = "validation" if normalize_text(record["text"]) in validation else "train"
                splits[split].append(record)
            for split, items in splits.items():
                self._write_jsonl(destination / f"{split}.jsonl", items)
        information = {"format_version": 1, "language": "Korean", "text_format": "plain", "training_model": "Qwen ASR",
                       "synthetic_audio": True, "tts_model": getattr(self.config, "model_name", self.config.model_file.name), "source_job_id": job_id,
                       "created_at": current_timestamp(), "count": len(records), "split_counts": {key: len(values) for key, values in splits.items()}}
        (destination / "dataset_info.json").write_text(json.dumps(information, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _copy_mapping_audio(self, destination, job_id, row):
        """원본 WAV를 보존하고 매핑 폴더에 검증된 복사본을 만든다."""
        source = self.audio_file(job_id, row["id"])
        relative = f"audio/{row['id']}.wav"
        copied = destination / relative
        shutil.copyfile(source, copied)
        return {"id": row["id"], "audio": relative, "text": row["normalized_text"],
                "spoken_text": row["spoken_text"], **inspect_wav(copied)}

    def _write_jsonl(self, path, records):
        """학습·매핑 목록을 UTF-8 JSONL로 기록한다."""
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")

    def export_training(self, job_id):
        """WAV와 현재 정답을 Qwen ASR 프로젝트가 읽는 train·validation ZIP으로 묶는다."""
        with self._lock:
            job = self.decorate_job(self.dataset_service.get_job(job_id))
            if self.is_active(job_id) or job["status"] in ACTIVE_STATUSES:
                raise RuntimeError("텍스트·음성 생성이 끝난 뒤 내보내세요.")
            rows = eligible_examples(job)
            if not rows or any(row["audio"]["status"] != "ready" for row in rows):
                raise ValueError("검토를 마친 채택 예제의 WAV를 모두 생성한 뒤 내보내세요.")
            validation = self._validation_groups(rows)
            return self._write_archive(job_id, rows, validation)

    def _validation_groups(self, rows):
        """같은 정답이 양쪽에 섞이지 않도록 정답 그룹의 약 10%를 검증용으로 고정한다."""
        groups = sorted({normalize_text(row["normalized_text"]) for row in rows},
                        key=lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
        if len(groups) < 2:
            raise ValueError("train·validation 분리를 위해 서로 다른 정답이 최소 2개 필요합니다.")
        return set(groups[:max(1, len(groups) // 10)])

    def _write_archive(self, job_id, rows, validation):
        """대용량 WAV를 메모리에 모으지 않고 임시 ZIP 파일에 순서대로 기록한다."""
        descriptor, name = tempfile.mkstemp(prefix="qwen-asr-", suffix=".zip", dir=self.dataset_service.data_dir / job_id)
        os.close(descriptor)
        destination = Path(name)
        try:
            self._fill_archive(destination, job_id, rows, validation)
            return destination
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    def _fill_archive(self, destination, job_id, rows, validation):
        """상대 WAV 경로와 접두어 없는 정답 전사를 학습 목록에 기록한다."""
        splits = {"train": [], "validation": []}
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for row in rows:
                self._archive_row(archive, job_id, row, validation, splits)
            for split, items in splits.items():
                archive.writestr(f"{split}.jsonl", "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
            information = {"format_version": 1, "language": "Korean", "text_format": "plain", "training_model": "Qwen ASR",
                           "tts_model": getattr(self.config, "model_name", self.config.model_file.name), "synthetic_audio": True, "source_job_id": job_id,
                           "created_at": current_timestamp(), "split_counts": {split: len(items) for split, items in splits.items()},
                           "split_policy": "동일 정답 그룹을 분리하지 않고 약 10%를 검증용으로 지정(최소 1그룹)"}
            archive.writestr("dataset_info.json", json.dumps(information, ensure_ascii=False, indent=2) + "\n")

    def _archive_row(self, archive, job_id, row, validation, splits):
        """검증한 WAV와 최신 정답을 동일 고유 ID로 연결한다."""
        path = self.audio_file(job_id, row["id"])
        relative = f"audio/{row['id']}.wav"
        split = "validation" if normalize_text(row["normalized_text"]) in validation else "train"
        splits[split].append({"id": row["id"], "audio": relative, "text": row["normalized_text"],
                              "spoken_text": row["spoken_text"], **inspect_wav(path)})
        archive.write(path, relative)

    def close(self):
        """앱 종료 시 음성 프로세스에 중지를 전달하고 저장 완료를 기다린다."""
        with self._lock:
            self._cancellation.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=15)
