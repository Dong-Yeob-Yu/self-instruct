"""LLM과 TTS가 동시에 모델 메모리를 사용하지 않도록 실행 구간을 관리한다."""

import threading
from contextlib import contextmanager


class ModelRuntime:
    def __init__(self):
        """모델 실행 잠금과 UI에 공개할 자원 상태를 준비한다."""
        self._execution_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._state = {"active_model": None, "status": "idle", "last_released": None}

    def snapshot(self):
        """현재 모델 점유 상태의 복사본을 반환한다."""
        with self._state_lock:
            return dict(self._state)

    @contextmanager
    def use(self, model, cancellation, cancellation_error):
        """이전 모델 실행 종료 후 진입하고 모든 종료 경로에서 점유를 해제한다."""
        while not self._execution_lock.acquire(timeout=0.1):
            if cancellation.is_set():
                raise cancellation_error()
        try:
            if cancellation.is_set():
                raise cancellation_error()
            with self._state_lock:
                self._state.update(active_model=model, status="running")
            yield
        finally:
            with self._state_lock:
                self._state.update(active_model=None, status="idle", last_released=model)
            self._execution_lock.release()
