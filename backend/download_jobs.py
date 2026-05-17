"""Background download jobs (e.g. YouTube) with progress tracking."""
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_jobs: Dict[str, dict] = {}
_lock = threading.Lock()
_JOB_TTL_SEC = 3600


def create_job(message: str = "Starting…") -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": message,
            "result": None,
            "error": None,
            "cancel_requested": False,
            "created_at": time.time(),
        }
    return job_id


def request_cancel(job_id: str) -> bool:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return False
        if job["status"] in ("done", "error", "cancelled"):
            return False
        job["cancel_requested"] = True
        job["message"] = "Cancelling…"
        return True


def is_cancelled(job_id: str) -> bool:
    with _lock:
        return bool(_jobs.get(job_id, {}).get("cancel_requested"))


def update_job(job_id: str, **kwargs: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def start_job(target: Callable, args: tuple = (), kwargs: Optional[dict] = None) -> str:
    job_id = create_job()
    thread = threading.Thread(
        target=target,
        args=(job_id, *args),
        kwargs=kwargs or {},
        daemon=True,
        name=f"download-{job_id}",
    )
    thread.start()
    return job_id
