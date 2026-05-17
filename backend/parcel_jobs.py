"""Background parcel-capture jobs."""
import json
import logging
import threading
from pathlib import Path
from typing import Any

from backend.config import PARCEL_META_DIR
from backend.infer_jobs import create_job, get_job, is_cancelled, update_job
from backend.parcel_capture import run_parcel_capture

PARCEL_META_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def run_parcel_job(
    job_id: str,
    video_path: Path,
    model_path: Path,
    *,
    button_film_sec: float,
    distance_m: float,
    speed_m_per_s: float,
    real_to_film_ratio: float,
    pre_window_sec: float,
    post_window_sec: float,
    top_clusters: int,
    conf: float,
    iou: float,
    frame_stride: int,
    delete_source_after: bool = False,
) -> None:
    try:
        if is_cancelled(job_id):
            update_job(job_id, status="cancelled", message="Cancelled")
            return

        update_job(job_id, status="running", message="Analyzing parcel window…", progress=0.0)

        def on_progress(pct: float, msg: str) -> None:
            if is_cancelled(job_id):
                return
            update_job(
                job_id,
                status="running",
                progress=min(0.99, pct),
                message=msg,
            )

        result = run_parcel_capture(
            video_path,
            model_path,
            button_film_sec=button_film_sec,
            distance_m=distance_m,
            speed_m_per_s=speed_m_per_s,
            real_to_film_ratio=real_to_film_ratio,
            pre_window_sec=pre_window_sec,
            post_window_sec=post_window_sec,
            top_clusters=top_clusters,
            conf=conf,
            iou=iou,
            frame_stride=frame_stride,
            progress_callback=on_progress,
        )

        if is_cancelled(job_id):
            update_job(job_id, status="cancelled", message="Cancelled")
            return

        meta_path = PARCEL_META_DIR / f"{result['capture_id']}.json"
        meta_path.write_text(
            json.dumps(
                {
                    "capture_id": result["capture_id"],
                    "video_path": str(video_path.resolve()),
                }
            ),
            encoding="utf-8",
        )

        update_job(
            job_id,
            status="done",
            progress=1.0,
            message="Complete",
            result=result,
        )
    except Exception as e:
        logger.exception("Parcel job %s failed", job_id)
        update_job(job_id, status="error", error=str(e), message="Failed")
    finally:
        if delete_source_after and video_path.exists():
            video_path.unlink(missing_ok=True)


def start_parcel_job(
    video_path: Path,
    model_path: Path,
    **kwargs: Any,
) -> str:
    job_id = create_job("Queued for parcel capture…")
    thread = threading.Thread(
        target=run_parcel_job,
        args=(job_id, video_path, model_path),
        kwargs=kwargs,
        daemon=True,
        name=f"parcel-{job_id}",
    )
    thread.start()
    return job_id
