"""Background inference jobs with progress tracking."""
import heapq
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from backend.infer_results import save_video_report, transcode_for_browser
from backend.infer_tracking import ObjectSpeedTracker, detections_from_result

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
            "current_frame": 0,
            "total_frames": 0,
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


_TOP_DETECTIONS_LIMIT = 100


def _push_top_detection(
    heap: List[Tuple[float, int, dict]],
    confidence: float,
    seq: int,
    detection: dict,
) -> None:
    entry = (confidence, seq, detection)
    if len(heap) < _TOP_DETECTIONS_LIMIT:
        heapq.heappush(heap, entry)
    elif confidence > heap[0][0]:
        heapq.heapreplace(heap, entry)


def _collect_frame_detections(
    result,
    frame_idx: int,
    class_stats: Dict[str, Dict[str, Any]],
    top_heaps_by_class: Dict[str, List[Tuple[float, int, dict]]],
    seq_counter: List[int],
) -> int:
    """Record box detections for summary counts and per-class top-N heaps."""
    if result.boxes is None:
        return 0

    names = result.names
    count = 0
    for box in result.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]

        if class_name not in class_stats:
            class_stats[class_name] = {"class_id": cls_id, "count": 0}
        class_stats[class_name]["count"] += 1

        seq_counter[0] += 1
        detection = {
            "frame": frame_idx,
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2],
        }
        if class_name not in top_heaps_by_class:
            top_heaps_by_class[class_name] = []
        _push_top_detection(top_heaps_by_class[class_name], conf, seq_counter[0], detection)
        count += 1
    return count


def _build_unique_class_totals(
    unique_by_class: Dict[str, int],
    class_stats: Dict[str, Dict[str, Any]],
) -> List[dict]:
    """Per-class unique object counts for totals overlay and report."""
    return sorted(
        [
            {
                "class_name": name,
                "class_id": class_stats.get(name, {}).get("class_id", -1),
                "count": count,
            }
            for name, count in unique_by_class.items()
        ],
        key=lambda x: (-x["count"], x["class_name"]),
    )


def _build_detection_report(
    class_stats: Dict[str, Dict[str, Any]],
    top_heaps_by_class: Dict[str, List[Tuple[float, int, dict]]],
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    summary = sorted(
        [
            {
                "class_name": name,
                "class_id": info["class_id"],
                "count": info["count"],
            }
            for name, info in class_stats.items()
        ],
        key=lambda x: (-x["count"], x["class_name"]),
    )
    top_by_class: Dict[str, List[dict]] = {}
    for name in sorted(top_heaps_by_class.keys()):
        heap = top_heaps_by_class[name]
        top_by_class[name] = [
            entry[2] for entry in sorted(heap, key=lambda x: (-x[0], x[1]))
        ]
    return summary, top_by_class


def _cleanup_old_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_SEC
    with _lock:
        stale = [jid for jid, j in _jobs.items() if j.get("created_at", 0) < cutoff]
        for jid in stale:
            del _jobs[jid]


def run_video_inference_job(
    job_id: str,
    video_path: Path,
    model_path: Path,
    out_path: Path,
    result_id: str,
    draw_fn,
    *,
    conf: float,
    iou: float,
    task: str,
    max_frames: int = 0,
    show_speed: bool = False,
    show_class_overlay: bool = False,
    show_class_total: bool = False,
    delete_source_after: bool = False,
) -> None:
    """Process video frames in a background thread."""

    def _progress(frame_idx: int, total: int, message: str) -> None:
        if total > 0:
            pct = min(1.0, frame_idx / total)
            update_job(
                job_id,
                status="running",
                progress=pct,
                current_frame=frame_idx,
                total_frames=total,
                message=message,
            )
        else:
            update_job(
                job_id,
                status="running",
                progress=0.0,
                current_frame=frame_idx,
                total_frames=0,
                message=message,
            )

    cap = None
    writer = None
    try:
        if is_cancelled(job_id):
            update_job(job_id, status="cancelled", message="Cancelled by user")
            return

        from ultralytics import YOLO

        update_job(job_id, status="running", message="Loading model…", progress=0.0)
        model = YOLO(str(model_path))

        if is_cancelled(job_id):
            update_job(job_id, status="cancelled", message="Cancelled by user")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if max_frames and total_frames:
            total_frames = min(total_frames, max_frames)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        if hasattr(draw_fn, "init"):
            draw_fn.init(fps)

        object_tracker = ObjectSpeedTracker(fps)
        frame_count = 0
        total_detections = 0
        cancelled = False
        class_stats: Dict[str, Dict[str, Any]] = {}
        top_heaps_by_class: Dict[str, List[Tuple[float, int, dict]]] = {}
        seq_counter = [0]

        _progress(0, total_frames, "Processing video…")

        while True:
            if is_cancelled(job_id):
                cancelled = True
                break

            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_count >= max_frames:
                break

            results = model.predict(frame, conf=conf, iou=iou, verbose=False)
            frame_dets: List[dict] = []
            for r in results:
                total_detections += _collect_frame_detections(
                    r, frame_count, class_stats, top_heaps_by_class, seq_counter
                )
                frame_dets.extend(detections_from_result(r, frame_count))
            enriched = object_tracker.update(frame_count, frame_dets)
            unique_totals = object_tracker.unique_objects_by_class()
            annotated = draw_fn(
                frame,
                results,
                task,
                cumulative_totals=unique_totals if show_class_total else None,
                enriched=enriched,
                tracker=object_tracker,
            )
            writer.write(annotated)
            frame_count += 1

            if frame_count % 5 == 0 or (total_frames and frame_count >= total_frames):
                _progress(
                    frame_count,
                    total_frames,
                    f"Frame {frame_count}" + (f" / {total_frames}" if total_frames else "…"),
                )

        if cancelled:
            if out_path.exists():
                out_path.unlink()
            update_job(job_id, status="cancelled", message="Cancelled by user")
            return

        transcode_for_browser(out_path)
        size_mb = out_path.stat().st_size / 1024 / 1024
        detection_summary, top_by_class = _build_detection_report(
            class_stats, top_heaps_by_class
        )
        class_totals = _build_unique_class_totals(
            object_tracker.unique_objects_by_class(), class_stats
        )
        result_payload = {
            "result_id": result_id,
            "frames_processed": frame_count,
            "total_detections": total_detections,
            "detection_summary": detection_summary,
            "class_totals": class_totals,
            "unique_objects": sum(row["count"] for row in class_totals),
            "top_detections_by_class": top_by_class,
            "fps": round(float(fps), 3),
            "size_mb": round(size_mb, 2),
            "result_video": f"/api/infer/result/{result_id}.mp4",
            "show_speed": show_speed,
            "show_class_overlay": show_class_overlay,
            "show_class_total": show_class_total,
        }
        if show_speed:
            result_payload["speed_summary"] = object_tracker.summary()
        save_video_report(result_id, result_payload)
        update_job(
            job_id,
            status="done",
            progress=1.0,
            current_frame=frame_count,
            total_frames=total_frames or frame_count,
            message="Complete",
            result=result_payload,
        )
    except Exception as e:
        logger.exception("Video inference job %s failed", job_id)
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        update_job(job_id, status="error", error=str(e), message="Failed")
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if delete_source_after and video_path.exists():
            video_path.unlink(missing_ok=True)
        _cleanup_old_jobs()


def start_video_job(
    video_path: Path,
    model_path: Path,
    out_path: Path,
    result_id: str,
    draw_fn,
    *,
    conf: float,
    iou: float,
    task: str,
    max_frames: int = 0,
    show_speed: bool = False,
    show_class_overlay: bool = False,
    show_class_total: bool = False,
    delete_source_after: bool = False,
) -> str:
    job_id = create_job("Queued for processing…")
    thread = threading.Thread(
        target=run_video_inference_job,
        args=(job_id, video_path, model_path, out_path, result_id, draw_fn),
        kwargs={
            "conf": conf,
            "iou": iou,
            "task": task,
            "max_frames": max_frames,
            "show_speed": show_speed,
            "show_class_overlay": show_class_overlay,
            "show_class_total": show_class_total,
            "delete_source_after": delete_source_after,
        },
        daemon=True,
        name=f"infer-video-{job_id}",
    )
    thread.start()
    return job_id
