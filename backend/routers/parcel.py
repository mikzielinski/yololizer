"""Parcel capture API — button time, distance offset, clip + top clusters."""
import io
import json
import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from backend.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    PARCEL_CLIPS_DIR,
    PARCEL_META_DIR,
    VIDEOS_DIR,
)
from backend.infer_jobs import get_job, request_cancel
from backend.infer_results import INFER_RESULTS_DIR
from backend.parcel_capture import (
    compute_arrival_film_sec,
    encode_video_frame_jpeg,
    real_seconds_to_film,
    travel_real_seconds,
)
from backend.parcel_jobs import start_parcel_job
from backend.routers.inference import INFER_MODELS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/parcel", tags=["parcel"])


def _load_capture_meta(capture_id: str) -> Dict[str, Any]:
    meta_path = PARCEL_META_DIR / f"{Path(capture_id).name}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Capture metadata not found")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _safe_zip_name(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s or "obj")[:40]


def _cluster_sort_key(cluster: Dict[str, Any], sort_id: str) -> float:
    c = cluster
    span = int(c.get("frame_end", 0)) - int(c.get("frame_start", 0))
    if sort_id == "time_asc":
        return float(c.get("best_frame", 0))
    if sort_id == "time_desc":
        return -float(c.get("best_frame", 0))
    if sort_id == "conf_desc":
        return -float(c.get("mean_confidence", 0))
    if sort_id == "conf_asc":
        return float(c.get("mean_confidence", 0))
    if sort_id == "score_asc":
        return float(c.get("score", 0))
    if sort_id == "detections_desc":
        return -float(c.get("detection_count", 0))
    if sort_id == "detections_asc":
        return float(c.get("detection_count", 0))
    if sort_id == "frame_span_desc":
        return -float(span)
    if sort_id == "rank":
        return float(c.get("rank", 999))
    return -float(c.get("score", 0))


def _sort_clusters(clusters: list, sort_id: str) -> list:
    allowed = {
        "score_desc", "score_asc", "time_asc", "time_desc",
        "conf_desc", "conf_asc", "detections_desc", "detections_asc",
        "frame_span_desc", "rank",
    }
    key = sort_id if sort_id in allowed else "score_desc"
    return sorted(clusters, key=lambda c: _cluster_sort_key(c, key))


@router.post("/preview-timing")
async def preview_timing(
    button_film_sec: Annotated[float, Form(description="Button press time on film timeline (seconds)")],
    distance_m: Annotated[float, Form(description="Distance from button to camera (meters)")] = 30.0,
    speed_m_per_s: Annotated[float, Form(description="Conveyor speed in real world (m/s)")] = 0.5,
    real_to_film_ratio: Annotated[
        float, Form(description="Film seconds per 1 real second (0.5 = 1s real → 0.5s on film)")
    ] = 0.5,
    pre_window_sec: Annotated[float, Form(description="Seconds before arrival to include")] = 2.0,
    post_window_sec: Annotated[float, Form(description="Seconds after arrival to include")] = 3.0,
):
    """Compute expected arrival on film timeline without running detection."""
    if speed_m_per_s <= 0:
        raise HTTPException(status_code=400, detail="speed_m_per_s must be > 0")
    if real_to_film_ratio <= 0:
        raise HTTPException(status_code=400, detail="real_to_film_ratio must be > 0")
    travel_real = travel_real_seconds(distance_m, speed_m_per_s)
    travel_film = real_seconds_to_film(travel_real, real_to_film_ratio)
    arrival = compute_arrival_film_sec(button_film_sec, distance_m, speed_m_per_s, real_to_film_ratio)
    return {
        "button_film_sec": button_film_sec,
        "travel_real_sec": round(travel_real, 3),
        "travel_film_sec": round(travel_film, 3),
        "arrival_film_sec": round(arrival, 3),
        "window_start_film_sec": round(max(0, arrival - pre_window_sec), 3),
        "window_end_film_sec": round(arrival + post_window_sec, 3),
        "real_to_film_ratio": real_to_film_ratio,
        "hint": f"1 s real = {real_to_film_ratio} s on film",
    }


@router.post("/analyze")
async def analyze_parcel(
    model_filename: Annotated[str, Form(description="`.pt` filename from GET /api/infer/models")],
    button_film_sec: Annotated[float, Form(description="Button press time on film timeline (seconds)")],
    distance_m: Annotated[float, Form(description="Distance button → camera (m)")] = 30.0,
    speed_m_per_s: Annotated[float, Form(description="Conveyor speed (m/s, real)")] = 0.5,
    real_to_film_ratio: Annotated[float, Form(description="Film seconds per 1 real second")] = 0.5,
    pre_window_sec: Annotated[float, Form(description="Pre-window (s)")] = 2.0,
    post_window_sec: Annotated[float, Form(description="Post-window (s)")] = 3.0,
    top_clusters: Annotated[int, Form(description="Number of top detection clusters to return")] = 4,
    conf: Annotated[float, Form(description="YOLO confidence threshold")] = 0.25,
    iou: Annotated[float, Form(description="YOLO NMS IoU threshold")] = 0.45,
    frame_stride: Annotated[int, Form(description="Process every Nth frame in window")] = 1,
    file: Annotated[Optional[UploadFile], File(description="Upload video (alternative to library/result)")] = None,
    source_filename: Annotated[Optional[str], Form(description="Video from library (GET /api/sources/videos)")] = None,
    result_video_id: Annotated[Optional[str], Form(description="Infer result video id")] = None,
):
    """
    Find parcel clusters in the video window after button press + travel offset.
    button_film_sec: button moment on the video timeline (seconds).
  real_to_film_ratio: seconds of film per 1 second real (0.5 = 1s real → 0.5s film).
    """
    model_path = INFER_MODELS_DIR / Path(model_filename).name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_filename}' not found")

    video_path: Optional[Path] = None
    delete_after = False

    if result_video_id:
        video_path = INFER_RESULTS_DIR / f"{Path(result_video_id).name}.mp4"
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Result video not found")
    elif source_filename:
        video_path = VIDEOS_DIR / Path(source_filename).name
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
    elif file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported video format")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            video_path = Path(tmp.name)
            delete_after = True
    else:
        raise HTTPException(status_code=400, detail="Provide video file, source_filename, or result_video_id")

    job_id = start_parcel_job(
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
        frame_stride=max(1, frame_stride),
        delete_source_after=delete_after,
    )
    return {"job_id": job_id, "status": "started"}


@router.get("/status/{job_id}")
async def parcel_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/cancel/{job_id}")
async def parcel_cancel(job_id: str):
    if not request_cancel(job_id):
        job = get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    return {"status": "cancel_requested", "job_id": job_id}


@router.get("/clip/{capture_id}.mp4")
async def get_clip(capture_id: str):
    path = PARCEL_CLIPS_DIR / f"{Path(capture_id).name}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(str(path), media_type="video/mp4", filename=f"parcel_{capture_id}.mp4")


@router.get("/meta/{capture_id}")
async def get_capture_meta(capture_id: str):
    """Saved capture metadata including top clusters (for UI refresh)."""
    return _load_capture_meta(capture_id)


@router.get("/frame/{capture_id}/{frame_index}")
async def get_capture_frame(
    capture_id: str,
    frame_index: int,
    x1: Optional[float] = None,
    y1: Optional[float] = None,
    x2: Optional[float] = None,
    y2: Optional[float] = None,
    download: bool = Query(False),
):
    """Frame from source video for a capture (metadata stores video ref)."""
    meta = _load_capture_meta(capture_id)
    video_path = Path(meta["video_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Source video missing")

    bbox = None
    if None not in (x1, y1, x2, y2):
        bbox = [x1, y1, x2, y2]
    try:
        jpg = encode_video_frame_jpeg(video_path, frame_index, bbox)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    headers = {}
    if download:
        headers["Content-Disposition"] = (
            f'attachment; filename="parcel_{Path(capture_id).name}_frame{frame_index}.jpg"'
        )
    return Response(content=jpg, media_type="image/jpeg", headers=headers)


@router.get("/frames-zip/{capture_id}")
async def download_top_frames_zip(
    capture_id: str,
    sort: str = Query(
        "score_desc",
        description="Cluster order: score_desc, time_asc, time_desc, conf_desc, conf_asc, detections_desc, frame_span_desc, rank",
    ),
):
    """ZIP of best-frame JPEGs for each saved top cluster."""
    meta = _load_capture_meta(capture_id)
    clusters = meta.get("clusters") or []
    if not clusters:
        raise HTTPException(status_code=404, detail="No clusters saved for this capture")

    video_path = Path(meta["video_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Source video missing")

    clusters = _sort_clusters(clusters, sort)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, c in enumerate(clusters, start=1):
            cls = _safe_zip_name(str(c.get("class_name", "obj")))
            frame_idx = int(c.get("best_frame", 0))
            arcname = f"{i:02d}_{cls}_frame{frame_idx}.jpg"
            try:
                jpg = encode_video_frame_jpeg(video_path, frame_idx, c.get("best_bbox"))
            except ValueError as e:
                logger.warning("Skip frame %s in zip: %s", frame_idx, e)
                continue
            zf.writestr(arcname, jpg)

    if not buf.getbuffer().nbytes:
        raise HTTPException(status_code=404, detail="Could not export any frames")

    buf.seek(0)
    cid = Path(capture_id).name
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="parcel_{cid}_top_frames.zip"',
        },
    )
