"""Parcel capture API — button time, distance offset, clip + top clusters."""
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from backend.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    PARCEL_CLIPS_DIR,
    PARCEL_META_DIR,
    VIDEOS_DIR,
)
from backend.infer_jobs import get_job, request_cancel
from backend.infer_results import INFER_RESULTS_DIR
from backend.parcel_capture import compute_arrival_film_sec, real_seconds_to_film, travel_real_seconds
from backend.parcel_jobs import start_parcel_job
from backend.routers.inference import INFER_MODELS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/parcel", tags=["parcel"])

@router.post("/preview-timing")
async def preview_timing(
    button_film_sec: float = Form(...),
    distance_m: float = Form(30.0),
    speed_m_per_s: float = Form(0.5),
    real_to_film_ratio: float = Form(0.5),
    pre_window_sec: float = Form(2.0),
    post_window_sec: float = Form(3.0),
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
    model_filename: str = Form(...),
    button_film_sec: float = Form(...),
    distance_m: float = Form(30.0),
    speed_m_per_s: float = Form(0.5),
    real_to_film_ratio: float = Form(0.5),
    pre_window_sec: float = Form(2.0),
    post_window_sec: float = Form(3.0),
    top_clusters: int = Form(4),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    frame_stride: int = Form(1),
    file: Optional[UploadFile] = File(None),
    source_filename: Optional[str] = Form(None),
    result_video_id: Optional[str] = Form(None),
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


@router.get("/frame/{capture_id}/{frame_index}")
async def get_capture_frame(
    capture_id: str,
    frame_index: int,
    x1: Optional[float] = None,
    y1: Optional[float] = None,
    x2: Optional[float] = None,
    y2: Optional[float] = None,
):
    """Frame from source video for a capture (metadata stores video ref)."""
    meta_path = PARCEL_META_DIR / f"{Path(capture_id).name}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Capture metadata not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    video_path = Path(meta["video_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Source video missing")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=404, detail="Frame not found")
        if None not in (x1, y1, x2, y2):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 80), 2)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise HTTPException(status_code=500, detail="Encode failed")
        return Response(content=jpg.tobytes(), media_type="image/jpeg")
    finally:
        cap.release()
