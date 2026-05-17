import io
import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from backend.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    DATA_DIR,
    VIDEOS_DIR,
)
from backend.infer_jobs import get_job as get_infer_job
from backend.infer_jobs import request_cancel as request_infer_cancel
from backend.infer_jobs import start_video_job
from backend.infer_box_draw import draw_nested_detection_boxes, track_id_map_from_enriched
from backend.infer_overlay import make_video_draw_fn

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/infer", tags=["inference"])

from backend.infer_results import (
    INFER_RESULTS_DIR,
    delete_result_files,
    load_video_report,
    transcode_for_browser,
)

INFER_DIR = DATA_DIR / "inference"
INFER_MODELS_DIR = INFER_DIR / "models"

for _d in [INFER_MODELS_DIR, INFER_RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def _form_bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "on", "yes")


def _draw_frame_index_badge(frame, frame_index: int) -> None:
    """Frame number in top-left (matches on-video player overlay)."""
    text = f"Frame {frame_index}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = 10, 28
    cv2.rectangle(frame, (x - 6, y - th - 10), (x + tw + 10, y + 8), (16, 16, 16), -1)
    cv2.rectangle(frame, (x - 6, y - th - 10), (x + tw + 10, y + 8), (99, 102, 241), 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (165, 243, 252), thickness, cv2.LINE_AA)


def _confidence_color(conf: float) -> tuple[int, int, int]:
    """Green → yellow → red based on confidence."""
    if conf >= 0.7:
        return (0, 220, 80)
    if conf >= 0.4:
        return (0, 200, 255)
    return (50, 80, 255)


def _draw_detections(
    img,
    results,
    task: str,
    *,
    box_speeds: Optional[dict] = None,
    enriched: Optional[List[dict]] = None,
):
    """Draw boxes / masks / keypoints on a cv2 image in-place."""
    import numpy as np

    box_speeds = box_speeds or {}
    track_map = track_id_map_from_enriched(enriched, results)
    box_items: List[dict] = []

    for ri, result in enumerate(results):
        names = result.names

        # --- segmentation masks ---
        if task == "segment" and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            h, w = img.shape[:2]
            for i, mask in enumerate(masks):
                mask_resized = cv2.resize(mask, (w, h))
                color_idx = i % 10
                hue = int(color_idx * 36)
                color = tuple(
                    int(c)
                    for c in cv2.cvtColor(
                        np.uint8([[[hue, 200, 180]]]), cv2.COLOR_HSV2BGR
                    )[0][0]
                )
                overlay = img.copy()
                overlay[mask_resized > 0.5] = (
                    overlay[mask_resized > 0.5] * 0.45 + np.array(color) * 0.55
                )
                cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        # --- bounding boxes (collected then drawn nested) ---
        if result.boxes is not None:
            for bi, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{names.get(cls_id, cls_id)} {conf:.2f}"
                sp = box_speeds.get((ri, bi))
                if sp and sp > 0:
                    label += f" {sp:.0f}px/s"
                box_items.append(
                    {
                        "ri": ri,
                        "bi": bi,
                        "xyxy": (x1, y1, x2, y2),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "label": label,
                        "color": _confidence_color(conf),
                    }
                )

        # --- keypoints (pose) ---
        if task == "pose" and result.keypoints is not None:
            kpts = result.keypoints.xy.cpu().numpy()
            for person in kpts:
                for kx, ky in person:
                    if kx > 0 and ky > 0:
                        cv2.circle(img, (int(kx), int(ky)), 4, (0, 255, 200), -1)

    draw_nested_detection_boxes(img, box_items, track_map)
    return img


# ---------------------------------------------------------------------------
# Upload a custom model
# ---------------------------------------------------------------------------
@router.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt model files are accepted")

    safe_name = Path(file.filename).name
    dest = INFER_MODELS_DIR / safe_name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size_mb = dest.stat().st_size / (1024 * 1024)
    logger.info("Uploaded inference model: %s (%.1f MB)", safe_name, size_mb)
    return {"filename": safe_name, "size_mb": round(size_mb, 2)}


@router.get("/models")
async def list_inference_models():
    """List uploaded .pt models available for inference."""
    models = []
    for p in sorted(INFER_MODELS_DIR.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True):
        models.append({"filename": p.name, "size_mb": round(p.stat().st_size / 1024 / 1024, 2)})
    return {"models": models}


@router.delete("/models/{filename}")
async def delete_inference_model(filename: str):
    path = INFER_MODELS_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    path.unlink()
    return {"deleted": filename}


# ---------------------------------------------------------------------------
# Run inference on an image
# ---------------------------------------------------------------------------
@router.post("/image")
async def infer_image(
    file: UploadFile = File(...),
    model_filename: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    task: str = Form("detect"),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    model_path = INFER_MODELS_DIR / Path(model_filename).name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_filename}' not found — upload it first")

    # Read image
    data = await file.read()
    import numpy as np
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        results = model.predict(img, conf=conf, iou=iou, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Build JSON detections
    detections = []
    for result in results:
        names = result.names
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
                detections.append({
                    "class_id": int(box.cls[0]),
                    "class_name": names.get(int(box.cls[0]), str(int(box.cls[0]))),
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [x1, y1, x2, y2],
                })

    # Draw and return annotated image
    annotated = _draw_detections(img, results, task)
    result_id = uuid.uuid4().hex[:8]
    out_path = INFER_RESULTS_DIR / f"{result_id}.jpg"
    cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {
        "result_id": result_id,
        "detections": detections,
        "count": len(detections),
        "result_image": f"/api/infer/result/{result_id}.jpg",
    }


# ---------------------------------------------------------------------------
# Run inference on a video (background job with progress)
# ---------------------------------------------------------------------------
@router.post("/video")
async def infer_video(
    model_filename: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    task: str = Form("detect"),
    max_frames: int = Form(0),  # 0 = unlimited
    show_speed: str = Form("0"),
    show_class_overlay: str = Form("0"),
    show_class_total: str = Form("0"),
    file: Optional[UploadFile] = File(None),
    source_filename: Optional[str] = Form(None),
):
    """Start video inference; poll GET /video/status/{job_id} for progress."""
    model_path = INFER_MODELS_DIR / Path(model_filename).name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_filename}' not found")

    video_path: Optional[Path] = None
    delete_after = False

    if source_filename:
        video_path = VIDEOS_DIR / Path(source_filename).name
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video '{source_filename}' not found")
        suffix = video_path.suffix.lower()
    elif file and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported video format")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            video_path = Path(tmp.name)
            delete_after = True
    else:
        raise HTTPException(status_code=400, detail="Provide a video file or source_filename")

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    result_id = uuid.uuid4().hex[:8]
    out_path = INFER_RESULTS_DIR / f"{result_id}.mp4"

    draw_fn = make_video_draw_fn(
        _draw_detections,
        show_speed=_form_bool(show_speed),
        show_class_overlay=_form_bool(show_class_overlay),
        show_class_total=_form_bool(show_class_total),
    )
    job_id = start_video_job(
        video_path,
        model_path,
        out_path,
        result_id,
        draw_fn,
        conf=conf,
        iou=iou,
        task=task,
        max_frames=max_frames,
        show_speed=_form_bool(show_speed),
        show_class_overlay=_form_bool(show_class_overlay),
        show_class_total=_form_bool(show_class_total),
        delete_source_after=delete_after,
    )
    return {"job_id": job_id, "status": "started"}


@router.get("/video/status/{job_id}")
async def infer_video_status(job_id: str):
    job = get_infer_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/video/cancel/{job_id}")
async def infer_video_cancel(job_id: str):
    if not request_infer_cancel(job_id):
        job = get_infer_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    return {"status": "cancel_requested", "job_id": job_id}


@router.get("/result/{result_id}/report")
async def get_result_report(result_id: str):
    report = load_video_report(result_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Detection report not found for this result")
    return report


@router.get("/result/{filename}")
async def get_result_file(filename: str):
    path = INFER_RESULTS_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    if filename.lower().endswith(".mp4"):
        return FileResponse(str(path), media_type="video/mp4")
    return FileResponse(str(path), media_type="image/jpeg")


@router.get("/result/{result_id}/frame/{frame_index}")
async def get_result_frame(
    result_id: str,
    frame_index: int,
    x1: Optional[float] = Query(None),
    y1: Optional[float] = Query(None),
    x2: Optional[float] = Query(None),
    y2: Optional[float] = Query(None),
):
    """Extract a single annotated frame from a result video (optional bbox highlight)."""
    video_path = INFER_RESULTS_DIR / f"{Path(result_id).name}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Result video not found")
    if frame_index < 0:
        raise HTTPException(status_code=400, detail="frame_index must be >= 0")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open result video")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise HTTPException(status_code=404, detail=f"Frame {frame_index} not found")

        _draw_frame_index_badge(frame, frame_index)

        if None not in (x1, y1, x2, y2):
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 220, 80), 2)
            label = f"frame {frame_index}"
            cv2.putText(
                frame,
                label,
                (ix1, max(iy1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 80),
                1,
                cv2.LINE_AA,
            )

        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode frame")
        return Response(content=jpg.tobytes(), media_type="image/jpeg")
    finally:
        cap.release()


@router.get("/results")
async def list_results():
    results = []
    for p in sorted(INFER_RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.suffix in {".jpg", ".mp4"}:
            result_id = p.stem
            results.append({
                "filename": p.name,
                "result_id": result_id,
                "type": "video" if p.suffix == ".mp4" else "image",
                "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                "mtime": p.stat().st_mtime,
                "url": f"/api/infer/result/{p.name}",
                "has_report": load_video_report(result_id) is not None,
            })
    return {"results": results}


@router.post("/results/{result_id}/transcode")
async def transcode_result_video(result_id: str):
    """Re-encode an existing result video for browser playback."""
    path = INFER_RESULTS_DIR / f"{Path(result_id).name}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result video not found")
    transcode_for_browser(path)
    return {
        "result_id": result_id,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "url": f"/api/infer/result/{path.name}",
    }


@router.delete("/results/{filename}")
async def delete_result(filename: str):
    path = INFER_RESULTS_DIR / Path(filename).name
    if not path.exists() and not load_video_report(Path(filename).stem):
        raise HTTPException(status_code=404, detail="Result not found")
    delete_result_files(filename)
    return {"deleted": filename}
