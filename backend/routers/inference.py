import io
import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from backend.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    DATA_DIR,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/infer", tags=["inference"])

INFER_DIR = DATA_DIR / "inference"
INFER_MODELS_DIR = INFER_DIR / "models"
INFER_RESULTS_DIR = INFER_DIR / "results"

for _d in [INFER_MODELS_DIR, INFER_RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def _confidence_color(conf: float) -> tuple[int, int, int]:
    """Green → yellow → red based on confidence."""
    if conf >= 0.7:
        return (0, 220, 80)
    if conf >= 0.4:
        return (0, 200, 255)
    return (50, 80, 255)


def _draw_detections(img, results, task: str):
    """Draw boxes / masks / keypoints on a cv2 image in-place."""
    import numpy as np

    for result in results:
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

        # --- bounding boxes ---
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{names.get(cls_id, cls_id)} {conf:.2f}"
                color = _confidence_color(conf)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )

        # --- keypoints (pose) ---
        if task == "pose" and result.keypoints is not None:
            kpts = result.keypoints.xy.cpu().numpy()
            for person in kpts:
                for kx, ky in person:
                    if kx > 0 and ky > 0:
                        cv2.circle(img, (int(kx), int(ky)), 4, (0, 255, 200), -1)

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


@router.get("/result/{filename}")
async def get_result_image(filename: str):
    path = INFER_RESULTS_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(str(path), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Run inference on a video — returns annotated video file
# ---------------------------------------------------------------------------
@router.post("/video")
async def infer_video(
    file: UploadFile = File(...),
    model_filename: str = Form(...),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    task: str = Form("detect"),
    max_frames: int = Form(0),  # 0 = unlimited
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    model_path = INFER_MODELS_DIR / Path(model_filename).name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_filename}' not found")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result_id = uuid.uuid4().hex[:8]
        out_path = INFER_RESULTS_DIR / f"{result_id}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_count >= max_frames:
                break

            results = model.predict(frame, conf=conf, iou=iou, verbose=False)
            annotated = _draw_detections(frame, results, task)
            writer.write(annotated)

            for r in results:
                if r.boxes is not None:
                    total_detections += len(r.boxes)
            frame_count += 1

        cap.release()
        writer.release()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video inference failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    size_mb = out_path.stat().st_size / 1024 / 1024
    return {
        "result_id": result_id,
        "frames_processed": frame_count,
        "total_detections": total_detections,
        "size_mb": round(size_mb, 2),
        "result_video": f"/api/infer/result/{result_id}.mp4",
    }


@router.get("/result/{filename}")
async def get_result_file(filename: str):
    path = INFER_RESULTS_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    media = "video/mp4" if filename.endswith(".mp4") else "image/jpeg"
    return FileResponse(str(path), media_type=media)


@router.get("/results")
async def list_results():
    results = []
    for p in sorted(INFER_RESULTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.suffix in {".jpg", ".mp4"}:
            results.append({
                "filename": p.name,
                "type": "video" if p.suffix == ".mp4" else "image",
                "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
                "mtime": p.stat().st_mtime,
                "url": f"/api/infer/result/{p.name}",
            })
    return {"results": results}


@router.delete("/results/{filename}")
async def delete_result(filename: str):
    path = INFER_RESULTS_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    path.unlink()
    return {"deleted": filename}
