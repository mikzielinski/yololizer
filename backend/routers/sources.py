import asyncio
import logging
import time
from pathlib import Path
from typing import List

import aiofiles
import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from backend.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    DEFAULT_FRAME_INTERVAL,
    FRAMES_DIR,
    UPLOADS_DIR,
    WEBCAM_INDEX,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sources", tags=["sources"])


def _is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def _is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


@router.post("/images")
async def upload_images(files: List[UploadFile] = File(...)):
    """Accept multipart image uploads, save to uploads/."""
    saved = []
    errors = []

    for upload in files:
        if not upload.filename:
            errors.append("Empty filename")
            continue

        if not _is_image(upload.filename):
            errors.append(f"{upload.filename}: not an allowed image type")
            continue

        dest = UPLOADS_DIR / Path(upload.filename).name
        # Avoid collisions
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            dest = UPLOADS_DIR / f"{stem}_{int(time.time())}{suffix}"

        try:
            async with aiofiles.open(dest, "wb") as f:
                content = await upload.read()
                await f.write(content)
            saved.append(dest.name)
        except Exception as e:
            logger.exception("Failed to save %s", upload.filename)
            errors.append(f"{upload.filename}: {str(e)}")

    return {"saved": saved, "errors": errors}


@router.post("/video")
async def upload_video(
    file: UploadFile = File(...),
    frame_interval: int = Form(DEFAULT_FRAME_INTERVAL),
):
    """Accept video upload, extract frames with OpenCV every N frames."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    if not _is_video(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Not an allowed video type. Allowed: {ALLOWED_VIDEO_EXTENSIONS}",
        )

    if frame_interval < 1:
        raise HTTPException(status_code=400, detail="frame_interval must be >= 1")

    # Save video temporarily
    tmp_path = UPLOADS_DIR / f"_tmp_{int(time.time())}_{Path(file.filename).name}"
    try:
        content = await file.read()
        async with aiofiles.open(tmp_path, "wb") as f:
            await f.write(content)

        # Extract frames in a thread to avoid blocking the event loop
        def _extract():
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                return [], "Failed to open video file"

            frames_saved = []
            frame_idx = 0
            video_stem = Path(file.filename).stem

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    frame_name = f"{video_stem}_frame{frame_idx:06d}.jpg"
                    frame_path = FRAMES_DIR / frame_name
                    cv2.imwrite(str(frame_path), frame)
                    frames_saved.append(frame_name)
                frame_idx += 1

            cap.release()
            return frames_saved, None

        loop = asyncio.get_event_loop()
        frames_saved, error = await loop.run_in_executor(None, _extract)

        if error:
            raise HTTPException(status_code=500, detail=error)

        return {
            "video": file.filename,
            "frame_interval": frame_interval,
            "frames_extracted": len(frames_saved),
            "frames": frames_saved,
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/webcam/stream")
async def webcam_stream():
    """MJPEG stream from webcam index 0."""

    def _generate():
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            raise HTTPException(status_code=503, detail="No webcam available at index 0")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                )
        finally:
            cap.release()

    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/webcam/capture")
async def webcam_capture():
    """Capture a single frame from the webcam and save it."""

    def _capture():
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            return None, "No webcam available at index 0"
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None, "Failed to read frame from webcam"
        fname = f"webcam_{int(time.time() * 1000)}.jpg"
        path = UPLOADS_DIR / fname
        cv2.imwrite(str(path), frame)
        return fname, None

    loop = asyncio.get_event_loop()
    fname, error = await loop.run_in_executor(None, _capture)
    if error:
        raise HTTPException(status_code=503, detail=error)

    return {"filename": fname}


@router.get("/images")
async def list_images():
    """List all uploaded images (including frames)."""
    images = []
    for ext in ALLOWED_IMAGE_EXTENSIONS:
        images.extend(UPLOADS_DIR.glob(f"*{ext}"))
        images.extend(UPLOADS_DIR.glob(f"*{ext.upper()}"))
        images.extend(FRAMES_DIR.glob(f"*{ext}"))
        images.extend(FRAMES_DIR.glob(f"*{ext.upper()}"))

    result = []
    for p in sorted(set(images)):
        result.append(
            {
                "filename": p.name,
                "path": str(p.relative_to(UPLOADS_DIR.parent.parent)),
                "is_frame": p.parent == FRAMES_DIR,
                "size": p.stat().st_size,
            }
        )
    return {"images": result, "count": len(result)}


@router.get("/image/{filename:path}")
async def serve_image(filename: str):
    """Serve an image file by name."""
    # Try uploads dir first, then frames dir
    for directory in [UPLOADS_DIR, FRAMES_DIR]:
        path = directory / filename
        if path.exists() and path.is_file():
            return FileResponse(str(path))

    raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
