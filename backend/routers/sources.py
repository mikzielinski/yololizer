import asyncio
import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import aiofiles
import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from backend.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    DEFAULT_FRAME_INTERVAL,
    FRAMES_DIR,
    UPLOADS_DIR,
    VIDEOS_DIR,
    WEBCAM_INDEX,
)
from backend import download_jobs
from backend.youtube import download_youtube_video, is_youtube_url

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sources", tags=["sources"])


def _is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def _is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^\w\-]+", "_", name.strip())[:80]
    return stem or "video"


def _delete_extracted_frames_for_stem(video_stem: str) -> int:
    """Remove annotation frames extracted from a video (prefix: {safe_stem}_frame*.jpg)."""
    prefix = f"{_safe_stem(video_stem)}_frame"
    deleted = 0
    for p in FRAMES_DIR.glob(f"{prefix}*.jpg"):
        if p.is_file():
            p.unlink(missing_ok=True)
            deleted += 1
    return deleted


def _extract_frames(
    video_path: Path,
    video_stem: str,
    frame_interval: int,
    cancel_check=None,
) -> Tuple[List[str], Optional[str]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], "Failed to open video file"

    frames_saved = []
    frame_idx = 0
    safe_stem = _safe_stem(video_stem)

    try:
        while True:
            if cancel_check and cancel_check():
                return frames_saved, "__cancelled__"
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frame_name = f"{safe_stem}_frame{frame_idx:06d}.jpg"
                frame_path = FRAMES_DIR / frame_name
                cv2.imwrite(str(frame_path), frame)
                frames_saved.append(frame_name)
            frame_idx += 1
    finally:
        cap.release()

    return frames_saved, None


def _download_youtube_to_videos(
    url: str,
    progress_callback=None,
) -> Tuple[Path, str, str]:
    """Download YouTube video into VIDEOS_DIR. Returns (path, filename, title)."""
    tmp_dir = UPLOADS_DIR / f"_yt_{int(time.time())}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        video_path, title = download_youtube_video(url, tmp_dir, progress_callback)
        dest = VIDEOS_DIR / video_path.name
        if dest.exists():
            dest = VIDEOS_DIR / f"{video_path.stem}_{int(time.time())}{video_path.suffix}"
        shutil.move(str(video_path), str(dest))
        return dest, dest.name, title
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _youtube_job_worker(job_id: str, url: str, mode: str, frame_interval: int) -> None:
    try:
        extract_after = mode in {"extract", "both"}
        download_weight = 0.55 if extract_after else 1.0

        def on_download(pct: float, msg: str) -> None:
            if download_jobs.is_cancelled(job_id):
                return
            download_jobs.update_job(
                job_id,
                status="running",
                progress=pct * download_weight,
                message=msg,
            )

        video_path, filename, title = _download_youtube_to_videos(url, on_download)

        if download_jobs.is_cancelled(job_id):
            video_path.unlink(missing_ok=True)
            download_jobs.update_job(job_id, status="cancelled", message="Cancelled by user")
            return

        frames_saved: List[str] = []
        if extract_after:
            download_jobs.update_job(
                job_id, status="running", progress=0.55, message="Extracting frames…"
            )
            frames_saved, error = _extract_frames(
                video_path,
                title,
                frame_interval,
                cancel_check=lambda: download_jobs.is_cancelled(job_id),
            )
            if error == "__cancelled__":
                download_jobs.update_job(job_id, status="cancelled", message="Cancelled by user")
                return
            if error:
                raise RuntimeError(error)

        result = {
            "video": title,
            "filename": filename,
            "source_url": url,
            "play_url": f"/api/sources/video/{filename}",
            "mode": mode,
            "frame_interval": frame_interval if extract_after else None,
            "frames_extracted": len(frames_saved),
            "frames": frames_saved,
        }
        download_jobs.update_job(
            job_id, status="done", progress=1.0, message="Complete", result=result
        )
    except Exception as e:
        logger.exception("YouTube job %s failed", job_id)
        download_jobs.update_job(job_id, status="error", error=str(e), message="Failed")


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

        loop = asyncio.get_event_loop()
        frames_saved, error = await loop.run_in_executor(
            None,
            _extract_frames,
            tmp_path,
            Path(file.filename).stem,
            frame_interval,
        )

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


@router.post("/youtube")
async def youtube_video(
    url: str = Form(...),
    mode: str = Form("watch"),
    frame_interval: int = Form(DEFAULT_FRAME_INTERVAL),
):
    """Download a YouTube video. mode: watch | extract | both."""
    url = url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    if not is_youtube_url(url):
        raise HTTPException(status_code=400, detail="Not a valid YouTube URL")

    mode = mode.strip().lower()
    if mode not in {"watch", "extract", "both"}:
        raise HTTPException(status_code=400, detail="mode must be watch, extract, or both")
    if mode != "watch" and frame_interval < 1:
        raise HTTPException(status_code=400, detail="frame_interval must be >= 1")

    job_id = download_jobs.create_job("Starting download…")
    thread = threading.Thread(
        target=_youtube_job_worker,
        args=(job_id, url, mode, frame_interval),
        daemon=True,
        name=f"youtube-{job_id}",
    )
    thread.start()
    return {"job_id": job_id, "status": "started"}


@router.get("/youtube/status/{job_id}")
async def youtube_job_status(job_id: str):
    job = download_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/youtube/cancel/{job_id}")
async def youtube_cancel_job(job_id: str):
    if not download_jobs.request_cancel(job_id):
        job = download_jobs.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    return {"status": "cancel_requested", "job_id": job_id}


@router.post("/videos/upload")
async def upload_video_to_library(file: UploadFile = File(...)):
    """Save an uploaded video file to the media library (no frame extraction)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")
    if not _is_video(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Not an allowed video type. Allowed: {ALLOWED_VIDEO_EXTENSIONS}",
        )

    safe_name = Path(file.filename).name
    dest = VIDEOS_DIR / safe_name
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        dest = VIDEOS_DIR / f"{stem}_{int(time.time())}{suffix}"

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        content = await file.read()
        async with aiofiles.open(dest, "wb") as f:
            await f.write(content)
    except Exception as e:
        logger.exception("Failed to save video %s", safe_name)
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "filename": dest.name,
        "size_mb": round(dest.stat().st_size / 1024 / 1024, 2),
        "play_url": f"/api/sources/video/{dest.name}",
    }


@router.get("/videos")
async def list_videos():
    """List downloaded videos available for playback and inference."""
    videos = []
    for p in sorted(VIDEOS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            st = p.stat()
            videos.append(
                {
                    "filename": p.name,
                    "size": st.st_size,
                    "size_mb": round(st.st_size / 1024 / 1024, 2),
                    "modified_at": st.st_mtime,
                    "play_url": f"/api/sources/video/{p.name}",
                }
            )
    return {"videos": videos, "count": len(videos)}


@router.delete("/videos/{filename}")
async def delete_video(
    filename: str,
    delete_frames: bool = Query(False, description="Also delete extracted annotation frames from this video"),
):
    """Remove a saved recording from the video library."""
    path = VIDEOS_DIR / Path(filename).name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {filename}")
    if path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a video file")

    frames_deleted = _delete_extracted_frames_for_stem(path.stem) if delete_frames else 0
    path.unlink(missing_ok=True)
    return {"deleted": path.name, "frames_deleted": frames_deleted}


@router.get("/video/{filename}")
async def serve_video(filename: str):
    """Serve a downloaded video file (supports browser playback)."""
    path = VIDEOS_DIR / Path(filename).name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {filename}")
    if path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not a video file")
    media = "video/mp4" if path.suffix.lower() == ".mp4" else "application/octet-stream"
    return FileResponse(str(path), media_type=media)


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
