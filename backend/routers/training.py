import asyncio
import csv
import io
import logging
import os
import queue
import random
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import (
    CLASSES_FILE,
    DATASET_DIR,
    LABELS_DIR,
    RUNS_DIR,
    UPLOADS_DIR,
    FRAMES_DIR,
    ALLOWED_IMAGE_EXTENSIONS,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])


class TrainingConfig(BaseModel):
    task: str = Field("detect", pattern="^(detect|segment|classify|pose)$")
    model_size: str = Field("n", pattern="^(n|s|m|l|x)$")
    epochs: int = Field(50, ge=1, le=1000)
    imgsz: int = Field(640, ge=32, le=1920)
    batch: int = Field(16, ge=-1)
    lr0: float = Field(0.01, gt=0, le=1.0)
    project_name: str = Field("yololizer_run")


# Global training state
_training_state = {
    "status": "idle",  # idle | preparing | training | done | error | stopped
    "config": None,
    "start_time": None,
    "end_time": None,
    "error": None,
    "run_dir": None,
}
_log_queue: queue.Queue = queue.Queue(maxsize=10000)
_training_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _enqueue(msg: str):
    try:
        _log_queue.put_nowait(msg)
    except queue.Full:
        pass


class _StreamCapture(io.TextIOBase):
    """Intercept writes to stdout/stderr and put them in the log queue."""

    def __init__(self, original):
        self._original = original

    def write(self, s: str) -> int:
        if s and s.strip():
            _enqueue(s.rstrip("\n"))
        return self._original.write(s)

    def flush(self):
        self._original.flush()


def _prepare_dataset(config: TrainingConfig):
    """Build YOLO dataset directory from uploaded images + label files."""
    _enqueue("[YOLOlizer] Preparing dataset...")

    # Gather all images that have a corresponding label file
    all_images = []
    for ext in ALLOWED_IMAGE_EXTENSIONS:
        all_images.extend(UPLOADS_DIR.glob(f"*{ext}"))
        all_images.extend(UPLOADS_DIR.glob(f"*{ext.upper()}"))
        all_images.extend(FRAMES_DIR.glob(f"*{ext}"))
        all_images.extend(FRAMES_DIR.glob(f"*{ext.upper()}"))

    labeled = [p for p in all_images if (LABELS_DIR / (p.stem + ".txt")).exists()]

    if not labeled:
        raise ValueError(
            "No labeled images found. Please annotate at least one image before training."
        )

    _enqueue(f"[YOLOlizer] Found {len(labeled)} labeled images")

    # Read classes
    if not CLASSES_FILE.exists():
        raise ValueError("No classes defined. Please add class names in the Annotate tab.")
    with open(CLASSES_FILE) as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError("Class list is empty.")

    _enqueue(f"[YOLOlizer] Classes: {classes}")

    # Shuffle and split 80/20
    random.shuffle(labeled)
    split = max(1, int(len(labeled) * 0.8))
    train_imgs = labeled[:split]
    val_imgs = labeled[split:] if split < len(labeled) else labeled[:1]  # at least 1 val image

    _enqueue(f"[YOLOlizer] Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    # Build dataset dirs
    for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            shutil.copy2(img_path, img_dir / img_path.name)
            lbl_src = LABELS_DIR / (img_path.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dir / lbl_src.name)

    # Write dataset.yaml
    yaml_content = {
        "path": str(DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(classes),
        "names": classes,
    }
    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    _enqueue(f"[YOLOlizer] Dataset YAML written: {yaml_path}")
    return str(yaml_path)


def _run_training(config: TrainingConfig):
    """Run in background thread."""
    global _training_state

    _training_state["status"] = "preparing"
    _training_state["start_time"] = time.time()
    _stop_event.clear()

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    try:
        yaml_path = _prepare_dataset(config)

        _training_state["status"] = "training"
        _enqueue("[YOLOlizer] Starting YOLO training...")

        # Redirect stdout/stderr to capture ultralytics output
        sys.stdout = _StreamCapture(orig_stdout)
        sys.stderr = _StreamCapture(orig_stderr)

        from ultralytics import YOLO

        model_name = f"yolov8{config.model_size}.pt"
        model = YOLO(model_name)

        results = model.train(
            data=yaml_path,
            task=config.task,
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            lr0=config.lr0,
            project=str(RUNS_DIR),
            name=config.project_name,
            exist_ok=True,
            verbose=True,
        )

        run_dir = RUNS_DIR / config.project_name
        _training_state["run_dir"] = str(run_dir)
        _training_state["status"] = "done"
        _enqueue(f"[YOLOlizer] Training complete! Results in: {run_dir}")

    except Exception as e:
        logger.exception("Training failed")
        _training_state["status"] = "error"
        _training_state["error"] = str(e)
        _enqueue(f"[YOLOlizer] ERROR: {e}")
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        _training_state["end_time"] = time.time()


@router.post("/start")
async def start_training(config: TrainingConfig):
    """Start YOLO training in a background thread."""
    global _training_thread, _training_state

    if _training_state["status"] in ("preparing", "training"):
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Reset state
    _training_state = {
        "status": "idle",
        "config": config.model_dump(),
        "start_time": None,
        "end_time": None,
        "error": None,
        "run_dir": None,
    }

    # Clear old logs
    while not _log_queue.empty():
        try:
            _log_queue.get_nowait()
        except queue.Empty:
            break

    _training_thread = threading.Thread(
        target=_run_training, args=(config,), daemon=True, name="yolo-training"
    )
    _training_thread.start()

    return {"status": "started", "config": config.model_dump()}


@router.get("/status")
async def get_status():
    """Return current training status."""
    state = dict(_training_state)
    elapsed = None
    if state.get("start_time"):
        end = state.get("end_time") or time.time()
        elapsed = round(end - state["start_time"], 1)
    return {**state, "elapsed_seconds": elapsed}


@router.post("/stop")
async def stop_training():
    """Request training to stop (best-effort)."""
    global _training_state

    if _training_state["status"] not in ("preparing", "training"):
        raise HTTPException(status_code=400, detail="No training in progress")

    _stop_event.set()
    _training_state["status"] = "stopped"
    _enqueue("[YOLOlizer] Stop requested by user.")
    return {"status": "stop_requested"}


@router.get("/logs")
async def stream_logs():
    """SSE endpoint: stream training log lines from the queue."""

    async def _event_generator():
        last_keepalive = time.time()
        while True:
            try:
                line = _log_queue.get_nowait()
                # Escape SSE special chars
                safe = line.replace("\n", " ").replace("\r", "")
                yield f"data: {safe}\n\n"
            except queue.Empty:
                now = time.time()
                if now - last_keepalive >= 15:
                    yield ": keepalive\n\n"
                    last_keepalive = now
                await asyncio.sleep(0.1)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
