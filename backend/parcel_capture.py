"""Parcel capture: button timestamp → travel offset → video window → top detection clusters."""
from __future__ import annotations

import logging
import math
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from backend.config import PARCEL_CLIPS_DIR
from backend.infer_tracking import ObjectSpeedTracker, detections_from_result

logger = logging.getLogger(__name__)


def travel_real_seconds(distance_m: float, speed_m_per_s: float) -> float:
    if speed_m_per_s <= 0:
        raise ValueError("conveyor_speed_m_per_s must be > 0")
    return distance_m / speed_m_per_s


def real_seconds_to_film(real_sec: float, real_to_film_ratio: float) -> float:
    """Map real-world seconds to video timeline (e.g. ratio 0.5 → 1s real = 0.5s on film)."""
    return real_sec * real_to_film_ratio


def compute_arrival_film_sec(
    button_film_sec: float,
    distance_m: float,
    speed_m_per_s: float,
    real_to_film_ratio: float,
) -> float:
    travel_film = real_seconds_to_film(travel_real_seconds(distance_m, speed_m_per_s), real_to_film_ratio)
    return button_film_sec + travel_film


def _cluster_score(track: Dict[str, Any], det_count: int) -> float:
    conf = float(track.get("confidence", 0))
    span = max(1, int(track.get("last_frame", 0)) - int(track.get("first_frame", 0)) + 1)
    return det_count * conf * math.sqrt(span)


def _build_clusters_from_archive(
    track_archive: Dict[int, Dict[str, Any]],
    det_counts: Dict[int, int],
    top_k: int,
) -> List[dict]:
    candidates: List[dict] = []
    for tid, tr in track_archive.items():
        first_f = int(tr.get("first_frame", tr.get("last_frame", 0)))
        last_f = int(tr.get("last_frame", first_f))
        count = det_counts.get(tid, 1)
        candidates.append(
            {
                "track_id": tid,
                "class_name": tr.get("class_name", "?"),
                "class_id": -1,
                "detection_count": count,
                "frame_start": first_f,
                "frame_end": last_f,
                "best_frame": int(tr.get("best_frame", (first_f + last_f) // 2)),
                "best_bbox": list(tr.get("bbox", [0, 0, 0, 0])),
                "mean_confidence": round(float(tr.get("confidence", 0)), 4),
                "score": round(_cluster_score(tr, count), 2),
            }
        )
    candidates.sort(key=lambda c: (-c["score"], -c["detection_count"]))
    for i, c in enumerate(candidates[:top_k]):
        c["rank"] = i + 1
    return candidates[:top_k]


def _archive_track(
    archive: Dict[int, Dict[str, Any]],
    tid: int,
    tr: Dict[str, Any],
    frame_idx: int,
    det: Optional[dict] = None,
) -> None:
    if tid not in archive:
        archive[tid] = {
            "track_id": tid,
            "class_name": tr.get("class_name", "?"),
            "bbox": list(tr.get("bbox", [0, 0, 0, 0])),
            "confidence": float(tr.get("confidence", 0)),
            "first_frame": frame_idx,
            "last_frame": frame_idx,
            "best_frame": frame_idx,
        }
    else:
        a = archive[tid]
        a["last_frame"] = max(int(a["last_frame"]), frame_idx)
        a["first_frame"] = min(int(a["first_frame"]), frame_idx)
        conf = float(tr.get("confidence", 0))
        if conf >= float(a.get("confidence", 0)):
            a["confidence"] = conf
            a["bbox"] = list(tr.get("bbox", a["bbox"]))
            a["best_frame"] = frame_idx
            a["class_name"] = tr.get("class_name", a["class_name"])


def extract_video_clip(
    video_path: Path,
    out_path: Path,
    start_sec: float,
    end_sec: float,
) -> None:
    """Cut [start_sec, end_sec] to a new mp4 (re-encode for accurate cuts)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end_sec - start_sec)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(max(0, start_sec)),
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )


def run_parcel_capture(
    video_path: Path,
    model_path: Path,
    *,
    button_film_sec: float,
    distance_m: float,
    speed_m_per_s: float,
    real_to_film_ratio: float = 0.5,
    pre_window_sec: float = 2.0,
    post_window_sec: float = 3.0,
    top_clusters: int = 4,
    conf: float = 0.25,
    iou: float = 0.45,
    frame_stride: int = 1,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Analyze video around expected parcel arrival at camera; return top-K object clusters.
    button_film_sec: button press position on the video timeline (seconds).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_film = total_frames / fps if total_frames else 0

    travel_real = travel_real_seconds(distance_m, speed_m_per_s)
    travel_film = real_seconds_to_film(travel_real, real_to_film_ratio)
    arrival_film = button_film_sec + travel_film
    win_start = max(0.0, arrival_film - pre_window_sec)
    win_end = min(duration_film if duration_film else arrival_film + post_window_sec, arrival_film + post_window_sec)
    if duration_film and win_end <= win_start:
        win_end = min(duration_film, win_start + 1.0)

    frame_start = int(win_start * fps)
    frame_end = int(win_end * fps)
    if total_frames:
        frame_end = min(total_frames - 1, frame_end)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    tracker = ObjectSpeedTracker(fps, max_gap_frames=120)
    track_archive: Dict[int, Dict[str, Any]] = {}
    det_counts: Dict[int, int] = {}
    frames_done = 0
    frames_total = max(1, (frame_end - frame_start) // max(1, frame_stride) + 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    frame_idx = frame_start
    while frame_idx <= frame_end:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - frame_start) % max(1, frame_stride) == 0:
            results = model.predict(frame, conf=conf, iou=iou, verbose=False)
            dets: List[dict] = []
            for r in results:
                dets.extend(detections_from_result(r, frame_idx))
            enriched = tracker.update(frame_idx, dets)
            for d in enriched:
                tid = d.get("track_id")
                if tid is not None:
                    det_counts[tid] = det_counts.get(tid, 0) + 1
                    tr = tracker._tracks.get(tid)
                    if tr:
                        _archive_track(track_archive, tid, tr, frame_idx, d)
            frames_done += 1
            if progress_callback:
                progress_callback(frames_done / frames_total, f"Frame {frame_idx} / {frame_end}")

        frame_idx += 1

    cap.release()

    clusters = _build_clusters_from_archive(track_archive, det_counts, top_clusters)

    capture_id = uuid.uuid4().hex[:10]
    clip_path = PARCEL_CLIPS_DIR / f"{capture_id}.mp4"
    try:
        extract_video_clip(video_path, clip_path, win_start, win_end)
        clip_url = f"/api/parcel/clip/{capture_id}.mp4"
        clip_size_mb = round(clip_path.stat().st_size / 1024 / 1024, 2)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Clip export failed: %s", e)
        clip_url = None
        clip_size_mb = 0

    return {
        "capture_id": capture_id,
        "fps": round(float(fps), 3),
        "video_duration_film_sec": round(duration_film, 3),
        "button_film_sec": round(button_film_sec, 3),
        "distance_m": distance_m,
        "speed_m_per_s": speed_m_per_s,
        "real_to_film_ratio": real_to_film_ratio,
        "travel_real_sec": round(travel_real, 3),
        "travel_film_sec": round(travel_film, 3),
        "arrival_film_sec": round(arrival_film, 3),
        "window_start_film_sec": round(win_start, 3),
        "window_end_film_sec": round(win_end, 3),
        "window_frame_start": frame_start,
        "window_frame_end": frame_end,
        "clip_url": clip_url,
        "clip_size_mb": clip_size_mb,
        "clusters": clusters,
        "tracks_found": len(det_counts),
    }
