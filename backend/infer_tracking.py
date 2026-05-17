"""Simple bbox tracking across frames to estimate object speed (px/s)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

Track = Dict[str, Any]


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detections_from_result(result, frame_idx: int) -> List[dict]:
    """Per-frame detection list for tracking."""
    if result.boxes is None:
        return []
    names = result.names
    out: List[dict] = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls[0])
        out.append(
            {
                "frame": frame_idx,
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return out


class ObjectSpeedTracker:
    """Match detections frame-to-frame; estimate speed in pixels per second."""

    def __init__(
        self,
        fps: float,
        *,
        iou_threshold: float = 0.25,
        max_gap_frames: int = 8,
        min_displacement_px: float = 2.0,
        speed_smooth: float = 0.65,
    ) -> None:
        self.fps = max(fps, 1.0)
        self.iou_threshold = iou_threshold
        self.max_gap_frames = max_gap_frames
        self.min_displacement_px = min_displacement_px
        self.speed_smooth = speed_smooth
        self._tracks: Dict[int, Track] = {}
        self._object_class: Dict[int, str] = {}  # track_id → class (one object counted once)
        self._next_id = 1
        self._frame_idx = -1
        self.avg_speed_px_s: float = 0.0
        self.active_track_count: int = 0

    def update(self, frame_idx: int, detections: List[dict]) -> List[dict]:
        """Associate detections with tracks; one speed per physical object (track_id).

        Matching uses IoU only — class labels may change on the same object.
        Multiple boxes for the same object in one frame share one track and one speed;
        only the highest-confidence box is marked to show the speed label.
        """
        self._frame_idx = frame_idx
        detections = sorted(detections, key=lambda d: -float(d.get("confidence", 0)))

        unmatched_tracks = set(self._tracks.keys())
        enriched: List[dict] = []
        speeds_this_frame: List[float] = []
        speed_shown_for_track: set[int] = set()

        for det in detections:
            bbox = det["bbox"]
            cls = det["class_name"]
            best_id: Optional[int] = None
            best_iou = self.iou_threshold
            duplicate_of_frame = False

            for tid in list(unmatched_tracks):
                tr = self._tracks[tid]
                gap = frame_idx - tr["last_frame"]
                if gap > self.max_gap_frames:
                    continue
                score = _iou(bbox, tr["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            # Same frame: overlapping box, different class → same object as earlier detection
            if best_id is None:
                for prev in enriched:
                    if _iou(bbox, prev["bbox"]) >= self.iou_threshold:
                        best_id = prev["track_id"]
                        duplicate_of_frame = True
                        break

            cx, cy = _bbox_center(bbox)
            speed_px_s = 0.0
            tid: int

            if best_id is not None and duplicate_of_frame:
                tr = self._tracks[best_id]
                speed_px_s = tr.get("speed_px_s") or 0.0
                tid = best_id
            elif best_id is not None:
                tr = self._tracks[best_id]
                unmatched_tracks.discard(best_id)
                prev_cx, prev_cy = tr["center"]
                gap = max(1, frame_idx - tr["last_frame"])
                dt = gap / self.fps
                dist = math.hypot(cx - prev_cx, cy - prev_cy)
                if dist >= self.min_displacement_px and dt > 0:
                    inst = dist / dt
                    prev_speed = tr.get("speed_px_s") or 0.0
                    speed_px_s = (
                        inst
                        if prev_speed <= 0
                        else self.speed_smooth * prev_speed + (1 - self.speed_smooth) * inst
                    )
                else:
                    speed_px_s = tr.get("speed_px_s") or 0.0

                if float(det.get("confidence", 0)) >= float(tr.get("confidence", 0)):
                    tr["class_name"] = cls
                tr.update(
                    bbox=bbox,
                    center=(cx, cy),
                    last_frame=frame_idx,
                    speed_px_s=speed_px_s,
                    confidence=max(float(tr.get("confidence", 0)), float(det.get("confidence", 0))),
                )
                tid = best_id
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "track_id": tid,
                    "class_name": cls,
                    "bbox": bbox,
                    "center": (cx, cy),
                    "last_frame": frame_idx,
                    "speed_px_s": 0.0,
                    "confidence": det.get("confidence", 0),
                }
                self._object_class[tid] = cls

            if best_id is not None and not duplicate_of_frame:
                self._object_class[tid] = self._tracks[tid]["class_name"]

            show_speed_label = False
            if speed_px_s > 0 and tid not in speed_shown_for_track:
                show_speed_label = True
                speed_shown_for_track.add(tid)
                speeds_this_frame.append(speed_px_s)

            out = dict(det)
            out["track_id"] = tid
            out["speed_px_s"] = round(speed_px_s, 1)
            out["show_speed_label"] = show_speed_label
            enriched.append(out)

        for tid in unmatched_tracks:
            tr = self._tracks[tid]
            if frame_idx - tr["last_frame"] > self.max_gap_frames:
                del self._tracks[tid]

        self.active_track_count = len({d["track_id"] for d in enriched})
        if speeds_this_frame:
            self.avg_speed_px_s = sum(speeds_this_frame) / len(speeds_this_frame)
        elif not enriched:
            self.avg_speed_px_s = 0.0

        return enriched

    def unique_objects_by_class(self) -> Dict[str, int]:
        """Count each tracked object once (by dominant class on its track)."""
        counts: Dict[str, int] = {}
        for name in self._object_class.values():
            counts[name] = counts.get(name, 0) + 1
        return counts

    def summary(self) -> dict:
        """Aggregate speeds for saved report."""
        speeds = [t["speed_px_s"] for t in self._tracks.values() if t.get("speed_px_s", 0) > 0]
        if not speeds:
            return {"avg_speed_px_s": 0.0, "max_speed_px_s": 0.0, "tracked_objects": 0}
        return {
            "avg_speed_px_s": round(sum(speeds) / len(speeds), 1),
            "max_speed_px_s": round(max(speeds), 1),
            "tracked_objects": len(speeds),
        }
