"""Nested bounding-box drawing when detections overlap (readable multi-class)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2

from backend.infer_tracking import _iou

_BOX_INSET_PX = 6
_GROUP_IOU = 0.25


def _box_area(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(0, x2 - x1) * max(0, y2 - y1)


def _group_detection_boxes(
    items: List[dict],
    track_by_index: Dict[Tuple[int, int], Optional[int]],
) -> List[List[dict]]:
    """Cluster overlapping boxes or boxes sharing a track_id."""
    n = len(items)
    if n <= 1:
        return [items] if items else []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            ti = track_by_index.get((items[i]["ri"], items[i]["bi"]))
            tj = track_by_index.get((items[j]["ri"], items[j]["bi"]))
            if ti is not None and ti == tj:
                union(i, j)
                continue
            if _iou(items[i]["bbox"], items[j]["bbox"]) >= _GROUP_IOU:
                union(i, j)

    groups: Dict[int, List[dict]] = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)
    return list(groups.values())


def _inset_box(x1: int, y1: int, x2: int, y2: int, inset: int) -> Tuple[int, int, int, int]:
    return x1 + inset, y1 + inset, x2 - inset, y2 - inset


def _draw_label(
    img,
    label: str,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    *,
    level: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5 if level else 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    # Stagger labels above the box so they do not cover each other
    lift = 6 + level * (th + 5)
    ly1 = max(0, y1 - lift - th - 4)
    cv2.rectangle(img, (x1, ly1), (x1 + tw + 4, ly1 + th + 4), color, -1)
    cv2.putText(
        img,
        label,
        (x1 + 2, ly1 + th + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_nested_detection_boxes(
    img,
    box_items: List[dict],
    track_by_index: Optional[Dict[Tuple[int, int], Optional[int]]] = None,
) -> None:
    """Draw detection boxes; overlapping groups become concentric frames."""
    if not box_items:
        return

    track_by_index = track_by_index or {}
    groups = _group_detection_boxes(box_items, track_by_index)

    for group in groups:
        if len(group) == 1:
            item = group[0]
            x1, y1, x2, y2 = item["xyxy"]
            color = item["color"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            _draw_label(img, item["label"], x1, y1, color, level=0)
            continue

        # Largest box outermost, each inner level inset
        sorted_group = sorted(
            group,
            key=lambda it: -_box_area(*it["xyxy"]),
        )
        for level, item in enumerate(sorted_group):
            x1, y1, x2, y2 = item["xyxy"]
            inset = level * _BOX_INSET_PX
            ix1, iy1, ix2, iy2 = _inset_box(x1, y1, x2, y2, inset)
            if ix2 - ix1 < 12 or iy2 - iy1 < 12:
                break
            color = item["color"]
            thickness = 2 if level == 0 else 1
            cv2.rectangle(img, (ix1, iy1), (ix2, iy2), color, thickness, cv2.LINE_AA)
            _draw_label(img, item["label"], ix1, iy1, color, level=level)


def track_id_map_from_enriched(
    enriched: Optional[List[dict]],
    results,
) -> Dict[Tuple[int, int], Optional[int]]:
    """Map (result_idx, box_idx) → track_id using IoU match."""
    if not enriched:
        return {}
    out: Dict[Tuple[int, int], Optional[int]] = {}
    for ri, result in enumerate(results):
        if result.boxes is None:
            continue
        for bi, box in enumerate(result.boxes):
            bbox = [round(v, 1) for v in box.xyxy[0].tolist()]
            best_tid, best_score = None, 0.2
            for det in enriched:
                score = _iou(bbox, det["bbox"])
                if score > best_score:
                    best_score = score
                    best_tid = det.get("track_id")
            out[(ri, bi)] = best_tid
    return out
