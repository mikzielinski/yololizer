"""Optional on-video overlays: class counter and speed HUD."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2

_MAX_CLASS_LINES = 8


def _panel_bg(img, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.62) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (16, 16, 16), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def counts_from_results(results) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in results:
        if result.boxes is None:
            continue
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = names.get(cls_id, str(cls_id))
            counts[name] = counts.get(name, 0) + 1
    return counts


def _draw_class_panel(
    img,
    class_counts: Dict[str, int],
    *,
    title: str,
    anchor: str,
    title_color: tuple[int, int, int] = (180, 180, 180),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    lines = _class_count_lines(class_counts)
    if not lines:
        return
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    line_h = 20
    pad = 8
    max_tw = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
    panel_w = max_tw + pad * 2
    header_h = 22
    panel_h = header_h + len(lines) * line_h + pad * 2
    margin = 10
    if anchor == "top_right":
        x2, y1 = w - margin, margin
        x1, y2 = x2 - panel_w, y1 + panel_h
    else:
        x2, y2 = w - margin, h - margin
        x1, y1 = x2 - panel_w, y2 - panel_h
    _panel_bg(img, x1, y1, x2, y2)
    cv2.putText(img, title, (x1 + pad, y1 + pad + 14), font, 0.45, title_color, 1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        y = y1 + pad + header_h + 6 + i * line_h
        cv2.putText(img, line, (x1 + pad, y), font, scale, text_color, thickness, cv2.LINE_AA)


def _class_count_lines(class_counts: Dict[str, int]) -> List[str]:
    if not class_counts:
        return []
    items = sorted(class_counts.items(), key=lambda x: (-x[1], x[0]))
    lines = [f"{name}: {n}" for name, n in items[:_MAX_CLASS_LINES]]
    if len(items) > _MAX_CLASS_LINES:
        rest = sum(n for _, n in items[_MAX_CLASS_LINES:])
        lines.append(f"+{len(items) - _MAX_CLASS_LINES} more ({rest})")
    return lines


def _measure_section(
    title: str,
    lines: List[str],
    *,
    font_scale: float = 0.48,
    line_h: int = 18,
) -> Tuple[int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    pad = 8
    header_h = 20
    max_tw = cv2.getTextSize(title, font, 0.45, thickness)[0][0]
    for line in lines:
        max_tw = max(max_tw, cv2.getTextSize(line, font, font_scale, thickness)[0][0])
    w = max_tw + pad * 2
    h = header_h + len(lines) * line_h + pad + (6 if lines else 0)
    return w, h


def draw_video_stats_hud(
    img,
    *,
    frame_counts: Optional[Dict[str, int]] = None,
    total_counts: Optional[Dict[str, int]] = None,
    avg_speed_px_s: Optional[float] = None,
    active_tracks: int = 0,
) -> None:
    """Single nested panel (top-right): speed, frame counts, total — all visible."""
    sections: List[dict] = []

    if avg_speed_px_s is not None:
        speed_lines = [f"Avg: {avg_speed_px_s:.0f} px/s"]
        if active_tracks:
            speed_lines.append(f"Tracks: {active_tracks}")
        sections.append(
            {
                "title": "Speed",
                "lines": speed_lines,
                "title_color": (100, 220, 255),
                "text_color": (210, 245, 255),
            }
        )

    frame_lines = _class_count_lines(frame_counts or {})
    if frame_lines:
        sections.append(
            {
                "title": "Frame",
                "lines": frame_lines,
                "title_color": (180, 180, 200),
                "text_color": (255, 255, 255),
            }
        )

    total_lines = _class_count_lines(total_counts or {})
    if total_lines:
        sections.append(
            {
                "title": "Total (objects)",
                "lines": total_lines,
                "title_color": (120, 200, 255),
                "text_color": (255, 235, 200),
            }
        )

    if not sections:
        return

    h_img, w_img = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = 10
    outer_pad = 10
    section_gap = 6
    inset = 5

    outer_title = "Stats"
    section_sizes = [_measure_section(s["title"], s["lines"]) for s in sections]
    inner_w = max(w for w, _ in section_sizes)
    inner_h = sum(sh for _, sh in section_sizes) + section_gap * (len(sections) - 1)
    header_outer = 26
    panel_w = inner_w + outer_pad * 2 + inset * 2
    panel_h = header_outer + inner_h + outer_pad * 2 + inset * 2

    x2 = w_img - margin
    y1 = margin
    x1 = x2 - panel_w
    y2 = y1 + panel_h

    _panel_bg(img, x1, y1, x2, y2, alpha=0.68)
    cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 110), 1, cv2.LINE_AA)

    cx = x1 + outer_pad
    cy = y1 + outer_pad
    cv2.putText(
        img,
        outer_title,
        (cx, y1 + outer_pad + 16),
        font,
        0.5,
        (220, 220, 230),
        1,
        cv2.LINE_AA,
    )

    inner_x1 = x1 + outer_pad + inset
    inner_x2 = x2 - outer_pad - inset
    inner_y1 = y1 + header_outer + inset
    inner_y2 = y2 - outer_pad - inset
    _panel_bg(img, inner_x1, inner_y1, inner_x2, inner_y2, alpha=0.45)
    cv2.rectangle(img, (inner_x1, inner_y1), (inner_x2, inner_y2), (70, 70, 85), 1, cv2.LINE_AA)

    y = inner_y1 + 8
    for i, (sec, (sec_w, sec_h)) in enumerate(zip(sections, section_sizes)):
        if i > 0:
            y += section_gap // 2
            cv2.line(img, (inner_x1 + 4, y), (inner_x2 - 4, y), (55, 55, 70), 1, cv2.LINE_AA)
            y += section_gap // 2

        box_x1 = inner_x1 + 4
        box_x2 = inner_x2 - 4
        box_y2 = y + sec_h
        cv2.rectangle(img, (box_x1, y), (box_x2, box_y2), (45, 45, 58), 1, cv2.LINE_AA)

        pad = 8
        cv2.putText(
            img,
            sec["title"],
            (box_x1 + pad, y + 14),
            font,
            0.45,
            sec["title_color"],
            1,
            cv2.LINE_AA,
        )
        line_y = y + 22
        for line in sec["lines"]:
            cv2.putText(
                img,
                line,
                (box_x1 + pad, line_y),
                font,
                0.48,
                sec["text_color"],
                1,
                cv2.LINE_AA,
            )
            line_y += 18
        y = box_y2


def draw_class_counter(img, class_counts: Dict[str, int]) -> None:
    _draw_class_panel(img, class_counts, title="Frame", anchor="top_right")


def draw_class_total_counter(img, class_counts: Dict[str, int]) -> None:
    _draw_class_panel(
        img,
        class_counts,
        title="Total (objects)",
        anchor="bottom_right",
        title_color=(255, 200, 120),
        text_color=(255, 230, 180),
    )


def draw_speed_hud(img, avg_speed_px_s: float, active_tracks: int = 0) -> None:
    h, w = img.shape[:2]
    line1 = f"Avg speed: {avg_speed_px_s:.0f} px/s"
    line2 = f"Tracks: {active_tracks}" if active_tracks else ""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    pad = 10
    lines = [line1] + ([line2] if line2 else [])
    max_tw = max(cv2.getTextSize(t, font, scale, thickness)[0][0] for t in lines)
    line_h = 22
    panel_h = len(lines) * line_h + pad * 2
    panel_w = max_tw + pad * 2
    x1, y2 = 10, h - 10
    y1 = y2 - panel_h
    x2 = x1 + panel_w
    _panel_bg(img, x1, y1, x2, y2)
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (x1 + pad, y1 + pad + 16 + i * line_h),
            font,
            scale,
            (100, 220, 255),
            thickness,
            cv2.LINE_AA,
        )


def make_video_draw_fn(
    base_draw,
    *,
    show_speed: bool = False,
    show_class_overlay: bool = False,
    show_class_total: bool = False,
):
    """Build draw callable with optional speed HUD and class counters."""
    from backend.infer_tracking import ObjectSpeedTracker

    fps_ref: List[float] = [25.0]

    def init(fps: float) -> None:
        fps_ref[0] = max(float(fps), 1.0)

    def draw(
        frame,
        results,
        task: str,
        *,
        cumulative_totals: Optional[Dict[str, int]] = None,
        enriched: Optional[List[dict]] = None,
        tracker: Optional[ObjectSpeedTracker] = None,
    ):
        use_speed = show_speed and enriched is not None
        box_speeds = speeds_by_box_index(enriched or [], results) if use_speed else {}
        img = base_draw(
            frame,
            results,
            task,
            box_speeds=box_speeds if use_speed else None,
            enriched=enriched,
        )

        if show_class_overlay:
            draw_class_counter(img, counts_from_results(results))
        if show_class_total and cumulative_totals:
            draw_class_total_counter(img, cumulative_totals)
        if show_speed and tracker:
            draw_speed_hud(img, tracker.avg_speed_px_s, tracker.active_track_count)

        draw.frame_idx += 1  # type: ignore[attr-defined]
        return img

    draw.frame_idx = 0  # type: ignore[attr-defined]
    draw.init = init
    return draw


def speeds_by_box_index(enriched: List[dict], results) -> Dict[Tuple[int, int], float]:
    """Map (result_index, box_index) -> speed_px_s; one label per track (object)."""
    from backend.infer_tracking import _iou

    if not enriched:
        return {}

    track_speed: Dict[int, float] = {}
    for det in enriched:
        tid = det.get("track_id")
        sp = det.get("speed_px_s") or 0
        if tid is None or sp <= 0:
            continue
        if det.get("show_speed_label"):
            track_speed[tid] = sp

    candidates: List[Tuple[float, int, int, int]] = []
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
            if best_tid is not None and best_tid in track_speed:
                candidates.append((best_score, ri, bi, best_tid))

    out: Dict[Tuple[int, int], float] = {}
    used_tracks: set[int] = set()
    for score, ri, bi, tid in sorted(candidates, key=lambda x: -x[0]):
        if tid in used_tracks:
            continue
        used_tracks.add(tid)
        out[(ri, bi)] = track_speed[tid]
    return out
