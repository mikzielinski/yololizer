import json
import logging
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import CLASSES_FILE, LABELS_DIR, UPLOADS_DIR, FRAMES_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/labels", tags=["labels"])


class BBox(BaseModel):
    """YOLO bbox: center x,y and width,height, all normalized 0–1."""

    cx: float
    cy: float
    w: float
    h: float


class Annotation(BaseModel):
    class_id: int
    type: str = "bbox"
    bbox: Optional[BBox] = None
    polygon: Optional[List[List[float]]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "class_id": 0,
                    "type": "bbox",
                    "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.3},
                }
            ]
        }
    }


class AnnotationSet(BaseModel):
    annotations: List[Annotation]


class ClassesBody(BaseModel):
    classes: List[str]


def _label_path(image_name: str) -> Path:
    return LABELS_DIR / (Path(image_name).stem + ".txt")


def _parse_label_file(path: Path) -> List[Annotation]:
    """Parse a YOLO-format label file into Annotation objects."""
    annotations = []
    if not path.exists():
        return annotations

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]

            # Detect polygon vs bbox by count: bbox = 4 values, polygon = even >= 6
            if len(values) == 4:
                cx, cy, w, h = values
                annotations.append(
                    Annotation(
                        class_id=class_id,
                        type="bbox",
                        bbox=BBox(cx=cx, cy=cy, w=w, h=h),
                    )
                )
            elif len(values) >= 6 and len(values) % 2 == 0:
                points = [[values[i], values[i + 1]] for i in range(0, len(values), 2)]
                annotations.append(
                    Annotation(
                        class_id=class_id,
                        type="polygon",
                        polygon=points,
                    )
                )

    return annotations


def _serialize_annotation(ann: Annotation) -> str:
    """Serialize an Annotation to a YOLO-format line."""
    if ann.type == "bbox" and ann.bbox:
        b = ann.bbox
        return f"{ann.class_id} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}"
    elif ann.type == "polygon" and ann.polygon:
        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in ann.polygon)
        return f"{ann.class_id} {flat}"
    else:
        raise ValueError(f"Cannot serialize annotation: {ann}")


@router.get("/classes")
async def get_classes():
    """Return the list of class names."""
    if not CLASSES_FILE.exists():
        return {"classes": []}
    async with aiofiles.open(CLASSES_FILE, "r") as f:
        content = await f.read()
    classes = [line.strip() for line in content.splitlines() if line.strip()]
    return {"classes": classes}


@router.post("/classes")
async def save_classes(body: ClassesBody):
    """Save the list of class names."""
    classes = body.classes
    content = "\n".join(str(c) for c in classes)
    async with aiofiles.open(CLASSES_FILE, "w") as f:
        await f.write(content)
    return {"classes": classes, "saved": True}


@router.get("/{image_name}")
async def get_annotations(image_name: str):
    """Get annotations for an image."""
    label_path = _label_path(image_name)
    annotations = _parse_label_file(label_path)
    return {
        "image_name": image_name,
        "annotations": [a.model_dump() for a in annotations],
    }


@router.post("/{image_name}")
async def save_annotations(image_name: str, body: AnnotationSet):
    """Save annotations for an image (overwrites existing label file)."""
    label_path = _label_path(image_name)
    lines = []
    for ann in body.annotations:
        try:
            lines.append(_serialize_annotation(ann))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async with aiofiles.open(label_path, "w") as f:
        await f.write("\n".join(lines))
        if lines:
            await f.write("\n")

    return {
        "image_name": image_name,
        "annotations_saved": len(lines),
        "label_file": str(label_path.name),
    }
