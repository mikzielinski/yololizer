# YOLOlizer

A local web service for training YOLO models — upload images, annotate bounding boxes/polygons, configure and train YOLO models, and download results. All runs locally with no cloud dependencies.

## Requirements

- Python 3.10+
- pip

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python run.py
```

Then open your browser to [http://localhost:8000](http://localhost:8000).

## Features

1. **Data Sources** — Upload images, extract frames from video, or capture from a webcam
2. **Annotate** — Draw bounding boxes or polygons, assign class labels, auto-save in YOLO format
3. **Train** — Configure task (detect/segment/classify/pose), model size, epochs, batch size, learning rate; stream real-time logs
4. **Models** — Download `best.pt`, view mAP charts

## Project Structure

```
yololizer/
  backend/
    main.py           # FastAPI application
    config.py         # Paths and constants
    routers/
      sources.py      # Image/video/webcam endpoints
      labels.py       # Annotation CRUD
      training.py     # Training control + SSE logs
      models.py       # Model listing and download
  frontend/
    index.html        # Single-page application
  data/               # Created automatically
    uploads/          # Uploaded images
    labels/           # YOLO .txt label files
    dataset/          # Prepared train/val split
    models/           # (reserved)
  runs/               # YOLO training outputs
  requirements.txt
  run.py
```

## Data Format

Labels are stored as standard YOLO `.txt` files (one file per image, same stem):
- Bounding box: `<class_id> <cx> <cy> <w> <h>` (all normalized 0-1)
- Polygon/segmentation: `<class_id> <x1> <y1> <x2> <y2> ...` (normalized)

## Notes

- Webcam support requires a physical webcam at index 0; the endpoint gracefully fails if none is available.
- Training runs in a background thread; the API remains responsive during training.
- SSE log stream sends keepalive comments every 15 seconds to prevent proxy timeouts.
