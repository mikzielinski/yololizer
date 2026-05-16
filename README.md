# YOLOlizer

A local web service for training and running YOLO models — upload images, annotate bounding boxes/polygons, train, and run inference on images or video. No cloud dependencies.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/mikzielinski/yololizer)

---

## Run options

### Option 1 — GitHub Codespaces (no install)

Click the badge above, or on the GitHub repo page go to **Code → Codespaces → Create codespace**.

The environment installs all dependencies automatically and starts the server. When the port-forwarding notification appears, click **Open in Browser** — YOLOlizer is ready.

> Note: Codespaces machines don't have a GPU by default, so training runs on CPU. For GPU training, use the Docker or local Python setup on a machine with a CUDA GPU.

---

### Option 2 — Docker (recommended for local use)

```bash
git clone https://github.com/mikzielinski/yololizer.git
cd yololizer
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000).

Your annotations, uploads, and trained models are persisted in `./data/` and `./runs/` on the host — they survive container restarts.

**GPU acceleration** (NVIDIA, requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):
Uncomment the `deploy.resources` block in `docker-compose.yml`, then re-run `docker compose up`.

---

### Option 3 — Local Python

```bash
git clone https://github.com/mikzielinski/yololizer.git
cd yololizer
pip install -r requirements.txt
python run.py
```

Open [http://localhost:8000](http://localhost:8000).

---

## Features

| Tab | What it does |
|-----|-------------|
| **Data Sources** | Upload images, extract frames from video, capture from webcam |
| **Annotate** | Draw bounding boxes or polygons, assign class labels, auto-save in YOLO format |
| **Train** | Pick task (detect / segment / classify / pose), model size, epochs, batch, LR; stream real-time logs and live metrics |
| **Models** | List training runs, download `best.pt`, view mAP chart |
| **Infer** | Upload any `.pt` model, run it on an image or video, download the annotated result |

---

## Project structure

```
yololizer/
  .devcontainer/
    devcontainer.json   # GitHub Codespaces config
  backend/
    main.py             # FastAPI application
    config.py           # Paths and constants
    routers/
      sources.py        # Image / video / webcam endpoints
      labels.py         # Annotation CRUD
      training.py       # Training control + SSE log stream
      models.py         # Model listing and download
      inference.py      # Run .pt model on image or video
  frontend/
    index.html          # Single-page application (no build step)
  data/                 # Created automatically
    uploads/            # Uploaded images and extracted frames
    labels/             # YOLO .txt label files
    dataset/            # Prepared train/val split
    inference/          # Uploaded models and result files
  runs/                 # YOLO training outputs (best.pt, results.csv, …)
  Dockerfile
  docker-compose.yml
  requirements.txt
  run.py
```

## Label format

Standard YOLO `.txt` (one file per image, same stem):
- Bounding box: `<class_id> <cx> <cy> <w> <h>` (normalized 0–1)
- Segmentation polygon: `<class_id> <x1> <y1> <x2> <y2> …` (normalized 0–1)

## Notes

- Webcam support requires a physical camera at index 0 and is not available inside Docker or Codespaces.
- Training runs in a background thread; the API stays responsive during training.
- The SSE log stream sends keepalive comments every 15 seconds to prevent proxy timeouts.
