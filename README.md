# YOLOlizer

A local web service for training and running YOLO models — upload images or video, annotate bounding boxes/polygons, train, and run inference. No cloud required for the API; optional static UI on GitHub Pages.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/mikzielinski/yololizer)

**Live UI (GitHub Pages):** [https://mikzielinski.github.io/yololizer/](https://mikzielinski.github.io/yololizer/) — connect to a local Docker API (see below).

---

## Run options

### Option 1 — GitHub Pages UI + local API (recommended for sharing the interface)

The frontend is deployed automatically to GitHub Pages on every push to `main`. The **API must run on your machine** (Docker or Python); Pages only hosts the HTML/JS.

1. **Enable Pages** (once): repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. **Run the API locally:**

```bash
git clone https://github.com/mikzielinski/yololizer.git
cd yololizer
docker compose up --build
```

3. Open [https://mikzielinski.github.io/yololizer/](https://mikzielinski.github.io/yololizer/).
4. In the header, set **API** to `http://127.0.0.1:8001` (default in `docs/config.js`) and click **Connect**.

> CORS is enabled on the API (`allow_origins=*`), so the Pages UI can talk to `localhost`.

**Same machine, no Pages:** open [http://localhost:8001](http://localhost:8001) after `docker compose up` (port **8001** on the host maps to 8000 in the container).

---

### Option 2 — GitHub Codespaces (no local install)

Click the badge above, or **Code → Codespaces → Create codespace**.

Dependencies install automatically and the server starts. Open the forwarded port when prompted.

> Codespaces has no GPU by default (CPU training). No webcam in cloud environments.

---

### Option 3 — Docker (full stack local)

```bash
git clone https://github.com/mikzielinski/yololizer.git
cd yololizer
docker compose up --build
```

Open [http://localhost:8001](http://localhost:8001).

Data in `./data/` and `./runs/` persists across restarts.

**GPU (NVIDIA):** install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), uncomment `deploy.resources` in `docker-compose.yml`, then `docker compose up` again.

---

### Option 4 — Local Python

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
|-----|----------------|
| **Data Sources** | Upload images/video, YouTube download, frame extraction, webcam capture |
| **Annotate** | Bounding boxes or polygons, class labels, auto-save YOLO `.txt` |
| **Train** | detect / segment / classify / pose; live logs and metrics |
| **Models** | List runs, download `best.pt`, mAP chart |
| **Infer** | Custom `.pt`, image or video inference, progress/cancel, nested bbox overlays, optional speed & class counters, detection modal with frame preview & download |

---

## Project structure

```
yololizer/
  .github/workflows/pages.yml   # Deploy UI to GitHub Pages
  .devcontainer/                # Codespaces
  backend/
    main.py                     # FastAPI + CORS + static frontend
    infer_jobs.py               # Video inference jobs
    infer_tracking.py           # Object tracking / speed
    infer_overlay.py            # On-video HUD
    infer_box_draw.py             # Nested detection boxes
    routers/                    # API routes
  frontend/
    index.html                  # Single-page app
    config.js                   # API base (empty = same origin)
  docs/                         # GitHub Pages output (index built in CI)
  scripts/build-pages.sh
  data/                         # Created at runtime (gitignored)
  runs/                         # Training outputs (gitignored)
  Dockerfile
  docker-compose.yml
  requirements.txt
  run.py
```

---

## GitHub Pages (maintainers)

- Workflow: `.github/workflows/pages.yml` runs on push to `main` / `master`.
- Build: `scripts/build-pages.sh` copies `frontend/index.html` → `docs/index.html`.
- Configure repo **Settings → Pages → Source: GitHub Actions**.
- After merge to `main`, the site is at `https://<user>.github.io/yololizer/`.

To preview Pages locally:

```bash
bash scripts/build-pages.sh
python -m http.server 8080 --directory docs
# open http://localhost:8080
```

---

## Label format

YOLO `.txt` per image (same stem):

- Box: `class_id cx cy w h` (normalized 0–1)
- Polygon: `class_id x1 y1 x2 y2 …` (normalized 0–1)

---

## Notes

- Webcam needs a local camera; not available in Docker without device passthrough or in Codespaces.
- Training runs in a background thread; API stays responsive.
- SSE log stream uses 15s keepalive comments for proxies.
- Video results are transcoded to H.264 for browser playback (`ffmpeg` in Docker image).
