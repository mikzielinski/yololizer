"""OpenAPI / Swagger metadata for YOLOlizer (single source of truth)."""

OPENAPI_TAGS = [
    {
        "name": "health",
        "description": "Service liveness check.",
    },
    {
        "name": "sources",
        "description": (
            "Upload images, extract frames from video, download YouTube, "
            "manage the **video library**, and webcam capture."
        ),
    },
    {
        "name": "labels",
        "description": "YOLO-format annotations (bbox / polygon) and class names.",
    },
    {
        "name": "training",
        "description": "Start/stop YOLO training, poll status, stream logs (SSE).",
    },
    {
        "name": "models",
        "description": "Trained runs under `runs/` — list, download `best.pt`, metrics CSV.",
    },
    {
        "name": "inference",
        "description": (
            "Run uploaded `.pt` models on images or videos. "
            "Video jobs are async — poll status until `done`."
        ),
    },
    {
        "name": "parcel",
        "description": (
            "Parcel capture: button time on film → conveyor offset → "
            "detect top object clusters in a time window."
        ),
    },
]

OPENAPI_DESCRIPTION = """
# YOLOlizer API

Local web service for **YOLO training**, **annotation**, **inference**, and **parcel capture**.

## Quick start

| Step | Action |
|------|--------|
| 1 | `GET /health` — verify API is up |
| 2 | Upload data via **sources** (images, YouTube, video library) |
| 3 | Annotate via **labels** |
| 4 | **training** → poll `/status`, stream `/logs` (SSE) |
| 5 | **inference** or **parcel** on videos |

## Interactive docs

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **OpenAPI JSON:** `/openapi.json`

## Background jobs

Several endpoints return `{ "job_id": "...", "status": "started" }`. Poll the matching status route until `status` is `done`, `error`, or `cancelled`:

| Feature | Start | Status | Cancel |
|---------|-------|--------|--------|
| YouTube download | `POST /api/sources/youtube` | `GET .../youtube/status/{job_id}` | `POST .../youtube/cancel/{job_id}` |
| Video inference | `POST /api/infer/video` | `GET .../video/status/{job_id}` | `POST .../video/cancel/{job_id}` |
| Parcel analyze | `POST /api/parcel/analyze` | `GET .../status/{job_id}` | `POST .../cancel/{job_id}` |

## Parcel timing

- `travel_real_sec = distance_m / speed_m_per_s`
- `travel_film_sec = travel_real_sec × real_to_film_ratio`
- `arrival_film_sec = button_film_sec + travel_film_sec`

`real_to_film_ratio`: seconds on **film** per 1 second **real** (e.g. `0.5` → 1 s real = 0.5 s on film).

## CORS

All origins allowed — the GitHub Pages UI can call a local API.
"""

OPENAPI_SERVERS = [
    {"url": "http://127.0.0.1:8001", "description": "Docker Compose (host port 8001)"},
    {"url": "http://127.0.0.1:8000", "description": "python run.py (default uvicorn port)"},
]

# (method_lower, path) → summary & description
OPERATION_DOCS: dict[tuple[str, str], dict] = {
    ("get", "/health"): {
        "summary": "Health check",
        "description": "Returns `{\"status\": \"ok\"}` when the API is running.",
    },
    # --- sources ---
    ("post", "/api/sources/images"): {
        "summary": "Upload images",
        "description": "Multipart upload of one or more images into `data/uploads/`.",
    },
    ("get", "/api/sources/images"): {
        "summary": "List images",
        "description": "All images in uploads and extracted frames (`data/frames/`).",
    },
    ("delete", "/api/sources/images"): {
        "summary": "Delete image (query)",
        "description": "Remove one upload or extracted frame. Pass `filename` query param (preferred for special characters).",
    },
    ("delete", "/api/sources/images/{filename}"): {
        "summary": "Delete image (path)",
        "description": "Remove one upload or extracted frame from Data Sources (`data/uploads/` or `data/frames/`).",
    },
    ("get", "/api/sources/image/{filename}"): {
        "summary": "Serve image",
        "description": "JPEG/PNG file by filename (searches uploads and frames).",
    },
    ("post", "/api/sources/video"): {
        "summary": "Upload video & extract frames",
        "description": "Upload a video file; extract every Nth frame to `data/frames/` for annotation. Does not add to video library.",
    },
    ("post", "/api/sources/youtube"): {
        "summary": "Download YouTube video",
        "description": (
            "**mode:** `watch` (save to library), `extract` (frames only), `both`. "
            "Returns `job_id` — poll `/youtube/status/{job_id}`."
        ),
    },
    ("get", "/api/sources/youtube/status/{job_id}"): {
        "summary": "YouTube job status",
        "description": "Fields: `status`, `progress` (0–1), `message`, `result` when done.",
    },
    ("post", "/api/sources/youtube/cancel/{job_id}"): {
        "summary": "Cancel YouTube job",
        "description": "Best-effort cancel for queued/running download.",
    },
    ("post", "/api/sources/videos/upload"): {
        "summary": "Upload video to library",
        "description": "Save a video file to the media library (`data/uploads/videos/`) without frame extraction.",
    },
    ("get", "/api/sources/videos"): {
        "summary": "List video library",
        "description": "Saved recordings in `data/uploads/videos/` (YouTube downloads and uploads).",
    },
    ("delete", "/api/sources/videos/{filename}/frames"): {
        "summary": "Delete extracted frames",
        "description": "Remove annotation frames linked to this library video from `data/frames/`. Keeps the video file.",
    },
    ("delete", "/api/sources/videos/{filename}"): {
        "summary": "Delete recording",
        "description": "Remove from library. Set `delete_frames=true` to also remove extracted annotation frames from Data Sources.",
    },
    ("get", "/api/sources/video/{filename}"): {
        "summary": "Stream library video",
        "description": "MP4 (or other allowed format) for browser playback.",
    },
    ("get", "/api/sources/webcam/stream"): {
        "summary": "Webcam MJPEG stream",
        "description": "Live multipart JPEG stream from camera index 0.",
    },
    ("post", "/api/sources/webcam/capture"): {
        "summary": "Capture webcam frame",
        "description": "Save a single JPEG snapshot to uploads.",
    },
    # --- labels ---
    ("get", "/api/labels/classes"): {
        "summary": "List class names",
        "description": "Names from `data/classes.txt` used for training and UI.",
    },
    ("post", "/api/labels/classes"): {
        "summary": "Save class names",
        "description": "Replace entire class list (order = class id).",
    },
    ("get", "/api/labels/{image_name}"): {
        "summary": "Get annotations",
        "description": "YOLO labels for one image (bbox or polygon per line).",
    },
    ("post", "/api/labels/{image_name}"): {
        "summary": "Save annotations",
        "description": "Overwrite `.txt` label file for the image stem.",
    },
    # --- training ---
    ("post", "/api/training/start"): {
        "summary": "Start training",
        "description": "Build dataset from labeled images, train YOLO in background. Returns immediately.",
    },
    ("get", "/api/training/status"): {
        "summary": "Training status",
        "description": "`status`: idle | preparing | training | done | error | stopped",
    },
    ("post", "/api/training/stop"): {
        "summary": "Stop training",
        "description": "Request graceful stop (best-effort).",
    },
    ("get", "/api/training/logs"): {
        "summary": "Training logs (SSE)",
        "description": "Server-Sent Events stream of stdout during training.",
    },
    # --- models (runs) ---
    ("get", "/api/models"): {
        "summary": "List training runs",
        "description": "Subdirectories of `runs/` with optional mAP from `results.csv`.",
    },
    ("get", "/api/models/download/{run_name}"): {
        "summary": "Download best.pt",
        "description": "Weights file for a completed run.",
    },
    ("get", "/api/models/{run_name}/metrics"): {
        "summary": "Training metrics CSV",
        "description": "Parsed `results.csv` as JSON rows.",
    },
    # --- inference ---
    ("post", "/api/infer/upload-model"): {
        "summary": "Upload .pt model",
        "description": "Store a custom weights file for inference / parcel.",
    },
    ("get", "/api/infer/models"): {
        "summary": "List inference models",
        "description": "All `.pt` files in `data/inference/models/`.",
    },
    ("delete", "/api/infer/models/{filename}"): {
        "summary": "Delete inference model",
        "description": "Remove an uploaded `.pt` file.",
    },
    ("post", "/api/infer/image"): {
        "summary": "Infer on image",
        "description": "Synchronous detection; returns annotated JPEG URL and detection list.",
    },
    ("post", "/api/infer/video"): {
        "summary": "Infer on video (async)",
        "description": (
            "Provide `file` **or** `source_filename` (library video). "
            "Optional overlays: `show_speed`, `show_class_overlay`, `show_class_total` (0/1). "
            "Poll `/video/status/{job_id}`."
        ),
    },
    ("get", "/api/infer/video/status/{job_id}"): {
        "summary": "Video inference job status",
        "description": "When `done`, `result` contains `result_id`, video URL, detection report.",
    },
    ("post", "/api/infer/video/cancel/{job_id}"): {
        "summary": "Cancel video inference",
        "description": "Cancel queued/running inference job.",
    },
    ("get", "/api/infer/results"): {
        "summary": "List past results",
        "description": "Saved images and videos under `data/inference/results/`.",
    },
    ("get", "/api/infer/result/{filename}"): {
        "summary": "Download result file",
        "description": "Annotated `.jpg` or `.mp4` by filename.",
    },
    ("get", "/api/infer/result/{result_id}/report"): {
        "summary": "Video detection report",
        "description": "JSON stats: per-frame detections, class totals, speeds (video results only).",
    },
    ("get", "/api/infer/result/{result_id}/frame/{frame_index}"): {
        "summary": "Extract result video frame",
        "description": "JPEG from annotated result video; optional bbox query params to highlight.",
    },
    ("post", "/api/infer/results/{result_id}/transcode"): {
        "summary": "Transcode result for browser",
        "description": "Re-encode MP4 to H.264 + faststart if playback fails.",
    },
    ("delete", "/api/infer/results/{filename}"): {
        "summary": "Delete result",
        "description": "Remove result media and associated report JSON.",
    },
    # --- parcel ---
    ("post", "/api/parcel/preview-timing"): {
        "summary": "Preview parcel timing",
        "description": "Compute arrival window on film timeline **without** running YOLO.",
    },
    ("post", "/api/parcel/analyze"): {
        "summary": "Analyze parcel window (async)",
        "description": (
            "Detect objects in time window around expected arrival. "
            "Video: `file`, `source_filename`, or `result_video_id`. Poll `/status/{job_id}`."
        ),
    },
    ("get", "/api/parcel/status/{job_id}"): {
        "summary": "Parcel job status",
        "description": "When `done`, `result` has `capture_id`, `clusters`, `clip_url`, timing fields.",
    },
    ("post", "/api/parcel/cancel/{job_id}"): {
        "summary": "Cancel parcel job",
        "description": "Cancel queued/running parcel analysis.",
    },
    ("get", "/api/parcel/meta/{capture_id}"): {
        "summary": "Parcel capture metadata",
        "description": "Saved clusters, fps, video path for a capture id.",
    },
    ("get", "/api/parcel/clip/{capture_id}.mp4"): {
        "summary": "Download parcel clip",
        "description": "Short MP4 cut around the analysis window.",
    },
    ("get", "/api/parcel/frame/{capture_id}/{frame_index}"): {
        "summary": "Parcel cluster frame",
        "description": "JPEG from source video; optional `x1,y1,x2,y2` to draw bbox. `download=true` for attachment.",
    },
    ("get", "/api/parcel/frames-zip/{capture_id}"): {
        "summary": "Download top frames ZIP",
        "description": (
            "ZIP of best-frame JPEGs for each saved cluster. "
            "Query `sort`: score_desc, time_asc, time_desc, conf_desc, conf_asc, "
            "detections_desc, detections_asc, frame_span_desc, rank."
        ),
    },
}


def apply_openapi(schema: dict) -> None:
    """Merge summaries/descriptions into generated OpenAPI schema."""
    schema["servers"] = OPENAPI_SERVERS

    paths = schema.get("paths", {})
    for path, path_item in paths.items():
        for method, operation in path_item.items():
            if method.startswith("x-") or not isinstance(operation, dict):
                continue
            key = (method.lower(), path)
            docs = OPERATION_DOCS.get(key)
            if not docs:
                continue
            if docs.get("summary"):
                operation["summary"] = docs["summary"]
            if docs.get("description"):
                operation["description"] = docs["description"].strip()
