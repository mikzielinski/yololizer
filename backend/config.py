from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
FRAMES_DIR = UPLOADS_DIR / "frames"
VIDEOS_DIR = UPLOADS_DIR / "videos"
LABELS_DIR = DATA_DIR / "labels"
MODELS_DIR = DATA_DIR / "models"
DATASET_DIR = DATA_DIR / "dataset"
RUNS_DIR = BASE_DIR / "runs"
PARCEL_DIR = DATA_DIR / "parcel"
PARCEL_CLIPS_DIR = PARCEL_DIR / "clips"
PARCEL_META_DIR = PARCEL_DIR / "meta"
INFER_DIR = DATA_DIR / "inference"
INFER_MODELS_DIR = INFER_DIR / "models"
CLASSES_FILE = LABELS_DIR / "classes.txt"

# Ensure directories exist at import time
for _dir in [
    UPLOADS_DIR, FRAMES_DIR, VIDEOS_DIR, LABELS_DIR, MODELS_DIR, DATASET_DIR, RUNS_DIR,
    PARCEL_DIR, PARCEL_CLIPS_DIR, PARCEL_META_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

DEFAULT_FRAME_INTERVAL = 30  # extract every Nth frame from video
WEBCAM_INDEX = 0
