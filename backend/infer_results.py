"""Persist inference result videos, browser-friendly encoding, and detection reports."""
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import DATA_DIR

logger = logging.getLogger(__name__)

INFER_RESULTS_DIR = DATA_DIR / "inference" / "results"
INFER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def transcode_for_browser(path: Path) -> None:
    """Re-encode OpenCV mp4v output to H.264 so HTML5 video can play it."""
    if not path.exists():
        return
    tmp = path.with_suffix(".browser.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp),
            ],
            check=True,
            capture_output=True,
        )
        tmp.replace(path)
    except FileNotFoundError:
        logger.warning("ffmpeg not found — result video may not play in the browser")
    except subprocess.CalledProcessError as e:
        logger.warning("ffmpeg transcode failed: %s", e.stderr.decode(errors="replace")[:300])
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def report_path(result_id: str) -> Path:
    return INFER_RESULTS_DIR / f"{Path(result_id).name}.json"


def save_video_report(result_id: str, report: Dict[str, Any]) -> None:
    path = report_path(result_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)


def load_video_report(result_id: str) -> Optional[Dict[str, Any]]:
    path = report_path(result_id)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Could not read report %s: %s", path, e)
        return None


def delete_result_files(filename: str) -> None:
    base = INFER_RESULTS_DIR / Path(filename).name
    if base.exists():
        base.unlink()
    stem = base.stem
    json_file = INFER_RESULTS_DIR / f"{stem}.json"
    if json_file.exists():
        json_file.unlink()
