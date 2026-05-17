import re
from pathlib import Path
from typing import Callable, Optional, Tuple

import yt_dlp

_YOUTUBE_HOST_RE = re.compile(
    r"(?:youtube\.com|youtu\.be|m\.youtube\.com|music\.youtube\.com)",
    re.IGNORECASE,
)


def is_youtube_url(url: str) -> bool:
    url = (url or "").strip()
    if not url:
        return False
    return bool(_YOUTUBE_HOST_RE.search(url))


def download_youtube_video(
    url: str,
    dest_dir: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Path, str]:
    """Download a YouTube video to dest_dir. Returns (file path, display title)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(dest_dir / "%(id)s.%(ext)s")

    def _progress_hook(data: dict) -> None:
        if not progress_callback:
            return
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            downloaded = data.get("downloaded_bytes") or 0
            if total:
                pct = min(1.0, downloaded / total)
                progress_callback(pct, f"Downloading… {int(pct * 100)}%")
            else:
                progress_callback(0.0, "Downloading…")
        elif status == "finished":
            progress_callback(1.0, "Finishing download…")

    ydl_opts = {
        "format": "best[ext=mp4]/best[height<=720]/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if progress_callback:
        ydl_opts["progress_hooks"] = [_progress_hook]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise ValueError("Could not fetch video information")

        if "entries" in info:
            info = info["entries"][0]

        video_path = Path(ydl.prepare_filename(info))
        if not video_path.exists():
            # prepare_filename may not match actual ext when format merges
            candidates = list(dest_dir.glob(f"{info['id']}.*"))
            if not candidates:
                raise ValueError("Download completed but video file was not found")
            video_path = candidates[0]

        title = info.get("title") or info.get("id") or "youtube"
        return video_path, title
