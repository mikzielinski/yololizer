import csv
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import RUNS_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])


def _find_best_pt(run_dir: Path) -> Path | None:
    """Search for best.pt inside a run directory."""
    for candidate in [
        run_dir / "weights" / "best.pt",
        run_dir / "best.pt",
    ]:
        if candidate.exists():
            return candidate
    # Recursive search (some yolo versions nest differently)
    matches = list(run_dir.rglob("best.pt"))
    return matches[0] if matches else None


def _find_results_csv(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("results.csv"))
    if candidates:
        return candidates[0]
    matches = list(run_dir.rglob("results.csv"))
    return matches[0] if matches else None


def _get_run_info(run_dir: Path) -> dict:
    info = {
        "name": run_dir.name,
        "path": str(run_dir),
        "has_best_pt": False,
        "mtime": None,
        "map50": None,
        "map50_95": None,
    }

    best_pt = _find_best_pt(run_dir)
    if best_pt:
        info["has_best_pt"] = True
        info["mtime"] = best_pt.stat().st_mtime

    results_csv = _find_results_csv(run_dir)
    if results_csv:
        try:
            with open(results_csv, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                last = rows[-1]
                # Column names vary; try common ones
                for k in last:
                    kstrip = k.strip()
                    if "map50-95" in kstrip.lower() or "map50_95" in kstrip.lower():
                        try:
                            info["map50_95"] = float(last[k])
                        except ValueError:
                            pass
                    elif "map50" in kstrip.lower() and "95" not in kstrip.lower():
                        try:
                            info["map50"] = float(last[k])
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning("Failed to parse results.csv for %s: %s", run_dir.name, e)

    return info


@router.get("")
async def list_models():
    """List all trained model runs found under runs/."""
    if not RUNS_DIR.exists():
        return {"models": []}

    models = []
    for run_dir in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if run_dir.is_dir():
            models.append(_get_run_info(run_dir))

    return {"models": models}


@router.get("/download/{run_name}")
async def download_model(run_name: str):
    """Download best.pt for a given run."""
    run_dir = RUNS_DIR / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' not found")

    best_pt = _find_best_pt(run_dir)
    if not best_pt:
        raise HTTPException(
            status_code=404,
            detail=f"best.pt not found in run '{run_name}'. Training may not be complete.",
        )

    return FileResponse(
        str(best_pt),
        media_type="application/octet-stream",
        filename=f"{run_name}_best.pt",
    )


@router.get("/{run_name}/metrics")
async def get_metrics(run_name: str):
    """Return results.csv parsed as JSON for a given run."""
    run_dir = RUNS_DIR / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' not found")

    results_csv = _find_results_csv(run_dir)
    if not results_csv:
        raise HTTPException(
            status_code=404,
            detail=f"results.csv not found in run '{run_name}'",
        )

    try:
        with open(results_csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
        return {"run_name": run_name, "metrics": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse metrics: {e}")
