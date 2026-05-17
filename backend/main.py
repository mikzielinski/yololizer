import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.openapi_config import OPENAPI_DESCRIPTION, OPENAPI_TAGS, apply_openapi
from backend.routers import inference, labels, models, parcel, sources, training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YOLOlizer API",
    description=OPENAPI_DESCRIPTION,
    version="1.0.0",
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={"name": "YOLOlizer", "url": "https://github.com/mikzielinski/yololizer"},
    license_info={"name": "MIT"},
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    apply_openapi(schema)
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(sources.router)
app.include_router(labels.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(inference.router)
app.include_router(parcel.router)

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Frontend not found"}, status_code=404)
    return FileResponse(str(index))


@app.get("/config.js", include_in_schema=False)
async def serve_frontend_config():
    """API base URL for UI (empty = same origin)."""
    path = FRONTEND_DIR / "config.js"
    if not path.exists():
        from fastapi.responses import Response
        return Response("window.YOLOlIZER_API = '';", media_type="application/javascript")
    return FileResponse(str(path), media_type="application/javascript")


@app.get("/health", tags=["health"])
async def health():
    """Check that the API process is running."""
    return {"status": "ok", "service": "YOLOlizer"}
