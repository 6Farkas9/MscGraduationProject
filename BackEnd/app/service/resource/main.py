from __future__ import annotations

import sys
from pathlib import Path

_this = Path(__file__).resolve()
project_root = _this.parents[3]  # webdemo -> resource_web_demo -> FrontEnd -> project_root
backend_root = project_root / "BackEnd"
if backend_root.exists():
    sys.path.insert(0, str(backend_root))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.service.resource.api.resource import router as api_router
from app.service.resource.ui.resource import router as ui_router

app = FastAPI(title="Resource Service Demo")

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(ui_router, prefix="/ui")
app.include_router(api_router, prefix="/api")
