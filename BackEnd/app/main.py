from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.interfaces.api.http.router import router as http_router

def create_app() -> FastAPI:
    app = FastAPI(title="Metaverse Education Personalized Service System")

    # Mount FrontEnd static (if any)
    static_dir = Path(__file__).resolve().parents[2] / "FrontEnd" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    app.include_router(http_router)
    return app

app = create_app()
