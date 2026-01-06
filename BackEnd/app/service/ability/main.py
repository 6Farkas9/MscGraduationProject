from __future__ import annotations

import sys
from pathlib import Path

# --- Permanent, zero-config import fix ---
# Expected layout:
# project_root/
#   BackEnd/      (contains app/domain/...)
#   FrontEnd/
#     xxx_web_demo/
#       webdemo/main.py  (this file)
#
# We add project_root/BackEnd to sys.path so backend package "app.*" can be imported.
_this = Path(__file__).resolve()
project_root = _this.parents[3]  # webdemo -> xxx_web_demo -> FrontEnd -> project_root
backend_root = project_root / "BackEnd"
if backend_root.exists():
    sys.path.insert(0, str(backend_root))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.service.ability.api.ability import router as ability_api_router
from app.service.ability.ui.ability import router as ability_ui_router

app = FastAPI(title="MLS Ability Visualization Demo")

app.mount("/static", StaticFiles(directory="webdemo/static"), name="static")

app.include_router(ability_ui_router, prefix="/ui")
app.include_router(ability_api_router, prefix="/api")
