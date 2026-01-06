from __future__ import annotations

from fastapi import APIRouter

from app.interfaces.api.http.routes.ui import router as ui_router
from app.interfaces.api.http.routes.ability import router as ability_router
from app.interfaces.api.http.routes.config import router as config_router
from app.interfaces.api.http.routes.resource import router as resource_router
from app.interfaces.api.http.routes.relation import router as relation_router
from app.interfaces.api.http.routes.manage import router as manage_router

router = APIRouter()

# UI pages
router.include_router(ui_router)

# API routes (保持原前端已验证的路径结构：各子模块内部仍是 /ability/* /config/* ...，统一由 prefix="/api" 暴露)
router.include_router(ability_router, prefix="/api")
router.include_router(config_router, prefix="/api")
router.include_router(resource_router, prefix="/api")
router.include_router(relation_router, prefix="/api")
router.include_router(manage_router, prefix="/api")
