from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()

# BackEnd/app/interfaces/api/http/routes -> go up to repo root then FrontEnd/templates
TEMPLATES_DIR = Path(__file__).resolve().parents[6] / "FrontEnd" / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@router.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/login")

@router.get("/ui/login", response_class=HTMLResponse)
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "active": "learner"})

@router.get("/ui/learner", response_class=HTMLResponse)
def learner_portal(request: Request, view: str = "ability"):
    view = view or "ability"
    iframe_map = {
        "ability": "/ui/ability",
        "config": "/ui/config",
        "resource": "/ui/resource",
        "relation": "/ui/relation",
    }
    iframe_src = iframe_map.get(view, "/ui/ability")
    return templates.TemplateResponse("learner_portal.html", {
        "request": request,
        "active": "learner",
        "view": view if view in iframe_map else "ability",
        "iframe_src": iframe_src,
    })

@router.get("/ui/manage", response_class=HTMLResponse)
def manage_portal(request: Request):
    return templates.TemplateResponse("manage_portal.html", {"request": request, "active": "manage"})

@router.get("/ui/ability", response_class=HTMLResponse)
def ability_page(request: Request):
    return templates.TemplateResponse("ability.html", {"request": request})

@router.get("/ui/config", response_class=HTMLResponse)
def config_page(request: Request):
    return templates.TemplateResponse("config.html", {"request": request})

@router.get("/ui/resource", response_class=HTMLResponse)
def resource_page(request: Request):
    return templates.TemplateResponse("resource.html", {"request": request})

@router.get("/ui/relation", response_class=HTMLResponse)
def relation_page(request: Request):
    return templates.TemplateResponse("relation.html", {"request": request})

@router.get("/ui/manage_inner", response_class=HTMLResponse)
def manage_inner(request: Request):
    return templates.TemplateResponse("manage.html", {"request": request})
