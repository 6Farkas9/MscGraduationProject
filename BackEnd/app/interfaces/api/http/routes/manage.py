from __future__ import annotations

import time
import uuid
from typing import List

from fastapi import APIRouter, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.service.manage.db.mysql_repo import search_concepts

router = APIRouter()
templates = Jinja2Templates(directory=str((Path(__file__).resolve().parents[6] / "FrontEnd" / "templates")))

@router.get("/manage/ping")
def ping():
    print("[PING] /api/manage/ping called")
    return {"ok": True}

@router.get("/manage/concepts", response_class=HTMLResponse)
def concept_suggest(request: Request, q: str = Query("", max_length=100)):
    rows = search_concepts(q=q, limit=20)
    return templates.TemplateResponse("concept_suggest_fragment.html", {"request": request, "rows": rows, "q": q})

@router.post("/manage/create", response_class=HTMLResponse)
def create_resource(
    request: Request,
    title: str = Form(...),
    resource_type: str = Form(...),
    difficulty_level: str = Form(...),
    time_estimate: int = Form(...),
    pedagogical_function: str = Form(...),
    interaction_level: str = Form(...),
    collaboration_mode: str = Form(...),
    concept_uids: str = Form(""),
    concept_names: str = Form(""),
    extra_tags: str = Form(""),
):
    # Fake create: return a pretty summary (no JSON)
    rid = "res_" + uuid.uuid4().hex[:10]
    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    concepts = []
    if concept_uids and concept_names:
        uids = [x for x in concept_uids.split(",") if x.strip()]
        names = [x for x in concept_names.split(",") if x.strip()]
        for uid, name in zip(uids, names):
            concepts.append({"uid": uid, "name": name})
    tags = [t.strip() for t in (extra_tags or "").split(",") if t.strip()]

    ctx = {
        "request": request,
        "rid": rid,
        "created_at": created_at,
        "title": title.strip(),
        "resource_type": resource_type,
        "difficulty_level": difficulty_level,
        "time_estimate": int(time_estimate),
        "pedagogical_function": pedagogical_function,
        "interaction_level": interaction_level,
        "collaboration_mode": collaboration_mode,
        "concepts": concepts,
        "tags": tags,
    }
    print(f"[MANAGE] fake create rid={rid} title={title} concepts={len(concepts)} tags={len(tags)}")
    return templates.TemplateResponse("create_result_fragment.html", ctx)
