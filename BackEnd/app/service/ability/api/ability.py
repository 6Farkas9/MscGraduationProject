from __future__ import annotations

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.service.ability.db import load_topic_concept_map, load_topics, load_concepts
from app.service.ability.adapters.ability_adapter import summarize_ability, topic_concepts_page, concept_trend

# 后端真实 pipeline
from app.domain.prediction import prediction_pipeline

router = APIRouter()
templates = Jinja2Templates(directory="webdemo/templates")

@router.post("/ability/run", response_class=HTMLResponse)
def run_ability(
    request: Request,
    learner_uid: str = Form(...),
    mode: str = Form("auto"),
    selected_topic_uid: str = Form(""),
):
    is_new_learner = None
    if mode == "existing":
        is_new_learner = False
    elif mode == "new":
        is_new_learner = True

    res = prediction_pipeline.analyze([learner_uid], is_new_learner=is_new_learner)
    errors = res.get("errors") or []
    results = (res.get("results") or {}).get(learner_uid) or {}
    kt = results.get("kt") or {}
    kt_history = results.get("kt_history") or []

    concept_to_topic = load_topic_concept_map()
    topic_meta = load_topics()
    concept_meta = load_concepts()

    view = summarize_ability(kt, kt_history, concept_to_topic, topic_meta)

    topic_uid = selected_topic_uid or (view["topic_rows"][0]["topic_uid"] if view["topic_rows"] else view["unknown_topic_uid"])
    topic_page = topic_concepts_page(
        kt, topic_uid, concept_to_topic, concept_meta,
        page=1, page_size=40, q=""
    )

    ctx = {
        "request": request,
        "learner_uid": learner_uid,
        "mode": mode,
        "errors": errors,
        "view": view,
        "topic_page": topic_page,
        "topic_uid": topic_uid,
        "kt_history": kt_history,
    }
    return templates.TemplateResponse("ability_result_fragment.html", ctx)

@router.get("/ability/topic_detail", response_class=HTMLResponse)
def ability_topic_detail(
    request: Request,
    learner_uid: str,
    topic_uid: str,
    mode: str = "auto",
    page: int = 1,
    q: str = "",
):
    is_new_learner = None
    if mode == "existing":
        is_new_learner = False
    elif mode == "new":
        is_new_learner = True

    res = prediction_pipeline.analyze([learner_uid], is_new_learner=is_new_learner)
    results = (res.get("results") or {}).get(learner_uid) or {}
    kt = results.get("kt") or {}

    concept_to_topic = load_topic_concept_map()
    concept_meta = load_concepts()

    topic_page = topic_concepts_page(
        kt, topic_uid, concept_to_topic, concept_meta,
        page=page, page_size=40, q=q
    )

    ctx = {"request": request, "topic_page": topic_page, "topic_uid": topic_uid, "learner_uid": learner_uid, "mode": mode}
    return templates.TemplateResponse("ability_topic_detail_fragment.html", ctx)

@router.get("/ability/concept_trend", response_class=HTMLResponse)
def ability_concept_trend(
    request: Request,
    learner_uid: str,
    concept_uid: str,
    mode: str = "auto",
):
    is_new_learner = None
    if mode == "existing":
        is_new_learner = False
    elif mode == "new":
        is_new_learner = True

    res = prediction_pipeline.analyze([learner_uid], is_new_learner=is_new_learner)
    results = (res.get("results") or {}).get(learner_uid) or {}
    kt_history = results.get("kt_history") or []

    trend = concept_trend(kt_history, concept_uid)
    ctx = {"request": request, "trend": trend, "concept_uid": concept_uid}
    return templates.TemplateResponse("ability_concept_trend_fragment.html", ctx)
