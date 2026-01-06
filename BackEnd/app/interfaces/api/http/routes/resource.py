from __future__ import annotations

import time
import traceback
import random
from typing import Any, Dict, Tuple

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.service.resource.adapters.resource_adapter import build_narrative, build_resource_cards
from app.service.resource.db.mysql_repo import fetch_concepts
from app.service.resource.db.mongo_repo import fetch_kt_for_learner

router = APIRouter()
templates = Jinja2Templates(directory=str((Path(__file__).resolve().parents[6] / "FrontEnd" / "templates")))

_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_CACHE_TS: Dict[Tuple[str, str], float] = {}
_CACHE_TTL_SEC = 60 * 60

def _cache_key(learner_uid: str, mode: str) -> Tuple[str, str]:
    return (learner_uid, mode)

def _cache_get(learner_uid: str, mode: str):
    k = _cache_key(learner_uid, mode)
    ts = _CACHE_TS.get(k)
    if ts is None:
        return None
    if time.time() - ts > _CACHE_TTL_SEC:
        _CACHE.pop(k, None)
        _CACHE_TS.pop(k, None)
        return None
    return _CACHE.get(k)

def _cache_set(learner_uid: str, mode: str, result: Dict[str, Any]):
    k = _cache_key(learner_uid, mode)
    _CACHE[k] = result
    _CACHE_TS[k] = time.time()

def _build_demo_inputs(learner_uid: str):
    # Demo mode follows test_orchestration_pipeline.py idea:
    # - read Concepts from MySQL
    # - simulate KT using real concept_uids
    # - simulate profile labels via PROFILE_LABELS/get_label
    from app.shared.models.profiles_labels import PROFILE_LABELS, get_label

    concepts = fetch_concepts(limit=10)
    if not concepts:
        raise RuntimeError("Concepts 表未读取到数据，请检查 MySQL 连接与 MLS_DB_NAME。")

    concept_uids = [c["uid"] for c in concepts if c.get("uid")]
    chosen = random.sample(concept_uids, min(len(concept_uids), 5))
    kt = {learner_uid: {cid: round(random.uniform(0.4, 0.95), 3) for cid in chosen}}

    profile_for_learner: Dict[str, Any] = {}
    for dim, cat_dict in PROFILE_LABELS.items():
        dim_entry: Dict[str, Any] = {}
        for category, code_to_label in cat_dict.items():
            codes = list(code_to_label.keys())
            if not codes:
                continue
            code = random.choice(codes)
            label_text = get_label(dim, category, code)
            if label_text is not None:
                dim_entry[category] = label_text
        if dim_entry:
            profile_for_learner[dim] = dim_entry

    profile = {learner_uid: profile_for_learner}
    return kt, profile

@router.get("/resource/ping")
def ping():
    print("[PING] /api/resource/ping called")
    return {"ok": True}

@router.post("/resource/run", response_class=HTMLResponse)
def run_resource(
    request: Request,
    learner_uid: str = Form(...),
    run_mode: str = Form("demo"),  # demo | real
):
    cached = _cache_get(learner_uid, run_mode)
    if cached is not None:
        ctx = dict(cached)
        ctx["request"] = request
        return templates.TemplateResponse("resource_result_fragment.html", ctx)

    start = time.time()
    print(f"[RESOURCE] start learner_uid={learner_uid} mode={run_mode}")

    try:
        from app.domain.orchestration.orchestration_pipeline import OrchestrationPipeline

        if run_mode == "real":
            # REAL MODE REQUIREMENTS (per user):
            # - KT from MongoDB: MLS.Learners[uid].KT
            # - profile computed via ProfilingPipeline
            kt_map = fetch_kt_for_learner(learner_uid)
            if not kt_map:
                raise RuntimeError("MongoDB 中未读取到该学习者的KT字段（请确认 uid 与 KT 字段存在）。")
            kt = {learner_uid: kt_map}

            from app.domain.profiling.profiling_pipeline import ProfilingPipeline
            prof = ProfilingPipeline().analyze([learner_uid])
            profile = {learner_uid: (prof.get(learner_uid, {}).get("overall") or {})}
        else:
            # DEMO MODE
            kt, profile = _build_demo_inputs(learner_uid)

        pipeline = OrchestrationPipeline()
        out = pipeline.analyze(learner_uids=[learner_uid], kt=kt, profile=profile)
        result = (out.get("results") or {}).get(learner_uid) or {}
        planning = result.get("planning") or {}
        orchestration = result.get("orchestration") or {}
        learning_path = result.get("learning_path") or ""

        if not planning or not orchestration:
            raise RuntimeError("pipeline returned empty results")

        narrative = build_narrative(planning)
        resources = build_resource_cards(orchestration)

        ctx = {
            "learner_uid": learner_uid,
            "run_mode": run_mode,
            "narrative": narrative,
            "resources": resources,
            "learning_path_md": learning_path,
            "elapsed_sec": round(time.time() - start, 2),
        }
        _cache_set(learner_uid, run_mode, dict(ctx))
        ctx["request"] = request
        print(f"[RESOURCE] done learner_uid={learner_uid} elapsed={ctx['elapsed_sec']}s")
        return templates.TemplateResponse("resource_result_fragment.html", ctx)

    except Exception as e:
        print(f"[RESOURCE] ERROR learner_uid={learner_uid} err={e!r}")
        traceback.print_exc()
        return HTMLResponse(
            f"""
            <div class="bg-white rounded-xl shadow-sm border border-rose-200 p-4">
              <div class="text-rose-700 font-semibold">生成失败</div>
              <div class="text-sm text-slate-600 mt-2">
                演示模式：从 Concepts 表读取真实知识点并模拟 KT/服务调节状态（再走真实资源服务链路）；
                真实模式：从 MongoDB 读取 KT，并调用服务调节状态 Pipeline 生成 profile。
              </div>
              <div class="text-xs text-slate-500 mt-2">错误：{str(e)}</div>
            </div>
            """,
            status_code=500,
        )
