from __future__ import annotations

import time
import traceback
import random
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.service.relation.db.mysql_repo import fetch_basic_learners, fetch_concepts
from app.service.relation.db.mongo_repo import fetch_kt_for_learner
from app.service.relation.adapters.relation_adapter import build_page_view

router = APIRouter()
templates = Jinja2Templates(directory=str((Path(__file__).resolve().parents[6] / "FrontEnd" / "templates")))

_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_CACHE_TS: Dict[Tuple[str, str], float] = {}
_CACHE_TTL_SEC = 60 * 30

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

def _sample_kt(concepts: List[str], style: str, target_kt: Dict[str, float] | None = None) -> Dict[str, float]:
    """Generate KT for candidates with different styles.
    - partner style: closer to target (if target_kt provided), with small noise
    - role_model style: higher mastery on target weak concepts and broader coverage
    """
    if not concepts:
        return {}
    if style == "partner" and target_kt:
        keys = list(target_kt.keys())
        if keys:
            chosen = random.sample(keys, k=min(len(keys), random.randint(60, 120)))
        else:
            chosen = random.sample(concepts, k=min(len(concepts), 80))
        kt = {}
        for c in chosen:
            base = float(target_kt.get(c, random.uniform(0.2, 0.9)))
            p = max(0.05, min(0.99, base + random.uniform(-0.08, 0.08)))
            kt[c] = float(round(p, 4))
        extras = random.sample(concepts, k=min(len(concepts), random.randint(10, 30)))
        for c in extras:
            kt.setdefault(c, float(round(random.uniform(0.2, 0.9), 4)))
        return kt

    if style == "role_model" and target_kt:
        sorted_weak = sorted(target_kt.items(), key=lambda kv: kv[1])[:80]
        weak_concepts = [k for k,_ in sorted_weak] or random.sample(concepts, k=min(len(concepts), 80))
        kt = {}
        for c in weak_concepts:
            base = float(target_kt.get(c, random.uniform(0.1, 0.7)))
            p = max(0.2, min(0.995, base + random.uniform(0.25, 0.55)))
            kt[c] = float(round(p, 4))
        more = random.sample(concepts, k=min(len(concepts), random.randint(100, 180)))
        for c in more:
            kt.setdefault(c, float(round(random.uniform(0.5, 0.99), 4)))
        return kt

    n = random.randint(min(60, len(concepts)), min(140, len(concepts)))
    chosen = random.sample(concepts, k=min(n, len(concepts)))
    return {c: float(round(random.uniform(0.1, 0.95), 4)) for c in chosen}

def _sample_profile_labels(style: str, target_profile: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Generate service-adjustment labels for candidates.
    - partner: closer to target labels distribution
    - role_model: more diverse labels
    """
    from app.shared.models.profiles_labels import PROFILE_LABELS, get_label
    out: Dict[str, Any] = {}

    def random_label(dim: str, cat: str):
        codes = list(PROFILE_LABELS.get(dim, {}).get(cat, {}).keys())
        if not codes:
            return None
        code = random.choice(codes)
        return get_label(dim, cat, code)

    if style == "partner" and target_profile:
        for dim, cats in PROFILE_LABELS.items():
            dim_entry: Dict[str, Any] = {}
            for cat in cats.keys():
                if dim in target_profile and cat in (target_profile.get(dim) or {}) and random.random() < 0.7:
                    dim_entry[cat] = (target_profile[dim][cat])
                else:
                    lbl = random_label(dim, cat)
                    if lbl is not None:
                        dim_entry[cat] = lbl
            if dim_entry:
                out[dim] = dim_entry
        return out

    for dim, cats in PROFILE_LABELS.items():
        dim_entry: Dict[str, Any] = {}
        for cat in cats.keys():
            lbl = random_label(dim, cat)
            if lbl is not None:
                dim_entry[cat] = lbl
        if dim_entry:
            out[dim] = dim_entry
    return out

def _get_target_real_inputs(learner_uid: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    kt = fetch_kt_for_learner(learner_uid) or {}
    from app.domain.profiling.profiling_pipeline import ProfilingPipeline
    prof = ProfilingPipeline().analyze([learner_uid])
    overall = (prof.get(learner_uid, {}).get("overall") or {})
    return kt, overall

def _get_target_demo_inputs() -> Tuple[Dict[str, float], Dict[str, Any]]:
    # Demo target is simulated (fast), but still used to drive candidate simulation.
    concepts = fetch_concepts(limit=600)
    if not concepts:
        concepts = [f"cpt_demo_{i:04d}" for i in range(1, 601)]
    target_kt = {c: float(round(random.uniform(0.15, 0.9), 4)) for c in random.sample(concepts, k=min(140, len(concepts)))}
    target_profile = _sample_profile_labels("partner", None)
    return target_kt, target_profile

def _build_simulated_pool(
    target_uid: str,
    n_candidates: int,
    target_kt: Dict[str, float],
    target_profile: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    rows = fetch_basic_learners(limit=max(n_candidates, 40))
    if not rows:
        raise RuntimeError("未从 BasicLearners 读取到 uid/name。请确认 mls_sample.BasicLearners 表存在且有数据。")
    uid2name = {str(r["uid"]): str(r["name"]) for r in rows if r.get("uid") and r.get("name")}

    concepts = fetch_concepts(limit=600)
    if not concepts:
        concepts = [f"cpt_demo_{i:04d}" for i in range(1, 601)]

    pool_uids = [u for u in uid2name.keys() if u != target_uid]
    random.shuffle(pool_uids)

    n_partners = max(10, int(n_candidates * 0.5))
    n_models = max(10, n_candidates - n_partners)

    partner_uids = pool_uids[:n_partners]
    model_uids = pool_uids[n_partners:n_partners+n_models]

    data: Dict[str, Any] = {}

    for u in partner_uids:
        data[u] = {
            "learner_profile": _sample_profile_labels("partner", target_profile),
            "knowledge_concepts": _sample_kt(concepts, "partner", target_kt),
        }

    for u in model_uids:
        data[u] = {
            "learner_profile": _sample_profile_labels("role_model", target_profile),
            "knowledge_concepts": _sample_kt(concepts, "role_model", target_kt),
        }

    data[target_uid] = {
        "learner_profile": target_profile,
        "knowledge_concepts": target_kt,
    }

    return data, uid2name

@router.get("/relation/ping")
def ping():
    print("[PING] /api/relation/ping called")
    return {"ok": True}

@router.post("/relation/run", response_class=HTMLResponse)
def run_relation(
    request: Request,
    learner_uid: str = Form(...),
    run_mode: str = Form("demo"),  # demo | real
    n_candidates: int = Form(80),
):
    cached = _cache_get(learner_uid, run_mode)
    if cached is not None:
        ctx = dict(cached)
        ctx["request"] = request
        return templates.TemplateResponse("relation_result_fragment.html", ctx)

    start = time.time()
    n_candidates = max(20, min(300, int(n_candidates)))
    print(f"[RELATION] start learner_uid={learner_uid} mode={run_mode} candidates={n_candidates}")

    try:
        # IMPORTANT (per requirement):
        # - demo: target inputs are simulated (fast)
        # - real: target inputs are real (Mongo KT + ProfilingPipeline)
        if run_mode == "real":
            target_kt, target_profile = _get_target_real_inputs(learner_uid)
            if not target_kt:
                # allow running even if target KT missing
                concepts = fetch_concepts(limit=600)
                target_kt = {c: float(round(random.uniform(0.2, 0.9), 4)) for c in random.sample(concepts, k=min(120, len(concepts)))}
            if not target_profile:
                target_profile = _sample_profile_labels("partner", None)
        else:
            target_kt, target_profile = _get_target_demo_inputs()

        data, uid2name = _build_simulated_pool(learner_uid, n_candidates, target_kt, target_profile)

        from app.domain.partner.partner_pipeline import PartnerRecommendationPipeline
        pipeline = PartnerRecommendationPipeline()
        out = pipeline.analyze(learner_uids=[learner_uid], data=data)
        rec = (out.get("results") or {}).get(learner_uid) or {}

        view = build_page_view(rec, uid2name)

        ctx = {
            "learner_uid": learner_uid,
            "learner_name": uid2name.get(learner_uid, learner_uid),
            "run_mode": run_mode,
            "n_candidates": n_candidates,
            "partners": view["partners"],
            "role_models": view["role_models"],
            "elapsed_sec": round(time.time() - start, 2),
        }
        _cache_set(learner_uid, run_mode, dict(ctx))
        ctx["request"] = request
        print(f"[RELATION] done learner_uid={learner_uid} elapsed={ctx['elapsed_sec']}s")
        return templates.TemplateResponse("relation_result_fragment.html", ctx)

    except Exception as e:
        print(f"[RELATION] ERROR learner_uid={learner_uid} err={e!r}")
        traceback.print_exc()
        return HTMLResponse(
            f"""
            <div class="bg-white rounded-xl shadow-sm border border-rose-200 p-4">
              <div class="text-rose-700 font-semibold">生成失败</div>
              <div class="text-sm text-slate-600 mt-2">
                候选学习者从数据库读取真实 uid/name（仅用于姓名展示），并分为伙伴池/榜样池分别模拟，以体现两类推荐差异。
              </div>
              <div class="text-xs text-slate-500 mt-2">错误：{str(e)}</div>
            </div>
            """,
            status_code=500,
        )
