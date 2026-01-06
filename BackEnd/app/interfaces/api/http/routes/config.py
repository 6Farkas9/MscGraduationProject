from __future__ import annotations

import time
import traceback
import sys
import io
from contextlib import redirect_stdout
from typing import Dict, Any, Tuple

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.service.config.adapters.config_adapter import build_overview, build_category_detail

from app.domain.profiling import profiling_pipeline

router = APIRouter()
templates = Jinja2Templates(directory=str((Path(__file__).resolve().parents[6] / "FrontEnd" / "templates")))

# Simple in-memory cache to avoid re-running slow pipeline on every click
_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_CACHE_TS: Dict[Tuple[str, str], float] = {}
_CACHE_TTL_SEC = 60 * 60  # 1 hour

def _cache_key(learner_uid: str, device: str) -> Tuple[str, str]:
    return (learner_uid, device.strip() or "__default__")

def _cache_get(learner_uid: str, device: str):
    k = _cache_key(learner_uid, device)
    ts = _CACHE_TS.get(k)
    if ts is None:
        return None
    if time.time() - ts > _CACHE_TTL_SEC:
        _CACHE.pop(k, None)
        _CACHE_TS.pop(k, None)
        return None
    return _CACHE.get(k)

def _cache_set(learner_uid: str, device: str, result: Dict[str, Any]):
    k = _cache_key(learner_uid, device)
    _CACHE[k] = result
    _CACHE_TS[k] = time.time()

@router.get("/config/ping")
def ping():
    print("[PING] /api/config/ping called")
    return {"ok": True}

@router.post("/config/run", response_class=HTMLResponse)
def run_config(
    request: Request,
    learner_uid: str = Form(...),
    device: str = Form(""),
):
    start = time.time()
    print(f"[CONFIG] start analyze learner_uid={learner_uid} device={device!r}")

    try:
        class _Tee(io.TextIOBase):
            def __init__(self, real):
                self.real = real
            def write(self, s):
                self.real.write(s)
                self.real.flush()
                return len(s)
            def flush(self):
                self.real.flush()

        tee = _Tee(sys.stdout)
        with redirect_stdout(tee):
            if device.strip():
                result = profiling_pipeline.analyze([learner_uid], device=device.strip())
            else:
                result = profiling_pipeline.analyze([learner_uid])

        _cache_set(learner_uid, device, result)

        block = result.get(learner_uid) or {}
        overview = build_overview(block)

        # default selection: first item with detail
        default_dim = None
        default_cat = None
        for d in overview["dimensions"]:
            if d["insufficient"]:
                continue
            for c in d["categories"]:
                if c["has_detail"]:
                    default_dim = d["dimension"]
                    default_cat = c["category"]
                    break
            if default_dim:
                break

        detail = None
        if default_dim and default_cat:
            detail = build_category_detail(block, default_dim, default_cat)

        cost = time.time() - start
        print(f"[CONFIG] done learner_uid={learner_uid} elapsed={cost:.2f}s")

        ctx = {
            "request": request,
            "learner_uid": learner_uid,
            "device": device,
            "overview": overview,
            "detail": detail,
            "default_dim": default_dim,
            "default_cat": default_cat,
        }
        return templates.TemplateResponse("config_result_fragment.html", ctx)

    except Exception as e:
        cost = time.time() - start
        print(f"[CONFIG] ERROR learner_uid={learner_uid} elapsed={cost:.2f}s err={e!r}")
        traceback.print_exc()
        return HTMLResponse(
            f"""
            <div class="bg-white rounded-xl shadow-sm border border-rose-200 p-4">
              <div class="text-rose-700 font-semibold">分析失败</div>
              <div class="text-sm text-slate-600 mt-2">后端发生异常，请查看控制台日志。</div>
              <div class="text-xs text-slate-500 mt-2">学习者UID：<span class="font-mono">{learner_uid}</span></div>
            </div>
            """,
            status_code=500,
        )

@router.get("/config/detail", response_class=HTMLResponse)
def config_detail(
    request: Request,
    learner_uid: str,
    dimension: str,
    category: str,
    device: str = "",
):
    # Use cached results from last run when possible
    cached = _cache_get(learner_uid, device)
    if cached is None:
        # fallback: recompute (slow)
        if device.strip():
            cached = profiling_pipeline.analyze([learner_uid], device=device.strip())
        else:
            cached = profiling_pipeline.analyze([learner_uid])
        _cache_set(learner_uid, device, cached)

    block = (cached or {}).get(learner_uid) or {}
    detail = build_category_detail(block, dimension, category)

    ctx = {"request": request, "learner_uid": learner_uid, "detail": detail}
    return templates.TemplateResponse("config_detail_fragment.html", ctx)
