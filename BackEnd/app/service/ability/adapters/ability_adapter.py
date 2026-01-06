from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def summarize_ability(
    kt: Dict[str, Any],
    kt_history: List[Dict[str, Any]],
    concept_to_topic: Dict[str, str],
    topic_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    mastery_by_concept: Dict[str, float] = {}
    for c_uid, v in (kt or {}).items():
        fv = _safe_float(v)
        if fv is not None:
            mastery_by_concept[str(c_uid)] = max(0.0, min(1.0, fv))

    mastery_values = list(mastery_by_concept.values())

    by_topic = defaultdict(list)
    unknown_topic = "__unknown__"
    for c_uid, m in mastery_by_concept.items():
        t_uid = concept_to_topic.get(c_uid, unknown_topic)
        by_topic[t_uid].append(m)

    topic_rows: List[Dict[str, Any]] = []
    for t_uid, arr in by_topic.items():
        n = len(arr)
        mean = sum(arr)/n if n else 0.0
        low_ratio = sum(1 for x in arr if x < 0.4)/n if n else 0.0
        high_ratio = sum(1 for x in arr if x > 0.8)/n if n else 0.0
        meta = topic_meta.get(t_uid, {"uid": t_uid, "name": t_uid, "explanation": None})
        topic_rows.append({
            "topic_uid": t_uid,
            "topic_name": meta.get("name") or t_uid,
            "count": n,
            "mean": mean,
            "low_ratio": low_ratio,
            "high_ratio": high_ratio,
        })
    topic_rows.sort(key=lambda r: (r["mean"], -r["count"]))

    trend_by_topic = defaultdict(list)
    for item in (kt_history or []):
        step = item.get("step_index")
        cm = item.get("concept_mastery", {}) or {}
        step_by_topic = defaultdict(list)
        for c_uid, v in cm.items():
            fv = _safe_float(v)
            if fv is None:
                continue
            fv = max(0.0, min(1.0, fv))
            t_uid = concept_to_topic.get(str(c_uid), unknown_topic)
            step_by_topic[t_uid].append(fv)
        for t_uid, arr in step_by_topic.items():
            if not arr:
                continue
            trend_by_topic[t_uid].append({"step": step, "mean": sum(arr)/len(arr)})

    topic_trends = {}
    for t_uid, series in trend_by_topic.items():
        series.sort(key=lambda x: (x["step"] if x["step"] is not None else 10**9))
        meta = topic_meta.get(t_uid, {"uid": t_uid, "name": t_uid, "explanation": None})
        topic_trends[t_uid] = {"topic_uid": t_uid, "topic_name": meta.get("name") or t_uid, "series": series}

    def percentile(vals: List[float], p: float) -> Optional[float]:
        if not vals:
            return None
        vals_sorted = sorted(vals)
        k = (len(vals_sorted)-1)*p
        f = int(k)
        c = min(f+1, len(vals_sorted)-1)
        if f == c:
            return vals_sorted[f]
        return vals_sorted[f] + (vals_sorted[c]-vals_sorted[f])*(k-f)

    stats = {
        "count": len(mastery_values),
        "mean": (sum(mastery_values)/len(mastery_values)) if mastery_values else None,
        "p10": percentile(mastery_values, 0.10),
        "p50": percentile(mastery_values, 0.50),
        "p90": percentile(mastery_values, 0.90),
        "ge_0_8": sum(1 for x in mastery_values if x >= 0.8),
        "lt_0_4": sum(1 for x in mastery_values if x < 0.4),
    }

    return {
        "stats": stats,
        "mastery_values": mastery_values,
        "topic_rows": topic_rows,
        "topic_trends": topic_trends,
        "unknown_topic_uid": unknown_topic,
    }

def topic_concepts_page(
    kt: Dict[str, Any],
    topic_uid: str,
    concept_to_topic: Dict[str, str],
    concept_meta: Dict[str, Dict[str, Any]],
    page: int = 1,
    page_size: int = 40,
    q: str = "",
) -> Dict[str, Any]:
    rows: List[Tuple[str, str, float]] = []
    q_lower = (q or "").strip().lower()

    for c_uid, v in (kt or {}).items():
        fv = _safe_float(v)
        if fv is None:
            continue
        fv = max(0.0, min(1.0, fv))
        c_uid = str(c_uid)
        t_uid = concept_to_topic.get(c_uid, "__unknown__")
        if t_uid != topic_uid:
            continue

        meta = concept_meta.get(c_uid, {})
        c_name = (meta.get("name") or "").strip()

        # 搜索：同时匹配 uid 和 name
        if q_lower:
            if (q_lower not in c_uid.lower()) and (q_lower not in c_name.lower()):
                continue

        rows.append((c_uid, c_name, fv))

    rows.sort(key=lambda x: x[2])  # mastery 从低到高
    total = len(rows)
    page = max(1, int(page))
    start = (page-1)*page_size
    end = start + page_size

    items = []
    for c_uid, c_name, mastery in rows[start:end]:
        items.append({
            "concept_uid": c_uid,
            "concept_name": c_name if c_name else "(未命名知识点)",
            "mastery": mastery,
        })

    return {
        "topic_uid": topic_uid,
        "q": q,
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items,
    }

def concept_trend(
    kt_history: List[Dict[str, Any]],
    concept_uid: str,
) -> Dict[str, Any]:
    series = []
    for item in (kt_history or []):
        step = item.get("step_index")
        cm = item.get("concept_mastery", {}) or {}
        if concept_uid in cm:
            v = _safe_float(cm.get(concept_uid))
            if v is None:
                continue
            series.append({"step": step, "mastery": max(0.0, min(1.0, v))})
    series.sort(key=lambda x: (x["step"] if x["step"] is not None else 10**9))
    return {"concept_uid": concept_uid, "series": series}
