from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def build_overview(result_for_learner: Dict[str, Any]) -> Dict[str, Any]:
    overall = (result_for_learner or {}).get("overall") or {}
    details = (result_for_learner or {}).get("details") or {}

    dimensions: List[Dict[str, Any]] = []
    all_labels: List[str] = []

    dim_keys = sorted(set(list(overall.keys()) + list(details.keys())))
    for dim_key in dim_keys:
        dim_overall = overall.get(dim_key) or {}
        dim_detail = details.get(dim_key) or {}
        insufficient = bool(dim_detail.get("insufficient_data"))
        insufficient_reason = dim_detail.get("insufficient_reason")

        cats: List[Dict[str, Any]] = []
        cat_keys = set(dim_overall.keys())
        for k, v in dim_detail.items():
            if isinstance(v, dict) and "final_code" in v:
                cat_keys.add(k)

        for cat in sorted(cat_keys):
            label = dim_overall.get(cat)
            if isinstance(label, str) and label.strip():
                all_labels.append(label.strip())

            cat_detail = dim_detail.get(cat) if isinstance(dim_detail.get(cat), dict) else {}
            final_code = None
            course_count = None
            if isinstance(cat_detail, dict):
                final_code = cat_detail.get("final_code")
                courses = cat_detail.get("courses") or {}
                if isinstance(courses, dict):
                    course_count = len(courses)

            cats.append({
                "category": cat,
                "label": label,
                "final_code": final_code,
                "course_count": course_count,
                "insufficient": insufficient,
                "has_detail": (not insufficient) and isinstance(cat_detail, dict) and ("courses" in cat_detail),
            })

        dimensions.append({
            "dimension": dim_key,
            "insufficient": insufficient,
            "insufficient_reason": insufficient_reason,
            "categories": cats,
        })

    stats = {
        "dimension_count": len(dimensions),
        "category_count": sum(len(d["categories"]) for d in dimensions),
        "insufficient_dimensions": sum(1 for d in dimensions if d["insufficient"]),
    }

    label_counter = Counter(all_labels)
    label_chips = [{"text": k, "count": v} for k, v in label_counter.most_common()]
    overall_text = "；".join([x["text"] for x in label_chips[:8]])

    return {"dimensions": dimensions, "stats": stats, "label_chips": label_chips, "overall_text": overall_text}

def build_category_detail(result_for_learner: Dict[str, Any], dimension: str, category: str) -> Dict[str, Any]:
    overall = (result_for_learner or {}).get("overall") or {}
    details = (result_for_learner or {}).get("details") or {}

    dim_detail = details.get(dimension) or {}
    if dim_detail.get("insufficient_data"):
        return {
            "dimension": dimension,
            "category": category,
            "insufficient": True,
            "reason": dim_detail.get("insufficient_reason"),
        }

    cat_detail = dim_detail.get(category) or {}
    label_text = (overall.get(dimension) or {}).get(category)

    final_code = cat_detail.get("final_code")
    overall_metrics = cat_detail.get("overall_metrics") or {}
    courses = cat_detail.get("courses") or {}

    metric_keys = set()
    for _, payload in courses.items():
        metrics = (payload or {}).get("metrics") or {}
        metric_keys.update(metrics.keys())
    metric_keys_sorted = sorted(metric_keys)

    preferred = ["E_att_norm", "performance", "relevant_ratio", "ui_ratio", "text_ratio", "visual_ratio", "example_ratio"]
    primary_metric = None
    for k in preferred:
        if k in metric_keys:
            primary_metric = k
            break
    if primary_metric is None and metric_keys_sorted:
        primary_metric = metric_keys_sorted[0]

    rows = []
    code_counter = Counter()
    for crs_uid, payload in courses.items():
        code = (payload or {}).get("code")
        metrics = (payload or {}).get("metrics") or {}
        code_counter[str(code)] += 1
        row = {"course_uid": crs_uid, "code": code}
        for k in metric_keys_sorted[:10]:
            row[k] = metrics.get(k)
        rows.append(row)

    # Sort rows: put same-code together, then by primary metric (if any)
    def _row_key(r):
        c = str(r.get("code"))
        v = _safe_float(r.get(primary_metric)) if primary_metric else None
        return (c, v if v is not None else 10**9, str(r.get("course_uid")))
    rows.sort(key=_row_key)

    for i, r in enumerate(rows, start=1):
        r["idx"] = i

    series = []
    if primary_metric:
        for r in rows:
            v = _safe_float(r.get(primary_metric))
            if v is None:
                continue
            series.append({"idx": r["idx"], "value": v, "hover": str(r.get("course_uid"))})

    code_chips = [{"code": k, "count": v} for k, v in code_counter.most_common()]

    return {
        "dimension": dimension,
        "category": category,
        "insufficient": False,
        "label_text": label_text,
        "final_code": final_code,
        "overall_metrics": overall_metrics,
        "rows": rows,
        "metric_keys": metric_keys_sorted[:10],
        "primary_metric": primary_metric,
        "series": series,
        "course_count": len(courses) if isinstance(courses, dict) else 0,
        "code_chips": code_chips,
    }
