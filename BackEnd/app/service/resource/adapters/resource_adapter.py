from __future__ import annotations
from typing import Any, Dict

def _pick(d: Dict[str, Any], key: str, default=None):
    v = d.get(key, default)
    return v if v is not None else default

def build_narrative(planning: Dict[str, Any]) -> Dict[str, Any]:
    concept_priority = _pick(planning, "concept_priority", []) or []
    type_priority = _pick(planning, "type_priority", []) or []
    overall_strategy = str(_pick(planning, "overall_strategy", "") or "")
    constraints = _pick(planning, "resource_constraints", {}) or {}

    weak_concepts = concept_priority[:5]
    primary_type = type_priority[0] if type_priority else None

    constraints_phrases = []
    mapping = {
        "difficulty_level": ("难度", {"easy":"偏基础", "medium":"中等", "hard":"偏进阶"}),
        "structure_level": ("结构", {"low":"更自由", "medium":"适中", "high":"更结构化"}),
        "guidance_level": ("引导", {"low":"少引导", "medium":"适中", "high":"强引导"}),
        "interaction_level": ("交互", {"none":"无交互", "low":"轻交互", "medium":"中等交互", "high":"高交互"}),
        "collaboration_mode": ("协作", {"none":"不强调协作", "pair":"两人协作", "group":"小组协作", "open":"开放式协作"}),
        "pedagogical_function": ("教学功能", {
            "concept_introduction":"概念引入", "practice":"练习巩固", "assessment":"测评检验",
            "feedback":"反馈纠错", "exploration":"探索学习", "collaboration":"协作学习", "reflection":"反思总结"
        }),
    }
    for k, (cn, mp) in mapping.items():
        if k in constraints and constraints.get(k) is not None:
            raw = str(constraints.get(k))
            constraints_phrases.append(f"{cn}：{mp.get(raw, raw)}")

    reason_parts = []
    if weak_concepts:
        reason_parts.append("你在以下知识点上更需要优先巩固：" + "、".join(weak_concepts))
    if constraints_phrases:
        reason_parts.append("同时我们会尽量匹配你的学习偏好（" + "；".join(constraints_phrases) + "）")
    if not reason_parts:
        reason_parts.append("根据你当前的学习情况，我们为你做了一个整体资源规划建议")

    suggest_parts = []
    if primary_type:
        suggest_parts.append(f"优先选择「{primary_type}」类资源")
    if type_priority:
        suggest_parts.append("资源类型优先级为：" + " → ".join(type_priority[:5]))
    if overall_strategy:
        suggest_parts.append(overall_strategy)

    return {
        "reason_text": "；".join(reason_parts),
        "suggest_text": "；".join([p for p in suggest_parts if p]),
        "weak_concepts": weak_concepts,
        "type_priority": type_priority[:6],
        "constraints_phrases": constraints_phrases,
    }

def build_resource_cards(orchestration: Dict[str, Any]) -> Dict[str, Any]:
    resources = orchestration.get("resources", []) or []
    top_k = orchestration.get("top_k")
    used_level = orchestration.get("used_relaxation_level")
    candidate_count = orchestration.get("candidate_count")

    cards = []
    for i, r in enumerate(resources, start=1):
        concepts = r.get("concepts", []) or []
        cards.append({
            "idx": i,
            "uid": r.get("uid"),
            "type": r.get("type"),
            "score": r.get("score"),
            "concepts": concepts[:8],
            "difficulty_level": r.get("difficulty_level"),
            "pedagogical_function": r.get("pedagogical_function"),
            "interaction_level": r.get("interaction_level"),
            "collaboration_mode": r.get("collaboration_mode"),
            "time_estimate": r.get("time_estimate"),
        })

    series = [{"idx": c["idx"], "score": float(c["score"] or 0), "time": c["time_estimate"] or 0} for c in cards]

    return {
        "cards": cards,
        "series": series,
        "top_k": top_k,
        "used_relaxation_level": used_level,
        "candidate_count": candidate_count,
    }
