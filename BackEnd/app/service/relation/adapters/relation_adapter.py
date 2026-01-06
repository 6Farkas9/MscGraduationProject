from __future__ import annotations
from typing import Any, Dict, List, Set

def map_name(uid: str, uid2name: Dict[str, str]) -> str:
    return uid2name.get(uid) or uid

def _ensure_unique_reason(items: List[Dict[str, Any]], fallback_bank: List[str]) -> List[Dict[str, Any]]:
    used: Set[str] = set()
    out = []
    for it in items:
        reason = str(it.get("explanation") or it.get("reason") or "").strip()
        if not reason:
            reason = random_choice(fallback_bank)
        if reason in used:
            # rotate a different one
            reason = next_diff(fallback_bank, used) or reason
        used.add(reason)
        it2 = dict(it)
        it2["reason"] = reason
        out.append(it2)
    return out

def random_choice(bank: List[str]) -> str:
    import random
    return random.choice(bank) if bank else "推荐理由：综合学习状态与学习需求匹配。"

def next_diff(bank: List[str], used: Set[str]) -> str:
    for r in bank:
        if r not in used:
            return r
    return ""

def normalize_recs(items: List[Dict[str, Any]], uid2name: Dict[str, str], reason_bank: List[str]) -> List[Dict[str, Any]]:
    out = []
    for it in items or []:
        uid = str(it.get("uid") or "")
        out.append({
            "uid": uid,
            "name": map_name(uid, uid2name),
            "explanation": it.get("explanation") or it.get("reason") or "",
        })
    out = _ensure_unique_reason(out, reason_bank)
    return out

def build_page_view(result_for_target: Dict[str, Any], uid2name: Dict[str, str]) -> Dict[str, Any]:
    # independent reason banks for partner vs role model
    partner_bank = [
        "学习节奏与您接近，适合组成互相督促的小组。",
        "知识点掌握分布相似，讨论同类问题更高效。",
        "在相近难度阶段推进，合作完成任务更顺畅。",
        "互动方式较一致，协作沟通成本更低。",
        "学习策略接近，适合一起制定与调整计划。",
    ]
    role_model_bank = [
        "在您当前薄弱知识点上更稳定，适合参考其学习路径。",
        "在多课程中表现更均衡，适合作为阶段性对标对象。",
        "擅长结构化推进任务，能提供可借鉴的节奏与方法。",
        "在高难度资源上更有经验，适合跟随其进阶顺序。",
        "在长期坚持方面更突出，适合参考其持续学习方式。",
    ]
    partners = normalize_recs(result_for_target.get("partner") or result_for_target.get("partners") or [], uid2name, partner_bank)
    role_models = normalize_recs(result_for_target.get("role_model") or result_for_target.get("role_models") or [], uid2name, role_model_bank)

    # ensure non-overlap just in case pipeline overlaps
    partner_uids = {p["uid"] for p in partners}
    role_models = [r for r in role_models if r["uid"] not in partner_uids]

    return {"partners": partners, "role_models": role_models}
