# -*- coding: utf-8 -*-
"""
HR-PRR：Hybrid Retrieval with Progressive Relaxation & Re-ranking

功能：
- 根据第一次大模型输出（target_concepts + resource_preferences），
  从 MongoDB.MLS.Fragments 中匹配资源分段，并进行多级检索：
    Stage 1: 严格概念 + 强偏好类型过滤
    Stage 2: 放宽类型，仅保留概念过滤
    Stage 3: 去掉概念过滤，仅用画像偏好 + 语义相似做兜底

- 所有阶段返回的候选，最终都用统一的多目标打分公式重排序：
    overall = alpha * concept + beta * type + gamma * feature + delta * semantic

依赖：
    pip install pymongo sentence-transformers numpy pymysql
"""

# HR-PRR：Hybrid Retrieval with Progressive Relaxation & Re-ranking
# （混合检索 + 渐进放宽 + 重排序）

# 核心思路：

# Stage 1 – Strict Concept & Strong-Type Filter

# 只要同时满足：

# 与目标知识点匹配（concepts.uid in target_concepts）

# 类型为偏好里 preference_level ∈ {high, medium} 的那些

# 如果候选数 ≥ 3 * top_k（理论上足够排序质量）→ 直接重排

# 否则触发回退

# Stage 2 – Relax Type, Keep Concept

# 只保留知识点过滤（concepts.uid in target_concepts），不过滤类型（所有 type 都可）

# 如果候选数 ≥ top_k → 重排

# 否则继续回退

# Stage 3 – Semantic & Profile-only Fallback

# 不再要求知识点匹配，只按画像偏好 + 语义相似度在全库里找若干候选

# 最后仍然用同一个多目标打分公式做排序

# 这样保证 “永远不会 0 结果”，但你能区分“概念匹配” vs “兜底泛资源”

# 阈值的理由：

# 对于排序问题，候选数最好 ≥ 3×top_k，这样多目标打分、语义重排才有意义（这个比例在许多学习排序/搜索重排的实践经验里都被认为比较稳健）；

# 若连 top_k（50）都凑不齐，就说明资源覆盖确实有限，必须放宽过滤，否则硬撑也没意义。

from typing import Dict, Any, List, Tuple
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pymysql


# =====================
# 全局配置
# =====================

# MongoDB
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "MLS"
FRAGMENTS_COLLECTION = "Fragments"

# 语义模型（无需训练）
SEM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# MySQL
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "mls",
    "charset": "utf8mb4",
}

# 检索参数
DEFAULT_TOPK = 50
MAX_CANDIDATES = 2000  # 每阶段最多拉这么多候选


# =====================
# 映射与工具
# =====================

PRIORITY_WEIGHT = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
}

GOAL_TYPE_MULTIPLIER = {
    "remedial": 1.3,
    "consolidation": 1.0,
    "new_learning": 1.1,
}

PREF_LEVEL_SCORE = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.2,
}


def safe_get(d: Dict, key: str, default=None):
    return d[key] if key in d else default


# =====================
# 1. 解析大模型输出
# =====================

def build_concept_weight_map(plan: Dict[str, Any]) -> Dict[str, float]:
    """
    根据 target_concepts 构建概念权重：
      w(concept) = priority_weight * goal_type_multiplier
    """
    weights: Dict[str, float] = {}
    for item in plan.get("target_concepts", []):
        cid = item.get("concept_uid")
        if not cid:
            continue
        priority = item.get("priority", "medium")
        goal_type = item.get("goal_type", "consolidation")
        pw = PRIORITY_WEIGHT.get(priority, 0.5)
        gm = GOAL_TYPE_MULTIPLIER.get(goal_type, 1.0)
        w = pw * gm
        weights[cid] = max(weights.get(cid, 0.0), w)
    return weights


def build_type_preference_map(plan: Dict[str, Any]) -> Dict[str, float]:
    """
    根据 unit_type_preferences 构建类型偏好：
      type -> [0,1] 分数
    """
    pref_map: Dict[str, float] = {}
    uprefs = safe_get(plan.get("resource_preferences", {}), "unit_type_preferences", [])
    for item in uprefs:
        t = (item.get("type") or "").lower()
        level = item.get("preference_level", "medium")
        score = PREF_LEVEL_SCORE.get(level, 0.5)
        if t:
            pref_map[t] = score

    # 没出现的类型默认中性偏好
    for t in ["video", "vr", "ar", "interact", "cooperate"]:
        pref_map.setdefault(t, 0.5)

    return pref_map


def build_feature_constraints(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    清洗 feature_constraints：
      - 保证有 name 和 desired_values
    """
    res = []
    fcs = safe_get(plan.get("resource_preferences", {}), "feature_constraints", [])
    for fc in fcs:
        name = fc.get("name")
        desired = fc.get("desired_values")
        if not name or desired is None:
            continue
        weight = float(fc.get("weight", 1.0))
        res.append({
            "name": name,
            "desired_values": desired,
            "weight": weight,
            "reason": fc.get("reason", ""),
        })
    return res


# =====================
# 2. 多级 MongoDB 粗筛（Progressive Relaxation）
# =====================

def build_query_stage(
    stage: int,
    concept_weights: Dict[str, float],
    type_prefs: Dict[str, float],
) -> Dict[str, Any]:
    """
    根据 stage 构造不同的查询条件：

    Stage 1（strict）:
        - concepts.uid in target_concepts
        - type in {pref_level in [high, medium]}

    Stage 2（relax type）:
        - concepts.uid in target_concepts
        - 不限制 type

    Stage 3（fallback）:
        - 不限制 concepts
        - 不限制 type
    """
    query: Dict[str, Any] = {}
    target_concepts = list(concept_weights.keys())

    if stage == 1:
        if target_concepts:
            query["concepts.uid"] = {"$in": target_concepts}
        strong_types = [t for t, s in type_prefs.items() if s >= 0.6]  # high or medium
        if strong_types:
            query["type"] = {"$in": strong_types}

    elif stage == 2:
        if target_concepts:
            query["concepts.uid"] = {"$in": target_concepts}
        # 不限制 type

    elif stage == 3:
        # 完全不加 concepts/type 限制，作为兜底
        pass

    return query


def multi_stage_fetch_candidates(
    mongo_client: MongoClient,
    concept_weights: Dict[str, float],
    type_prefs: Dict[str, float],
    top_k: int,
    max_candidates: int = MAX_CANDIDATES
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Progressive Relaxation 策略的候选集获取：

    Stage 1:
      - 概念 + 强偏好类型过滤
      - 若 count >= 3 * top_k 则直接采样
      - 否则继续

    Stage 2:
      - 仅按概念过滤
      - 若 count >= top_k (50) 或 count > 0，则使用
      - 否则继续

    Stage 3:
      - 不加概念/类型过滤，拉全库的 max_candidates 作为兜底
    """
    db = mongo_client[MONGO_DB_NAME]
    coll = db[FRAGMENTS_COLLECTION]

    # 阈值：3倍 top_k 作为“足够排序”的标准
    L1_min = 3 * top_k
    L2_min = top_k

    # ---------- Stage 1 ----------
    q1 = build_query_stage(1, concept_weights, type_prefs)
    if q1:
        count1 = coll.count_documents(q1)
        if count1 > 0:
            limit1 = min(count1, max_candidates)
            candidates1 = list(coll.find(q1).limit(limit1))
        else:
            candidates1 = []
    else:
        candidates1, count1 = [], 0

    if count1 >= L1_min:
        return candidates1, 1  # 使用 stage 1 的结果

    # ---------- Stage 2 ----------
    q2 = build_query_stage(2, concept_weights, type_prefs)
    if q2:
        count2 = coll.count_documents(q2)
        if count2 > 0:
            limit2 = min(count2, max_candidates)
            candidates2 = list(coll.find(q2).limit(limit2))
        else:
            candidates2 = []
    else:
        candidates2, count2 = [], 0

    if candidates2:
        # 如果 stage 1 有结果，就把 stage 1 + stage 2 合并去重
        if candidates1:
            existing_ids = {c["_id"] for c in candidates1}
            extra = [c for c in candidates2 if c["_id"] not in existing_ids]
            merged = candidates1 + extra
            return merged, 2
        else:
            return candidates2, 2

    # ---------- Stage 3 (fallback) ----------
    q3 = build_query_stage(3, concept_weights, type_prefs)
    count3 = coll.count_documents(q3)
    limit3 = min(count3, max_candidates)
    candidates3 = list(coll.find(q3).limit(limit3))

    return candidates3, 3


# =====================
# 3. 标签/类型打分
# =====================

def score_concept_match(doc: Dict[str, Any], concept_weights: Dict[str, float]) -> float:
    """
    概念相关性得分：累加 doc 中概念的权重，并截断到 [0,1]
    """
    if not concept_weights:
        return 0.0
    total = 0.0
    for c in doc.get("concepts", []):
        cid = c.get("uid")
        if cid in concept_weights:
            total += concept_weights[cid]
    return min(1.0, total)


def score_type_preference(doc: Dict[str, Any], type_prefs: Dict[str, float]) -> float:
    """
    类型偏好得分：根据 type_prefs 映射
    """
    t = (doc.get("type") or "").lower()
    return type_prefs.get(t, 0.5)


def evaluate_numeric_constraint(value, cond: Dict[str, Any]) -> bool:
    if value is None:
        return False
    op = cond.get("operator")
    v = cond.get("value")
    if v is None:
        return False
    try:
        val = float(value)
        v = float(v)
    except Exception:
        return False
    if op == "<=":
        return val <= v
    elif op == "<":
        return val < v
    elif op == ">=":
        return val >= v
    elif op == ">":
        return val > v
    elif op == "==":
        return val == v
    return False


def score_feature_match(doc: Dict[str, Any], feature_constraints: List[Dict[str, Any]]) -> float:
    """
    特征匹配得分：
      - 枚举/布尔字段： doc[name] ∈ desired_values => 1，否则 0
      - list 字段（如 role_requirement）：有交集则 1
      - 数值字段（task_steps/time_estimate）：desired_values 中可以是 {operator, value}，满足任一即 1
    """
    if not feature_constraints:
        return 0.0

    total_weight = 0.0
    score_sum = 0.0

    for fc in feature_constraints:
        name = fc["name"]
        desired = fc["desired_values"]
        w = float(fc["weight"])
        val = doc.get(name)

        # 数值或对象约束
        if name in ["task_steps", "time_estimate"] or any(isinstance(v, dict) for v in desired):
            matched = False
            for cond in desired:
                if isinstance(cond, dict) and evaluate_numeric_constraint(val, cond):
                    matched = True
                    break
            s = 1.0 if matched else 0.0
        else:
            # 枚举/布尔/List 字段
            if isinstance(val, list):
                if any(x in val for x in desired):
                    s = 1.0
                else:
                    s = 0.0
            else:
                s = 1.0 if val in desired else 0.0

        total_weight += w
        score_sum += s * w

    if total_weight == 0:
        return 0.0
    return score_sum / total_weight


# =====================
# 4. 语义相似度
# =====================

_sem_model: SentenceTransformer = None  # 懒加载


def get_semantic_model() -> SentenceTransformer:
    global _sem_model
    if _sem_model is None:
        _sem_model = SentenceTransformer(SEM_MODEL_NAME)
    return _sem_model


def build_doc_text_repr(doc: Dict[str, Any]) -> str:
    """
    把文档若干字段拼成语义表示用的文本
    """
    parts = []
    t = doc.get("type", "")
    if t:
        parts.append(f"type: {t}")

    cpt_names = [c.get("name", "") for c in doc.get("concepts", [])]
    if cpt_names:
        parts.append("concepts: " + ", ".join(cpt_names))

    for key in [
        "pedagogical_function", "difficulty_level", "cognitive_load",
        "guidance_level", "interaction_level", "social_intensity",
        "environment_complexity", "immersion_level"
    ]:
        val = doc.get(key)
        if val is not None:
            parts.append(f"{key}: {val}")

    content = doc.get("content")
    if content:
        parts.append("content: " + content)

    return "\n".join(parts)


def build_query_text_repr(plan: Dict[str, Any]) -> str:
    """
    把 target_concepts + 偏好拼成一个“查询文本”，用于语义编码
    """
    parts = []

    tc_list = plan.get("target_concepts", [])
    if tc_list:
        names = []
        for item in tc_list:
            cname = item.get("concept_name", "")
            gtype = item.get("goal_type", "")
            if cname:
                names.append(f"{cname} ({gtype})")
        if names:
            parts.append("target concepts: " + ", ".join(names))

    uprefs = safe_get(plan.get("resource_preferences", {}), "unit_type_preferences", [])
    if uprefs:
        desc = []
        for item in uprefs:
            t = item.get("type")
            pl = item.get("preference_level")
            if t and pl:
                desc.append(f"{t}: {pl} preference")
        if desc:
            parts.append("unit type preferences: " + "; ".join(desc))

    fcs = safe_get(plan.get("resource_preferences", {}), "feature_constraints", [])
    if fcs:
        for fc in fcs:
            name = fc.get("name")
            desired = fc.get("desired_values")
            if not name or desired is None:
                continue
            parts.append(f"prefer {name} in {desired}")

    notes = plan.get("strategy_notes", [])
    for n in notes:
        parts.append("strategy: " + str(n))

    return "\n".join(parts)


def compute_semantic_scores(
    plan: Dict[str, Any],
    docs: List[Dict[str, Any]]
) -> List[float]:
    """
    使用 Sentence-Transformers 计算语义相似度分数（映射到 0~1）
    """
    if not docs:
        return []

    model = get_semantic_model()
    query_text = build_query_text_repr(plan)
    query_emb = model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)[0]
    doc_texts = [build_doc_text_repr(d) for d in docs]
    doc_embs = model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=False)

    cos_scores = util.cos_sim(query_emb, doc_embs)[0].cpu().numpy()
    sem_scores = (cos_scores + 1.0) / 2.0
    return sem_scores.tolist()


# =====================
# 5. 融合总分
# =====================

def compute_overall_scores(
    plan: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    concept_weights: Dict[str, float],
    type_prefs: Dict[str, float],
    feature_constraints: List[Dict[str, Any]],
    alpha: float = 0.35,
    beta: float = 0.15,
    gamma: float = 0.25,
    delta: float = 0.25,
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    overall = alpha * concept + beta * type + gamma * feature + delta * semantic
    """
    if not candidates:
        return []

    semantic_scores = compute_semantic_scores(plan, candidates)
    scored_docs: List[Tuple[float, Dict[str, Any]]] = []

    for doc, sem in zip(candidates, semantic_scores):
        c_score = score_concept_match(doc, concept_weights)
        t_score = score_type_preference(doc, type_prefs)
        f_score = score_feature_match(doc, feature_constraints)
        overall = alpha * c_score + beta * t_score + gamma * f_score + delta * sem

        doc["_score"] = {
            "overall": overall,
            "concept_score": c_score,
            "type_score": t_score,
            "feature_score": f_score,
            "semantic_score": sem,
        }
        scored_docs.append((overall, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return scored_docs


# =====================
# 6. 主匹配接口
# =====================

def match_resources_hybrid(plan: Dict[str, Any], top_k: int = DEFAULT_TOPK) -> List[Dict[str, Any]]:
    """
    主入口：给第一次大模型输出的 plan，返回排序后的 top_k 个资源分段。
    内部使用 HR-PRR 的三阶段检索策略。
    """
    concept_weights = build_concept_weight_map(plan)
    type_prefs = build_type_preference_map(plan)
    feature_constraints = build_feature_constraints(plan)

    client = MongoClient(MONGO_URI)
    try:
        candidates, stage_used = multi_stage_fetch_candidates(
            client,
            concept_weights,
            type_prefs,
            top_k,
            max_candidates=MAX_CANDIDATES
        )
    finally:
        client.close()

    scored = compute_overall_scores(
        plan,
        candidates,
        concept_weights,
        type_prefs,
        feature_constraints
    )

    # 把使用的 stage 也塞到每个 doc 里，便于你调试和分析
    for _, doc in scored:
        doc["_retrieval_stage"] = stage_used

    return [doc for _, doc in scored[:top_k]]


# =====================
# 7. 辅助：从 MySQL 取真实概念做测试
# =====================

def get_example_concepts_from_mysql(limit: int = 10) -> List[Dict[str, str]]:
    """
    从 MySQL.mls 中读取“真实存在且在 Unit_Concept 中出现过”的若干概念：
      SELECT DISTINCT c.uid, c.name
      FROM Concepts c
      JOIN Unit_Concept uc ON uc.cpt_uid = c.uid
      LIMIT ...

    返回：[{uid, name}, ...]
    """
    conn = pymysql.connect(**MYSQL_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT DISTINCT c.uid AS cpt_uid, c.name AS cpt_name
                FROM Concepts c
                JOIN Unit_Concept uc ON uc.cpt_uid = c.uid
                LIMIT %s
            """
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            return [{"uid": r["cpt_uid"], "name": r["cpt_name"]} for r in rows]
    finally:
        conn.close()


# =====================
# 8. 调试输出
# =====================

def pretty_print_results(title: str, results: List[Dict[str, Any]], top_show: int = 5):
    print("=" * 80)
    print(f"Test case: {title}")
    print("=" * 80)
    if not results:
        print("No results found.\n")
        return

    for i, doc in enumerate(results[:top_show], start=1):
        s = doc["_score"]
        stage = doc.get("_retrieval_stage", "?")
        print(f"#{i}: uid={doc.get('uid')} type={doc.get('type')} "
              f"[stage={stage}] "
              f"concepts={[c.get('name') for c in doc.get('concepts', [])]}")
        print(f"    overall={s['overall']:.4f}, "
              f"concept={s['concept_score']:.4f}, "
              f"type={s['type_score']:.4f}, "
              f"feature={s['feature_score']:.4f}, "
              f"semantic={s['semantic_score']:.4f}")
        print(f"    pedagogical_function={doc.get('pedagogical_function')}, "
              f"difficulty={doc.get('difficulty_level')}, "
              f"guidance={doc.get('guidance_level')}, "
              f"cognitive_load={doc.get('cognitive_load')}")
        print()


# =====================
# 9. __main__：用真实 Concepts 构造 3 个场景测试
# =====================

if __name__ == "__main__":
    example_concepts = get_example_concepts_from_mysql(limit=12)

    if len(example_concepts) == 0:
        print("从 MySQL 中没有读到任何 Concepts + Unit_Concept 数据，无法构造测试用例。")
    else:
        def pick_concept(idx: int) -> Dict[str, str]:
            return example_concepts[idx % len(example_concepts)]

        c1 = pick_concept(0)
        c2 = pick_concept(1)
        c3 = pick_concept(2)
        c4 = pick_concept(3)
        c5 = pick_concept(4)
        c6 = pick_concept(5)

        # ---- 场景 1：基础补救 + 视频优先 ----
        plan_basic_remedial = {
            "target_concepts": [
                {
                    "concept_uid": c1["uid"],
                    "concept_name": c1["name"],
                    "predicted_accuracy": 0.35,
                    "status": "learned",
                    "goal_type": "remedial",
                    "priority": "high",
                    "target_accuracy": 0.8,
                    "reason": f"{c1['name']} 是后续内容的基础，该学习者掌握较弱"
                },
                {
                    "concept_uid": c2["uid"],
                    "concept_name": c2["name"],
                    "predicted_accuracy": 0.6,
                    "status": "learned",
                    "goal_type": "consolidation",
                    "priority": "medium",
                    "target_accuracy": 0.85,
                    "reason": f"{c2['name']} 与 {c1['name']} 高度相关，需要适度巩固"
                }
            ],
            "resource_preferences": {
                "unit_type_preferences": [
                    {"type": "video", "preference_level": "high", "reason": "结构化讲解适合补救"},
                    {"type": "interact", "preference_level": "medium", "reason": "适量交互练习"},
                    {"type": "vr", "preference_level": "low", "reason": "当前不强调沉浸式场景"},
                    {"type": "ar", "preference_level": "low", "reason": ""},
                    {"type": "cooperate", "preference_level": "low", "reason": ""}
                ],
                "feature_constraints": [
                    {
                        "name": "guidance_level",
                        "desired_values": ["high"],
                        "weight": 1.2,
                        "reason": "需要高引导性资源"
                    },
                    {
                        "name": "difficulty_level",
                        "desired_values": ["basic", "intermediate"],
                        "weight": 1.0,
                        "reason": "以基础/中阶补救为主"
                    },
                    {
                        "name": "example_included",
                        "desired_values": [True],
                        "weight": 1.0,
                        "reason": "示例有助于理解"
                    },
                    {
                        "name": "cognitive_load",
                        "desired_values": ["low", "medium"],
                        "weight": 0.8,
                        "reason": "避免过高认知负荷"
                    },
                    {
                        "name": "time_estimate",
                        "desired_values": [
                            {"operator": "<=", "value": 90}
                        ],
                        "weight": 0.5,
                        "reason": "控制在 1.5 分钟以内"
                    }
                ]
            },
            "strategy_notes": [
                f"先用高引导性的短视频补救 {c1['name']} 与 {c2['name']}，再配合少量交互练习。"
            ]
        }

        # ---- 场景 2：高探索型学习者 + VR/AR + 高交互 ----
        plan_exploration_vr = {
            "target_concepts": [
                {
                    "concept_uid": c3["uid"],
                    "concept_name": c3["name"],
                    "predicted_accuracy": 0.7,
                    "status": "learned",
                    "goal_type": "consolidation",
                    "priority": "medium",
                    "target_accuracy": 0.9,
                    "reason": f"{c3['name']} 是后续路径/搜索相关内容的基础"
                },
                {
                    "concept_uid": c4["uid"],
                    "concept_name": c4["name"],
                    "predicted_accuracy": -1,
                    "status": "not_learned",
                    "goal_type": "new_learning",
                    "priority": "high",
                    "target_accuracy": 0.75,
                    "reason": f"前驱知识已具备，可以在空间导航任务中引入 {c4['name']}"
                }
            ],
            "resource_preferences": {
                "unit_type_preferences": [
                    {"type": "vr", "preference_level": "high", "reason": "适合利用空间导航场景"},
                    {"type": "ar", "preference_level": "high", "reason": "增强现实中的可视化"},
                    {"type": "interact", "preference_level": "medium", "reason": "交互任务辅助理解"},
                    {"type": "video", "preference_level": "medium", "reason": "作为补充讲解"},
                    {"type": "cooperate", "preference_level": "low", "reason": "当前以个人探索为主"}
                ],
                "feature_constraints": [
                    {
                        "name": "interaction_level",
                        "desired_values": ["medium", "high"],
                        "weight": 1.2,
                        "reason": "高探索倾向学习者偏好操作性强的资源"
                    },
                    {
                        "name": "exploration_freedom",
                        "desired_values": ["high"],
                        "weight": 1.2,
                        "reason": "允许自由尝试不同路径/方案"
                    },
                    {
                        "name": "environment_complexity",
                        "desired_values": ["moderate", "complex"],
                        "weight": 0.8,
                        "reason": "适度复杂的环境更能体现差异"
                    },
                    {
                        "name": "spatial_navigation_demand",
                        "desired_values": ["medium", "high"],
                        "weight": 0.8,
                        "reason": "鼓励在空间中进行导航与搜索"
                    },
                    {
                        "name": "cognitive_load",
                        "desired_values": ["medium"],
                        "weight": 0.5,
                        "reason": "保持中等认知负荷"
                    }
                ]
            },
            "strategy_notes": [
                f"通过 VR/AR 场景中的探索任务巩固 {c3['name']}，并引入 {c4['name']}。"
            ]
        }

        # ---- 场景 3：高社交型学习者 + 协作任务优先 ----
        c5 = pick_concept(4)
        c6 = pick_concept(5)
        plan_social_cooperate = {
            "target_concepts": [
                {
                    "concept_uid": c5["uid"],
                    "concept_name": c5["name"],
                    "predicted_accuracy": 0.4,
                    "status": "learned",
                    "goal_type": "remedial",
                    "priority": "high",
                    "target_accuracy": 0.8,
                    "reason": f"{c5['name']} 与团队项目密切相关，但该学习者掌握较弱"
                },
                {
                    "concept_uid": c6["uid"],
                    "concept_name": c6["name"],
                    "predicted_accuracy": -1,
                    "status": "not_learned",
                    "goal_type": "new_learning",
                    "priority": "medium",
                    "target_accuracy": 0.7,
                    "reason": f"可在协作任务中自然引入 {c6['name']}"
                }
            ],
            "resource_preferences": {
                "unit_type_preferences": [
                    {"type": "cooperate", "preference_level": "high", "reason": "社交倾向强，适合协作任务"},
                    {"type": "interact", "preference_level": "medium", "reason": "个体操作配合协作"},
                    {"type": "video", "preference_level": "medium", "reason": "流程讲解作为背景知识"},
                    {"type": "vr", "preference_level": "low", "reason": ""},
                    {"type": "ar", "preference_level": "low", "reason": ""}
                ],
                "feature_constraints": [
                    {
                        "name": "social_intensity",
                        "desired_values": ["high"],
                        "weight": 1.2,
                        "reason": "需要高社交互动"
                    },
                    {
                        "name": "collaboration_mode",
                        "desired_values": ["group"],
                        "weight": 1.0,
                        "reason": "适合小组协作任务"
                    },
                    {
                        "name": "role_requirement",
                        "desired_values": ["leader", "coordinator", "executor"],
                        "weight": 0.8,
                        "reason": "鼓励在协作中扮演不同角色"
                    },
                    {
                        "name": "pedagogical_function",
                        "desired_values": ["practice", "demonstration"],
                        "weight": 0.8,
                        "reason": "通过实践与示范任务体验协作流程"
                    }
                ]
            },
            "strategy_notes": [
                f"通过协作式任务（如多人模拟 {c5['name']} 流程），在实践中补救并引入 {c6['name']}。"
            ]
        }

        print("Running HR-PRR hybrid matching with real Concepts from MySQL...\n")

        res1 = match_resources_hybrid(plan_basic_remedial, top_k=20)
        pretty_print_results("Scenario 1: Basic remedial + video-first", res1)

        res2 = match_resources_hybrid(plan_exploration_vr, top_k=20)
        pretty_print_results("Scenario 2: High-exploration VR/AR", res2)

        res3 = match_resources_hybrid(plan_social_cooperate, top_k=20)
        pretty_print_results("Scenario 3: High-social cooperative", res3)
