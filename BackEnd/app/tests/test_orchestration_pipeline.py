# test_orchestration_pipeline.py
# -*- coding: utf-8 -*-

"""
测试 OrchestrationPipeline 的端到端脚本（虚拟学习者 + 真实知识点/关系 + 随机学习状态/得分）。

运行前：
1) 确保 MySQL 可连接：
   - host=127.0.0.1, port=3306
   - user=root, password=123456
   - database=mls
2) 确保大模型环境变量已配置（至少 key/base_url/provider）：
   - 例如使用 Aizex：
       export LLM_PROVIDER=aizex
       export LLM_AIZEX_BASE_URL=https://aizex.top/v1
       export LLM_AIZEX_API_KEY=sk-xxxx
       export LLM_AIZEX_DEFAULT_MODEL=gpt-4.1-nano
3) 确保工程 import 路径正确（能 import app.xxx）

注意：
- 本脚本的重点是“pipeline 编排能跑通”，不是评测质量。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import pymysql

from app.domain.orchestration.orchestration_pipeline import OrchestrationPipeline
from app.shared.models.profiles_labels import PROFILE_LABELS, get_label  # :contentReference[oaicite:1]{index=1}


# ----------------------------
# MySQL 读取：Concepts 与 Concept_Concept
# ----------------------------
def fetch_concepts_and_relations(
    host: str = "127.0.0.1",
    port: int = 3306,
    user: str = "root",
    password: str = "123456",
    database: str = "mls",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    返回：
      - concepts_by_uid: uid -> {"uid","name","explanation"}
      - predecessors_map: uid -> [pre_uid,...]
      - successors_map: uid -> [aft_uid,...]
    """
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT uid, name, explanation FROM Concepts")
            rows = cur.fetchall()

            concepts_by_uid: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                uid = r["uid"]
                concepts_by_uid[uid] = {
                    "concept_uid": uid,
                    "concept_name": r["name"],
                    "explanation": r.get("explanation"),
                }

            cur.execute("SELECT pre_uid, aft_uid FROM Concept_Concept")
            edges = cur.fetchall()

            predecessors_map: Dict[str, List[str]] = {}
            successors_map: Dict[str, List[str]] = {}

            for e in edges:
                pre = e["pre_uid"]
                aft = e["aft_uid"]
                if not pre or not aft:
                    continue

                # 仅保留在 concepts 表中存在的 uid
                if pre not in concepts_by_uid or aft not in concepts_by_uid:
                    continue

                successors_map.setdefault(pre, []).append(aft)
                predecessors_map.setdefault(aft, []).append(pre)

        return concepts_by_uid, predecessors_map, successors_map
    finally:
        conn.close()


# ----------------------------
# 随机生成学习者画像（转成文本标签）
# ----------------------------
def random_profile_labels() -> Dict[str, Any]:
    """
    从 PROFILE_LABELS 随机抽样，生成一个“文本化的画像标签结构”。

    输出结构示例：
    {
      "attention_allocation": {"efficiency": "...", "style": "..."},
      "social_learning": {"role": "..."},
      ...
    }
    """
    profile: Dict[str, Any] = {}

    for dimension, dim_cfg in PROFILE_LABELS.items():
        profile[dimension] = {}
        for category, mapping in dim_cfg.items():
            codes = list(mapping.keys())
            code = random.choice(codes)
            profile[dimension][category] = get_label(dimension, category, code)

    return profile


# ----------------------------
# 构建 knowledge_concepts（真实概念 + 随机学习状态/得分）
# ----------------------------
def build_knowledge_concepts(
    concepts_by_uid: Dict[str, Dict[str, Any]],
    predecessors_map: Dict[str, List[str]],
    successors_map: Dict[str, List[str]],
    learned_ratio: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    将数据库真实概念与关系转换为 pipeline/LLM 输入结构：
      - concept_uid, concept_name
      - status: learned / not_learned
      - predicted_accuracy: learned 才给 [0,1] 随机；not_learned -> None（或不传）
      - predecessors/successors: 真实关系
    """
    knowledge_concepts: List[Dict[str, Any]] = []

    for uid, base in concepts_by_uid.items():
        status = "learned" if random.random() < learned_ratio else "not_learned"

        item = {
            "concept_uid": base["concept_uid"],
            "concept_name": base["concept_name"],
            "status": status,
            "predecessors": predecessors_map.get(uid, []),
            "successors": successors_map.get(uid, []),
        }

        if status == "learned":
            # learned 才给预测分
            item["predicted_accuracy"] = round(random.uniform(0.15, 0.98), 4)
        else:
            # not_learned 不需要 -1；可不给，也可置 None
            item["predicted_accuracy"] = None

        knowledge_concepts.append(item)

    return knowledge_concepts


def main() -> None:
    # 1) 读取真实知识点与关系
    concepts_by_uid, predecessors_map, successors_map = fetch_concepts_and_relations()

    # 2) 构建虚拟学习者
    learner_uid = "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e"
    learner_profile = random_profile_labels()

    # 3) 构建知识状态（真实概念 + 随机学习状态）
    knowledge_concepts = build_knowledge_concepts(
        concepts_by_uid,
        predecessors_map,
        successors_map,
        learned_ratio=0.35,
    )

    # 4) 运行 pipeline
    pipeline = OrchestrationPipeline(llm_provider=None, device="cpu")

    result = pipeline.analyze(
        learner_uid=learner_uid,
        learner_profile=learner_profile,
        knowledge_concepts=knowledge_concepts,
        # 允许测试时手动切换模型（可填 None 用默认）
        llm1_model=None,
        llm2_model=None,
        top_k=20,
        constraints={
            "max_total_time": 60,   # 例：希望总时长不超过 60 分钟（第二次LLM可参考）
            "max_steps": 8,         # 例：最多 8 步
        },
    )

    # 5) 打印关键结果（避免输出过大）
    print("\n=== SUMMARY ===")
    summary = result.get("summary", {})
    print("Targets (count):", len(summary.get("target_concepts", [])))
    print("Top resources:", len(summary.get("recommended_resources_top10", [])))
    print("Steps:", len(summary.get("learning_steps", [])))
    print("Overview:", summary.get("path_overview"))

    # 如果你想看每步理由：
    print("\n=== STEP REASONS ===")
    for s in summary.get("learning_steps", []):
        print(f"- Step {s.get('step_index')} | goal={s.get('goal')} | time={s.get('time_estimate')}min")
        print("  resource_uids:", s.get("resource_uids"))
        print("  why:", s.get("why"))
        print()

    # 如果你想看原始输出（可能很大），可自行 print(result)


if __name__ == "__main__":
    main()
