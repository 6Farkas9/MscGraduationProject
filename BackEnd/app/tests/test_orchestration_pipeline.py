# test_orchestration_pipeline.py
# -*- coding: utf-8 -*-

"""
简单测试脚本：
- 不依赖 pytest；
- 不引用 mysql_base_repository；
- 直接连接本地 MySQL（mls.Concepts）获取真实知识点；
- 使用 profiles_labels 生成真实画像标签；
- 调用 OrchestrationPipeline.analyze 并打印结果摘要。
"""

from __future__ import annotations
import random
from typing import Dict, Any, List

import pymysql

from app.domain.orchestration.orchestration_pipeline import OrchestrationPipeline
from app.shared.models.profiles_labels import PROFILE_LABELS, get_label


# -------------------------------------------------------------
# 1. 从本地 MySQL mls.Concepts 表读取真实知识点
# -------------------------------------------------------------
def fetch_real_concepts_from_mysql(
    host: str = "localhost",
    port: int = 3306,
    user: str = "root",
    password: str = "123456",
    database: str = "mls",
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    直接通过 pymysql 从本地 mls.Concepts 读取真实知识点。
    不使用 mysql_base_repository。
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
        with conn.cursor() as cursor:
            cursor.execute("SELECT uid, name FROM Concepts LIMIT %s", (limit,))
            rows = cursor.fetchall()
    finally:
        conn.close()

    return rows


# -------------------------------------------------------------
# 2. 构造模拟 KT：使用真实 concept_uid，概率随机
# -------------------------------------------------------------
def build_mock_kt(learner_uids: List[str], concepts: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    为每个学习者构造 KT：
    { learner_uid: { concept_uid: prob, ... } }
    概率在 [0.4, 0.95] 之间随机。
    """
    kt: Dict[str, Dict[str, float]] = {}

    concept_uids = [c["uid"] for c in concepts if c.get("uid")]

    for uid in learner_uids:
        cur_kt: Dict[str, float] = {}
        # 随机选取若干个知识点（例如 5 个以内）
        chosen = random.sample(concept_uids, min(len(concept_uids), 5))
        for cid in chosen:
            prob = round(random.uniform(0.4, 0.95), 3)
            cur_kt[cid] = prob
        kt[uid] = cur_kt

    return kt


# -------------------------------------------------------------
# 3. 构造模拟 Profile：使用 profiles_labels 的真实标签
# -------------------------------------------------------------
def build_mock_profile(learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    为每个学习者构造画像：
    { learner_uid: { dimension: { category: label_text, ... }, ... } }

    - dimension / category 使用 PROFILE_LABELS 的真实配置；
    - code 随机选一个有效 code，通过 get_label 转成文本描述；
    - 与用户之前给的示例格式一致（直接使用中文描述文本）。
    """
    profiles: Dict[str, Dict[str, Any]] = {}

    for uid in learner_uids:
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

        profiles[uid] = profile_for_learner

    return profiles


# -------------------------------------------------------------
# 4. 主测试流程
# -------------------------------------------------------------
def main():
    # 为了在测试输出中稳定一点，固定随机种子（如果你想每次都不同，可以注释掉）
    random.seed(42)

    # 4.1 准备学习者 UID
    learner_uids = ["learner_001", "learner_002"]

    # 4.2 从 MySQL 读取真实概念
    concepts = fetch_real_concepts_from_mysql(limit=10)
    if not concepts:
        print("从数据库 mls.Concepts 中没有读取到任何知识点，请检查数据库配置。")
        return

    print(f"从数据库中读取到 {len(concepts)} 个知识点示例：")
    for row in concepts:
        print(f"  uid={row['uid']}, name={row['name']}")

    # 4.3 构造 KT 和 Profile
    kt = build_mock_kt(learner_uids, concepts)
    profile = build_mock_profile(learner_uids)

    print("\n模拟 KT 输入示例（仅展示第一个学习者）：")
    first_uid = learner_uids[0]
    print(f"  learner={first_uid}, KT={kt[first_uid]}")

    print("\n模拟 Profile 输入示例（仅展示第一个学习者的一部分）：")
    first_profile_dim_sample = dict(list(profile[first_uid].items())[:2])  # 截两项维度示例
    for dim, cat in first_profile_dim_sample.items():
        print(f"  [{dim}] -> {cat}")

    # 4.4 调用 OrchestrationPipeline
    pipeline = OrchestrationPipeline()
    result = pipeline.analyze(
        learner_uids=learner_uids,
        kt=kt,
        profile=profile,
    )

    # 4.5 打印结果摘要
    print("\n=== Pipeline Engine Status ===")
    for k, v in result.get("engine_status", {}).items():
        print(f"- {k}: {v}")

    print("\n=== Pipeline Results (per learner) ===")
    for uid in learner_uids:
        r = result["results"].get(uid)
        if not r:
            print(f"\n[ {uid} ] 无结果")
            continue

        print(f"\n[ {uid} ]")
        # 只打印规划的一小部分和资源数量，避免输出过长
        planning = r.get("planning") or {}
        orchestration = r.get("orchestration") or {}
        learning_path_text = r.get("learning_path")

        print("  - Planning (keys):", list(planning.keys()))
        print("  - Orchestration candidate_count:",
              orchestration.get("candidate_count"),
              "top_k:", orchestration.get("top_k"),
              "used_relaxation_level:", orchestration.get("used_relaxation_level"))

        resources = orchestration.get("resources") or []
        print("  - Orchestration first resource (if any):")
        if resources:
            print("    ", resources[0])
        else:
            print("    无资源匹配到。")

        print("  - Learning Path (first 400 chars)：")
        if isinstance(learning_path_text, str):
            preview = learning_path_text[:400]
            print("    ", preview.replace("\n", "\\n"))
        else:
            print("    无学习路线文本。")


if __name__ == "__main__":
    main()
