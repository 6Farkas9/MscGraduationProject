# app/tests/test_partner_pipeline.py
# -*- coding: utf-8 -*-
"""
简单测试 PartnerRecommendationPipeline 的脚本。

特点
----
- 不依赖 pytest，只做：
    1. 构造一批模拟学习者画像 + 知识点预测数据（全在内存中）；
    2. 调用 PartnerRecommendationPipeline.analyze(learner_uids, data)；
    3. 做一些基本的结果检查（用 assert）；
    4. 打印耗时和部分结果，粗略观察性能与输出结构。

使用方式
--------
python -m app.tests.test_partner_pipeline
或
python app/tests/test_partner_pipeline.py
"""

import random
import time
from typing import Any, Dict, List

from app.domain.partner.partner_pipeline import PartnerRecommendationPipeline

# 为了测试可重复，固定随机种子
random.seed(2025)


# ------------------------ 模拟画像标签空间（仿 profiles_labels） ------------------------

PROFILE_LABEL_SPACE: Dict[str, Dict[str, List[str]]] = {
    "attention_allocation": {
        "level": [
            "高度专注型",
            "中等专注型",
            "容易分心型",
        ],
        "style": [
            "均衡型",
            "任务切换型",
            "爆发冲刺型",
        ],
        "efficiency": [
            "高效",
            "中等",
            "低效",
        ],
    },
    "engagement_persistence": {
        "level": [
            "高投入",
            "中等投入",
            "低投入",
        ],
        "pattern": [
            "持续型",
            "间歇型",
            "短冲刺型",
        ],
    },
    "feedback_orientation": {
        "level": [
            "积极接纳反馈",
            "选择性接纳反馈",
            "回避反馈",
        ],
    },
    "collaborative_role_contribution": {
        "role": [
            "组织者",
            "协调者",
            "执行者",
            "观察者",
        ],
        "contribution_type": [
            "解释与总结",
            "资源分享",
            "提问与澄清",
            "情绪支持",
        ],
    },
    "contribution_reputation": {
        "level": [
            "高声誉贡献者",
            "中等贡献者",
            "低贡献者",
        ],
    },
    "interaction_style": {
        "style": [
            "高频互动型",
            "中频互动型",
            "低频互动型",
            "潜水型",
        ],
    },
    "reflection_value_evolution": {
        "level": [
            "高反思成长型",
            "稳定反思型",
            "浅层反思型",
        ],
    },
    "social_learning": {
        "role": [
            "积极社会学习型",
            "跟随型学习者",
            "独立学习型",
        ],
    },
    "exploration_orientation": {
        "level": [
            "高探索倾向",
            "适度探索",
            "保守稳健",
        ],
    },
    "srl_helpseeking": {
        "level": [
            "主动求助型",
            "选择性求助型",
            "少求助型",
        ],
    },
    "task_efficiency": {
        "level": [
            "高效完成者",
            "按时完成者",
            "拖延型完成者",
        ],
    },
}


# ------------------------ 构造模拟数据的工具函数 ------------------------

def generate_concept_uids(n_concepts: int = 80) -> List[str]:
    """
    构造一组模拟的知识点 uid。
    示例：["kp_001", "kp_002", ...]
    """
    return [f"kp_{i:03d}" for i in range(1, n_concepts + 1)]


def sample_kt_for_learner(concept_uids: List[str]) -> Dict[str, float]:
    """
    为一个学习者生成 KT 字典：
    - 从全部知识点中抽一部分；
    - 使用不同能力段 + Beta 分布生成预测准确率，避免纯随机。
    """
    kt: Dict[str, float] = {}

    if not concept_uids:
        return kt

    total_concepts = len(concept_uids)
    min_concepts = min(20, total_concepts)
    max_concepts = min(60, total_concepts)
    if max_concepts < min_concepts:
        chosen_uids = concept_uids
    else:
        k = random.randint(min_concepts, max_concepts)
        chosen_uids = random.sample(concept_uids, k=k)

    # 给学习者分配一个能力段
    ability_group = random.choices(
        population=["high", "medium", "low"],
        weights=[3, 4, 3],
        k=1,
    )[0]

    if ability_group == "high":
        alpha, beta_param = 8.0, 3.0
    elif ability_group == "medium":
        alpha, beta_param = 5.0, 5.0
    else:
        alpha, beta_param = 3.0, 7.0

    for cu in chosen_uids:
        p = random.betavariate(alpha, beta_param)
        p = max(0.05, min(0.99, p))
        kt[cu] = float(round(p, 4))

    return kt


def sample_profile_for_learner() -> Dict[str, Any]:
    """
    按照 PROFILE_LABEL_SPACE 为单个学习者生成完整的 11 维画像。
    """
    profiles: Dict[str, Any] = {}

    for dim, subspace in PROFILE_LABEL_SPACE.items():
        dim_dict: Dict[str, Any] = {}
        for sub_key, labels in subspace.items():
            if not labels:
                continue
            dim_dict[sub_key] = random.choice(labels)
        profiles[dim] = dim_dict

    return profiles


def build_mock_inputs(
    n_learners: int = 1000,
    n_concepts: int = 80,
    n_targets: int = 20,
):
    """
    构造模拟的三类输入：
    - learner_uids: 目标学习者列表（长度 n_targets）
    - data: 所有学习者的聚合输入（给 pipeline / engine 使用）

    data 结构与实际调用保持一致：
    data = {
      "<uid>": {
        "learner_profile": {...},        # 画像（11 维）
        "knowledge_concepts": {...},     # KT 向量
      },
      ...
    }
    """
    concept_uids = generate_concept_uids(n_concepts)

    data: Dict[str, Any] = {}
    all_uids: List[str] = []

    for i in range(1, n_learners + 1):
        uid = f"lrn_test_{i:05d}"
        all_uids.append(uid)

        data[uid] = {
            "learner_profile": sample_profile_for_learner(),
            "knowledge_concepts": sample_kt_for_learner(concept_uids),
        }

    # 从所有学习者中随机选出部分作为“需要推荐”的目标学习者
    n_targets = min(n_targets, len(all_uids))
    target_uids = random.sample(all_uids, k=n_targets)

    return target_uids, data


# ------------------------ 简单跑通 + 性能测试 ------------------------

def simple_run():
    """
    主入口：构造模拟数据，跑一遍 pipeline，打印耗时和部分结果。
    """
    # 可以在这里调规模，看看性能大致情况
    N_LEARNERS = 2000      # 总学习者数（候选池规模）
    N_CONCEPTS = 120       # 知识点数
    N_TARGETS = 30         # 目标学习者数（一次推荐的人数）

    print(
        f"[INFO] 构造模拟数据: learners={N_LEARNERS}, concepts={N_CONCEPTS}, targets={N_TARGETS}"
    )
    t0 = time.perf_counter()
    learner_uids, data = build_mock_inputs(
        n_learners=N_LEARNERS,
        n_concepts=N_CONCEPTS,
        n_targets=N_TARGETS,
    )
    t1 = time.perf_counter()
    print(f"[INFO] 数据构造完成，耗时 {t1 - t0:.3f} 秒")

    print("[INFO] 初始化 PartnerRecommendationPipeline...")
    pipeline = PartnerRecommendationPipeline()

    print("[INFO] 调用 pipeline.analyze(learner_uids, data) ...")
    t2 = time.perf_counter()
    result = pipeline.analyze(
        learner_uids=learner_uids,
        data=data,
    )
    t3 = time.perf_counter()

    elapsed = t3 - t2
    print(f"[INFO] pipeline.analyze 完成，耗时 {elapsed:.3f} 秒")
    if elapsed > 0:
        print(
            f"[INFO] 平均每个目标学习者耗时约 {elapsed / len(learner_uids):.6f} 秒"
        )

    # 基本结构检查（用 assert 即可）
    assert isinstance(result, dict), "pipeline 输出必须是 dict"
    assert "engine_status" in result, "缺少 engine_status 字段"
    assert "results" in result, "缺少 results 字段"

    results_map = result["results"]
    assert isinstance(results_map, dict), "results 必须是 dict"

    for uid in learner_uids:
        assert uid in results_map, f"结果中缺少目标学习者 {uid}"
        rec = results_map[uid]
        assert "partner" in rec, f"{uid} 结果中缺少 'partner' 字段"
        assert "role_model" in rec, f"{uid} 结果中缺少 'role_model' 字段"
        assert isinstance(rec["partner"], list)
        assert isinstance(rec["role_model"], list)

    # 打印前若干个学习者的结果做人工检查
    from pprint import pprint

    print("\n=== Engine Status ===")
    pprint(result.get("engine_status"))

    print("\n=== 部分学习者推荐结果示例 ===")
    show_num = min(3, len(learner_uids))
    for i in range(show_num):
        uid = learner_uids[i]
        print(f"\n--- Learner: {uid} ---")
        pprint(results_map[uid])


if __name__ == "__main__":
    simple_run()
