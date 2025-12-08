# generate_fake_learner_profiles.py
# -*- coding: utf-8 -*-
"""
批量生成模拟 LearnerProfile 文档，写入本地 MongoDB 的 MLS.LearnerProfile 集合。

前置条件：
- 本地 MongoDB，无认证：
    URI: mongodb://localhost:27017
    DB:  MLS
    Collection: LearnerProfile
- 本地 MySQL：
    Host: localhost
    Port: 3306
    User: root
    Password: 123456
    DB: mls
    Table: Concepts(uid, name, ...)

脚本功能：
- 从 MySQL.mls.Concepts 中读取所有知识点 uid；
- 基于这些知识点，为约 3000 个模拟学习者生成：
    * uid: lrn_sim_XXXXXX
    * KT: { concept_uid: predicted_accuracy }
    * updated_time: 最近 60 天内的随机时间
    * profiles: 11 个维度的画像标签，按照 profiles_labels 覆盖所有维度
- 批量插入至 MongoDB 的 LearnerProfile 集合中。
"""

import random
import string
from datetime import datetime, timedelta
from typing import Dict, Any, List

from pymongo import MongoClient
import pymysql


# ------------------------ 配置区域 ------------------------

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "MLS"
MONGO_COLLECTION_NAME = "LearnerProfile"

MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "123456"
MYSQL_DB = "mls"

TARGET_LEARNER_COUNT = 3000

# 为了可重复性，可以固定随机种子；如果你希望每次都不一样，可以注释掉这一行
random.seed(2024)


# ------------------------ 画像标签空间（仿 profiles_labels） ------------------------

# 维度和子键结构与之前的 profiles_labels 规范一致
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


# ------------------------ 工具函数 ------------------------

def get_mysql_concept_uids() -> List[str]:
    """
    从 MySQL.mls.Concepts 表中取出所有知识点 uid。
    """
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT uid FROM Concepts")
            rows = cursor.fetchall()
            concept_uids = [row["uid"] for row in rows if row.get("uid")]
            print(f"[INFO] 从 MySQL Concepts 中读取到 {len(concept_uids)} 个知识点 uid")
            return concept_uids
    finally:
        conn.close()


def connect_mongo():
    """
    连接 MongoDB 并返回 collection 句柄。
    """
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]
    return client, collection


def generate_sim_learner_uid(index: int) -> str:
    """
    生成模拟学习者 uid，避免和真实学习者混淆。
    例如：lrn_sim_000001
    """
    return f"lrn_sim_{index:06d}"


def sample_kt_for_learner(concept_uids: List[str]) -> Dict[str, float]:
    """
    为一个学习者生成 KT 向量：
    - 从所有知识点中抽样一部分作为“已学/有记录”的知识点；
    - 为这些知识点生成 [0,1] 的预测正确率；
    - 使用不同能力段 + Beta 分布，避免纯随机导致分布不自然。
    """
    kt: Dict[str, float] = {}

    if not concept_uids:
        return kt

    total_concepts = len(concept_uids)

    # 每个学习者只覆盖部分知识点，模拟“选修/学习路径不同”
    min_concepts = min(30, total_concepts)
    max_concepts = min(150, total_concepts)
    if max_concepts < min_concepts:
        # 知识点很少的情况下，全部都算进去
        chosen_uids = concept_uids
    else:
        k = random.randint(min_concepts, max_concepts)
        chosen_uids = random.sample(concept_uids, k=k)

    # 为了制造“高水平 / 中等 / 较弱”的分群，每个学习者先抽一个能力段
    ability_group = random.choices(
        population=["high", "medium", "low"],
        weights=[3, 4, 3],  # 稍微偏向中等
        k=1,
    )[0]

    # 不同能力段使用不同的 Beta 分布参数
    if ability_group == "high":
        alpha, beta_param = 8.0, 3.0   # 均值大约在 0.7~0.8
    elif ability_group == "medium":
        alpha, beta_param = 5.0, 5.0   # 均值大约在 0.5
    else:  # "low"
        alpha, beta_param = 3.0, 7.0   # 均值大约在 0.3 左右

    for cu in chosen_uids:
        p = random.betavariate(alpha, beta_param)
        # 限制在 [0.05, 0.99] 避免极端 0 或 1
        p = max(0.05, min(0.99, p))
        kt[cu] = float(round(p, 4))

    return kt


def sample_profile_for_learner() -> Dict[str, Any]:
    """
    根据 PROFILE_LABEL_SPACE 为单个学习者生成完整的 11 维画像。
    每个维度下的子键从该维度预定义标签中均匀随机采样。

    由于有 3000 个学习者，且每个维度的标签通常只有 3~4 种，
    这种均匀随机基本可以保证每种标签都有大量样本，利于后续匹配。
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


def generate_single_learner_doc(index: int, concept_uids: List[str]) -> Dict[str, Any]:
    """
    生成一个完整的 LearnerProfile 文档（不带 _id，交给 Mongo 自动生成）。
    """
    uid = generate_sim_learner_uid(index)

    kt = sample_kt_for_learner(concept_uids)
    profiles = sample_profile_for_learner()

    # 最近 N 天内的随机时间
    now = datetime.utcnow()
    max_days_ago = 60
    days_ago = random.randint(0, max_days_ago)
    seconds_ago = random.randint(0, 24 * 3600 - 1)
    updated_time = now - timedelta(days=days_ago, seconds=seconds_ago)

    doc = {
        "uid": uid,
        "KT": kt,
        "updated_time": updated_time,
        "profiles": profiles,
    }
    return doc


def get_existing_sim_uids(collection) -> List[str]:
    """
    查一下集合中已经存在的模拟学习者 uid（以 lrn_sim_ 前缀区分）。
    如果你想重复跑脚本，又不想插入重复 uid，可以用这个做简单防重复。
    """
    existing = collection.find(
        {"uid": {"$regex": r"^lrn_sim_"}},
        {"uid": 1, "_id": 0},
    )
    return [doc["uid"] for doc in existing if "uid" in doc]


def main():
    # 1. 读取知识点 uid
    concept_uids = get_mysql_concept_uids()
    if not concept_uids:
        print("[WARN] MySQL Concepts 表中没有任何 uid，KT 向量将为空。")
    else:
        print(f"[INFO] 将基于 {len(concept_uids)} 个知识点构造 KT 向量")

    # 2. 连接 MongoDB
    client, collection = connect_mongo()

    try:
        # 已经存在的模拟 uid，防止重复生成
        existing_sim_uids = set(get_existing_sim_uids(collection))
        print(f"[INFO] 当前集合中已存在 {len(existing_sim_uids)} 个模拟学习者 uid")

        docs_to_insert = []
        created_count = 0
        index = 1

        # 控制目标总量为 TARGET_LEARNER_COUNT（不包含已经存在的）
        while created_count < TARGET_LEARNER_COUNT:
            uid = generate_sim_learner_uid(index)
            index += 1

            if uid in existing_sim_uids:
                continue

            doc = generate_single_learner_doc(index=created_count + 1, concept_uids=concept_uids)
            # 为了安全起见，还是用 doc 里的 uid 覆盖一下
            doc["uid"] = uid

            docs_to_insert.append(doc)
            created_count += 1

        if not docs_to_insert:
            print("[INFO] 没有新文档需要插入。")
            return

        # 3. 批量插入（一次性 3000 条 MongoDB 完全没问题）
        result = collection.insert_many(docs_to_insert)
        print(f"[INFO] 成功插入 {len(result.inserted_ids)} 条模拟 LearnerProfile 文档。")

    finally:
        client.close()


if __name__ == "__main__":
    main()
