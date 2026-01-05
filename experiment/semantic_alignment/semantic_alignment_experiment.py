# -*- coding: utf-8 -*-
"""
实验：语义类别分布与场景对齐性分析（最终版动词映射）

数据源：
    本地 MongoDB，无认证
    库：MLS
    集合：Interaction

目标：
    对 5 类学习单元（video, vr, ar, interact, cooperate）中的动词，
    按最终版语义类别映射统计：
        count(u, c)
        M_{u,c} = count(u, c) / sum_{c'} count(u, c')

输出：
    semantic_distribution_counts.csv     —— 计数矩阵
    semantic_distribution_matrix.csv     —— 概率矩阵 M_{u,c}
"""

from collections import defaultdict
from pymongo import MongoClient
import pandas as pd

# ===================== 配置 =====================

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "MLS"
COLLECTION_NAME = "Interaction"

# 统一使用小写形式的 _type 作为内部编码
UNIT_TYPES_DB = ["video", "vr", "ar", "interact", "cooperate"]

# 显示在结果中的行标签
UNIT_LABELS = {
    "video": "video",
    "vr": "VR",
    "ar": "AR",
    "interact": "interact",
    "cooperate": "cooperate",
}

# 列的顺序（语义类别）
SEMANTIC_CATEGORIES = [
    "空间行为",
    "对象操作行为",
    "注意、状态与认知加工行为",
    "协作与社会交互行为",
    "任务与结果行为",
]

# ===== 最终版：verb（URL 最后那段） → 语义类别 =====
# 对应你给出的 LaTeX 表格
VERB_TO_CATEGORY = {
    # 空间行为
    "navigated-to-space": "空间行为",
    "teleported-to-space": "空间行为",

    # 对象操作行为
    "manipulated-object": "对象操作行为",
    "performed-procedure-step": "对象操作行为",
    "contributed-resource": "对象操作行为",
    "exchanged-value": "对象操作行为",

    # 注意、状态与认知加工行为
    "focused-on-resource": "注意、状态与认知加工行为",
    "reviewed-feedback": "注意、状态与认知加工行为",
    "explored-extension": "注意、状态与认知加工行为",
    "reflected-on-activity": "注意、状态与认知加工行为",
    "remained-idle": "注意、状态与认知加工行为",

    # 协作与社会交互行为
    "collaborated-on-activity": "协作与社会交互行为",
    "co-edited-artifact": "协作与社会交互行为",
    "observed-peer": "协作与社会交互行为",
    "requested-support": "协作与社会交互行为",

    # 任务与结果行为
    "experienced": "任务与结果行为",
    "initialized": "任务与结果行为",
    "completed": "任务与结果行为",
    "answered": "任务与结果行为",
    "passed": "任务与结果行为",
    "failed": "任务与结果行为",
}

VERB_KEYS = list(VERB_TO_CATEGORY.keys())


# ===================== 聚合管道构造 =====================

def build_pipeline():
    """
    聚合步骤：
      1) 从 _type 提取单位类型（小写） -> unit
      2) 从 verb.id 提取最后一段 -> verb_key
      3) 只保留 unit ∈ UNIT_TYPES_DB 且 verb_key ∈ VERB_KEYS 的文档
      4) 按 (unit, verb_key) 分组计数
    随后在 Python 端再把 verb_key 映射到语义类别。
    """
    pipeline = [
        # 1. 先投影出 unit 和 verb_id
        {
            "$project": {
                "unit": {"$toLower": "$_type"},
                "verb_id": "$verb.id",
            }
        },
        # 2. 从 verb_id 中提取最后一段作为 verb_key
        {
            "$project": {
                "unit": 1,
                "verb_key": {
                    "$arrayElemAt": [
                        {"$split": ["$verb_id", "/"]},
                        -1
                    ]
                }
            }
        },
        # 3. 过滤出关心的学习单元类型和动词
        {
            "$match": {
                "unit": {"$in": UNIT_TYPES_DB},
                "verb_key": {"$in": VERB_KEYS},
            }
        },
        # 4. 按 (unit, verb_key) 分组计数
        {
            "$group": {
                "_id": {
                    "unit": "$unit",
                    "verb_key": "$verb_key",
                },
                "count": {"$sum": 1},
            }
        },
    ]
    return pipeline


# ===================== 主流程 =====================

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    coll = db[COLLECTION_NAME]

    pipeline = build_pipeline()

    print("运行 MongoDB 聚合管道，统计 (unit, verb_key) 频次 ...")
    cursor = coll.aggregate(
        pipeline,
        allowDiskUse=True,
        batchSize=20000,
    )

    # (unit_db, category) -> count
    count_uc = defaultdict(int)
    total_by_unit = defaultdict(int)

    # 先聚合为 unit × category
    for doc in cursor:
        _id = doc["_id"]
        unit_db = _id["unit"]         # video/vr/ar/interact/cooperate（小写）
        verb_key = _id["verb_key"]
        cnt = doc["count"]

        category = VERB_TO_CATEGORY.get(verb_key)
        if category is None:
            continue

        count_uc[(unit_db, category)] += cnt
        total_by_unit[unit_db] += cnt

    # ===================== 构建矩阵 =====================

    # 行标签按指定顺序
    index_labels = [UNIT_LABELS[u] for u in UNIT_TYPES_DB]

    # 计数矩阵
    df_counts = pd.DataFrame(
        0,
        index=index_labels,
        columns=SEMANTIC_CATEGORIES,
        dtype="int64"
    )

    db2label = UNIT_LABELS

    for (unit_db, category), cnt in count_uc.items():
        row_label = db2label.get(unit_db)
        if row_label is None:
            continue
        if category not in df_counts.columns:
            continue
        df_counts.loc[row_label, category] = cnt

    # 概率矩阵 M_{u,c}
    df_probs = df_counts.astype("float64").copy()
    for row in df_probs.index:
        s = df_probs.loc[row].sum()
        if s > 0:
            df_probs.loc[row] = df_probs.loc[row] / s

    # ===================== 打印与保存 =====================

    print("\n=== 每个学习单元类型的样本总数（归入语义体系的行为数） ===")
    for unit_db in UNIT_TYPES_DB:
        label = db2label[unit_db]
        print(f"{label}: {total_by_unit.get(unit_db, 0)}")

    print("\n=== 语义类别计数矩阵 count(u, c) ===")
    with pd.option_context("display.width", 200):
        print(df_counts)

    print("\n=== 语义类别分布矩阵 M_{u,c}（行归一化） ===")
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(df_probs)

    counts_csv = "semantic_distribution_counts.csv"
    probs_csv = "semantic_distribution_matrix.csv"

    df_counts.to_csv(counts_csv, encoding="utf-8-sig")
    df_probs.to_csv(probs_csv, encoding="utf-8-sig")

    print(f"\n计数矩阵已保存到：{counts_csv}")
    print(f"分布矩阵 M_u,c 已保存到：{probs_csv}")


if __name__ == "__main__":
    main()
