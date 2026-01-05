# -*- coding: utf-8 -*-
"""
实验三：Context / Extensions 语义构件利用度统计脚本

目标：
1. 统计细粒度 xAPI 语句中 Context 与 Extensions 语义字段的使用情况：
   - 对每个字段 f，统计在所有语句中的非空出现次数 count(f)
   - 计算全局利用度 U(f) = count(f) / N

2. 统计字段在不同学习单元类型 u 中的条件利用度：
   - 对每个字段 f、每个单元类型 u（即 _type），统计 count(f, u)
   - 计算 U(f | u) = count(f, u) / sum_{f'} count(f', u)

说明：
- 数据来源：MongoDB.MLS.Interaction
- 会话结构、动词等与前两个实验一致，这里只关心 context / extensions。
- 字段集合 F 自动从实际数据中“扫”出来：
  - context.contextActivities.parent
  - context.contextActivities.grouping
  - context.extensions 下所有具体 key（完整 IRI）

输出：
1) field_usage_global.csv
   - 列：field_name, count, U_global

2) field_usage_by_type_counts.csv
   - 列：field_name, <unit_type_1>, <unit_type_2>, ...

3) field_usage_by_type_probs.csv
   - 列：field_name, <unit_type_1>, <unit_type_2>, ...
   - 值：U(f | u)
"""

from pymongo import MongoClient, ASCENDING
from collections import defaultdict
from tqdm import tqdm
import csv

# ===================== 配置 =====================

MONGO_CONFIG = {
    "host": "localhost",
    "port": 27017,
    "db_name": "MLS",
    "xapi_collection": "Interaction",
}

# 输出文件
GLOBAL_USAGE_CSV = "field_usage_global.csv"
BY_TYPE_COUNTS_CSV = "field_usage_by_type_counts.csv"
BY_TYPE_PROBS_CSV = "field_usage_by_type_probs.csv"

MONGO_BATCH_SIZE = 10000


# ===================== 主逻辑 =====================

def main():
    # 1. 连接 MongoDB
    client = MongoClient(MONGO_CONFIG["host"], MONGO_CONFIG["port"])
    db = client[MONGO_CONFIG["db_name"]]
    col = db[MONGO_CONFIG["xapi_collection"]]

    # 估计文档总数，用于进度条
    total_docs = col.estimated_document_count()
    print(f"预计将处理细粒度语句数量 N = {total_docs}")

    # 2. 准备游标：只取本实验需要的字段
    cursor = col.find(
        {},
        projection={
            "_id": 0,
            "_type": 1,
            "context": 1,
        },
    ).sort([
        ("_lrn_uid", ASCENDING),
        ("_session_id", ASCENDING),
        ("timestamp", ASCENDING),
    ]).batch_size(MONGO_BATCH_SIZE)

    # 3. 统计结构
    total_N = 0  # 全部语句数量 N
    field_counts = defaultdict(int)  # count(f)
    field_counts_by_type = defaultdict(lambda: defaultdict(int))  # count(f, u)
    unit_types_set = set()  # 收集所有出现过的 _type

    # 4. 遍历语句
    with tqdm(total=total_docs, desc="统计 Context / Extensions 利用度") as pbar:
        for doc in cursor:
            pbar.update(1)
            total_N += 1

            u_type = doc.get("_type", "unknown")
            unit_types_set.add(u_type)

            context = doc.get("context") or {}
            ctx_acts = context.get("contextActivities") or {}
            extensions = context.get("extensions") or {}

            # ---- ContextActivities: parent / grouping ----
            parent = ctx_acts.get("parent")
            if parent:
                # 非空 parent 视为字段 "context.contextActivities.parent" 被使用
                field_name = "context.contextActivities.parent"
                field_counts[field_name] += 1
                field_counts_by_type[u_type][field_name] += 1

            grouping = ctx_acts.get("grouping")
            if grouping:
                # 非空 grouping 视为字段 "context.contextActivities.grouping" 被使用
                field_name = "context.contextActivities.grouping"
                field_counts[field_name] += 1
                field_counts_by_type[u_type][field_name] += 1

            # ---- Extensions: 每个 key 单独作为一个字段 f ----
            # 例如：
            # "https://legend-meta.com/xapi/ext/space-id"
            # "https://legend-meta.com/xapi/ext/focus-target-id"
            for key, val in extensions.items():
                # 非空（非 None、非空字符串）则认为使用
                if val is None:
                    continue
                if isinstance(val, str) and val.strip() == "":
                    continue

                field_name = key  # 直接使用完整 IRI 作为字段名
                field_counts[field_name] += 1
                field_counts_by_type[u_type][field_name] += 1

    print(f"实际统计语句数量 N = {total_N}")
    print(f"发现字段数量 |F| = {len(field_counts)}")
    print(f"发现单元类型（_type）数量 = {len(unit_types_set)} ：{sorted(unit_types_set)}")

    # 将字段集合、类型集合排序，便于输出
    all_fields = sorted(field_counts.keys())
    unit_types = sorted(unit_types_set)

    # 5. 计算全局利用度 U(f)
    # U(f) = count(f) / N
    print("计算全局利用度 U(f)...")
    global_usage_rows = []
    for f in all_fields:
        cnt = field_counts[f]
        uf = cnt / total_N if total_N > 0 else 0.0
        global_usage_rows.append((f, cnt, uf))

    # 输出：field_usage_global.csv
    with open(GLOBAL_USAGE_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["field_name", "count", "U_global"])
        for f, cnt, uf in global_usage_rows:
            writer.writerow([f, cnt, uf])

    print(f"全局利用度结果已写入：{GLOBAL_USAGE_CSV}")

    # 6. 按单元类型统计 count(f, u) 与 U(f | u)
    # U(f | u) = count(f, u) / sum_{f'} count(f', u)
    print("计算按单元类型的条件利用度 U(f | u)...")

    # 先计算每个 u 的分母：sum_{f'} count(f', u)
    denom_by_type = {}
    for u in unit_types:
        denom_by_type[u] = sum(field_counts_by_type[u].values())

    # 准备用于输出的矩阵
    # counts_matrix[f][u] = count(f, u)
    # probs_matrix[f][u]  = U(f | u)
    counts_matrix = {}
    probs_matrix = {}

    for f in all_fields:
        counts_matrix[f] = {}
        probs_matrix[f] = {}
        for u in unit_types:
            c_fu = field_counts_by_type[u].get(f, 0)
            counts_matrix[f][u] = c_fu

            denom = denom_by_type.get(u, 0)
            if denom > 0:
                probs_matrix[f][u] = c_fu / denom
            else:
                probs_matrix[f][u] = 0.0

    # 7. 输出 count(f, u) 矩阵
    with open(BY_TYPE_COUNTS_CSV, "w", newline="", encoding="utf-8") as f_counts:
        writer = csv.writer(f_counts)
        header = ["field_name"] + unit_types
        writer.writerow(header)

        for f in all_fields:
            row = [f] + [counts_matrix[f][u] for u in unit_types]
            writer.writerow(row)

    print(f"字段×单元类型的出现次数 count(f, u) 已写入：{BY_TYPE_COUNTS_CSV}")

    # 8. 输出 U(f | u) 矩阵
    with open(BY_TYPE_PROBS_CSV, "w", newline="", encoding="utf-8") as f_probs:
        writer = csv.writer(f_probs)
        header = ["field_name"] + unit_types
        writer.writerow(header)

        for f in all_fields:
            row = [f] + [probs_matrix[f][u] for u in unit_types]
            writer.writerow(row)

    print(f"字段×单元类型的条件利用度 U(f | u) 已写入：{BY_TYPE_PROBS_CSV}")
    print("实验三统计完成。")


if __name__ == "__main__":
    main()
