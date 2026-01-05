# -*- coding: utf-8 -*-
"""
实验二：基于细粒度 xAPI 语句的序列结构分析脚本

功能：
1. 连接 MongoDB.MLS.Interaction，按 (_lrn_uid, _session_id, timestamp) 顺序扫描所有事件。
2. 在每个会话内部按时间顺序构建动词序列，统计：
   - 相邻动词对 (v_i, v_j) 的转移次数 count(v_i, v_j)
   - 每个会话的长度 L（事件数）与不同动词种类数 D
3. 构建一阶动词转移矩阵 T_{i,j}，并输出为 CSV。
4. 将会话级统计 (L, D) 输出为 CSV。

说明：
- 假设数据已由前面的生成脚本写入：
  - 数据库：MLS
  - 集合：Interaction
- 会话 ID 字段：_session_id（int，真实会话 + persona 虚拟会话）
- 已存在索引：(_lrn_uid, _session_id, timestamp)，用于高效按序遍历。
"""

from pymongo import MongoClient, ASCENDING
from tqdm import tqdm
from collections import defaultdict
import csv

# ===================== 配置 =====================

MONGO_CONFIG = {
    "host": "localhost",
    "port": 27017,
    "db_name": "MLS",
    "xapi_collection": "Interaction",
}

# 输出文件名
SESSION_STATS_CSV = "session_stats.csv"
VERB_COUNTS_CSV = "verb_transition_counts.csv"
VERB_PROBS_CSV = "verb_transition_probs.csv"

# 每次从 Mongo 取多少条文档
MONGO_BATCH_SIZE = 10000


# ===================== 主逻辑 =====================

def main():
    # 1. 连接 MongoDB
    client = MongoClient(MONGO_CONFIG["host"], MONGO_CONFIG["port"])
    db = client[MONGO_CONFIG["db_name"]]
    col = db[MONGO_CONFIG["xapi_collection"]]

    # 估计文档总数，用于进度条（不要求非常精确）
    total_docs = col.estimated_document_count()
    print(f"预计将处理文档数：{total_docs}")

    # 2. 准备游标：按 (_lrn_uid, _session_id, timestamp) 排序
    # 利用此前创建的索引 idx_lrn_session_time(_lrn_uid, _session_id, timestamp)
    cursor = col.find(
        {},
        projection={
            "_id": 0,
            "_lrn_uid": 1,
            "_session_id": 1,
            "_course_uid": 1,
            "_type": 1,
            "timestamp": 1,
            "verb.id": 1,
        },
    ).sort([
        ("_lrn_uid", ASCENDING),
        ("_session_id", ASCENDING),
        ("timestamp", ASCENDING),
    ]).batch_size(MONGO_BATCH_SIZE)

    # 3. 统计结构准备
    # 全局：转移次数统计 & 动词集合
    transition_counts = defaultdict(int)   # key: (v_i, v_j) -> count
    verbs_set = set()

    # 会话级：当前会话的状态（在线维护，结束时写入 CSV）
    current_session_id = None
    current_lrn_uid = None
    current_course_uid = None
    current_session_type = None
    current_verb_set = set()
    current_length = 0
    prev_verb = None

    # 4. 打开会话统计 CSV，边遍历边写，避免内存占用
    with open(SESSION_STATS_CSV, "w", newline="", encoding="utf-8") as f_sess:
        sess_writer = csv.writer(f_sess)
        sess_writer.writerow([
            "session_id",
            "lrn_uid",
            "course_uid",
            "session_type",
            "length_L",
            "distinct_verbs_D",
        ])

        # 遍历所有文档
        with tqdm(total=total_docs, desc="遍历 xAPI 事件") as pbar:
            for doc in cursor:
                pbar.update(1)

                lrn_uid = doc.get("_lrn_uid")
                session_id = doc.get("_session_id")
                course_uid = doc.get("_course_uid")
                session_type = doc.get("_type")

                verb_obj = doc.get("verb") or {}
                verb_id = verb_obj.get("id")
                if verb_id is None:
                    # 没有 verb.id 的事件，直接跳过
                    continue

                # 维护动词全集
                verbs_set.add(verb_id)

                # 判断是否进入了一个新的会话
                if current_session_id is None:
                    # 第一个会话初始化
                    current_session_id = session_id
                    current_lrn_uid = lrn_uid
                    current_course_uid = course_uid
                    current_session_type = session_type
                    current_verb_set = set()
                    current_length = 0
                    prev_verb = None
                elif session_id != current_session_id or lrn_uid != current_lrn_uid:
                    # 会话切换：先写出上一会话的统计
                    D = len(current_verb_set)
                    sess_writer.writerow([
                        current_session_id,
                        current_lrn_uid,
                        current_course_uid,
                        current_session_type,
                        current_length,
                        D,
                    ])
                    # 然后重置状态为新会话
                    current_session_id = session_id
                    current_lrn_uid = lrn_uid
                    current_course_uid = course_uid
                    current_session_type = session_type
                    current_verb_set = set()
                    current_length = 0
                    prev_verb = None

                # 当前会话中更新 L / D / 转移计数
                current_length += 1
                current_verb_set.add(verb_id)

                if prev_verb is not None:
                    transition_counts[(prev_verb, verb_id)] += 1
                prev_verb = verb_id

            # for 循环结束后，别忘了写出最后一个会话
            if current_session_id is not None:
                D = len(current_verb_set)
                sess_writer.writerow([
                    current_session_id,
                    current_lrn_uid,
                    current_course_uid,
                    current_session_type,
                    current_length,
                    D,
                ])

    print(f"会话级统计已写入：{SESSION_STATS_CSV}")

    # 5. 构建动词列表和转移矩阵
    verbs = sorted(verbs_set)
    verb_to_idx = {v: i for i, v in enumerate(verbs)}
    n_verbs = len(verbs)
    print(f"共发现动词种类数 |V| = {n_verbs}")

    # 初始化计数矩阵
    counts_matrix = [[0 for _ in range(n_verbs)] for _ in range(n_verbs)]
    for (v_i, v_j), c in transition_counts.items():
        i = verb_to_idx[v_i]
        j = verb_to_idx[v_j]
        counts_matrix[i][j] = c

    # 6. 归一化得到转移概率矩阵 T
    probs_matrix = []
    for i in range(n_verbs):
        row_counts = counts_matrix[i]
        row_sum = sum(row_counts)
        if row_sum > 0:
            row_probs = [cnt / row_sum for cnt in row_counts]
        else:
            row_probs = [0.0] * n_verbs
        probs_matrix.append(row_probs)

    # 7. 写出转移次数矩阵 CSV
    with open(VERB_COUNTS_CSV, "w", newline="", encoding="utf-8") as f_counts:
        writer = csv.writer(f_counts)
        header = ["from_verb"] + verbs
        writer.writerow(header)
        for i, v_i in enumerate(verbs):
            row = [v_i] + counts_matrix[i]
            writer.writerow(row)

    print(f"动词转移次数矩阵已写入：{VERB_COUNTS_CSV}")

    # 8. 写出转移概率矩阵 CSV
    with open(VERB_PROBS_CSV, "w", newline="", encoding="utf-8") as f_probs:
        writer = csv.writer(f_probs)
        header = ["from_verb"] + verbs
        writer.writerow(header)
        for i, v_i in enumerate(verbs):
            row = [v_i] + probs_matrix[i]
            writer.writerow(row)

    print(f"动词转移概率矩阵已写入：{VERB_PROBS_CSV}")
    print("实验二统计完成。")


if __name__ == "__main__":
    main()
