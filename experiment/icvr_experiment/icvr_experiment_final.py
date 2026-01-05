# -*- coding: utf-8 -*-
"""
实验：行为统计分布的区间合理性分析（ICVR）——按比值边界过滤版

边界规则（你指定）：
- 对所有 ratio 类指标：r = additioninfo2 / additioninfo1
- 仅保留 0 <= r <= 2 的记录
- 超出范围的样本视为无效数据，不参与区间 I 的 min/max

数据源：
- MySQL: mls.Interaction（粗粒度）
- MongoDB: MLS.Interaction（细粒度 xAPI）

约定：
- 学习单元 uid 前缀：unt_
- 题目 uid 前缀：qus_
- 单元行为：additioninfo1=总时长，additioninfo2=有效/集中/观看等时长
- 题目行为：一行=一次尝试（orig）；gen 优先用 _session_id

输出：
- icvr_results.csv
"""

import math
from typing import List, Tuple
import pymysql
from pymongo import MongoClient
import pandas as pd


# -----------------------
# 配置
# -----------------------
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "123456"
MYSQL_DB = "mls"

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "MLS"
MONGO_COLLECTION = "Interaction"

OUTPUT_CSV = "icvr_results.csv"

SESSION_FIELD = "_session_id"

# 你指定的边界
RATIO_MIN = 0.0
RATIO_MAX = 1.0


# -----------------------
# 工具：ICVR
# -----------------------
def safe_interval(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    return (float(min(xs)), float(max(xs)))


def interval_coverage_rate(i_orig: Tuple[float, float], i_gen: Tuple[float, float]) -> float:
    """
    ICVR = |I_orig ∩ I_gen| / |I_orig|
    """
    o_min, o_max = i_orig
    g_min, g_max = i_gen
    if any(math.isnan(x) for x in [o_min, o_max, g_min, g_max]):
        return float("nan")

    if o_min > o_max:
        o_min, o_max = o_max, o_min
    if g_min > g_max:
        g_min, g_max = g_max, g_min

    inter_min = max(o_min, g_min)
    inter_max = min(o_max, g_max)
    inter_len = max(0.0, inter_max - inter_min)
    orig_len = max(0.0, o_max - o_min)

    if orig_len == 0.0:
        return 1.0 if (g_min <= o_min <= g_max) else 0.0
    return inter_len / orig_len


def keep_ratio(r: float) -> bool:
    return (math.isfinite(r) and (RATIO_MIN <= r <= RATIO_MAX))


# -----------------------
# MySQL 读取
# -----------------------
def mysql_connect():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )


def load_mysql_unit_ratios(conn, unit_type: str) -> Tuple[List[float], int]:
    """
    从 MySQL 计算：r = additioninfo2/additioninfo1，并应用边界过滤 0<=r<=2
    返回：(有效样本比值列表, 被过滤掉的样本数)
    """
    sql = """
    SELECT i.additioninfo1 AS total_len, i.additioninfo2 AS eff_len
    FROM Interaction i
    JOIN Units u ON u.uid = i.unt_uid
    WHERE LEFT(i.unt_uid, 4) = 'unt_'
      AND LOWER(u.type) = %s
      AND i.additioninfo1 IS NOT NULL
      AND i.additioninfo2 IS NOT NULL
      AND i.additioninfo1 > 0
    """
    xs: List[float] = []
    dropped = 0

    with conn.cursor() as cur:
        cur.execute(sql, (unit_type.lower(),))
        for row in cur.fetchall():
            total_len = float(row["total_len"])
            eff_len = float(row["eff_len"])
            if total_len <= 0:
                dropped += 1
                continue
            r = eff_len / total_len
            if keep_ratio(r):
                xs.append(r)
            else:
                dropped += 1

    return xs, dropped


def load_mysql_question_attempt_counts(conn) -> List[float]:
    """
    orig：题目尝试次数 n_attempt(u,q)
    - qus_ 前缀
    - 一行=一次尝试
    => COUNT(*) by (lrn_uid, unt_uid)
    """
    sql = """
    SELECT lrn_uid, unt_uid, COUNT(*) AS cnt
    FROM Interaction
    WHERE LEFT(unt_uid, 4) = 'qus_'
    GROUP BY lrn_uid, unt_uid
    """
    xs: List[float] = []
    with conn.cursor() as cur:
        cur.execute(sql)
        for row in cur.fetchall():
            cnt = row["cnt"]
            if cnt is None:
                continue
            xs.append(float(cnt))
    return xs


# -----------------------
# Mongo 读取（细粒度）
# -----------------------
def ensure_session_field_exists(mcol) -> bool:
    return mcol.find_one({SESSION_FIELD: {"$exists": True}}, projection={SESSION_FIELD: 1}) is not None


def compute_gen_question_attempt_counts(mcol) -> Tuple[List[float], str]:
    """
    gen：题目尝试次数
    - 有 _session_id：按 distinct session_id 计数
    - 无 _session_id：fallback 为计数 answered/failed/passed（近似）
    """
    has_session = ensure_session_field_exists(mcol)

    if has_session:
        pipeline = [
            {"$match": {"_type": "question", "_unt_uid": {"$regex": r"^qus_"}, SESSION_FIELD: {"$exists": True}}},
            {"$group": {"_id": {"lrn": "$_lrn_uid", "qus": "$_unt_uid"}, "sessions": {"$addToSet": f"${SESSION_FIELD}"}}},
            {"$project": {"cnt": {"$size": "$sessions"}}},
        ]
        xs: List[float] = []
        for doc in mcol.aggregate(pipeline, allowDiskUse=True, batchSize=20000):
            xs.append(float(doc["cnt"]))
        return xs, "gen(question)=distinct _session_id per (lrn,qus)"

    pipeline = [
        {"$match": {"_type": "question", "_unt_uid": {"$regex": r"^qus_"}, "verb.id": {"$exists": True}}},
        {"$project": {"lrn": "$_lrn_uid", "qus": "$_unt_uid", "vk": {"$arrayElemAt": [{"$split": ["$verb.id", "/"]}, -1]}}},
        {"$match": {"vk": {"$in": ["answered", "failed", "passed"]}}},
        {"$group": {"_id": {"lrn": "$lrn", "qus": "$qus"}, "cnt": {"$sum": 1}}},
    ]
    xs: List[float] = []
    for doc in mcol.aggregate(pipeline, allowDiskUse=True, batchSize=20000):
        xs.append(float(doc["cnt"]))
    return xs, "gen(question)=count(answered/failed/passed) per (lrn,qus) [fallback;建议加入_session_id]"


def compute_gen_expand_density(mcol) -> Tuple[List[float], str]:
    """
    gen：展开密度 E_expand = 每条粗粒度交互对应的 xAPI 数
    需要 _session_id
    """
    has_session = ensure_session_field_exists(mcol)
    if not has_session:
        return [], "gen(expand)=无法计算（缺少 _session_id），请先重生成数据并写入 _session_id"

    pipeline = [
        {"$match": {SESSION_FIELD: {"$exists": True}}},
        {"$group": {"_id": f"${SESSION_FIELD}", "n": {"$sum": 1}}},
    ]
    xs: List[float] = []
    for doc in mcol.aggregate(pipeline, allowDiskUse=True, batchSize=20000):
        xs.append(float(doc["n"]))
    return xs, "gen(expand)=count(statements) per _session_id"


# -----------------------
# 主实验
# -----------------------
def main():
    # -------- MySQL --------
    print("Reading MySQL data ...")
    conn = mysql_connect()
    try:
        # proxy-origin：video watch ratio
        x_orig_watch, drop_orig_watch = load_mysql_unit_ratios(conn, "video")

        # gen：各类单元比值（同口径，来自 MySQL）
        x_gen_video, drop_gen_video = load_mysql_unit_ratios(conn, "video")
        x_gen_vr, drop_gen_vr = load_mysql_unit_ratios(conn, "vr")
        x_gen_ar, drop_gen_ar = load_mysql_unit_ratios(conn, "ar")
        x_gen_interact, drop_gen_interact = load_mysql_unit_ratios(conn, "interact")
        x_gen_cooperate, drop_gen_cooperate = load_mysql_unit_ratios(conn, "cooperate")

        # question attempts (orig)
        x_orig_q_attempt = load_mysql_question_attempt_counts(conn)

        # expand orig：粗粒度每条 interaction 视为 1（区间 [1,1]）
        i_orig_expand = (1.0, 1.0)
    finally:
        conn.close()

    # 区间构造
    i_orig_watch = safe_interval(x_orig_watch)
    i_gen_video = safe_interval(x_gen_video)
    i_gen_vr_ar = safe_interval(x_gen_vr + x_gen_ar)
    i_gen_interact = safe_interval(x_gen_interact)
    i_gen_cooperate = safe_interval(x_gen_cooperate)

    i_orig_q_attempt = safe_interval(x_orig_q_attempt)

    # -------- Mongo --------
    print("Reading Mongo data (question attempts & expand density) ...")
    mongo = MongoClient(MONGO_URI)
    mcol = mongo[MONGO_DB][MONGO_COLLECTION]

    x_gen_q_attempt, note_q_gen = compute_gen_question_attempt_counts(mcol)
    i_gen_q_attempt = safe_interval(x_gen_q_attempt)

    x_gen_expand, note_expand_gen = compute_gen_expand_density(mcol)
    i_gen_expand = safe_interval(x_gen_expand)

    # -------- 汇总 --------
    rows = []

    def add_row(metric: str, i_orig, i_gen, note: str):
        rows.append({
            "metric": metric,
            "I_orig_min": i_orig[0],
            "I_orig_max": i_orig[1],
            "I_gen_min": i_gen[0],
            "I_gen_max": i_gen[1],
            "ICVR": interval_coverage_rate(i_orig, i_gen),
            "note": note,
        })

    # 统一 note：说明边界过滤
    boundary_note = f"ratio filter applied: keep {RATIO_MIN}<=r<= {RATIO_MAX}; out-of-range dropped"

    add_row(
        "Video 观看投入比 r_watch",
        i_orig_watch,
        i_gen_video,
        f"orig=MySQL(video) add2/add1; gen=MySQL(video) add2/add1; {boundary_note}; dropped(orig)={drop_orig_watch}, dropped(gen)={drop_gen_video}"
    )

    add_row(
        "VR/AR 沉浸聚焦比 r_focus",
        i_orig_watch,
        i_gen_vr_ar,
        f"orig=复用视频 r_watch 区间; gen=MySQL(vr/ar) add2/add1; {boundary_note}; dropped(vr)={drop_gen_vr}, dropped(ar)={drop_gen_ar}"
    )

    add_row(
        "Interact 有效操作比 r_op",
        i_orig_watch,
        i_gen_interact,
        f"orig=复用视频 r_watch 区间; gen=MySQL(interact) add2/add1; {boundary_note}; dropped(interact)={drop_gen_interact}"
    )

    add_row(
        "Cooperate 有效协作比 r_eff",
        i_orig_watch,
        i_gen_cooperate,
        f"orig=复用视频 r_watch 区间; gen=MySQL(cooperate) add2/add1; {boundary_note}; dropped(cooperate)={drop_gen_cooperate}"
    )

    add_row(
        "题目单题尝试次数 n_attempt(u,q)",
        i_orig_q_attempt,
        i_gen_q_attempt,
        f"orig=MySQL题目(qus_)按(lrn,qus)计数行数; {note_q_gen}"
    )

    add_row(
        "粗细粒度展开密度 E_expand",
        i_orig_expand,
        i_gen_expand,
        f"orig=粗粒度每条Interaction视为1; {note_expand_gen}"
    )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n=== ICVR 汇总 ===")
    with pd.option_context("display.width", 200, "display.precision", 6):
        print(df[["metric", "I_orig_min", "I_orig_max", "I_gen_min", "I_gen_max", "ICVR"]])

    print(f"\n结果已保存到：{OUTPUT_CSV}")

    if "无法计算" in str(note_expand_gen):
        print("\n[提示] Mongo 细粒度日志缺少 _session_id，无法精确计算展开密度，题目尝试(gen)也只能用近似口径。")
        print("     建议重生成：把 MySQL Interaction.id 写入每条 xAPI 的 _session_id，并创建索引。")


if __name__ == "__main__":
    main()
