#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 session_stats.csv（300w+ 行）分块处理，输出给 Origin 直接用的 CSV：
- origin_plot1_length_by_type.csv    (session_type, length_L, log10_length_L)
- origin_plot2_distinct_by_type.csv  (session_type, distinct_verbs_D, log10_distinct_verbs_D)
- origin_plot3_L_vs_D_by_type.csv    (session_type, length_L, distinct_verbs_D, log10_length_L, log10_distinct_verbs_D)

新增规则：
- session_type 只保留：vr/interact/cooperate/video/ar/question
- 丢弃：course-level
- 其它未知值也丢弃（防止脏数据混入）

说明：
- log10(x) 仅对 x>0 计算；否则为 NaN（空）
- 采用 chunksize 流式写出，避免一次性占用大量内存
"""

import os
import argparse
import pandas as pd
import numpy as np


KEEP_SESSION_TYPES = {"vr", "interact", "cooperate", "video", "ar", "question"}
DROP_SESSION_TYPES = {"course-level"}  # 显式声明（实际过滤由 KEEP_SESSION_TYPES 控制）


def safe_log10(arr: pd.Series) -> pd.Series:
    """对 >0 的值取 log10，否则 NaN。"""
    a = pd.to_numeric(arr, errors="coerce")
    out = pd.Series(np.nan, index=a.index, dtype="float64")
    m = a > 0
    out.loc[m] = np.log10(a.loc[m].astype("float64"))
    return out


def normalize_session_type(s: pd.Series) -> pd.Series:
    """
    规范化 session_type：
    - 转字符串
    - 去首尾空格
    - 小写化
    """
    # 注意：dtype 已读为 string，但仍做一遍稳健处理
    return s.astype("string").str.strip().str.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="session_stats.csv", help="输入文件名（默认 session_stats.csv）")
    parser.add_argument("--chunksize", type=int, default=500_000, help="分块行数（默认 500000）")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="图3输出抽样比例(0~1]，默认1.0=全量；例如0.2会抽20%%以减小Origin压力",
    )
    args = parser.parse_args()

    if not (0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac 必须在 (0, 1] 范围内")

    in_path = args.input
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"找不到输入文件：{in_path}")

    out1 = "origin_plot1_length_by_type.csv"
    out2 = "origin_plot2_distinct_by_type.csv"
    out3 = "origin_plot3_L_vs_D_by_type.csv"

    # 若已存在旧输出，先删掉，避免重复 append
    for p in (out1, out2, out3):
        if os.path.exists(p):
            os.remove(p)

    usecols = ["session_type", "length_L", "distinct_verbs_D"]

    # 指定 dtype 以减少内存（length_L / distinct_verbs_D 可能有缺失，先用 float）
    dtypes = {
        "session_type": "string",
        "length_L": "float64",
        "distinct_verbs_D": "float64",
    }

    first_write_1 = True
    first_write_2 = True
    first_write_3 = True

    total_in = 0
    total_after_type_filter = 0
    total_kept_3 = 0

    for chunk in pd.read_csv(
        in_path,
        usecols=usecols,
        dtype=dtypes,
        chunksize=args.chunksize,
        low_memory=True,
    ):
        total_in += len(chunk)

        # 清理：session_type 为空的行去掉
        chunk = chunk.dropna(subset=["session_type"])
        if chunk.empty:
            continue

        # 规范化 + 过滤 session_type
        chunk["session_type"] = normalize_session_type(chunk["session_type"])
        chunk = chunk[chunk["session_type"].isin(KEEP_SESSION_TYPES)]
        total_after_type_filter += len(chunk)

        if chunk.empty:
            continue

        # ---- 图1：length_L + log10(length_L)
        df1 = chunk[["session_type", "length_L"]].copy()
        df1["log10_length_L"] = safe_log10(df1["length_L"])
        df1.to_csv(out1, index=False, mode="a", header=first_write_1)
        first_write_1 = False

        # ---- 图2：distinct_verbs_D + log10(distinct_verbs_D)
        df2 = chunk[["session_type", "distinct_verbs_D"]].copy()
        df2["log10_distinct_verbs_D"] = safe_log10(df2["distinct_verbs_D"])
        df2.to_csv(out2, index=False, mode="a", header=first_write_2)
        first_write_2 = False

        # ---- 图3：length_L vs distinct_verbs_D + logs（可选抽样减小体量）
        df3 = chunk[["session_type", "length_L", "distinct_verbs_D"]].copy()

        if args.sample_frac < 1.0:
            # 每块独立随机抽样，整体近似 sample_frac
            df3 = df3.sample(frac=args.sample_frac, random_state=42)

        df3["log10_length_L"] = safe_log10(df3["length_L"])
        df3["log10_distinct_verbs_D"] = safe_log10(df3["distinct_verbs_D"])

        total_kept_3 += len(df3)
        df3.to_csv(out3, index=False, mode="a", header=first_write_3)
        first_write_3 = False

    print("完成！")
    print(f"输入总行数（读入）：{total_in:,}")
    print(f"过滤 session_type 后行数（保留 {sorted(KEEP_SESSION_TYPES)}）：{total_after_type_filter:,}")
    print(f"图3输出行数：{total_kept_3:,}（sample-frac={args.sample_frac}）")
    print("输出文件：")
    print(f" - {out1}")
    print(f" - {out2}")
    print(f" - {out3}")


if __name__ == "__main__":
    main()
