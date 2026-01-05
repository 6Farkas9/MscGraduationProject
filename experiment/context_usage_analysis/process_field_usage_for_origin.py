#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
处理两个CSV并输出给 Origin 直接用的格式：

共同处理（两文件都做）：
1) 行标题规范化：
   - context.contextActivities.parent -> context
   - https://legend-meta.com/xapi/ext/xxx -> xxx   （注意：不再是 ext/xxx）

global 文件额外处理：
2) 计算对数变换（按你的公式）：
   U'(f)=log10( (U(f)+ε) / (U_min+ε) )
   - 其中 U_min 为 global 表中 U_global 的最小值（忽略缺失/非数）
   - ε 默认 1e-12，可通过参数 --epsilon 调整
3) 按 U_global 从大到小排序
4) 输出 origin_field_usage_global_processed.csv

type 文件额外处理：
5) 输出 origin_field_usage_by_type_processed.csv
   - 默认按 global 的排序顺序对齐（可用 --no-align 关闭）
"""

import argparse
import math
from collections import Counter
from typing import Any, List, Optional

import pandas as pd


EXT_PREFIX = "https://legend-meta.com/xapi/ext/"
CONTEXT_PARENT = "context.contextActivities.parent"

DEFAULT_GLOBAL_IN = "field_usage_global.csv"
DEFAULT_TYPE_IN = "field_usage_by_type_probs.csv"
DEFAULT_GLOBAL_OUT = "origin_field_usage_global_processed.csv"
DEFAULT_TYPE_OUT = "origin_field_usage_by_type_processed.csv"


def normalize_field_name(x: Any) -> str:
    """按需求规范化行标题。"""
    s = "" if x is None else str(x).strip()
    if not s:
        return s
    if s == CONTEXT_PARENT:
        return "context"
    if s.startswith(EXT_PREFIX):
        # 规则修改：提取成 xxx（不带 ext/）
        return s[len(EXT_PREFIX):].lstrip("/")
    return s


def make_unique(labels: List[str]) -> List[str]:
    """如果规范化后重名，自动加 __2, __3..."""
    seen = Counter()
    out = []
    for x in labels:
        x = "" if x is None else str(x)
        seen[x] += 1
        out.append(x if seen[x] == 1 else f"{x}__{seen[x]}")
    return out


def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def u_prime_series(u: pd.Series, eps: float) -> pd.Series:
    """
    U'(f)=log10( (U(f)+eps) / (U_min+eps) )
    - U_min 基于该列最小有限值（忽略 NaN）
    """
    u = to_float_series(u)
    finite = u[~u.isna()]
    if finite.empty:
        # 全是缺失，直接返回 NaN
        return pd.Series([float("nan")] * len(u), index=u.index, dtype="float64")

    u_min = float(finite.min())
    denom = u_min + eps
    if denom <= 0:
        # 极端情况：如果 u_min 很负导致 denom<=0，改用最小“正denom”的保底
        # 这里不猜测业务数据，直接返回 NaN 以避免产生误导值
        return pd.Series([float("nan")] * len(u), index=u.index, dtype="float64")

    num = u + eps
    # num 可能 <=0（如果 U 有负且绝对值大），这些点置 NaN
    out = pd.Series(float("nan"), index=u.index, dtype="float64")
    mask = num > 0
    out.loc[mask] = (num.loc[mask] / denom).map(lambda z: math.log10(z))
    return out + 0.1


def process_global(global_path: str, out_path: str, eps: float) -> List[str]:
    """
    返回：排序后的 field 列表（用于对齐 type 文件）。
    """
    df = pd.read_csv(global_path)

    if "field_name" not in df.columns:
        raise ValueError(f"[global] 找不到列 field_name，实际列：{list(df.columns)}")
    if "U_global" not in df.columns:
        raise ValueError(f"[global] 找不到列 U_global，实际列：{list(df.columns)}")

    df["field"] = df["field_name"].map(normalize_field_name)
    df["field"] = make_unique(df["field"].tolist())

    # 计算 U'(f)
    df["U_prime"] = u_prime_series(df["U_global"], eps=eps)

    # 按 U_global 降序排序
    df["U_global"] = to_float_series(df["U_global"])
    df = df.sort_values(by="U_global", ascending=False, kind="mergesort").reset_index(drop=True)

    # 输出：field | U_global | U_prime | count(如存在)
    cols = ["field", "U_global", "U_prime"]
    if "count" in df.columns:
        cols.append("count")

    df_out = df[cols].copy()
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    return df["field"].tolist()


def process_type(type_path: str, out_path: str, align_order: Optional[List[str]], align_to_global: bool):
    df = pd.read_csv(type_path)

    if "field_name" not in df.columns:
        raise ValueError(f"[type] 找不到列 field_name，实际列：{list(df.columns)}")

    df["field"] = df["field_name"].map(normalize_field_name)
    df["field"] = make_unique(df["field"].tolist())

    value_cols = [c for c in df.columns if c not in ("field_name", "field")]
    df_out = df[["field"] + value_cols].copy()

    if align_to_global and align_order:
        order_index = {k: i for i, k in enumerate(align_order)}
        df_out["_order"] = df_out["field"].map(lambda x: order_index.get(x, 10**12))
        df_out = df_out.sort_values(by="_order", ascending=True, kind="mergesort").drop(columns=["_order"])

    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-in", default=DEFAULT_GLOBAL_IN, help="global 输入CSV")
    ap.add_argument("--type-in", default=DEFAULT_TYPE_IN, help="type 输入CSV")
    ap.add_argument("--global-out", default=DEFAULT_GLOBAL_OUT, help="global 输出CSV")
    ap.add_argument("--type-out", default=DEFAULT_TYPE_OUT, help="type 输出CSV")
    ap.add_argument("--epsilon", type=float, default=1e-12, help="公式中的 ε（默认 1e-12）")
    ap.add_argument("--no-align", action="store_true", help="不把 type 文件按 global 的排序对齐（默认会对齐）")
    args = ap.parse_args()

    if args.epsilon <= 0:
        raise ValueError("--epsilon 必须为正数")

    ordered_fields = process_global(args.global_in, args.global_out, eps=args.epsilon)

    process_type(
        args.type_in,
        args.type_out,
        align_order=ordered_fields,
        align_to_global=(not args.no_align),
    )

    print("完成！输出文件：")
    print(f" - {args.global_out}")
    print(f" - {args.type_out}")
    print(f"使用 epsilon = {args.epsilon:g}")
    if args.no_align:
        print("说明：type 输出未按 global 排序对齐（你用了 --no-align）。")
    else:
        print("说明：type 输出已按 global 的 field 顺序对齐。")


if __name__ == "__main__":
    main()
