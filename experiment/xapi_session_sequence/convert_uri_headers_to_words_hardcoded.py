#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把矩阵CSV（行/列标题都是 xAPI verb 的 URI）转换为“单词/连字符”标题，并输出新 CSV。

默认：
- 输入：verb_transition_probs.csv
- 输出：verb_transition_probs_readable.csv

用法：
python convert_uri_headers_to_words_hardcoded.py --input verb_transition_probs.csv --output verb_transition_probs_readable.csv
"""

import argparse
from collections import Counter
from typing import Any, Dict

import pandas as pd


# ====== 硬编码映射：URI -> 单词/连字符（来自 xAPI_interaction_profile.py 中 VERB_BASE + VERBS） ======
URI_TO_WORD: Dict[str, str] = {
    # 任务与结果行为
    "https://legend-meta.com/xapi/verb/experienced": "experienced",
    "https://legend-meta.com/xapi/verb/initialized": "initialized",
    "https://legend-meta.com/xapi/verb/completed": "completed",
    "https://legend-meta.com/xapi/verb/answered": "answered",
    "https://legend-meta.com/xapi/verb/passed": "passed",
    "https://legend-meta.com/xapi/verb/failed": "failed",

    # 空间行为
    "https://legend-meta.com/xapi/verb/navigated-to-space": "navigated-to-space",
    "https://legend-meta.com/xapi/verb/teleported-to-space": "teleported-to-space",

    # 对象操作行为
    "https://legend-meta.com/xapi/verb/manipulated-object": "manipulated-object",
    "https://legend-meta.com/xapi/verb/performed-procedure-step": "performed-procedure-step",
    "https://legend-meta.com/xapi/verb/contributed-resource": "contributed-resource",
    "https://legend-meta.com/xapi/verb/exchanged-value": "exchanged-value",

    # 注意、状态与认知加工行为
    "https://legend-meta.com/xapi/verb/focused-on-resource": "focused-on-resource",
    "https://legend-meta.com/xapi/verb/reviewed-feedback": "reviewed-feedback",
    "https://legend-meta.com/xapi/verb/explored-extension": "explored-extension",
    "https://legend-meta.com/xapi/verb/reflected-on-activity": "reflected-on-activity",
    "https://legend-meta.com/xapi/verb/remained-idle": "remained-idle",

    # 协作与社会交互行为
    "https://legend-meta.com/xapi/verb/collaborated-on-activity": "collaborated-on-activity",
    "https://legend-meta.com/xapi/verb/co-edited-artifact": "co-edited-artifact",
    "https://legend-meta.com/xapi/verb/observed-peer": "observed-peer",
    "https://legend-meta.com/xapi/verb/requested-support": "requested-support",
}


def _strip_uri(x: Any) -> str:
    return "" if x is None else str(x).strip()


def _uri_tail(uri: str) -> str:
    """兜底：取 URI 最后一个 path 段。"""
    uri = _strip_uri(uri)
    if not uri:
        return ""
    return uri.rstrip("/").split("/")[-1]


def make_unique(labels):
    """处理重名：重复项加后缀 __2, __3..."""
    labels = [str(x) for x in labels]
    seen = Counter()
    out = []
    for x in labels:
        seen[x] += 1
        out.append(x if seen[x] == 1 else f"{x}__{seen[x]}")
    return out


def map_uri_to_word(x: Any) -> str:
    uri = _strip_uri(x)
    if not uri:
        return ""
    # 直接命中
    if uri in URI_TO_WORD:
        return URI_TO_WORD[uri]
    # 去掉末尾 /
    uri2 = uri.rstrip("/")
    if uri2 in URI_TO_WORD:
        return URI_TO_WORD[uri2]
    # 兜底：取 tail
    return _uri_tail(uri).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="verb_transition_probs.csv", help="输入矩阵CSV（行列标题是URI）")
    ap.add_argument("--output", default="verb_transition_probs_readable.csv", help="输出CSV（行列标题为单词/连字符）")
    args = ap.parse_args()

    # 读取矩阵：假定第一列是行标题（index）
    df = pd.read_csv(args.input, index_col=0)

    # 映射行列标题
    new_index = [map_uri_to_word(v) for v in df.index]
    new_cols = [map_uri_to_word(v) for v in df.columns]

    # 去重避免覆盖/导入歧义
    df.index = make_unique(new_index)
    df.columns = make_unique(new_cols)

    # 输出（带 BOM，兼容部分 Windows/Origin 导入）
    df.to_csv(args.output, index=True, encoding="utf-8-sig")

    print("完成！")
    print(f"输入: {args.input}")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
