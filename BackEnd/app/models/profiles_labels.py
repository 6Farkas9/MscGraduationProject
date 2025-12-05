# app/models/profiles_labels.py
# -*- coding: utf-8 -*-

"""
画像标签规范（可扩展）

目标：
- 各个 engine 返回“数值标签 code”（如 0/1/2），而不是把文字写死在 engine 里；
- 系统层面统一在这里维护 label 文案（中英文等），方便全局替换与扩展；
- 支持“并列时选择含义更好”的 tie-break（比如 rank 越高越好）。

用法示例：
from app.models.profiles_labels import (
    DIM_ATTENTION_ALLOCATION_EFFICIENCY,
    DIM_ATTENTION_ALLOCATION_STYLE,
    get_profile_label,
    better_code,
)
label_zh = get_profile_label(DIM_ATTENTION_ALLOCATION_EFFICIENCY, 2, lang="zh")
best = better_code(DIM_ATTENTION_ALLOCATION_EFFICIENCY, [0,2])  # -> 2
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any, Iterable, List


# -----------------------------
# 维度 ID（建议全局唯一、稳定）
# -----------------------------

DIM_ATTENTION_ALLOCATION_EFFICIENCY = "attention_allocation.efficiency"
DIM_ATTENTION_ALLOCATION_STYLE = "attention_allocation.style"


@dataclass(frozen=True)
class LabelPack:
    """单个标签 code 的多语言文案"""
    zh: str
    en: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DimensionSpec:
    """一个维度的标签定义"""
    name: str
    labels: Dict[int, LabelPack]
    # “哪个更好”的排序：越靠后越好（用于并列时/择优）
    better_order: Optional[List[int]] = None


PROFILES_LABELS: Dict[str, DimensionSpec] = {
    # 注意效率（聚类结果 rank）
    DIM_ATTENTION_ALLOCATION_EFFICIENCY: DimensionSpec(
        name="注意力分配-注意效率",
        labels={
            0: LabelPack(
                zh="低效注意策略（任务表现较低、任务相关注意比例较低且在非任务 UI 区域停留较多）",
                en="Low efficiency attention strategy",
            ),
            1: LabelPack(
                zh="中等注意策略（任务相关注意与表现处于中间水平）",
                en="Medium efficiency attention strategy",
            ),
            2: LabelPack(
                zh="高效注意策略（在关键资源上集中注意、较少停留在无关 UI，且表现较好）",
                en="High efficiency attention strategy",
            ),
        },
        better_order=[0, 1, 2],
    ),

    # 信息加工风格（规则分类结果）
    # 注意：style 没有“更好更差”的强业务含义，因此 better_order 不设置（或你也可以给一个业务偏好）
    DIM_ATTENTION_ALLOCATION_STYLE: DimensionSpec(
        name="注意力分配-信息加工风格",
        labels={
            0: LabelPack(zh="文本优先型加工（进入或整体上更偏向文字信息）", en="Text-first processing"),
            1: LabelPack(zh="图像/模型优先型加工（进入或整体上更偏向图像/3D 模型）", en="Visual-first processing"),
            2: LabelPack(zh="示例/演示优先型加工（更偏向提示、示例或同伴演示）", en="Example-first processing"),
            3: LabelPack(zh="均衡整合型加工（在文本/图像/示例之间较为均衡地分配注意）", en="Balanced-integrative processing"),
            4: LabelPack(zh="加工风格未明（数据不足或注意非常分散）", en="Undefined/insufficient data"),
        }
    ),
}


def get_profile_label(dim: str, code: Optional[int], lang: str = "zh", default: Optional[str] = None) -> Optional[str]:
    """
    根据维度 dim 与 code 获取文案。
    - code 为 None -> 返回 default
    - lang: "zh" / "en"
    """
    if code is None:
        return default
    spec = PROFILES_LABELS.get(dim)
    if not spec:
        return default
    pack = spec.labels.get(int(code))
    if not pack:
        return default
    if lang == "en":
        return pack.en or default or pack.zh
    return pack.zh


def better_code(dim: str, candidates: Iterable[int]) -> Optional[int]:
    """
    在多个候选 code 中选择“更好”的那个：
    - 若维度配置了 better_order，则按 better_order 选最后一个（最好）
    - 否则默认选 max(code)
    """
    cands = [int(x) for x in candidates if x is not None]
    if not cands:
        return None

    spec = PROFILES_LABELS.get(dim)
    if not spec or not spec.better_order:
        return max(cands)

    rank = {code: idx for idx, code in enumerate(spec.better_order)}
    # 未出现在 better_order 的 code 视为最差（idx = -1）
    best = None
    best_idx = -10**9
    for c in cands:
        idx = rank.get(c, -10**6)
        if idx > best_idx:
            best_idx = idx
            best = c
    return best
