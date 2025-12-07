# BackEnd/app/shared/utils/stats_utils.py
# -*- coding: utf-8 -*-
"""
通用统计与聚类工具函数：
- compute_mean_std: 简单均值 + 总体标准差
- kmeans_1d: 一维 k-means 聚类
- choose_final_code: 带优先级的“众数”选择
"""

from math import sqrt
from typing import Dict, Iterable, Optional, List, Tuple
from collections import Counter

def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """
    计算一组数值的均值和总体标准差（σ，而非样本标准差）。

    返回:
        (mean, std)
        当列表为空时返回 (0.0, 0.0)
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_v = sum(values) / float(n)
    if n == 1:
        return mean_v, 0.0
    var = sum((v - mean_v) ** 2 for v in values) / float(n)
    return mean_v, sqrt(var)


def kmeans_1d(values: List[float], k: int = 3, max_iter: int = 50) -> List[int]:
    """
    一维 k-means 聚类，返回每个值所属的簇编号（0..k-1）。

    约定:
        - 若 n == 0 则返回空列表；
        - 若 k <= 0，则视为 k = 1；
        - 若 n < k，则自动将 k 限制为 n；
        - 当所有值几乎相等时（极小方差），统一归为同一簇 0；
        - 聚类完成后，将簇中心从小到大映射到 0..k-1，使得“code 越大数值越高”。
    """
    n = len(values)
    if n == 0:
        return []
    if k <= 0:
        k = 1
    if n < k:
        k = n

    v_min, v_max = min(values), max(values)
    if abs(v_max - v_min) < 1e-6:
        return [0 for _ in values]

    # 初始化中心：在 [v_min, v_max] 上均匀取 k 个点
    centers = [
        v_min + (v_max - v_min) * (i + 0.5) / float(k) for i in range(k)
    ]

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for idx, v in enumerate(values):
            best_c = 0
            best_dist = abs(v - centers[0])
            for ci in range(1, k):
                d = abs(v - centers[ci])
                if d < best_dist:
                    best_dist = d
                    best_c = ci
            clusters[best_c].append(idx)

        new_centers = centers[:]
        for ci in range(k):
            if clusters[ci]:
                new_centers[ci] = sum(
                    values[i] for i in clusters[ci]
                ) / float(len(clusters[ci]))
        max_shift = max(abs(new_centers[ci] - centers[ci]) for ci in range(k))
        centers = new_centers
        if max_shift < 1e-4:
            break

    # 根据最终中心，把每个值分配到最近的簇
    assignments: List[int] = []
    for v in values:
        best_c = 0
        best_dist = abs(v - centers[0])
        for ci in range(1, k):
            d = abs(v - centers[ci])
            if d < best_dist:
                best_dist = d
                best_c = ci
        assignments.append(best_c)

    # 中心从小到大映射到 0..k-1，保证 code 越大“数值越高”
    sorted_idx = sorted(range(k), key=lambda i: centers[i])
    cluster_to_rank = {cluster_idx: rank for rank, cluster_idx in enumerate(sorted_idx)}
    return [cluster_to_rank[c] for c in assignments]


def choose_final_code(codes: Iterable[int], code_priority: Dict[int, int]) -> Optional[int]:
    """
    从一组 code 中选出最终 code：
      1. 先按出现次数（众数）筛选候选；
      2. 候选中按 priority 最大优先；
      3. 若 priority 也相同，则选择 code 数值更大的那个。
    """
    codes = [c for c in codes if c is not None]
    if not codes:
        return None

    counter = Counter(codes)
    max_count = max(counter.values())
    candidates = [c for c, cnt in counter.items() if cnt == max_count]

    if len(candidates) == 1:
        return candidates[0]

    best_code = None
    best_pri = -1
    for c in candidates:
        pri = code_priority.get(c, 0)
        if pri > best_pri or (pri == best_pri and (best_code is None or c > best_code)):
            best_code = c
            best_pri = pri
    return best_code
