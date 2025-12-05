# attention_allocation_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

from app.repositories.attention_allocation_repository import (
    AttentionAllocationRepository,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 通用小工具
# ----------------------------------------------------------------------
def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """简单均值 + 总体标准差。"""
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

    # 中心从小到大映射到 0..k-1，code 越大代表“越高”
    sorted_idx = sorted(range(k), key=lambda i: centers[i])
    cluster_to_rank = {cluster_idx: rank for rank, cluster_idx in enumerate(sorted_idx)}
    return [cluster_to_rank[c] for c in assignments]


def classify_style_code(metrics: Dict[str, float]) -> int:
    """
    根据 AOI 比例和首注视比例，给出 style 数值 code：
        0: 文本优先
        1: 图像/模型优先
        2: 示例/演示优先
        3: 均衡整合
        4: 未定义/数据不足

    与 app/models/profiles_labels.py 中 attention_allocation.style 定义保持一致。
    """
    tr = metrics.get("text_ratio", 0.0)
    vr = metrics.get("visual_ratio", 0.0)
    er = metrics.get("example_ratio", 0.0)
    rr = metrics.get("relevant_ratio", 0.0)
    ftr = metrics.get("first_text_ratio", 0.0)
    fvr = metrics.get("first_visual_ratio", 0.0)
    fer = metrics.get("first_example_ratio", 0.0)

    max_ratio = max(tr, vr, er)
    first_max = max(ftr, fvr, fer)

    # 文本优先
    if (tr == max_ratio and tr >= 0.55) or (ftr == first_max and ftr > 0):
        return 0

    # 图像/模型优先
    if (vr == max_ratio and vr >= 0.55) or (fvr == first_max and fvr > 0):
        return 1

    # 示例/演示优先
    if (er == max_ratio and er >= 0.45) or (fer == first_max and fer > 0):
        return 2

    # 均衡整合：任务相关比例高且三类较均衡
    if rr >= 0.7 and max_ratio <= 0.6:
        return 3

    return 4  # 未定义 / 数据不足


def choose_final_code(codes: List[int], code_priority: Dict[int, int]) -> Optional[int]:
    """
    从一组课程级标签 code 中选整体标签：
    - 先看出现次数（众数）；
    - 若有并列，则用优先级（越大越“好”）来打破平局。
    """
    if not codes:
        return None
    counter = Counter(codes)
    max_count = max(counter.values())
    candidates = [c for c, cnt in counter.items() if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]

    best_code = None
    best_priority = None
    for c in candidates:
        pr = code_priority.get(c, 0)
        if best_code is None or pr > best_priority:
            best_code = c
            best_priority = pr
    return best_code


# ----------------------------------------------------------------------
# Engine 主体
# ----------------------------------------------------------------------
class AttentionAllocationEngine:
    """
    注意力分配画像分析引擎。

    对外只暴露一个接口:
        analyze(learner_uids: List[str]) -> Dict[str, Any]

    - Repository 只提供原始聚合数据；
    - Engine 在课程内部完成 E_att / 聚类，并在学习者层面汇总整体标签。
    """

    DIMENSION_KEY = "attention_allocation"
    EFFICIENCY_CATEGORY = "efficiency"
    STYLE_CATEGORY = "style"

    def __init__(
        self,
        repository: Optional[AttentionAllocationRepository] = None,
        efficiency_cluster_k: int = 3,
        min_learners_per_course: Optional[int] = None,
    ):
        """
        Args:
            repository: 可注入自定义 Repository 实现，默认使用 AttentionAllocationRepository。
            efficiency_cluster_k: 每门课程做效率聚类的簇数（默认 3）。
            min_learners_per_course: 单门课程参与聚类的最小学习者数量，
                                     默认与 efficiency_cluster_k 相同。
        """
        self.repo = repository or AttentionAllocationRepository()
        self.efficiency_cluster_k = efficiency_cluster_k
        self.min_learners_per_course = (
            min_learners_per_course or efficiency_cluster_k
        )

        # 效率标签：code 越大越“好”
        self.efficiency_priority = {0: 0, 1: 1, 2: 2}
        # 风格标签优先级：3(均衡) > 0/1/2(单一偏好) > 4(未定义)
        self.style_priority = {3: 2, 0: 1, 1: 1, 2: 1, 4: 0}

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        对一批学习者进行注意力分配画像分析。

        Args:
            learner_uids: 学习者 uid 列表（可包含多个）。

        Returns:
            { learner_uid: { "attention_allocation": { ... } } }
        """
        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        if not learner_uids:
            logger.info("AttentionAllocationEngine.analyze: 空的学习者列表，直接返回。")
            return {}

        logger.info(
            "AttentionAllocationEngine: 开始分析，学习者数: %d",
            len(learner_uids),
        )

        # 1. 仓库层：准备原始聚合数据
        (
            raw_by_lc,
            learners_per_course_count,
            learner_courses_map,
        ) = self.repo.load_metrics_for_learners(learner_uids)

        if not raw_by_lc:
            logger.info("AttentionAllocationEngine: 所有学习者均无有效原始数据。")
            result = {}
            for uid in learner_uids:
                result[uid] = {
                    self.DIMENSION_KEY: {
                        "insufficient_data": True,
                        "insufficient_reason": "no_data_in_interaction",
                    }
                }
            return result

        logger.info(
            "AttentionAllocationEngine: 收到原始 (lrn, crs) 对数: %d，课程数: %d",
            len(raw_by_lc),
            len(learners_per_course_count),
        )

        # 2. 根据原始统计计算每个 (lrn, crs) 的基础数值指标（比例等）
        base_metrics_by_lc: Dict[Tuple[str, str], Dict[str, float]] = {}
        for (lrn_uid, crs_uid), raw in raw_by_lc.items():
            durations = raw.get("durations") or {}
            text_dur = float(durations.get("text", 0.0))
            visual_dur = float(durations.get("visual", 0.0))
            example_dur = float(durations.get("example", 0.0))
            ui_dur = float(durations.get("ui_other", 0.0))
            total_dur = text_dur + visual_dur + example_dur + ui_dur
            if total_dur <= 0:
                continue

            text_ratio = text_dur / total_dur
            visual_ratio = visual_dur / total_dur
            example_ratio = example_dur / total_dur
            ui_ratio = ui_dur / total_dur
            relevant_ratio = (text_dur + visual_dur + example_dur) / total_dur

            first_counts = raw.get("first_counts") or {}
            ft = int(first_counts.get("text", 0))
            fv = int(first_counts.get("visual", 0))
            fe = int(first_counts.get("example", 0))
            first_total = ft + fv + fe
            if first_total > 0:
                first_text_ratio = ft / float(first_total)
                first_visual_ratio = fv / float(first_total)
                first_example_ratio = fe / float(first_total)
            else:
                first_text_ratio = first_visual_ratio = first_example_ratio = 0.0

            perf_sum = float(raw.get("perf_sum", 0.0))
            perf_cnt = int(raw.get("perf_cnt", 0))
            performance = perf_sum / perf_cnt if perf_cnt > 0 else None

            base_metrics_by_lc[(lrn_uid, crs_uid)] = {
                "text_ratio": text_ratio,
                "visual_ratio": visual_ratio,
                "example_ratio": example_ratio,
                "ui_ratio": ui_ratio,
                "relevant_ratio": relevant_ratio,
                "first_text_ratio": first_text_ratio,
                "first_visual_ratio": first_visual_ratio,
                "first_example_ratio": first_example_ratio,
                "performance": performance,
            }

        logger.info(
            "AttentionAllocationEngine: 完成基础比例指标计算，有效 (lrn, crs) 条目: %d",
            len(base_metrics_by_lc),
        )

        # 3. 以课程为单位组织数据
        course_entries: Dict[str, List[Tuple[str, Dict[str, float]]]] = defaultdict(list)
        for (lrn_uid, crs_uid), metrics in base_metrics_by_lc.items():
            course_entries[crs_uid].append((lrn_uid, metrics))

        logger.info(
            "AttentionAllocationEngine: 参与分析的课程数: %d",
            len(course_entries),
        )

        # 4. 在课程内部计算 E_att / E_att_norm，并做聚类得到 efficiency_code
        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]] = {}
        total_courses = len(course_entries)
        processed_courses = 0

        for crs_uid, entries in course_entries.items():
            processed_courses += 1
            learner_count = learners_per_course_count.get(crs_uid, len(entries))
            if learner_count < self.min_learners_per_course:
                logger.info(
                    "AttentionAllocationEngine: 课程 %s 学习者数 %d 少于聚类簇数 %d，跳过该课程。",
                    crs_uid,
                    learner_count,
                    self.efficiency_cluster_k,
                )
                continue

            perf_vals: List[float] = []
            rel_vals: List[float] = []
            ui_vals: List[float] = []

            for _, m in entries:
                perf = m.get("performance")
                if perf is None:
                    perf = 0.5  # 无表现数据，视为中性
                perf_vals.append(perf)
                rel_vals.append(m.get("relevant_ratio", 0.0))
                ui_vals.append(m.get("ui_ratio", 0.0))

            mean_perf, std_perf = compute_mean_std(perf_vals)
            mean_rel, std_rel = compute_mean_std(rel_vals)
            mean_ui, std_ui = compute_mean_std(ui_vals)

            E_vals: List[float] = []
            z_cache: List[Tuple[float, float, float]] = []

            for idx, (_, m) in enumerate(entries):
                perf = perf_vals[idx]
                rel = rel_vals[idx]
                ui = ui_vals[idx]

                z_perf = (perf - mean_perf) / std_perf if std_perf > 1e-6 else 0.0
                z_rel = (rel - mean_rel) / std_rel if std_rel > 1e-6 else 0.0
                z_ui = (ui - mean_ui) / std_ui if std_ui > 1e-6 else 0.0

                E_att = (z_perf + z_rel - z_ui) / sqrt(3.0)
                E_vals.append(E_att)
                z_cache.append((z_perf, z_rel, z_ui))

            if not E_vals:
                continue

            E_min, E_max = min(E_vals), max(E_vals)
            span = E_max - E_min if E_max > E_min else 0.0
            E_norm_vals: List[float] = []
            for E_att in E_vals:
                if span > 1e-6:
                    E_norm_vals.append((E_att - E_min) / span)
                else:
                    E_norm_vals.append(0.5)

            cluster_codes = kmeans_1d(E_norm_vals, k=self.efficiency_cluster_k)

            for idx, (lrn_uid, base_metrics) in enumerate(entries):
                z_perf, z_rel, z_ui = z_cache[idx]
                E_att = E_vals[idx]
                E_norm = E_norm_vals[idx]
                eff_code = cluster_codes[idx]
                style_code = classify_style_code(base_metrics)

                lc_key = (lrn_uid, crs_uid)
                per_lc_result[lc_key] = {
                    "course_uid": crs_uid,
                    "learner_uid": lrn_uid,
                    "efficiency_code": eff_code,
                    "style_code": style_code,
                    "E_att": E_att,
                    "E_att_norm": E_norm,
                    "z_perf": z_perf,
                    "z_rel": z_rel,
                    "z_ui": z_ui,
                    **base_metrics,
                }

            if processed_courses % 50 == 0 or processed_courses == total_courses:
                logger.info(
                    "AttentionAllocationEngine: 已完成课程级分析 %d/%d",
                    processed_courses,
                    total_courses,
                )

        logger.info(
            "AttentionAllocationEngine: 课程级分析完成，有课程级结果的 (lrn, crs) 条目: %d",
            len(per_lc_result),
        )

        # 5. 逐个学习者汇总课程级结果，构建最终结构
        result: Dict[str, Any] = {}
        for lrn_uid in learner_uids:
            dim_result: Dict[str, Any] = {}
            courses_for_learner = learner_courses_map.get(lrn_uid, set())
            learner_course_results = {
                crs_uid: per_lc_result.get((lrn_uid, crs_uid))
                for crs_uid in courses_for_learner
                if (lrn_uid, crs_uid) in per_lc_result
            }

            if not learner_course_results:
                dim_result["insufficient_data"] = True
                dim_result["insufficient_reason"] = "no_valid_courses"
                result[lrn_uid] = {self.DIMENSION_KEY: dim_result}
                continue

            dim_result["insufficient_data"] = False
            dim_result["insufficient_reason"] = None

            eff_codes: List[int] = []
            eff_overall_metrics = {
                "E_att_norm_mean": 0.0,
                "performance_mean": 0.0,
                "relevant_ratio_mean": 0.0,
                "ui_ratio_mean": 0.0,
            }
            eff_courses_detail: Dict[str, Any] = {}

            style_codes: List[int] = []
            style_overall_metrics = {
                "text_ratio_mean": 0.0,
                "visual_ratio_mean": 0.0,
                "example_ratio_mean": 0.0,
                "first_text_ratio_mean": 0.0,
                "first_visual_ratio_mean": 0.0,
                "first_example_ratio_mean": 0.0,
            }
            style_courses_detail: Dict[str, Any] = {}

            eff_n = 0
            style_n = 0

            for crs_uid, lc_res in learner_course_results.items():
                if lc_res is None:
                    continue

                eff_code = lc_res["efficiency_code"]
                style_code = lc_res["style_code"]
                eff_codes.append(eff_code)
                style_codes.append(style_code)

                E_norm = lc_res.get("E_att_norm", 0.0)
                perf = lc_res.get("performance")
                if perf is None:
                    perf = 0.5
                rel_r = lc_res.get("relevant_ratio", 0.0)
                ui_r = lc_res.get("ui_ratio", 0.0)

                eff_n += 1
                eff_overall_metrics["E_att_norm_mean"] += E_norm
                eff_overall_metrics["performance_mean"] += perf
                eff_overall_metrics["relevant_ratio_mean"] += rel_r
                eff_overall_metrics["ui_ratio_mean"] += ui_r

                eff_courses_detail[crs_uid] = {
                    "code": eff_code,
                    "metrics": {
                        "E_att": lc_res.get("E_att", 0.0),
                        "E_att_norm": E_norm,
                        "performance": perf,
                        "relevant_ratio": rel_r,
                        "ui_ratio": ui_r,
                        "text_ratio": lc_res.get("text_ratio", 0.0),
                        "visual_ratio": lc_res.get("visual_ratio", 0.0),
                        "example_ratio": lc_res.get("example_ratio", 0.0),
                    },
                }

                tr = lc_res.get("text_ratio", 0.0)
                vr = lc_res.get("visual_ratio", 0.0)
                er = lc_res.get("example_ratio", 0.0)
                ftr = lc_res.get("first_text_ratio", 0.0)
                fvr = lc_res.get("first_visual_ratio", 0.0)
                fer = lc_res.get("first_example_ratio", 0.0)

                style_n += 1
                style_overall_metrics["text_ratio_mean"] += tr
                style_overall_metrics["visual_ratio_mean"] += vr
                style_overall_metrics["example_ratio_mean"] += er
                style_overall_metrics["first_text_ratio_mean"] += ftr
                style_overall_metrics["first_visual_ratio_mean"] += fvr
                style_overall_metrics["first_example_ratio_mean"] += fer

                style_courses_detail[crs_uid] = {
                    "code": style_code,
                    "metrics": {
                        "text_ratio": tr,
                        "visual_ratio": vr,
                        "example_ratio": er,
                        "ui_ratio": ui_r,
                        "relevant_ratio": rel_r,
                        "first_text_ratio": ftr,
                        "first_visual_ratio": fvr,
                        "first_example_ratio": fer,
                    },
                }

            if eff_n > 0:
                for k in eff_overall_metrics:
                    eff_overall_metrics[k] /= float(eff_n)
            if style_n > 0:
                for k in style_overall_metrics:
                    style_overall_metrics[k] /= float(style_n)

            final_eff_code = choose_final_code(
                eff_codes, self.efficiency_priority
            )
            final_style_code = choose_final_code(
                style_codes, self.style_priority
            )

            dim_result[self.EFFICIENCY_CATEGORY] = {
                "final_code": final_eff_code,
                "overall_metrics": eff_overall_metrics,
                "courses": eff_courses_detail,
            }
            dim_result[self.STYLE_CATEGORY] = {
                "final_code": final_style_code,
                "overall_metrics": style_overall_metrics,
                "courses": style_courses_detail,
            }

            result[lrn_uid] = {self.DIMENSION_KEY: dim_result}

        logger.info(
            "AttentionAllocationEngine: 完成 %d 个学习者的注意力分配画像分析。",
            len(result),
        )

        return result


# ----------------------------------------------------------------------
# main: 简单测试（包括标签文本演示）
# ----------------------------------------------------------------------
def _print_with_text_labels(engine_result: Dict[str, Any]) -> None:
    """
    演示如何结合 profiles_labels 把数值标签转换成文本标签并打印。
    """
    from app.models.profiles_labels import get_label  # 只在测试/展示时使用

    dim = AttentionAllocationEngine.DIMENSION_KEY
    for learner_uid, dim_map in engine_result.items():
        aa = dim_map.get(dim) or {}
        if aa.get("insufficient_data"):
            print(f"[{learner_uid}] 数据不足：{aa.get('insufficient_reason')}")
            continue

        eff = aa.get("efficiency") or {}
        sty = aa.get("style") or {}

        eff_code = eff.get("final_code")
        sty_code = sty.get("final_code")

        eff_label = get_label(dim, "efficiency", eff_code)
        sty_label = get_label(dim, "style", sty_code)

        print(f"\n=== 学习者 {learner_uid} 的注意力分配画像 ===")
        print(f"- 整体效率标签 (code={eff_code}): {eff_label}")
        print(f"- 整体加工风格标签 (code={sty_code}): {sty_label}")
        print("  效率整体数值指标:", eff.get("overall_metrics"))
        print("  风格整体数值指标:", sty.get("overall_metrics"))


if __name__ == "__main__":
    # 使用两个真实 uid 做简单测试
    test_learners = [
        "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
        "lrn_004a9c3f5bf246faab3d390ce716e658",
    ]

    engine = AttentionAllocationEngine()
    numeric_result = engine.analyze(test_learners)

    # 1) 直接打印数值结构的 keys
    print("=== 原始数值结果结构（只展示顶层 keys） ===")
    for uid, payload in numeric_result.items():
        print(uid, "=>", list(payload.keys()))

    # 2) 使用标签配置文件输出带文本标签的整体结果
    print("\n=== 转换为带标签文本的整体结果 ===")
    _print_with_text_labels(numeric_result)
