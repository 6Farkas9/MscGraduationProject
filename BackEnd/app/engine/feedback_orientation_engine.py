# feedback_orientation_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.repositories.feedback_orientation_repository import (
    FeedbackOrientationRepository,
)

logger = logging.getLogger(__name__)


class FeedbackOrientationEngine:
    """
    反馈敏感度与数据使用能力（Feedback Orientation & Data Use Literacy）分析引擎。

    - Repository 负责准备课程级行为统计指标：
        feedback_view_rate / support_view_rate / improvement_after_feedback 等；
    - Engine 在“课程内部”基于上述指标构建 FO 指数，并进行 1D k-means 聚类，
      得到三档反馈敏感度与数据使用水平标签（level_code 0/1/2）；
    - Engine 只返回数值 code，具体文案由 app.models.profiles_labels 负责映射。
    """

    DIMENSION_KEY = "feedback_orientation"

    def __init__(
        self,
        repository: Optional[FeedbackOrientationRepository] = None,
        level_cluster_k: int = 3,
        min_learners_per_course: Optional[int] = None,
    ):
        """
        Args:
            repository: 数据准备仓库实例，默认自动构造。
            level_cluster_k: 课程内聚类簇数（对应 level code 0/1/2）。
            min_learners_per_course: 单门课程参与学习者数量下限，
                若 None 则默认 = level_cluster_k。
        """
        self.repository = repository or FeedbackOrientationRepository()
        self.level_cluster_k = max(2, min(level_cluster_k, 3))
        self.min_learners_per_course = (
            min_learners_per_course or self.level_cluster_k
        )

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        对若干学习者进行“反馈敏感度与数据使用能力”分析。

        返回结构示意：
        {
          learner_uid: {
            "feedback_orientation": {
              "insufficient_data": bool,
              "insufficient_reason": Optional[str],
              "level": {
                "final_code": Optional[int],
                "overall_metrics": {...},
                "courses": {
                  crs_uid: {
                    "code": int,
                    "metrics": {...}
                  },
                  ...
                },
              },
            }
          }
        }
        """
        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        logger.info(
            "FeedbackOrientationEngine.analyze: 开始分析，学习者数量: %d",
            len(learner_uids),
        )

        if not learner_uids:
            return {}

        # 1) 仓库层：准备课程级基础指标
        (
            metrics_by_lc,
            learners_per_course,
            learner_courses_map,
        ) = self.repository.load_metrics_for_learners(learner_uids)

        logger.info(
            "FeedbackOrientationEngine.analyze: Repository 返回 (lrn, crs) 条目数: %d，课程数: %d",
            len(metrics_by_lc),
            len(learners_per_course),
        )

        if not metrics_by_lc:
            # 所有人都没有可用数据
            results: Dict[str, Any] = {}
            for uid in learner_uids:
                results[uid] = {
                    self.DIMENSION_KEY: {
                        "insufficient_data": True,
                        "insufficient_reason": "该学习者在反馈相关课程中没有可用数据。",
                        "level": None,
                    }
                }
            return results

        # 2) 课程内 FO 指数计算 + 聚类
        per_lc_result = self._analyze_per_course(metrics_by_lc, learners_per_course)

        logger.info(
            "FeedbackOrientationEngine.analyze: 课程内聚类完成，(lrn, crs) 有效条目数: %d",
            len(per_lc_result),
        )

        # 3) 学习者维度聚合
        final_results: Dict[str, Any] = {}
        for lrn_uid in learner_uids:
            dim_result = self._build_dimension_result_for_learner(
                learner_uid=lrn_uid,
                learner_courses=learner_courses_map.get(lrn_uid, set()),
                per_lc_result=per_lc_result,
            )
            final_results[lrn_uid] = {self.DIMENSION_KEY: dim_result}

        logger.info(
            "FeedbackOrientationEngine.analyze: 分析完成，返回学习者数: %d",
            len(final_results),
        )
        return final_results

    # ------------------------------------------------------------------
    # 课程内部 FO 计算 + 聚类
    # ------------------------------------------------------------------
    def _analyze_per_course(
        self,
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]],
        learners_per_course: Dict[str, int],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在单门课程内部完成：
        - 针对 feedback_view_rate / support_view_rate / improvement_after_feedback
          进行课程内 z 标准化；
        - 计算反馈敏感度与数据使用指数 FO，并在课程级做 min-max 归一化得到 FO_norm_course；
        - 对 FO_norm_course 进行 1D k-means 聚类，得到 level_code（0/1/2）。
        """
        course_entries: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for (lrn_uid, crs_uid), metrics in metrics_by_lc.items():
            course_entries[crs_uid].append((lrn_uid, metrics))

        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_entries.items():
            num_learners = learners_per_course.get(crs_uid, len(entries))

            if num_learners < self.min_learners_per_course:
                logger.info(
                    "FeedbackOrientationEngine: 课程 %s 学习者数 %d < 聚类数 %d，跳过课程级聚类。",
                    crs_uid,
                    num_learners,
                    self.level_cluster_k,
                )
                continue

            logger.info(
                "FeedbackOrientationEngine: 开始对课程 %s 做反馈敏感度聚类，学习者数: %d",
                crs_uid,
                num_learners,
            )

            fv_rates: List[float] = []
            sp_rates: List[float] = []
            imps: List[float] = []

            for _, m in entries:
                fv_rates.append(float(m.get("feedback_view_rate", 0.0)))
                sp_rates.append(float(m.get("support_view_rate", 0.0)))
                imps.append(float(m.get("improvement_after_feedback", 0.0)))

            m_fv, s_fv = self._mean_std(fv_rates)
            m_sp, s_sp = self._mean_std(sp_rates)
            m_im, s_im = self._mean_std(imps)

            # 课程内 FO & FO_norm_course
            FO_vals: List[float] = []
            for i in range(len(entries)):
                z_fv = self._z(fv_rates[i], m_fv, s_fv)
                z_sp = self._z(sp_rates[i], m_sp, s_sp)
                z_im = self._z(imps[i], m_im, s_im)
                FO = (z_fv + z_sp + z_im) / 3.0
                FO_vals.append(FO)

            FO_norm = self._min_max_norm(FO_vals)

            # 一维 k-means 聚类（课程内部）
            k = min(self.level_cluster_k, num_learners)
            cluster_ids = self._kmeans_1d(FO_norm, k=k, max_iter=50)
            cluster_centers = self._compute_cluster_centers(cluster_ids, FO_norm, k)

            # cluster 中心从小到大排序映射到 level_code 0/1/2
            ordered_clusters = sorted(
                range(k), key=lambda cid: cluster_centers.get(cid, 0.0)
            )
            cluster_to_level_code: Dict[int, int] = {}
            for rank, cid in enumerate(ordered_clusters):
                level_code = min(rank, 2)
                cluster_to_level_code[cid] = level_code

            # 汇总为 (lrn, crs) 级结果
            for idx, (lrn_uid, metrics) in enumerate(entries):
                cid = cluster_ids[idx]
                level_code = cluster_to_level_code.get(cid, 1)  # 默认中档

                key = (lrn_uid, crs_uid)
                per_lc_result[key] = {
                    "course_uid": crs_uid,
                    "learner_uid": lrn_uid,
                    "level_code": level_code,
                    "FO": FO_vals[idx],
                    "FO_norm_course": FO_norm[idx],
                    # 回写基础指标，便于后续聚合与展示
                    "feedback_view_count": int(metrics.get("feedback_view_count", 0)),
                    "support_view_count": int(metrics.get("support_view_count", 0)),
                    "feedback_view_rate": float(metrics.get("feedback_view_rate", 0.0)),
                    "support_view_rate": float(metrics.get("support_view_rate", 0.0)),
                    "improvement_after_feedback": float(
                        metrics.get("improvement_after_feedback", 0.0)
                    ),
                    "feedback_view_type_dist": metrics.get(
                        "feedback_view_type_dist", {}
                    ),
                    "opportunity_count": int(metrics.get("opportunity_count", 0)),
                }

        return per_lc_result

    # ------------------------------------------------------------------
    # 学习者维度聚合
    # ------------------------------------------------------------------
    def _build_dimension_result_for_learner(
        self,
        learner_uid: str,
        learner_courses: set,
        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        将 (lrn, crs) 级结果聚合为学习者维度的总结果。
        """
        course_keys = [
            (learner_uid, crs_uid)
            for crs_uid in learner_courses
            if (learner_uid, crs_uid) in per_lc_result
        ]

        if not course_keys:
            return {
                "insufficient_data": True,
                "insufficient_reason": "该学习者在反馈相关课程中的样本量不足，无法进行课程内聚类分析。",
                "level": None,
            }

        level_codes: Dict[str, int] = {}
        level_course_metrics: Dict[str, Dict[str, Any]] = {}

        for (_, crs_uid) in course_keys:
            item = per_lc_result[(learner_uid, crs_uid)]
            level_codes[crs_uid] = int(item.get("level_code", 0))

            level_course_metrics[crs_uid] = {
                "FO_norm_course": float(item.get("FO_norm_course", 0.0)),
                "FO": float(item.get("FO", 0.0)),
                "feedback_view_rate": float(item.get("feedback_view_rate", 0.0)),
                "support_view_rate": float(item.get("support_view_rate", 0.0)),
                "improvement_after_feedback": float(
                    item.get("improvement_after_feedback", 0.0)
                ),
                "feedback_view_count": int(item.get("feedback_view_count", 0)),
                "support_view_count": int(item.get("support_view_count", 0)),
                "opportunity_count": int(item.get("opportunity_count", 0)),
                # 类型分布保留原始 dict，整体聚合时不做数值平均
                "feedback_view_type_dist": item.get("feedback_view_type_dist", {}),
            }

        # 最终 level：出现次数最多；并列时 code 越大越好
        final_level_code = self._choose_final_code_with_priority(
            level_codes.values(),
            priority_map={0: 0, 1: 1, 2: 2},
        )

        level_overall_metrics = self._aggregate_overall_level_metrics(
            level_course_metrics
        )

        level_courses_dict = {
            crs_uid: {
                "code": level_codes[crs_uid],
                "metrics": level_course_metrics[crs_uid],
            }
            for crs_uid in level_codes
        }

        return {
            "insufficient_data": False,
            "insufficient_reason": None,
            "level": {
                "final_code": final_level_code,
                "overall_metrics": level_overall_metrics,
                "courses": level_courses_dict,
            },
        }

    # ------------------------------------------------------------------
    # 工具函数：统计 / 聚类 / 聚合
    # ------------------------------------------------------------------
    @staticmethod
    def _mean_std(values: List[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 0.0
        n = len(values)
        mean_v = sum(values) / float(n)
        if n <= 1:
            return mean_v, 0.0
        var = sum((v - mean_v) ** 2 for v in values) / float(n)
        return mean_v, math.sqrt(var)

    @staticmethod
    def _z(v: float, mean_v: float, std_v: float) -> float:
        if std_v <= 1e-6:
            return 0.0
        return (v - mean_v) / float(std_v)

    @staticmethod
    def _min_max_norm(vals: List[float]) -> List[float]:
        if not vals:
            return []
        v_min = min(vals)
        v_max = max(vals)
        if abs(v_max - v_min) <= 1e-9:
            return [0.5 for _ in vals]
        return [(v - v_min) / (v_max - v_min) for v in vals]

    def _kmeans_1d(
        self, values: List[float], k: int, max_iter: int = 50
    ) -> List[int]:
        """
        一维 KMeans，用于课程内 FO_norm_course 聚类。
        返回：每个样本的 cluster id（0 ~ k-1）。
        """
        n = len(values)
        if n == 0:
            return []
        if k <= 1:
            return [0] * n
        k = min(k, n)

        # 初始化中心：按分位点从数据中取 k 个初始值，避免随机不稳定
        sorted_vals = sorted(values)
        centers = [
            sorted_vals[int(i * (n - 1) / float(k - 1))]
            for i in range(k)
        ]

        for _ in range(max_iter):
            clusters: List[List[float]] = [[] for _ in range(k)]
            assignments: List[int] = []
            for v in values:
                dists = [abs(v - c) for c in centers]
                cid = min(range(k), key=lambda i: dists[i])
                assignments.append(cid)
                clusters[cid].append(v)

            new_centers = []
            for cid in range(k):
                if clusters[cid]:
                    new_centers.append(
                        sum(clusters[cid]) / float(len(clusters[cid]))
                    )
                else:
                    new_centers.append(centers[cid])

            if all(
                abs(new_centers[i] - centers[i]) <= 1e-4
                for i in range(k)
            ):
                centers = new_centers
                break
            centers = new_centers

        return assignments

    @staticmethod
    def _compute_cluster_centers(
        cluster_ids: List[int], values: List[float], k: int
    ) -> Dict[int, float]:
        sums = [0.0] * k
        cnts = [0] * k
        for cid, v in zip(cluster_ids, values):
            if 0 <= cid < k:
                sums[cid] += v
                cnts[cid] += 1
        centers: Dict[int, float] = {}
        for cid in range(k):
            if cnts[cid] > 0:
                centers[cid] = sums[cid] / float(cnts[cid])
            else:
                centers[cid] = 0.0
        return centers

    @staticmethod
    def _choose_final_code_with_priority(
        codes: Iterable[int], priority_map: Dict[int, int]
    ) -> Optional[int]:
        codes = [c for c in codes if c is not None]
        if not codes:
            return None
        counter = Counter(codes)
        max_count = max(counter.values())
        candidate_codes = [c for c, cnt in counter.items() if cnt == max_count]

        if len(candidate_codes) == 1:
            return candidate_codes[0]

        best_code = None
        best_pri = -1
        for c in candidate_codes:
            pri = priority_map.get(c, 0)
            if pri > best_pri or (pri == best_pri and (best_code is None or c > best_code)):
                best_code = c
                best_pri = pri
        return best_code

    @staticmethod
    def _aggregate_overall_level_metrics(
        level_course_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        if not level_course_metrics:
            return {}

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        FO_norm_vals = [
            float(m.get("FO_norm_course", 0.0))
            for m in level_course_metrics.values()
        ]
        FO_vals = [float(m.get("FO", 0.0)) for m in level_course_metrics.values()]
        fv_vals = [
            float(m.get("feedback_view_rate", 0.0))
            for m in level_course_metrics.values()
        ]
        sp_vals = [
            float(m.get("support_view_rate", 0.0))
            for m in level_course_metrics.values()
        ]
        imp_vals = [
            float(m.get("improvement_after_feedback", 0.0))
            for m in level_course_metrics.values()
        ]

        return {
            "FO_norm_mean": mean(FO_norm_vals),
            "FO_mean": mean(FO_vals),
            "feedback_view_rate_mean": mean(fv_vals),
            "support_view_rate_mean": mean(sp_vals),
            "improvement_after_feedback_mean": mean(imp_vals),
            "courses_count": len(level_course_metrics),
        }


# ----------------------------------------------------------------------
# main：简单测试（数值结果 + 文本标签）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import pprint
    from app.models.profiles_labels import get_label

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = FeedbackOrientationEngine()

    # 使用你指定的两个真实 UID 做测试
    test_learners = [
        "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
        "lrn_004a9c3f5bf246faab3d390ce716e658",
    ]

    print("=" * 80)
    print("1) 数值型原始结果（结构示意，仅打印顶层维度 key）：")
    numeric_result = engine.analyze(test_learners)
    pprint.pprint(
        {uid: list(numeric_result.get(uid, {}).keys()) for uid in test_learners},
        width=120,
    )

    print("\n" + "=" * 80)
    print("2) 带文本标签的整体结果（逐个学习者打印）：\n")

    dim_key = FeedbackOrientationEngine.DIMENSION_KEY

    for uid in test_learners:
        dim_data = numeric_result.get(uid, {}).get(dim_key)
        print(f"\n>>> 学习者 {uid}")
        if not dim_data or dim_data.get("insufficient_data"):
            print("  - 数据不足，无法进行反馈敏感度与数据使用能力分析。")
            continue

        level_info = dim_data.get("level") or {}
        level_code = level_info.get("final_code")
        level_label = get_label(dim_key, "level", level_code)

        print(f"  - 反馈敏感度与数据使用水平（level_code={level_code}）: {level_label}")
        print("  - 整体指标:")
        pprint.pprint(level_info.get("overall_metrics"), indent=4, width=120)

        # 展示部分课程标签，方便检查课程内聚类效果
        print("  - 课程级标签与指标（部分字段）：")
        courses = level_info.get("courses") or {}
        for crs_uid, c in courses.items():
            code = c.get("code")
            label = get_label(dim_key, "level", code)
            metrics = c.get("metrics") or {}
            print(
                f"    · 课程 {crs_uid}: level={code}({label}), "
                f"FO_norm={metrics.get('FO_norm_course', 0.0):.3f}, "
                f"fv_rate={metrics.get('feedback_view_rate', 0.0):.3f}, "
                f"sp_rate={metrics.get('support_view_rate', 0.0):.3f}, "
                f"improve={metrics.get('improvement_after_feedback', 0.0):+.3f}"
            )
