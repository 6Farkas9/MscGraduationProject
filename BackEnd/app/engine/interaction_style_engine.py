# interaction_style_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.repositories.interaction_style_repository import (
    InteractionStyleRepository,
)

logger = logging.getLogger(__name__)


class InteractionStyleEngine:
    """
    交互与操作熟练度 / 风格（interaction_style）分析引擎。

    - Repository 负责准备课程级行为统计指标：
        freq_per_minute / step_success_rate / unit_success_rate /
        performance_score / x / y；
    - Engine 在“课程内部”基于 (x, y) 进行 2D k-means 聚类，
      自动映射为三档交互风格标签（style_code 0/1/2）；
    - Engine 只返回数值 code，具体文案由 app.models.profiles_labels 负责映射。
    """

    DIMENSION_KEY = "interaction_style"

    def __init__(
        self,
        repository: Optional[InteractionStyleRepository] = None,
        style_cluster_k: int = 3,
        min_learners_per_course: Optional[int] = None,
    ):
        """
        Args:
            repository: 数据准备仓库实例，默认自动构造。
            style_cluster_k: 课程内聚类簇数（对应 style code 0/1/2）。
            min_learners_per_course: 单门课程参与学习者数量下限，
                若 None 则默认 = style_cluster_k。
        """
        self.repository = repository or InteractionStyleRepository()
        self.style_cluster_k = max(2, min(style_cluster_k, 3))
        self.min_learners_per_course = (
            min_learners_per_course or self.style_cluster_k
        )

        # 数值化风格指数映射（用于整体指标）
        self.style_index_map = {
            0: 0.3,  # 随便乱点型
            1: 0.7,  # 多试多练型
            2: 0.9,  # 少操作但准确型
        }

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        对若干学习者进行“交互与操作熟练度 / 风格”分析。

        返回结构示意：
        {
          learner_uid: {
            "interaction_style": {
              "insufficient_data": bool,
              "insufficient_reason": Optional[str],
              "style": {
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
            "InteractionStyleEngine.analyze: 开始分析，学习者数量: %d",
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
            "InteractionStyleEngine.analyze: Repository 返回 (lrn, crs) 条目数: %d，课程数: %d",
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
                        "insufficient_reason": "该学习者在交互相关课程中没有可用数据。",
                        "style": None,
                    }
                }
            return results

        # 2) 课程内风格聚类
        per_lc_result = self._analyze_per_course(metrics_by_lc, learners_per_course)

        logger.info(
            "InteractionStyleEngine.analyze: 课程内聚类完成，(lrn, crs) 有效条目数: %d",
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
            "InteractionStyleEngine.analyze: 分析完成，返回学习者数: %d",
            len(final_results),
        )
        return final_results

    # ------------------------------------------------------------------
    # 课程内部风格聚类
    # ------------------------------------------------------------------
    def _analyze_per_course(
        self,
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]],
        learners_per_course: Dict[str, int],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在单门课程内部完成：
        - 基于 (x, y) = (log(1+freq), 1-performance_score) 做二维 k-means 聚类；
        - 自动将 cluster 映射到三档风格 code（0/1/2）；
        - 生成 (lrn, crs) 级别的风格结果。
        """
        course_entries: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for (lrn_uid, crs_uid), metrics in metrics_by_lc.items():
            course_entries[crs_uid].append((lrn_uid, metrics))

        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_entries.items():
            num_learners = learners_per_course.get(crs_uid, len(entries))

            if num_learners < self.min_learners_per_course:
                logger.info(
                    "InteractionStyleEngine: 课程 %s 学习者数 %d < 聚类数 %d，跳过课程级聚类。",
                    crs_uid,
                    num_learners,
                    self.style_cluster_k,
                )
                continue

            logger.info(
                "InteractionStyleEngine: 开始对课程 %s 做交互风格聚类，学习者数: %d",
                crs_uid,
                num_learners,
            )

            points: List[Tuple[float, float]] = []
            for _, m in entries:
                x = float(m.get("x", 0.0))
                y = float(m.get("y", 0.0))
                points.append((x, y))

            k = min(self.style_cluster_k, num_learners)
            centers, assignments = self._kmeans_2d(points, k=k, max_iter=50)

            if not centers:
                logger.info(
                    "InteractionStyleEngine: 课程 %s k-means 结果为空，跳过。",
                    crs_uid,
                )
                continue

            # 根据中心位置将 cluster 映射到三档风格 code（0: 随便乱点型, 1: 多试多练型, 2: 少操作但准确型）
            cluster_to_style_code: Dict[int, int] = self._map_clusters_to_style_codes(
                centers
            )

            # 写回 (lrn, crs) 级结果
            for (lrn_uid, metrics), cluster_idx in zip(entries, assignments):
                style_code = cluster_to_style_code.get(cluster_idx, 1)  # 默认多试多练型
                style_index = float(self.style_index_map.get(style_code, 0.7))

                key = (lrn_uid, crs_uid)
                per_lc_result[key] = {
                    "course_uid": crs_uid,
                    "learner_uid": lrn_uid,
                    "style_code": style_code,
                    "style_index": style_index,
                    "x": float(metrics.get("x", 0.0)),
                    "y": float(metrics.get("y", 0.0)),
                    "freq_per_minute": float(metrics.get("freq_per_minute", 0.0)),
                    "step_success_rate": float(
                        metrics.get("step_success_rate", 0.0)
                    ),
                    "unit_success_rate": float(
                        metrics.get("unit_success_rate", 0.0)
                    ),
                    "performance_score": float(
                        metrics.get("performance_score", 0.0)
                    ),
                }

        return per_lc_result

    def _map_clusters_to_style_codes(
        self, centers: List[Tuple[float, float]]
    ) -> Dict[int, int]:
        """
        根据聚类中心 (x, y) 自动映射到三档风格 code：

        - 随便乱点型（code=0）：错误/不熟练程度 y 最大；
        - 少操作但准确型（code=2）：在剩余中心中 x+y 最小（低强度+高成功率）；
        - 多试多练型（code=1）：剩下的那个中心。
        """
        k = len(centers)
        if k == 0:
            return {}

        # 1) 找 y 最大的中心 → 随便乱点型
        idx_random = max(range(k), key=lambda i: centers[i][1])
        remain_idx = [i for i in range(k) if i != idx_random]

        if not remain_idx:
            # 只有一个簇，全部视为多试多练型
            return {0: 1}

        if len(remain_idx) == 1:
            idx_precise = remain_idx[0]
            # 两个簇：一个乱点，一个少操作但准确
            cluster_to_code = {
                idx_random: 0,
                idx_precise: 2,
            }
            return cluster_to_code

        # 剩余两个中心：选 x+y 较小的作为“少操作但准确型”
        i1, i2 = remain_idx
        s1 = centers[i1][0] + centers[i1][1]
        s2 = centers[i2][0] + centers[i2][1]
        if s1 <= s2:
            idx_precise = i1
            idx_practice = i2
        else:
            idx_precise = i2
            idx_practice = i1

        cluster_to_code = {
            idx_random: 0,   # 随便乱点型
            idx_practice: 1, # 多试多练型
            idx_precise: 2,  # 少操作但准确型
        }
        return cluster_to_code

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
                "insufficient_reason": "该学习者在交互相关课程中的样本量不足，无法进行课程内聚类分析。",
                "style": None,
            }

        style_codes: Dict[str, int] = {}
        style_course_metrics: Dict[str, Dict[str, Any]] = {}

        for (_, crs_uid) in course_keys:
            item = per_lc_result[(learner_uid, crs_uid)]
            style_codes[crs_uid] = int(item.get("style_code", 1))

            style_course_metrics[crs_uid] = {
                "style_index": float(item.get("style_index", 0.7)),
                "freq_per_minute": float(item.get("freq_per_minute", 0.0)),
                "step_success_rate": float(item.get("step_success_rate", 0.0)),
                "unit_success_rate": float(item.get("unit_success_rate", 0.0)),
                "performance_score": float(item.get("performance_score", 0.0)),
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
            }

        # 最终 style：出现次数最多；并列时 code 越大越好（2>1>0）
        final_style_code = self._choose_final_code_with_priority(
            style_codes.values(),
            priority_map={0: 0, 1: 2, 2: 3},
        )

        style_overall_metrics = self._aggregate_overall_style_metrics(
            style_course_metrics
        )

        style_courses_dict = {
            crs_uid: {
                "code": style_codes[crs_uid],
                "metrics": style_course_metrics[crs_uid],
            }
            for crs_uid in style_codes
        }

        return {
            "insufficient_data": False,
            "insufficient_reason": None,
            "style": {
                "final_code": final_style_code,
                "overall_metrics": style_overall_metrics,
                "courses": style_courses_dict,
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

    def _kmeans_2d(
        self, points: List[Tuple[float, float]], k: int, max_iter: int = 50
    ) -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        二维 k-means 聚类，返回 (centers, assignments)。
        """
        if not points:
            return [], []

        n = len(points)
        if n < k:
            k = n

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # 所有点几乎重合的退化情况
        if abs(x_max - x_min) < 1e-6 and abs(y_max - y_min) < 1e-6:
            centers = [(x_min, y_min) for _ in range(k)]
            assignments = [0 for _ in range(n)]
            return centers, assignments

        # 初始化中心：在对角线上均匀放置 k 个中心
        centers: List[Tuple[float, float]] = []
        for i in range(k):
            alpha = (i + 0.5) / float(k)
            cx = x_min + (x_max - x_min) * alpha
            cy = y_min + (y_max - y_min) * alpha
            centers.append((cx, cy))

        for _ in range(max_iter):
            clusters: List[List[int]] = [[] for _ in range(k)]
            for idx, (x, y) in enumerate(points):
                best_c = 0
                cx, cy = centers[0]
                best_dist = (x - cx) ** 2 + (y - cy) ** 2
                for ci in range(1, k):
                    cx, cy = centers[ci]
                    d = (x - cx) ** 2 + (y - cy) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_c = ci
                clusters[best_c].append(idx)

            new_centers = list(centers)
            for ci in range(k):
                if clusters[ci]:
                    sum_x = sum(points[idx][0] for idx in clusters[ci])
                    sum_y = sum(points[idx][1] for idx in clusters[ci])
                    cnt = float(len(clusters[ci]))
                    new_centers[ci] = (sum_x / cnt, sum_y / cnt)
                else:
                    new_centers[ci] = centers[ci]

            max_shift = 0.0
            for (ox, oy), (nx, ny) in zip(centers, new_centers):
                shift = (ox - nx) ** 2 + (oy - ny) ** 2
                if shift > max_shift:
                    max_shift = shift
            centers = new_centers
            if max_shift < 1e-4:
                break

        # 最终 assignments
        assignments: List[int] = []
        for (x, y) in points:
            best_c = 0
            cx, cy = centers[0]
            best_dist = (x - cx) ** 2 + (y - cy) ** 2
            for ci in range(1, k):
                cx, cy = centers[ci]
                d = (x - cx) ** 2 + (y - cy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_c = ci
            assignments.append(best_c)

        return centers, assignments

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
    def _aggregate_overall_style_metrics(
        style_course_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        if not style_course_metrics:
            return {}

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        idx_vals = [
            float(m.get("style_index", 0.7))
            for m in style_course_metrics.values()
        ]
        freq_vals = [
            float(m.get("freq_per_minute", 0.0))
            for m in style_course_metrics.values()
        ]
        step_vals = [
            float(m.get("step_success_rate", 0.0))
            for m in style_course_metrics.values()
        ]
        unit_vals = [
            float(m.get("unit_success_rate", 0.0))
            for m in style_course_metrics.values()
        ]
        perf_vals = [
            float(m.get("performance_score", 0.0))
            for m in style_course_metrics.values()
        ]

        return {
            "style_index_mean": mean(idx_vals),
            "freq_per_minute_mean": mean(freq_vals),
            "step_success_rate_mean": mean(step_vals),
            "unit_success_rate_mean": mean(unit_vals),
            "performance_score_mean": mean(perf_vals),
            "courses_count": len(style_course_metrics),
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

    engine = InteractionStyleEngine()

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

    dim_key = InteractionStyleEngine.DIMENSION_KEY

    for uid in test_learners:
        dim_data = numeric_result.get(uid, {}).get(dim_key)
        print(f"\n>>> 学习者 {uid}")
        if not dim_data or dim_data.get("insufficient_data"):
            print("  - 数据不足，无法进行交互与操作风格分析。")
            continue

        style_info = dim_data.get("style") or {}
        style_code = style_info.get("final_code")
        style_label = get_label(dim_key, "style", style_code)

        print(f"  - 交互与操作风格（style_code={style_code}）: {style_label}")
        print("  - 整体指标:")
        pprint.pprint(style_info.get("overall_metrics"), indent=4, width=120)

        print("  - 课程级标签与指标（部分字段）：")
        courses = style_info.get("courses") or {}
        for crs_uid, c in courses.items():
            code = c.get("code")
            label = get_label(dim_key, "style", code)
            metrics = c.get("metrics") or {}
            print(
                f"    · 课程 {crs_uid}: style={code}({label}), "
                f"freq={metrics.get('freq_per_minute', 0.0):.3f}, "
                f"step_suc={metrics.get('step_success_rate', 0.0):.3f}, "
                f"unit_suc={metrics.get('unit_success_rate', 0.0):.3f}, "
                f"perf={metrics.get('performance_score', 0.0):.3f}"
            )
