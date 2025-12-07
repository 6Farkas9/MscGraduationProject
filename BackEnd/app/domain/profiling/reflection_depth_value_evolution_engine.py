# BackEnd/app/domain/profiling/reflection_depth_value_evolution_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.data_access.profiling.reflection_depth_value_evolution_repository import (
    ReflectionDepthValueEvolutionRepository,
)
from app.domain.common.base_engine import BaseEngine
from app.shared.utils.stats_utils import choose_final_code

logger = logging.getLogger(__name__)


class ReflectionDepthValueEvolutionEngine(BaseEngine):
    """
    反思深度与价值观演变（reflection_value_evolution）分析引擎。

    - Repository 负责准备课程级行为统计指标：
        reflection_count / depth_score_avg / value_evolution_score /
        reflection_to_action_rate / reflection_index_norm 等；
    - Engine 在“课程内部”基于
        (reflection_index_norm, value_evolution_score) 做 2D k-means 聚类，
      自动映射为三档反思水平标签（level_code 0/1/2）；
    - Engine 只返回数值 code，具体文案由 app.shared.models.profiles_labels 负责映射。
    """

    # 注意：维度 key 与 profiles_labels 中保持一致
    DIMENSION_KEY = "reflection_value_evolution"

    def __init__(
        self,
        repository: Optional[ReflectionDepthValueEvolutionRepository] = None,
        level_cluster_k: int = 3,
        min_learners_per_course: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            repository: 数据准备仓库实例，默认自动构造。
            level_cluster_k: 课程内聚类簇数（对应 level code 0/1/2）。
            min_learners_per_course: 单门课程参与学习者数量下限，
                若 None 则默认 = level_cluster_k。
            device: 运行设备标记（与 BaseEngine 保持统一接口）。
        """
        super().__init__(device=device)

        self.repository = repository or ReflectionDepthValueEvolutionRepository()
        self.level_cluster_k = max(2, min(level_cluster_k, 3))
        self.min_learners_per_course = (
            min_learners_per_course or self.level_cluster_k
        )

    # ------------------------------------------------------------------
    # BaseEngine 接口
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """
        当前引擎没有额外模型需要加载，保持接口以便未来扩展。
        """
        if self.is_initialized:
            return True
        self.is_initialized = True
        logger.info("%s 初始化完成。", self.__class__.__name__)
        return True

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        对若干学习者进行“反思深度与价值观演变”分析。

        返回结构示意：
        {
          learner_uid: {
            "reflection_value_evolution": {
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
        # 确保已初始化
        if not self.ensure_initialized():
            logger.error("ReflectionDepthValueEvolutionEngine 初始化失败。")
            return {}

        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        logger.info(
            "ReflectionDepthValueEvolutionEngine.analyze: 开始分析，学习者数量: %d",
            len(learner_uids),
        )

        if not learner_uids:
            return {}

        (
            metrics_by_lc,
            learners_per_course,
            learner_courses_map,
        ) = self.repository.load_metrics_for_learners(learner_uids)

        logger.info(
            "ReflectionDepthValueEvolutionEngine.analyze: Repository 返回 (lrn, crs) 条目数: %d，课程数: %d",
            len(metrics_by_lc),
            len(learners_per_course),
        )

        if not metrics_by_lc:
            results: Dict[str, Any] = {}
            for uid in learner_uids:
                results[uid] = {
                    self.DIMENSION_KEY: {
                        "insufficient_data": True,
                        "insufficient_reason": "该学习者在反思相关课程中没有可用数据。",
                        "level": None,
                    }
                }
            return results

        per_lc_result = self._analyze_per_course(metrics_by_lc, learners_per_course)

        logger.info(
            "ReflectionDepthValueEvolutionEngine.analyze: 课程内聚类完成，(lrn, crs) 有效条目数: %d",
            len(per_lc_result),
        )

        final_results: Dict[str, Any] = {}
        for lrn_uid in learner_uids:
            dim_result = self._build_dimension_result_for_learner(
                learner_uid=lrn_uid,
                learner_courses=learner_courses_map.get(lrn_uid, set()),
                per_lc_result=per_lc_result,
            )
            final_results[lrn_uid] = {self.DIMENSION_KEY: dim_result}

        logger.info(
            "ReflectionDepthValueEvolutionEngine.analyze: 分析完成，返回学习者数: %d",
            len(final_results),
        )
        return final_results

    # ------------------------------------------------------------------
    # 课程内部聚类
    # ------------------------------------------------------------------
    def _analyze_per_course(
        self,
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]],
        learners_per_course: Dict[str, int],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在单门课程内部完成：
        - 基于 (reflection_index_norm, value_evolution_score) 做二维 k-means 聚类；
        - 自动将 cluster 映射到三档水平 code（0/1/2）；
        - 生成 (lrn, crs) 级别的结果。
        """
        course_entries: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for (lrn_uid, crs_uid), metrics in metrics_by_lc.items():
            course_entries[crs_uid].append((lrn_uid, metrics))

        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_entries.items():
            num_learners = learners_per_course.get(crs_uid, len(entries))

            if num_learners < self.min_learners_per_course:
                logger.info(
                    "ReflectionDepthValueEvolutionEngine: 课程 %s 学习者数 %d < 聚类数 %d，跳过课程级聚类。",
                    crs_uid,
                    num_learners,
                    self.level_cluster_k,
                )
                continue

            logger.info(
                "ReflectionDepthValueEvolutionEngine: 开始对课程 %s 做反思维度聚类，学习者数: %d",
                crs_uid,
                num_learners,
            )

            points: List[Tuple[float, float]] = []
            for _, m in entries:
                idx_norm = float(m.get("reflection_index_norm", 0.0))
                ve = float(m.get("value_evolution_score", 0.0))
                points.append((idx_norm, ve))

            k = min(self.level_cluster_k, num_learners)
            centers, assignments = self._kmeans_2d(points, k=k, max_iter=50)

            if not centers:
                logger.info(
                    "ReflectionDepthValueEvolutionEngine: 课程 %s k-means 结果为空，跳过。",
                    crs_uid,
                )
                continue

            cluster_to_level_code: Dict[int, int] = self._map_clusters_to_level_codes(
                centers
            )

            for (lrn_uid, metrics), cluster_idx in zip(entries, assignments):
                level_code = cluster_to_level_code.get(cluster_idx, 1)

                key = (lrn_uid, crs_uid)
                per_lc_result[key] = {
                    "course_uid": crs_uid,
                    "learner_uid": lrn_uid,
                    "level_code": level_code,
                    "reflection_index_norm": float(
                        metrics.get("reflection_index_norm", 0.0)
                    ),
                    "reflection_index": float(metrics.get("reflection_index", 0.0)),
                    "reflection_count": int(metrics.get("reflection_count", 0)),
                    "depth_score_avg": float(metrics.get("depth_score_avg", 0.0)),
                    "value_evolution_score": float(
                        metrics.get("value_evolution_score", 0.0)
                    ),
                    "reflection_to_action_rate": float(
                        metrics.get("reflection_to_action_rate", 0.0)
                    ),
                    # 全局归一化特征一起带出，方便后续整体指标计算
                    "freq_norm": float(metrics.get("freq_norm", 0.0)),
                    "depth_norm": float(metrics.get("depth_norm", 0.0)),
                    "value_growth_norm": float(
                        metrics.get("value_growth_norm", 0.0)
                    ),
                    "action_norm": float(metrics.get("action_norm", 0.0)),
                }

        return per_lc_result

    def _map_clusters_to_level_codes(
        self, centers: List[Tuple[float, float]]
    ) -> Dict[int, int]:
        """
        根据聚类中心 (idx_norm, value_evolution_score) 自动映射到三档 level code：

        设计思路：
        - idx_norm 越大 → 反思频率 & 深度整体越高；
        - value_evolution_score 越大 → 价值语汇“后期 - 早期”越正向；
        - 组合成 score = 0.7 * idx_norm + 0.3 * max(value_evolution_score, 0)
          用于排序和分档。

        映射规则：
        - score 最低 → code=0（浅层或不稳定反思者）
        - 中间 → code=1（稳定深度反思者）
        - 最高 → code=2（成长型价值反思者）
        """
        k = len(centers)
        if k == 0:
            return {}

        scores: List[float] = []
        for (idx_norm, ve) in centers:
            s = 0.7 * float(idx_norm) + 0.3 * max(float(ve), 0.0)
            scores.append(s)

        ordered = sorted(range(k), key=lambda i: scores[i])
        cluster_to_code: Dict[int, int] = {}

        if k == 1:
            cluster_to_code[ordered[0]] = 1
        elif k == 2:
            cluster_to_code[ordered[0]] = 0
            cluster_to_code[ordered[1]] = 2
        else:
            cluster_to_code[ordered[0]] = 0
            cluster_to_code[ordered[1]] = 1
            cluster_to_code[ordered[2]] = 2

        return cluster_to_code

    # ------------------------------------------------------------------
    # 学习者维度聚合
    # ------------------------------------------------------------------
    def _build_dimension_result_for_learner(
        self,
        learner_uid: str,
        learner_courses: Iterable[str],
        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        将 (lrn, crs) 级结果聚合为学习者维度的总结果。
        """
        learner_courses = set(learner_courses or [])
        course_keys = [
            (learner_uid, crs_uid)
            for crs_uid in learner_courses
            if (learner_uid, crs_uid) in per_lc_result
        ]

        if not course_keys:
            return {
                "insufficient_data": True,
                "insufficient_reason": "该学习者在反思相关课程中的样本量不足，无法进行课程内聚类分析。",
                "level": None,
            }

        level_codes: Dict[str, int] = {}
        course_metrics: Dict[str, Dict[str, Any]] = {}

        for (_, crs_uid) in course_keys:
            item = per_lc_result[(learner_uid, crs_uid)]
            level_codes[crs_uid] = int(item.get("level_code", 0))

            course_metrics[crs_uid] = {
                "reflection_index_norm": float(
                    item.get("reflection_index_norm", 0.0)
                ),
                "reflection_index": float(item.get("reflection_index", 0.0)),
                "reflection_count": int(item.get("reflection_count", 0)),
                "depth_score_avg": float(item.get("depth_score_avg", 0.0)),
                "value_evolution_score": float(
                    item.get("value_evolution_score", 0.0)
                ),
                "reflection_to_action_rate": float(
                    item.get("reflection_to_action_rate", 0.0)
                ),
                "freq_norm": float(item.get("freq_norm", 0.0)),
                "depth_norm": float(item.get("depth_norm", 0.0)),
                "value_growth_norm": float(
                    item.get("value_growth_norm", 0.0)
                ),
                "action_norm": float(item.get("action_norm", 0.0)),
            }

        # 最终 level：出现次数最多；并列时 code 越大越好（2>1>0）
        final_code = choose_final_code(
            list(level_codes.values()),
            code_priority={0: 0, 1: 1, 2: 2},
        )

        overall_metrics = self._aggregate_overall_level_metrics(course_metrics)

        level_courses_dict = {
            crs_uid: {
                "code": level_codes[crs_uid],
                "metrics": course_metrics[crs_uid],
            }
            for crs_uid in level_codes
        }

        return {
            "insufficient_data": False,
            "insufficient_reason": None,
            "level": {
                "final_code": final_code,
                "overall_metrics": overall_metrics,
                "courses": level_courses_dict,
            },
        }

    # ------------------------------------------------------------------
    # 工具函数：统计 / 聚类 / 聚合
    # ------------------------------------------------------------------
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
    def _aggregate_overall_level_metrics(
        course_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        if not course_metrics:
            return {}

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        idx_vals = [
            float(m.get("reflection_index_norm", 0.0))
            for m in course_metrics.values()
        ]
        raw_idx_vals = [
            float(m.get("reflection_index", 0.0)) for m in course_metrics.values()
        ]
        depth_vals = [
            float(m.get("depth_score_avg", 0.0)) for m in course_metrics.values()
        ]
        ve_vals = [
            float(m.get("value_evolution_score", 0.0))
            for m in course_metrics.values()
        ]
        act_vals = [
            float(m.get("reflection_to_action_rate", 0.0))
            for m in course_metrics.values()
        ]

        return {
            "reflection_index_norm_mean": mean(idx_vals),
            "reflection_index_mean": mean(raw_idx_vals),
            "depth_score_mean": mean(depth_vals),
            "value_evolution_score_mean": mean(ve_vals),
            "reflection_to_action_rate_mean": mean(act_vals),
            "courses_count": len(course_metrics),
        }


# ----------------------------------------------------------------------
# main：简单测试（数值结果 + 文本标签）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import pprint
    from app.shared.models.profiles_labels import get_label

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = ReflectionDepthValueEvolutionEngine()

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

    dim_key = ReflectionDepthValueEvolutionEngine.DIMENSION_KEY

    for uid in test_learners:
        dim_data = numeric_result.get(uid, {}).get(dim_key)
        print(f"\n>>> 学习者 {uid}")
        if not dim_data or dim_data.get("insufficient_data"):
            print("  - 数据不足，无法进行反思深度与价值观演变分析。")
            continue

        level_info = dim_data.get("level") or {}
        level_code = level_info.get("final_code")
        level_label = get_label(dim_key, "level", level_code)

        print(f"  - 反思深度与价值观演变水平（level_code={level_code}）: {level_label}")
        print("  - 整体指标:")
        pprint.pprint(level_info.get("overall_metrics"), indent=4, width=120)

        print("  - 课程级标签与指标（部分字段）：")
        courses = level_info.get("courses") or {}
        for crs_uid, c in courses.items():
            code = c.get("code")
            label = get_label(dim_key, "level", code)
            metrics = c.get("metrics") or {}
            print(
                f"    · 课程 {crs_uid}: level={code}({label}), "
                f"idx_norm={metrics.get('reflection_index_norm', 0.0):.3f}, "
                f"depth={metrics.get('depth_score_avg', 0.0):.3f}, "
                f"val_evo={metrics.get('value_evolution_score', 0.0):+.3f}, "
                f"to_action={metrics.get('reflection_to_action_rate', 0.0):.3f}"
            )
