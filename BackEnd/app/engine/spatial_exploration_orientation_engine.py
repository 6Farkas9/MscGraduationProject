# spatial_exploration_orientation_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.repositories.spatial_exploration_orientation_repository import (
    SpatialExplorationOrientationRepository,
)

logger = logging.getLogger(__name__)


class SpatialExplorationOrientationEngine:
    """
    空间与资源探索倾向（exploration_orientation）分析引擎。

    - Repository 负责准备课程级行为统计指标：
        unique_spaces / unique_resources / has_extension / path_jump /
        teleport_ratio / 各自 z 分数 / exploration_index_norm 等；
    - Engine 在“课程内部”基于 exploration_index_norm 做一维 k-means 聚类，
      自动映射为三档探索水平标签（level_code 0/1/2）；
    - Engine 只返回数值 code，具体文案由 app.models.profiles_labels 负责映射。
    """

    DIMENSION_KEY = "exploration_orientation"

    def __init__(
        self,
        repository: Optional[SpatialExplorationOrientationRepository] = None,
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
        self.repository = repository or SpatialExplorationOrientationRepository()
        self.level_cluster_k = max(2, min(level_cluster_k, 3))
        self.min_learners_per_course = (
            min_learners_per_course or self.level_cluster_k
        )

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        对若干学习者进行“空间与资源探索倾向”分析。

        返回结构示例：
        {
          learner_uid: {
            "exploration_orientation": {
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
            "SpatialExplorationOrientationEngine.analyze: 开始分析，学习者数量: %d",
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
            "SpatialExplorationOrientationEngine.analyze: Repository 返回 (lrn, crs) 条目数: %d，课程数: %d",
            len(metrics_by_lc),
            len(learners_per_course),
        )

        if not metrics_by_lc:
            results: Dict[str, Any] = {}
            for uid in learner_uids:
                results[uid] = {
                    self.DIMENSION_KEY: {
                        "insufficient_data": True,
                        "insufficient_reason": "该学习者在空间/资源探索相关课程中没有可用数据。",
                        "level": None,
                    }
                }
            return results

        per_lc_result = self._analyze_per_course(metrics_by_lc, learners_per_course)

        logger.info(
            "SpatialExplorationOrientationEngine.analyze: 课程内聚类完成，(lrn, crs) 有效条目数: %d",
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
            "SpatialExplorationOrientationEngine.analyze: 分析完成，返回学习者数: %d",
            len(final_results),
        )
        return final_results

    # ------------------------------------------------------------------
    # 课程内部聚类：一维 k-means on exploration_index_norm
    # ------------------------------------------------------------------
    def _analyze_per_course(
        self,
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]],
        learners_per_course: Dict[str, int],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在单门课程内部完成：
        - 基于 exploration_index_norm 做一维 k-means 聚类；
        - 根据簇中心大小映射为三档 level_code（0/1/2）；
        - 生成 (lrn, crs) 级别结果。
        """
        course_entries: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for (lrn_uid, crs_uid), metrics in metrics_by_lc.items():
            course_entries[crs_uid].append((lrn_uid, metrics))

        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_entries.items():
            num_learners = learners_per_course.get(crs_uid, len(entries))

            # 健壮性：若课程学习者数少于聚类簇数，则跳过整门课程
            if num_learners < self.min_learners_per_course:
                logger.info(
                    "SpatialExplorationOrientationEngine: 课程 %s 学习者数 %d < 聚类数 %d，跳过课程级聚类。",
                    crs_uid,
                    num_learners,
                    self.level_cluster_k,
                )
                continue

            logger.info(
                "SpatialExplorationOrientationEngine: 开始对课程 %s 做探索水平聚类，学习者数: %d",
                crs_uid,
                num_learners,
            )

            values: List[float] = []
            for _, m in entries:
                v = float(m.get("exploration_index_norm", 0.0))
                values.append(v)

            k = min(self.level_cluster_k, num_learners)
            centers, labels = self._kmeans_1d(values, k=k, max_iter=50)

            if not centers:
                logger.info(
                    "SpatialExplorationOrientationEngine: 课程 %s k-means 结果为空，跳过。",
                    crs_uid,
                )
                continue

            cluster_to_level_code = self._map_clusters_to_level_codes(centers)

            for (lrn_uid, metrics), cluster_idx in zip(entries, labels):
                level_code = cluster_to_level_code.get(cluster_idx, 1)

                key = (lrn_uid, crs_uid)
                per_lc_result[key] = {
                    "course_uid": crs_uid,
                    "learner_uid": lrn_uid,
                    "level_code": level_code,
                    "exploration_index_norm": float(
                        metrics.get("exploration_index_norm", 0.0)
                    ),
                    "exploration_index": float(
                        metrics.get("exploration_index", 0.0)
                    ),
                    "unique_spaces": int(metrics.get("unique_spaces", 0)),
                    "unique_resources": int(metrics.get("unique_resources", 0)),
                    "has_extension": int(metrics.get("has_extension", 0)),
                    "path_jump": int(metrics.get("path_jump", 0)),
                    "teleport_ratio": float(metrics.get("teleport_ratio", 0.0)),
                }

        return per_lc_result

    def _map_clusters_to_level_codes(
        self, centers: List[float]
    ) -> Dict[int, int]:
        """
        根据聚类中心（exploration_index_norm）大小映射为三档 level code：
        - center 越小，level 越低；
        - center 越大，level 越高。

        规则：
        - k == 1: 唯一簇 -> code = 1（中等）
        - k == 2: 小 -> 0， 大 -> 2
        - k >= 3: 按从小到大排序，依次映射 0/1/2
        """
        k = len(centers)
        if k == 0:
            return {}

        ordered = sorted(range(k), key=lambda i: centers[i])
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
                "insufficient_reason": "该学习者在空间/资源探索相关课程中的样本量不足，无法进行课程内聚类分析。",
                "level": None,
            }

        level_codes: Dict[str, int] = {}
        course_metrics: Dict[str, Dict[str, Any]] = {}

        for (_, crs_uid) in course_keys:
            item = per_lc_result[(learner_uid, crs_uid)]
            level_codes[crs_uid] = int(item.get("level_code", 0))

            course_metrics[crs_uid] = {
                "exploration_index_norm": float(
                    item.get("exploration_index_norm", 0.0)
                ),
                "exploration_index": float(
                    item.get("exploration_index", 0.0)
                ),
                "unique_spaces": int(item.get("unique_spaces", 0)),
                "unique_resources": int(item.get("unique_resources", 0)),
                "has_extension": int(item.get("has_extension", 0)),
                "path_jump": int(item.get("path_jump", 0)),
                "teleport_ratio": float(item.get("teleport_ratio", 0.0)),
            }

        # 最终 level：出现次数最多；并列时 code 越大越好（2>1>0）
        final_code = self._choose_final_code_with_priority(
            level_codes.values(),
            priority_map={0: 0, 1: 1, 2: 2},
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
    # 工具函数：一维 k-means / 统计
    # ------------------------------------------------------------------
    def _kmeans_1d(
        self, values: List[float], k: int, max_iter: int = 50
    ) -> Tuple[List[float], List[int]]:
        """
        一维 k-means 聚类（Lloyd 算法），用于课程内 exploration_index_norm 聚类。
        """
        import random

        if not values:
            return [], []

        n = len(values)
        if n < k:
            k = n

        v_min, v_max = min(values), max(values)
        if abs(v_max - v_min) < 1e-6:
            centers = [v_min for _ in range(k)]
            assignments = [0 for _ in range(n)]
            return centers, assignments

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

            new_centers: List[float] = []
            for ci in range(k):
                if clusters[ci]:
                    new_centers.append(
                        sum(values[idx] for idx in clusters[ci])
                        / float(len(clusters[ci]))
                    )
                else:
                    new_centers.append(random.choice(values))

            shift = sum(
                abs(a - b) for a, b in zip(centers, new_centers)
            )
            centers = new_centers
            if shift < 1e-6:
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
    def _aggregate_overall_level_metrics(
        course_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        if not course_metrics:
            return {}

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        E_norm_vals = [
            float(m.get("exploration_index_norm", 0.0))
            for m in course_metrics.values()
        ]
        E_vals = [
            float(m.get("exploration_index", 0.0))
            for m in course_metrics.values()
        ]
        space_vals = [
            float(m.get("unique_spaces", 0.0))
            for m in course_metrics.values()
        ]
        res_vals = [
            float(m.get("unique_resources", 0.0))
            for m in course_metrics.values()
        ]
        ext_vals = [
            float(m.get("has_extension", 0.0))
            for m in course_metrics.values()
        ]
        path_vals = [
            float(m.get("path_jump", 0.0))
            for m in course_metrics.values()
        ]
        tp_vals = [
            float(m.get("teleport_ratio", 0.0))
            for m in course_metrics.values()
        ]

        return {
            "exploration_index_norm_mean": mean(E_norm_vals),
            "exploration_index_mean": mean(E_vals),
            "unique_spaces_mean": mean(space_vals),
            "unique_resources_mean": mean(res_vals),
            "has_extension_mean": mean(ext_vals),
            "path_jump_mean": mean(path_vals),
            "teleport_ratio_mean": mean(tp_vals),
            "courses_count": len(course_metrics),
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

    engine = SpatialExplorationOrientationEngine()

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

    dim_key = SpatialExplorationOrientationEngine.DIMENSION_KEY

    for uid in test_learners:
        dim_data = numeric_result.get(uid, {}).get(dim_key)
        print(f"\n>>> 学习者 {uid}")
        if not dim_data or dim_data.get("insufficient_data"):
            print("  - 数据不足，无法进行空间与资源探索倾向分析。")
            continue

        level_info = dim_data.get("level") or {}
        level_code = level_info.get("final_code")
        level_label = get_label(dim_key, "level", level_code)

        print(f"  - 探索倾向水平（level_code={level_code}）: {level_label}")
        print("  - 整体指标:")
        pprint.pprint(level_info.get("overall_metrics"), indent=4, width=120)

        print("  - 课程级标签与指标（部分字段）：")
        courses = level_info.get("courses") or {}
        for crs_uid, c in courses.items():
            code = c.get("code")
            label = get_label(dim_key, "level", code)
            m = c.get("metrics") or {}
            print(
                f"    · 课程 {crs_uid}: level={code}({label}), "
                f"E_norm={m.get('exploration_index_norm', 0.0):.3f}, "
                f"E={m.get('exploration_index', 0.0):.3f}, "
                f"spaces={m.get('unique_spaces', 0)}, "
                f"resources={m.get('unique_resources', 0)}, "
                f"ext={m.get('has_extension', 0)}, "
                f"path_jump={m.get('path_jump', 0)}, "
                f"tp_ratio={m.get('teleport_ratio', 0.0):.2f}"
            )
