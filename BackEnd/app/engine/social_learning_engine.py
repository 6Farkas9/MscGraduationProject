# social_learning_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.repositories.social_learning_repository import SocialLearningRepository

logger = logging.getLogger(__name__)


class SocialLearningEngine:
    """
    社会性学习与同伴取向（social_learning）分析引擎。

    - Repository 负责准备课程级行为统计指标：
        obs/collab 次数与时长、观摩同伴数、social_index_norm 等；
    - Engine 在“课程内部”基于 (social_index_norm, ratio_obs) 做 2D k-means 聚类，
      并将 cluster 映射为四档社会角色 code（0~3）：
        0: 低社交参与型
        1: 观察型（观摩为主）
        2: 协作导向型（协作为主）
        3: 积极社会学习型（观摩+协作均衡且总体较高）；
    - Engine 只返回数值 code，具体文案由 app.models.profiles_labels 负责映射。
    """

    DIMENSION_KEY = "social_learning"

    def __init__(
        self,
        repository: Optional[SocialLearningRepository] = None,
        role_cluster_k: int = 4,
        min_learners_per_course: Optional[int] = None,
    ):
        """
        Args:
            repository: 数据准备仓库实例，默认自动构造。
            role_cluster_k: 课程内聚类簇数（对应 role code 0/1/2/3）。
            min_learners_per_course: 单门课程参与学习者数量下限，
                若 None 则默认 = role_cluster_k。
        """
        self.repository = repository or SocialLearningRepository()
        # 该维度设计为 4 类角色
        self.role_cluster_k = max(2, min(role_cluster_k, 4))
        self.min_learners_per_course = (
            min_learners_per_course or self.role_cluster_k
        )

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        对若干学习者进行“社会性学习与同伴取向”分析。

        返回结构示例：
        {
          learner_uid: {
            "social_learning": {
              "insufficient_data": bool,
              "insufficient_reason": Optional[str],
              "role": {
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
            "SocialLearningEngine.analyze: 开始分析，学习者数量: %d",
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
            "SocialLearningEngine.analyze: Repository 返回 (lrn, crs) 条目数: %d，课程数: %d",
            len(metrics_by_lc),
            len(learners_per_course),
        )

        if not metrics_by_lc:
            results: Dict[str, Any] = {}
            for uid in learner_uids:
                results[uid] = {
                    self.DIMENSION_KEY: {
                        "insufficient_data": True,
                        "insufficient_reason": "该学习者在社会性相关课程中没有可用数据。",
                        "role": None,
                    }
                }
            return results

        per_lc_result = self._analyze_per_course(metrics_by_lc, learners_per_course)

        logger.info(
            "SocialLearningEngine.analyze: 课程内聚类完成，(lrn, crs) 有效条目数: %d",
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
            "SocialLearningEngine.analyze: 分析完成，返回学习者数: %d",
            len(final_results),
        )
        return final_results

    # ------------------------------------------------------------------
    # 课程内部聚类：2D k-means on (social_index_norm, ratio_obs)
    # ------------------------------------------------------------------
    def _analyze_per_course(
        self,
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]],
        learners_per_course: Dict[str, int],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在单门课程内部完成：
        - 基于 (social_index_norm, ratio_obs) 做二维 k-means 聚类；
        - 根据簇中心的「总体社会性水平 + 观摩占比」映射为 4 档角色 code（0/1/2/3）；
        - 生成 (lrn, crs) 级结果。
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
                    "SocialLearningEngine: 课程 %s 学习者数 %d < 聚类数 %d，跳过课程级聚类。",
                    crs_uid,
                    num_learners,
                    self.role_cluster_k,
                )
                continue

            logger.info(
                "SocialLearningEngine: 开始对课程 %s 做社会性角色聚类，学习者数: %d",
                crs_uid,
                num_learners,
            )

            points: List[Tuple[float, float]] = []
            total_times: List[float] = []

            for _, m in entries:
                obs_time = float(m.get("obs_total_time", 0.0))
                collab_time = float(m.get("collab_total_time", 0.0))
                total_time = obs_time + collab_time
                total_times.append(total_time)

                s_norm = float(m.get("social_index_norm", 0.0))
                if total_time > 0:
                    ratio_obs = obs_time / total_time
                else:
                    ratio_obs = 0.5  # 无数据时给中间值
                points.append((s_norm, ratio_obs))

            k = min(self.role_cluster_k, num_learners)
            centers, assignments = self._kmeans_2d(points, k=k, max_iter=50)

            if not centers:
                logger.info(
                    "SocialLearningEngine: 课程 %s k-means 结果为空，跳过。",
                    crs_uid,
                )
                continue

            # 聚类后，统计每个簇的平均 social_index_norm / ratio_obs / total_time
            cluster_stats: Dict[int, Dict[str, float]] = defaultdict(
                lambda: {"sum_s": 0.0, "sum_ratio": 0.0, "sum_time": 0.0, "cnt": 0.0}
            )
            for (s_norm, ratio_obs), total_time, c_idx in zip(
                points, total_times, assignments
            ):
                st = cluster_stats[c_idx]
                st["sum_s"] += s_norm
                st["sum_ratio"] += ratio_obs
                st["sum_time"] += total_time
                st["cnt"] += 1.0

            for ci, st in cluster_stats.items():
                if st["cnt"] > 0:
                    st["avg_s"] = st["sum_s"] / st["cnt"]
                    st["avg_ratio"] = st["sum_ratio"] / st["cnt"]
                    st["avg_time"] = st["sum_time"] / st["cnt"]
                else:
                    st["avg_s"] = 0.0
                    st["avg_ratio"] = 0.5
                    st["avg_time"] = 0.0

            cluster_to_role_code = self._map_clusters_to_role_codes(cluster_stats)

            for (lrn_uid, metrics), c_idx in zip(entries, assignments):
                role_code = cluster_to_role_code.get(c_idx, 0)

                obs_time = float(metrics.get("obs_total_time", 0.0))
                collab_time = float(metrics.get("collab_total_time", 0.0))
                total_time = obs_time + collab_time
                if total_time > 0:
                    ratio_obs = obs_time / total_time
                else:
                    ratio_obs = 0.5

                key = (lrn_uid, crs_uid)
                per_lc_result[key] = {
                    "course_uid": crs_uid,
                    "learner_uid": lrn_uid,
                    "role_code": role_code,
                    "social_index_norm": float(
                        metrics.get("social_index_norm", 0.0)
                    ),
                    "social_index": float(metrics.get("social_index", 0.0)),
                    "obs_count": int(metrics.get("obs_count", 0)),
                    "obs_total_time": obs_time,
                    "obs_unique_peers": int(metrics.get("obs_unique_peers", 0)),
                    "collab_count": int(metrics.get("collab_count", 0)),
                    "collab_total_time": collab_time,
                    "ratio_obs": ratio_obs,
                    "total_social_time": total_time,
                }

        return per_lc_result

    def _map_clusters_to_role_codes(
        self,
        cluster_stats: Dict[int, Dict[str, float]],
    ) -> Dict[int, int]:
        """
        根据每个簇的平均 social_index_norm / ratio_obs / total_time
        将簇映射为四类角色 code：

        设计思路（在单门课程内部）：
        - avg_time 越小 → 越像低社交参与型；
        - 在剩余簇中：
            * avg_ratio 越大 → 越观察型；
            * avg_ratio 越小 → 越协作导向；
            * avg_ratio 居中且 avg_s 高 → 积极社会学习型。
        """
        nonempty_clusters = [ci for ci, st in cluster_stats.items() if st["cnt"] > 0]
        if not nonempty_clusters:
            return {}

        # 1) 低社交参与型：total time 最小的簇
        low_ci = min(
            nonempty_clusters, key=lambda ci: cluster_stats[ci].get("avg_time", 0.0)
        )

        cluster_to_code: Dict[int, int] = {low_ci: 0}
        remaining = [ci for ci in nonempty_clusters if ci != low_ci]
        if not remaining:
            return cluster_to_code

        # 2) 观察型：观摩占比最高
        obs_ci = max(
            remaining, key=lambda ci: cluster_stats[ci].get("avg_ratio", 0.0)
        )
        cluster_to_code[obs_ci] = 1
        remaining = [ci for ci in remaining if ci != obs_ci]
        if not remaining:
            return cluster_to_code

        # 3) 协作导向型：观摩占比最低
        collab_ci = min(
            remaining, key=lambda ci: cluster_stats[ci].get("avg_ratio", 0.0)
        )
        cluster_to_code[collab_ci] = 2
        remaining = [ci for ci in remaining if ci != collab_ci]
        if not remaining:
            return cluster_to_code

        # 4) 积极社会学习型：在剩余簇中 social_index_norm 平均值最高
        balanced_ci = max(
            remaining, key=lambda ci: cluster_stats[ci].get("avg_s", 0.0)
        )
        cluster_to_code[balanced_ci] = 3

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
                "insufficient_reason": "该学习者在社会性相关课程中的样本量不足，无法进行课程内聚类分析。",
                "role": None,
            }

        role_codes: Dict[str, int] = {}
        course_metrics: Dict[str, Dict[str, Any]] = {}

        for (_, crs_uid) in course_keys:
            item = per_lc_result[(learner_uid, crs_uid)]
            role_codes[crs_uid] = int(item.get("role_code", 0))

            course_metrics[crs_uid] = {
                "social_index_norm": float(
                    item.get("social_index_norm", 0.0)
                ),
                "social_index": float(item.get("social_index", 0.0)),
                "obs_count": int(item.get("obs_count", 0)),
                "obs_total_time": float(item.get("obs_total_time", 0.0)),
                "obs_unique_peers": int(item.get("obs_unique_peers", 0)),
                "collab_count": int(item.get("collab_count", 0)),
                "collab_total_time": float(item.get("collab_total_time", 0.0)),
                "ratio_obs": float(item.get("ratio_obs", 0.5)),
                "total_social_time": float(item.get("total_social_time", 0.0)),
            }

        # 最终 role：出现次数最多；并列时 code 越大越好（3>2>1>0）
        final_code = self._choose_final_code_with_priority(
            role_codes.values(),
            priority_map={0: 0, 1: 1, 2: 2, 3: 3},
        )

        overall_metrics = self._aggregate_overall_role_metrics(course_metrics)

        role_courses_dict = {
            crs_uid: {
                "code": role_codes[crs_uid],
                "metrics": course_metrics[crs_uid],
            }
            for crs_uid in role_codes
        }

        return {
            "insufficient_data": False,
            "insufficient_reason": None,
            "role": {
                "final_code": final_code,
                "overall_metrics": overall_metrics,
                "courses": role_courses_dict,
            },
        }

    # ------------------------------------------------------------------
    # 工具函数：2D k-means / 统计
    # ------------------------------------------------------------------
    def _kmeans_2d(
        self,
        points: List[Tuple[float, float]],
        k: int,
        max_iter: int = 50,
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

        if abs(x_max - x_min) < 1e-6 and abs(y_max - y_min) < 1e-6:
            centers = [(x_min, y_min) for _ in range(k)]
            assignments = [0 for _ in range(n)]
            return centers, assignments

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
    def _aggregate_overall_role_metrics(
        course_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        if not course_metrics:
            return {}

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        s_norm_vals = [
            float(m.get("social_index_norm", 0.0))
            for m in course_metrics.values()
        ]
        s_vals = [
            float(m.get("social_index", 0.0))
            for m in course_metrics.values()
        ]
        obs_time_vals = [
            float(m.get("obs_total_time", 0.0))
            for m in course_metrics.values()
        ]
        collab_time_vals = [
            float(m.get("collab_total_time", 0.0))
            for m in course_metrics.values()
        ]
        obs_cnt_vals = [
            float(m.get("obs_count", 0.0))
            for m in course_metrics.values()
        ]
        collab_cnt_vals = [
            float(m.get("collab_count", 0.0))
            for m in course_metrics.values()
        ]
        peers_vals = [
            float(m.get("obs_unique_peers", 0.0))
            for m in course_metrics.values()
        ]
        ratio_vals = [
            float(m.get("ratio_obs", 0.0))
            for m in course_metrics.values()
        ]

        return {
            "social_index_norm_mean": mean(s_norm_vals),
            "social_index_mean": mean(s_vals),
            "obs_total_time_mean": mean(obs_time_vals),
            "collab_total_time_mean": mean(collab_time_vals),
            "obs_count_mean": mean(obs_cnt_vals),
            "collab_count_mean": mean(collab_cnt_vals),
            "obs_unique_peers_mean": mean(peers_vals),
            "ratio_obs_mean": mean(ratio_vals),
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

    engine = SocialLearningEngine()

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

    dim_key = SocialLearningEngine.DIMENSION_KEY

    for uid in test_learners:
        dim_data = numeric_result.get(uid, {}).get(dim_key)
        print(f"\n>>> 学习者 {uid}")
        if not dim_data or dim_data.get("insufficient_data"):
            print("  - 数据不足，无法进行社会性学习与同伴取向分析。")
            continue

        role_info = dim_data.get("role") or {}
        role_code = role_info.get("final_code")
        role_label = get_label(dim_key, "role", role_code)

        print(f"  - 社会性学习角色（role_code={role_code}）: {role_label}")
        print("  - 角色整体指标:")
        pprint.pprint(role_info.get("overall_metrics"), indent=4, width=120)

        print("  - 课程级角色与指标（部分字段）：")
        courses = role_info.get("courses") or {}
        for crs_uid, c in courses.items():
            code = c.get("code")
            label = get_label(dim_key, "role", code)
            m = c.get("metrics") or {}
            print(
                f"    · 课程 {crs_uid}: role={code}({label}), "
                f"S_norm={m.get('social_index_norm', 0.0):.3f}, "
                f"obs_t={m.get('obs_total_time', 0.0):.1f}s, "
                f"collab_t={m.get('collab_total_time', 0.0):.1f}s, "
                f"ratio_obs={m.get('ratio_obs', 0.0):.2f}, "
                f"obs_cnt={m.get('obs_count', 0)}, "
                f"collab_cnt={m.get('collab_count', 0)}, "
                f"peers={m.get('obs_unique_peers', 0)}"
            )
