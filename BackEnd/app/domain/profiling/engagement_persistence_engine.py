# BackEnd/app/domain/profiling/engagement_persistence_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.data_access.profiling.engagement_persistence_repository import (
    EngagementPersistenceRepository,
)
from app.domain.common.base_engine import BaseEngine
from app.shared.utils.stats_utils import kmeans_1d, choose_final_code

logger = logging.getLogger(__name__)


class EngagementPersistenceEngine(BaseEngine):
    """
    行为投入度与坚持性（Behavioral Engagement & Persistence）分析引擎。

    - Repository 负责准备课程级行为统计指标：
        completion_rate / interaction_per_unit / retry_rate /
        extension_rate / idle_ratio / value_rate
    - Engine 在“课程内部”基于上述指标构建 EP 指数，并进行 1D k-means 聚类，
      得到三档行为投入度与坚持性水平标签（level_code 0/1/2）；
    - Engine 只返回数值 code，具体文案由 app.shared.models.profiles_labels 负责映射。
    """

    DIMENSION_KEY = "engagement_persistence"

    def __init__(
        self,
        repository: Optional[EngagementPersistenceRepository] = None,
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

        self.repository = repository or EngagementPersistenceRepository()
        self.level_cluster_k = max(2, min(level_cluster_k, 3))
        self.min_learners_per_course = (
            min_learners_per_course or self.level_cluster_k
        )

        # EP 权重设定，与原脚本保持一致语义
        self.w_completion = 1.5
        self.w_retry = 1.5
        self.w_extension = 1.2
        self.w_value = 1.0
        self.w_interact = 1.0
        self.w_idle = 1.2
        self._ep_weight_norm = math.sqrt(
            self.w_completion**2
            + self.w_retry**2
            + self.w_extension**2
            + self.w_value**2
            + self.w_interact**2
            + self.w_idle**2
        )

    # ------------------------------------------------------------------
    # BaseEngine 接口实现
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
        对若干学习者进行“行为投入度与坚持性”分析。

        返回结构示意：
        {
          learner_uid: {
            "engagement_persistence": {
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
            logger.error("EngagementPersistenceEngine 初始化失败。")
            return {}

        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        logger.info(
            "EngagementPersistenceEngine.analyze: 开始分析，学习者数量: %d",
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
            "EngagementPersistenceEngine.analyze: Repository 返回 (lrn, crs) 条目数: %d，课程数: %d",
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
                        "insufficient_reason": "该学习者在行为投入度与坚持性相关的课程中没有可用数据。",
                        "level": None,
                    }
                }
            return results

        # 2) 课程内 EP 指数计算 + 聚类
        per_lc_result = self._analyze_per_course(metrics_by_lc, learners_per_course)

        logger.info(
            "EngagementPersistenceEngine.analyze: 课程内聚类完成，(lrn, crs) 有效条目数: %d",
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
            "EngagementPersistenceEngine.analyze: 分析完成，返回学习者数: %d",
            len(final_results),
        )
        return final_results

    # ------------------------------------------------------------------
    # 课程内部 EP 计算 + 聚类
    # ------------------------------------------------------------------
    def _analyze_per_course(
        self,
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, float]],
        learners_per_course: Dict[str, int],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在单门课程内部完成：
        - 针对 completion_rate / interaction_per_unit / retry_rate /
          extension_rate / idle_ratio / value_rate 进行课程内 z 标准化；
        - 计算行为投入度与坚持性指数 EP，并在课程级做 min-max 归一化得到 EP_norm_course；
        - 对 EP_norm_course 进行 1D k-means 聚类，得到 level_code（0/1/2）。
        """
        course_entries: Dict[str, List[Tuple[str, Dict[str, float]]]] = defaultdict(list)
        for (lrn_uid, crs_uid), metrics in metrics_by_lc.items():
            course_entries[crs_uid].append((lrn_uid, metrics))

        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_entries.items():
            num_learners = learners_per_course.get(crs_uid, len(entries))

            if num_learners < self.min_learners_per_course:
                logger.info(
                    "EngagementPersistenceEngine: 课程 %s 学习者数 %d < 聚类数 %d，跳过课程级聚类。",
                    crs_uid,
                    num_learners,
                    self.level_cluster_k,
                )
                continue

            logger.info(
                "EngagementPersistenceEngine: 开始对课程 %s 做行为投入度聚类，学习者数: %d",
                crs_uid,
                num_learners,
            )

            # 收集课程内各指标列表
            comp_vals: List[float] = []
            inter_vals: List[float] = []
            retry_vals: List[float] = []
            ext_vals: List[float] = []
            idle_vals: List[float] = []
            value_vals: List[float] = []

            for _, m in entries:
                comp_vals.append(float(m.get("completion_rate", 0.0)))
                inter_vals.append(float(m.get("interaction_per_unit", 0.0)))
                retry_vals.append(float(m.get("retry_rate", 0.0)))
                ext_vals.append(float(m.get("extension_rate", 0.0)))
                idle_vals.append(float(m.get("idle_ratio", 0.0)))
                value_vals.append(float(m.get("value_rate", 0.0)))

            mean_comp, std_comp = self._mean_std(comp_vals)
            mean_inter, std_inter = self._mean_std(inter_vals)
            mean_retry, std_retry = self._mean_std(retry_vals)
            mean_ext, std_ext = self._mean_std(ext_vals)
            mean_idle, std_idle = self._mean_std(idle_vals)
            mean_value, std_value = self._mean_std(value_vals)

            # 课程内 EP & EP_norm_course
            EP_vals: List[float] = []
            for i in range(len(entries)):
                z_c = self._z(comp_vals[i], mean_comp, std_comp)
                z_i = self._z(inter_vals[i], mean_inter, std_inter)
                z_r = self._z(retry_vals[i], mean_retry, std_retry)
                z_e = self._z(ext_vals[i], mean_ext, std_ext)
                z_idle = self._z(idle_vals[i], mean_idle, std_idle)
                z_v = self._z(value_vals[i], mean_value, std_value)

                EP = (
                    self.w_completion * z_c
                    + self.w_retry * z_r
                    + self.w_extension * z_e
                    + self.w_value * z_v
                    + self.w_interact * z_i
                    - self.w_idle * z_idle
                )
                if self._ep_weight_norm > 0:
                    EP /= self._ep_weight_norm
                EP_vals.append(EP)

            EP_norm = self._min_max_norm(EP_vals)

            # 一维 k-means 聚类（课程内部）
            k = min(self.level_cluster_k, num_learners)
            cluster_ids = kmeans_1d(EP_norm, k=k, max_iter=50)
            cluster_centers = self._compute_cluster_centers(cluster_ids, EP_norm, k)

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

                per_lc_result[(lrn_uid, crs_uid)] = {
                    "course_uid": crs_uid,
                    "learner_uid": lrn_uid,
                    "level_code": level_code,
                    "EP": EP_vals[idx],
                    "EP_norm_course": EP_norm[idx],
                    # 回写基础指标，便于后续聚合与展示
                    "completion_rate": float(metrics.get("completion_rate", 0.0)),
                    "interaction_per_unit": float(
                        metrics.get("interaction_per_unit", 0.0)
                    ),
                    "retry_rate": float(metrics.get("retry_rate", 0.0)),
                    "extension_rate": float(metrics.get("extension_rate", 0.0)),
                    "idle_ratio": float(metrics.get("idle_ratio", 0.0)),
                    "value_rate": float(metrics.get("value_rate", 0.0)),
                }

        return per_lc_result

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
                "insufficient_reason": "该学习者在行为投入度与坚持性相关课程中的样本量不足，无法进行课程内聚类分析。",
                "level": None,
            }

        level_codes: Dict[str, int] = {}
        level_course_metrics: Dict[str, Dict[str, float]] = {}

        for (_, crs_uid) in course_keys:
            item = per_lc_result[(learner_uid, crs_uid)]
            level_codes[crs_uid] = int(item.get("level_code", 0))

            level_course_metrics[crs_uid] = {
                "EP_norm_course": float(item.get("EP_norm_course", 0.0)),
                "EP": float(item.get("EP", 0.0)),
                "completion_rate": float(item.get("completion_rate", 0.0)),
                "interaction_per_unit": float(
                    item.get("interaction_per_unit", 0.0)
                ),
                "retry_rate": float(item.get("retry_rate", 0.0)),
                "extension_rate": float(item.get("extension_rate", 0.0)),
                "idle_ratio": float(item.get("idle_ratio", 0.0)),
                "value_rate": float(item.get("value_rate", 0.0)),
            }

        # 最终 level：出现次数最多；并列时 code 越大越好
        final_level_code = choose_final_code(
            list(level_codes.values()),
            code_priority={0: 0, 1: 1, 2: 2},
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
    def _aggregate_overall_level_metrics(
        level_course_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        if not level_course_metrics:
            return {}

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        EP_norm_vals = [
            m.get("EP_norm_course", 0.0) for m in level_course_metrics.values()
        ]
        EP_vals = [m.get("EP", 0.0) for m in level_course_metrics.values()]
        comp_vals = [
            m.get("completion_rate", 0.0)
            for m in level_course_metrics.values()
        ]
        inter_vals = [
            m.get("interaction_per_unit", 0.0)
            for m in level_course_metrics.values()
        ]
        retry_vals = [
            m.get("retry_rate", 0.0) for m in level_course_metrics.values()
        ]
        ext_vals = [
            m.get("extension_rate", 0.0)
            for m in level_course_metrics.values()
        ]
        idle_vals = [
            m.get("idle_ratio", 0.0) for m in level_course_metrics.values()
        ]
        val_vals = [
            m.get("value_rate", 0.0) for m in level_course_metrics.values()
        ]

        return {
            "EP_norm_mean": mean(EP_norm_vals),
            "EP_mean": mean(EP_vals),
            "completion_rate_mean": mean(comp_vals),
            "interaction_per_unit_mean": mean(inter_vals),
            "retry_rate_mean": mean(retry_vals),
            "extension_rate_mean": mean(ext_vals),
            "idle_ratio_mean": mean(idle_vals),
            "value_rate_mean": mean(val_vals),
            "courses_count": len(level_course_metrics),
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

    engine = EngagementPersistenceEngine()

    # 按你的要求，使用这两个真实 UID 测试
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

    dim_key = EngagementPersistenceEngine.DIMENSION_KEY

    for uid in test_learners:
        dim_data = numeric_result.get(uid, {}).get(dim_key)
        print(f"\n>>> 学习者 {uid}")
        if not dim_data or dim_data.get("insufficient_data"):
            print("  - 数据不足，无法进行行为投入度与坚持性分析。")
            continue

        level_info = dim_data.get("level") or {}
        level_code = level_info.get("final_code")
        level_label = get_label(dim_key, "level", level_code)

        print(f"  - 行为投入度与坚持性水平（level_code={level_code}）: {level_label}")
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
                f"EP_norm={metrics.get('EP_norm_course', 0.0):.3f}, "
                f"completion={metrics.get('completion_rate', 0.0):.3f}, "
                f"retry={metrics.get('retry_rate', 0.0):.3f}, "
                f"ext={metrics.get('extension_rate', 0.0):.3f}, "
                f"idle={metrics.get('idle_ratio', 0.0):.3f}"
            )
