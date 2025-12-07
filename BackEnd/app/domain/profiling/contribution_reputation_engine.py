# BackEnd/app/domain/profiling/contribution_reputation_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.data_access.profiling.contribution_reputation_repository import (
    ContributionReputationRepository,
)
from app.domain.common.base_engine import BaseEngine
from app.shared.utils.stats_utils import kmeans_1d, choose_final_code

logger = logging.getLogger(__name__)


class ContributionReputationEngine(BaseEngine):
    """
    元宇宙价值贡献与声望（Metaverse Value Contribution & Reputation）维度分析引擎。

    - Repository 负责准备课程级价值与贡献行为统计；
    - Engine 在“课程内部”基于综合指数进行聚类，得到价值贡献水平标签；
    - 同时根据资源 vs 协作构成计算贡献风格标签；
    - Engine 只返回数值 code，文案由 app.shared.models.profiles_labels 负责映射。
    """

    DIMENSION_KEY = "contribution_reputation"

    def __init__(
        self,
        repository: Optional[ContributionReputationRepository] = None,
        level_cluster_k: int = 3,
        min_learners_per_course: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            repository: 数据准备仓库实例，默认自动构造。
            level_cluster_k: 课程内价值贡献水平聚类簇数（对应 level code 0/1/2）。
            min_learners_per_course: 单门课程参与学习者数量下限，
                若 None 则默认 = level_cluster_k。
            device: 运行设备标记（与 BaseEngine 保持统一接口）。
        """
        super().__init__(device=device)

        self.repository = repository or ContributionReputationRepository()
        self.level_cluster_k = max(2, min(level_cluster_k, 3))
        self.min_learners_per_course = (
            min_learners_per_course or self.level_cluster_k
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
    # 公共接口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        对若干学习者进行“价值贡献与声望”分析。

        返回结构示意：
        {
          learner_uid: {
            "contribution_reputation": {
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
        # 确保已初始化
        if not self.ensure_initialized():
            logger.error("ContributionReputationEngine 初始化失败。")
            return {}

        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        logger.info(
            "ContributionReputationEngine.analyze: 开始分析，学习者数量: %d",
            len(learner_uids),
        )

        if not learner_uids:
            return {}

        # 1) 仓库层：准备课程级基础数据
        (
            value_stats_by_lc,
            learners_per_course,
            learner_courses_map,
        ) = self.repository.load_metrics_for_learners(learner_uids)

        logger.info(
            "ContributionReputationEngine.analyze: Repository 返回 (lrn, crs) 条目数: %d，课程数: %d",
            len(value_stats_by_lc),
            len(learners_per_course),
        )

        # 2) 课程内聚类 + 课程级风格计算
        per_lc_result = self._analyze_per_course(
            value_stats_by_lc, learners_per_course
        )

        logger.info(
            "ContributionReputationEngine.analyze: 课程内聚类与风格计算完成，(lrn, crs) 有效条目数: %d",
            len(per_lc_result),
        )

        # 3) 按学习者聚合成最终结构
        final_results: Dict[str, Any] = {}

        for lrn_uid in learner_uids:
            dim_result = self._build_dimension_result_for_learner(
                learner_uid=lrn_uid,
                learner_courses=learner_courses_map.get(lrn_uid, set()),
                per_lc_result=per_lc_result,
            )
            final_results[lrn_uid] = {self.DIMENSION_KEY: dim_result}

        logger.info(
            "ContributionReputationEngine.analyze: 分析完成，返回学习者数: %d",
            len(final_results),
        )

        return final_results

    # ------------------------------------------------------------------
    # 课程内部分析
    # ------------------------------------------------------------------
    def _analyze_per_course(
        self,
        value_stats_by_lc: Dict[Tuple[str, str], Dict[str, Any]],
        learners_per_course: Dict[str, int],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在单门课程内部完成：
        - 基于 token_gain + 贡献次数构建课程内的价值贡献指数；
        - 使用 1D k-means 在课程内进行聚类，得到 level_code（0/1/2）；
        - 根据资源 vs 协作构成计算 style_code（0/1/2/3）。
        """
        course_entries: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for (lrn_uid, crs_uid), stats in value_stats_by_lc.items():
            course_entries[crs_uid].append((lrn_uid, stats))

        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_entries.items():
            num_learners = learners_per_course.get(crs_uid, len(entries))

            if num_learners < self.min_learners_per_course:
                logger.info(
                    "ContributionReputationEngine: 课程 %s 学习者数 %d < 聚类数 %d，跳过课程级聚类。",
                    crs_uid,
                    num_learners,
                    self.level_cluster_k,
                )
                continue

            logger.info(
                "ContributionReputationEngine: 开始对课程 %s 做价值贡献聚类，学习者数: %d",
                crs_uid,
                num_learners,
            )

            # 1) 为课程内所有学习者构建 token_gain / contrib_total 序列
            token_gains: List[float] = []
            contrib_totals: List[float] = []

            for _, stats in entries:
                token_gains.append(float(stats.get("token_gain", 0.0)))
                contrib_totals.append(float(stats.get("contrib_total", 0.0)))

            # 2) 课程内均值 / 标准差 & z-score
            mean_token, std_token = self._mean_std(token_gains)
            mean_contrib, std_contrib = self._mean_std(contrib_totals)

            z_token: List[float] = []
            z_contrib: List[float] = []
            for tg, ct in zip(token_gains, contrib_totals):
                if std_token > 1e-6:
                    zt = (tg - mean_token) / std_token
                else:
                    zt = 0.0
                if std_contrib > 1e-6:
                    zc = (ct - mean_contrib) / std_contrib
                else:
                    zc = 0.0
                z_token.append(zt)
                z_contrib.append(zc)

            # 3) 课程内综合指数 C + 课程内 min-max 归一化得到 C_norm_course
            C_vals: List[float] = []
            for zt, zc in zip(z_token, z_contrib):
                C_vals.append((zt + zc) / math.sqrt(2.0))

            C_norm = self._min_max_norm(C_vals)

            # 4) 一维 k-means （在课程内部）
            k = min(self.level_cluster_k, num_learners)
            cluster_ids = kmeans_1d(C_norm, k=k, max_iter=50)
            cluster_centers = self._compute_cluster_centers(cluster_ids, C_norm, k)

            # cluster 中心从小到大排序映射到 level_code 0/1/2
            ordered_clusters = sorted(
                range(k), key=lambda cid: cluster_centers.get(cid, 0.0)
            )
            cluster_to_level_code: Dict[int, int] = {}
            for rank, cid in enumerate(ordered_clusters):
                # rank ∈ [0, k-1]，映射到 level_code 0/1/2
                level_code = min(rank, 2)
                cluster_to_level_code[cid] = level_code

            # 5) 为课程内每个学习者生成 (lrn, crs) 级结果
            for idx, (lrn_uid, stats) in enumerate(entries):
                cid = cluster_ids[idx]
                level_code = cluster_to_level_code.get(cid, 1)

                style_code, style_metrics = self._compute_style_for_course(stats)

                per_lc_result[(lrn_uid, crs_uid)] = {
                    "course_uid": crs_uid,
                    "learner_uid": lrn_uid,
                    "level_code": level_code,
                    "style_code": style_code,
                    "C_norm_course": C_norm[idx],
                    "token_gain": float(stats.get("token_gain", 0.0)),
                    "contrib_total": float(stats.get("contrib_total", 0.0)),
                    "value_events": int(stats.get("value_events", 0)),
                    "style_metrics": style_metrics,
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
                "insufficient_reason": "该学习者在价值贡献相关课程中的样本量不足，无法进行课程内聚类分析。",
                "level": None,
                "style": None,
            }

        level_codes: Dict[str, int] = {}
        style_codes: Dict[str, int] = {}
        level_course_metrics: Dict[str, Dict[str, float]] = {}
        style_course_metrics: Dict[str, Dict[str, float]] = {}

        for (_, crs_uid) in course_keys:
            item = per_lc_result[(learner_uid, crs_uid)]

            level_codes[crs_uid] = int(item.get("level_code", 0))
            style_codes[crs_uid] = int(item.get("style_code", 0))

            level_course_metrics[crs_uid] = {
                "C_norm_course": float(item.get("C_norm_course", 0.0)),
                "token_gain": float(item.get("token_gain", 0.0)),
                "contrib_total": float(item.get("contrib_total", 0.0)),
                "value_events": int(item.get("value_events", 0)),
            }

            style_course_metrics[crs_uid] = {
                **(item.get("style_metrics") or {}),
                "value_events": int(item.get("value_events", 0)),
            }

        # 整体 level 最终 code：出现次数最多，若并列选“更好”的（数值越大越好）
        final_level_code = choose_final_code(
            list(level_codes.values()),
            code_priority={0: 0, 1: 1, 2: 2},
        )

        # 整体 style 最终 code：
        # 平衡型(1) 优先，其次协作型(2) / 资源型(3)，最后未定义(0)
        final_style_code = choose_final_code(
            list(style_codes.values()),
            code_priority={0: 0, 1: 3, 2: 2, 3: 2},
        )

        level_overall_metrics = self._aggregate_overall_level_metrics(
            level_course_metrics
        )
        style_overall_metrics = self._aggregate_overall_style_metrics(
            style_course_metrics
        )

        level_courses_dict = {
            crs_uid: {
                "code": level_codes[crs_uid],
                "metrics": level_course_metrics[crs_uid],
            }
            for crs_uid in level_codes
        }
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
            "level": {
                "final_code": final_level_code,
                "overall_metrics": level_overall_metrics,
                "courses": level_courses_dict,
            },
            "style": {
                "final_code": final_style_code,
                "overall_metrics": style_overall_metrics,
                "courses": style_courses_dict,
            },
        }

    # ------------------------------------------------------------------
    # 工具：样本统计 / 风格计算 / 聚类辅助
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

    def _compute_style_for_course(
        self, stats: Dict[str, Any]
    ) -> Tuple[int, Dict[str, float]]:
        """
        根据课程级贡献行为构成判定贡献风格 style_code。

        对应 profiles_labels.contribution_reputation.style 的 code 约定：
            0: 未定义/数据不足
            1: 平衡型
            2: 协作驱动型（co-edit + collaborated 占比较高）
            3: 资源驱动型（resource 占比较高）
        """
        resource_cnt = float(stats.get("resource_contrib_count", 0.0))
        coedit_cnt = float(stats.get("coedit_count", 0.0))
        collab_cnt = float(stats.get("collab_count", 0.0))

        total = resource_cnt + coedit_cnt + collab_cnt
        if total <= 0:
            return 0, {
                "resource_share": 0.0,
                "coedit_share": 0.0,
                "collab_share": 0.0,
                "total_actions": 0.0,
            }

        resource_share = resource_cnt / total
        coedit_share = coedit_cnt / total
        collab_share = collab_cnt / total
        collab_like_share = coedit_share + collab_share

        # 平衡型：整体没有明显偏向，并且资源 vs 协作比较接近
        max_share = max(resource_share, collab_like_share)
        if max_share < 0.6 and abs(resource_share - collab_like_share) < 0.15:
            style_code = 1
        else:
            if collab_like_share >= resource_share:
                style_code = 2
            else:
                style_code = 3

        style_metrics = {
            "resource_share": resource_share,
            "coedit_share": coedit_share,
            "collab_share": collab_share,
            "total_actions": total,
        }
        return style_code, style_metrics

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

        C_vals = [
            m.get("C_norm_course", 0.0) for m in level_course_metrics.values()
        ]
        tg_vals = [m.get("token_gain", 0.0) for m in level_course_metrics.values()]
        ct_vals = [m.get("contrib_total", 0.0) for m in level_course_metrics.values()]

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        return {
            "C_norm_course_mean": mean(C_vals),
            "token_gain_mean": mean(tg_vals),
            "contrib_total_mean": mean(ct_vals),
            "courses_count": len(level_course_metrics),
        }

    @staticmethod
    def _aggregate_overall_style_metrics(
        style_course_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        if not style_course_metrics:
            return {}

        rs = [m.get("resource_share", 0.0) for m in style_course_metrics.values()]
        cs = [m.get("coedit_share", 0.0) for m in style_course_metrics.values()]
        ls = [m.get("collab_share", 0.0) for m in style_course_metrics.values()]

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        return {
            "resource_share_mean": mean(rs),
            "coedit_share_mean": mean(cs),
            "collab_share_mean": mean(ls),
            "courses_count": len(style_course_metrics),
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

    engine = ContributionReputationEngine()

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

    dim_key = ContributionReputationEngine.DIMENSION_KEY

    for uid in test_learners:
        dim_data = numeric_result.get(uid, {}).get(dim_key)
        print(f"\n>>> 学习者 {uid}")
        if not dim_data or dim_data.get("insufficient_data"):
            print("  - 数据不足，无法进行价值贡献与声望分析。")
            continue

        level_info = dim_data.get("level") or {}
        style_info = dim_data.get("style") or {}

        level_code = level_info.get("final_code")
        style_code = style_info.get("final_code")

        level_label = get_label(dim_key, "level", level_code)
        style_label = get_label(dim_key, "style", style_code)

        print(f"  - 价值贡献水平（level_code={level_code}）: {level_label}")
        print(f"  - 价值贡献风格（style_code={style_code}）: {style_label}")

        print("  - 水平整体指标:")
        pprint.pprint(level_info.get("overall_metrics"), indent=4, width=120)

        print("  - 风格整体指标:")
        pprint.pprint(style_info.get("overall_metrics"), indent=4, width=120)
