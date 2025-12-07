# BackEnd/app/domain/profiling/slr_helpseeking_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.data_access.profiling.srl_helpseeking_repository import SRLHelpseekingRepository
from app.domain.common.base_engine import BaseEngine
from app.shared.utils.stats_utils import kmeans_1d, choose_final_code

logger = logging.getLogger(__name__)


class SRLHelpseekingEngine(BaseEngine):
    """
    自我调节与求助策略（srl_helpseeking）分析引擎。

    - Repository 负责准备课程级行为统计指标：
        srl_help_index / help_need_rate / early_help_ratio / ...
    - Engine 在“课程内部”基于 srl_help_index 做一维 k-means 聚类，
      自动映射为三档水平标签（level_code 0/1/2）；
    - Engine 只返回数值 code，具体文案由 app.shared.models.profiles_labels 负责映射。
    """

    DIMENSION_KEY = "srl_helpseeking"

    def __init__(
        self,
        repository: Optional[SRLHelpseekingRepository] = None,
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

        self.repository = repository or SRLHelpseekingRepository()
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
    # 对外主接口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        对若干学习者进行“自我调节与求助策略”分析。

        返回结构示例：
        {
          learner_uid: {
            "srl_helpseeking": {
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
            logger.error("SRLHelpseekingEngine 初始化失败。")
            return {}

        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        logger.info(
            "SRLHelpseekingEngine.analyze: 开始分析，学习者数量: %d",
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
            "SRLHelpseekingEngine.analyze: Repository 返回 (lrn, crs) 条目数: %d，课程数: %d",
            len(metrics_by_lc),
            len(learners_per_course),
        )

        if not metrics_by_lc:
            results: Dict[str, Any] = {}
            for uid in learner_uids:
                results[uid] = {
                    self.DIMENSION_KEY: {
                        "insufficient_data": True,
                        "insufficient_reason": "该学习者在自我调节/求助相关课程中没有可用数据。",
                        "level": None,
                    }
                }
            return results

        per_lc_result = self._analyze_per_course(metrics_by_lc, learners_per_course)

        logger.info(
            "SRLHelpseekingEngine.analyze: 课程内聚类完成，(lrn, crs) 有效条目数: %d",
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
            "SRLHelpseekingEngine.analyze: 分析完成，返回学习者数: %d",
            len(final_results),
        )
        return final_results

    # ------------------------------------------------------------------
    # 课程内部聚类：一维 k-means on srl_help_index
    # ------------------------------------------------------------------
    def _analyze_per_course(
        self,
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]],
        learners_per_course: Dict[str, int],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在单门课程内部完成：
        - 基于 srl_help_index 做一维 k-means 聚类；
        - 根据簇中心大小映射为三档 level_code（0/1/2）；
        - 生成 (lrn, crs) 级别结果。
        """
        course_entries: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for (lrn_uid, crs_uid), metrics in metrics_by_lc.items():
            course_entries[crs_uid].append((lrn_uid, metrics))

        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_entries.items():
            num_learners = learners_per_course.get(crs_uid, len(entries))

            if num_learners < self.min_learners_per_course:
                logger.info(
                    "SRLHelpseekingEngine: 课程 %s 学习者数 %d < 聚类数 %d，跳过课程级聚类。",
                    crs_uid,
                    num_learners,
                    self.level_cluster_k,
                )
                continue

            logger.info(
                "SRLHelpseekingEngine: 开始对课程 %s 做 SRL_help 聚类，学习者数: %d",
                crs_uid,
                num_learners,
            )

            values: List[float] = []
            for _, m in entries:
                v = float(m.get("srl_help_index", 0.0))
                values.append(v)

            k = min(self.level_cluster_k, num_learners)

            # 使用公共 kmeans_1d 得到簇分配，再在本地计算簇中心
            cluster_ids = kmeans_1d(values, k=k, max_iter=50)
            if not cluster_ids:
                logger.info(
                    "SRLHelpseekingEngine: 课程 %s k-means 结果为空，跳过。",
                    crs_uid,
                )
                continue

            centers = self._compute_cluster_centers(cluster_ids, values, k)
            cluster_to_level_code = self._map_clusters_to_level_codes(centers)

            for (lrn_uid, metrics), cluster_idx in zip(entries, cluster_ids):
                level_code = cluster_to_level_code.get(cluster_idx, 1)

                key = (lrn_uid, crs_uid)
                per_lc_result[key] = {
                    "course_uid": crs_uid,
                    "learner_uid": lrn_uid,
                    "level_code": level_code,
                    "srl_help_index": float(metrics.get("srl_help_index", 0.0)),
                    "help_need_rate": float(metrics.get("help_need_rate", 0.0)),
                    "early_help_ratio": float(
                        metrics.get("early_help_ratio", 0.0)
                    ),
                    "no_help_success_ratio": float(
                        metrics.get("no_help_success_ratio", 0.0)
                    ),
                    "feedback_density": float(
                        metrics.get("feedback_density", 0.0)
                    ),
                    "extension_flag": int(metrics.get("extension_flag", 0)),
                    "reflection_flag": int(metrics.get("reflection_flag", 0)),
                }

        return per_lc_result

    @staticmethod
    def _compute_cluster_centers(
        cluster_ids: List[int], values: List[float], k: int
    ) -> List[float]:
        """
        根据一维聚类分配结果计算每个簇的中心（均值）。
        """
        sums = [0.0] * k
        counts = [0] * k
        for cid, v in zip(cluster_ids, values):
            if 0 <= cid < k:
                sums[cid] += v
                counts[cid] += 1

        centers: List[float] = []
        for i in range(k):
            if counts[i] > 0:
                centers.append(sums[i] / float(counts[i]))
            else:
                centers.append(0.0)
        return centers

    def _map_clusters_to_level_codes(
        self, centers: List[float]
    ) -> Dict[int, int]:
        """
        根据聚类中心（srl_help_index）大小映射为三档 level code：
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
                "insufficient_reason": "该学习者在自我调节/求助相关课程中的样本量不足，无法进行课程内聚类分析。",
                "level": None,
            }

        level_codes: Dict[str, int] = {}
        course_metrics: Dict[str, Dict[str, Any]] = {}

        for (_, crs_uid) in course_keys:
            item = per_lc_result[(learner_uid, crs_uid)]
            level_codes[crs_uid] = int(item.get("level_code", 0))

            course_metrics[crs_uid] = {
                "srl_help_index": float(item.get("srl_help_index", 0.0)),
                "help_need_rate": float(item.get("help_need_rate", 0.0)),
                "early_help_ratio": float(
                    item.get("early_help_ratio", 0.0)
                ),
                "no_help_success_ratio": float(
                    item.get("no_help_success_ratio", 0.0)
                ),
                "feedback_density": float(
                    item.get("feedback_density", 0.0)
                ),
                "extension_flag": int(item.get("extension_flag", 0)),
                "reflection_flag": int(item.get("reflection_flag", 0)),
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
    # 工具函数：统计
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_overall_level_metrics(
        course_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        if not course_metrics:
            return {}

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        srl_vals = [
            float(m.get("srl_help_index", 0.0))
            for m in course_metrics.values()
        ]
        need_vals = [
            float(m.get("help_need_rate", 0.0))
            for m in course_metrics.values()
        ]
        early_vals = [
            float(m.get("early_help_ratio", 0.0))
            for m in course_metrics.values()
        ]
        nohelp_vals = [
            float(m.get("no_help_success_ratio", 0.0))
            for m in course_metrics.values()
        ]
        fb_vals = [
            float(m.get("feedback_density", 0.0))
            for m in course_metrics.values()
        ]
        ext_vals = [
            float(m.get("extension_flag", 0.0))
            for m in course_metrics.values()
        ]
        ref_vals = [
            float(m.get("reflection_flag", 0.0))
            for m in course_metrics.values()
        ]

        return {
            "srl_help_index_mean": mean(srl_vals),
            "help_need_rate_mean": mean(need_vals),
            "early_help_ratio_mean": mean(early_vals),
            "no_help_success_ratio_mean": mean(nohelp_vals),
            "feedback_density_mean": mean(fb_vals),
            "extension_flag_mean": mean(ext_vals),
            "reflection_flag_mean": mean(ref_vals),
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

    engine = SRLHelpseekingEngine()

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

    dim_key = SRLHelpseekingEngine.DIMENSION_KEY

    for uid in test_learners:
        dim_data = numeric_result.get(uid, {}).get(dim_key)
        print(f"\n>>> 学习者 {uid}")
        if not dim_data or dim_data.get("insufficient_data"):
            print("  - 数据不足，无法进行自我调节与求助策略分析。")
            continue

        level_info = dim_data.get("level") or {}
        level_code = level_info.get("final_code")
        level_label = get_label(dim_key, "level", level_code)

        print(f"  - 自我调节与求助策略水平（level_code={level_code}）: {level_label}")
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
                f"srl={m.get('srl_help_index', 0.0):.3f}, "
                f"need={m.get('help_need_rate', 0.0):.3f}, "
                f"early={m.get('early_help_ratio', 0.0):.3f}, "
                f"no_help_succ={m.get('no_help_success_ratio', 0.0):.3f}, "
                f"fb={m.get('feedback_density', 0.0):.3f}, "
                f"ext={m.get('extension_flag', 0)}, "
                f"refl={m.get('reflection_flag', 0)}"
            )
