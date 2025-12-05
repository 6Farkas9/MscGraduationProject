# collaborative_role_contribution_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict, Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.repositories.collaborative_role_contribution_repository import (
    CollaborativeRoleContributionRepository,
)

logger = logging.getLogger(__name__)


class CollaborativeRoleContributionEngine:
    """
    协作角色与贡献类型（Collaborative Role & Contribution Type）画像维度的分析引擎。

    - Repository 只负责准备数值特征（课程级协作份额 + 贡献构成）；
    - Engine 在课程内部进行聚类，得到“协作角色”标签（role_code）；
    - 同时根据贡献构成计算“贡献类型”标签（contribution_type_code）；
    - 引擎只返回数值型代码，具体文案由 app.models.profiles_labels 映射。
    """

    DIMENSION_KEY = "collaborative_role_contribution"

    def __init__(
        self,
        repository: Optional[CollaborativeRoleContributionRepository] = None,
        role_cluster_k: int = 5,
        min_learners_per_course: Optional[int] = None,
    ):
        """
        Args:
            repository: 数据准备仓库实例，默认自动构造。
            role_cluster_k: 课程内协作角色聚类簇数（分类数，最多 5）。
            min_learners_per_course: 单门课程参与学习者数量下限，
                若 None 则默认 = role_cluster_k。
        """
        self.repository = repository or CollaborativeRoleContributionRepository()
        self.role_cluster_k = max(2, min(role_cluster_k, 5))  # role code 1~5
        self.min_learners_per_course = (
            min_learners_per_course or self.role_cluster_k
        )

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        对若干学习者进行协作角色与贡献类型分析。

        返回结构：
        {
          learner_uid: {
            "collaborative_role_contribution": {
              "insufficient_data": bool,
              "insufficient_reason": Optional[str],
              "role": { ... },
              "contribution_type": { ... },
            }
          },
          ...
        }
        """
        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        logger.info(
            "CollaborativeRoleContributionEngine.analyze: 开始分析，学习者数量: %d",
            len(learner_uids),
        )

        if not learner_uids:
            return {}

        # 1) 仓库层：准备课程级基础数据
        (
            course_metrics_by_lc,
            learners_per_course,
            learner_courses_map,
        ) = self.repository.load_metrics_for_learners(learner_uids)

        logger.info(
            "CollaborativeRoleContributionEngine.analyze: Repository 返回 (lrn, crs) 条目数: %d，课程数: %d",
            len(course_metrics_by_lc),
            len(learners_per_course),
        )

        # 2) 课程内聚类 + 课程级 contribution_type 计算
        per_lc_result = self._analyze_per_course(
            course_metrics_by_lc, learners_per_course
        )

        logger.info(
            "CollaborativeRoleContributionEngine.analyze: 课程内聚类与贡献类型计算完成，(lrn, crs) 有效条目数: %d",
            len(per_lc_result),
        )

        # 3) 按学习者聚合为最终结构
        final_results: Dict[str, Any] = {}

        for lrn_uid in learner_uids:
            dim_result = self._build_dimension_result_for_learner(
                learner_uid=lrn_uid,
                learner_courses=learner_courses_map.get(lrn_uid, set()),
                per_lc_result=per_lc_result,
            )
            final_results[lrn_uid] = {self.DIMENSION_KEY: dim_result}

        logger.info(
            "CollaborativeRoleContributionEngine.analyze: 分析完成，返回学习者数: %d",
            len(final_results),
        )

        return final_results

    # ------------------------------------------------------------------
    # 课程内聚类 + 课程级标签
    # ------------------------------------------------------------------
    def _analyze_per_course(
        self,
        course_metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]],
        learners_per_course: Dict[str, int],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在课程内部进行聚类得到协作角色标签，同时基于贡献构成判定贡献类型。
        返回 (learner, course) 级别的中间结果。
        """
        course_entries: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
        for (lrn_uid, crs_uid), metrics in course_metrics_by_lc.items():
            course_entries[crs_uid].append((lrn_uid, metrics))

        per_lc_result: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_entries.items():
            num_learners_in_course = learners_per_course.get(
                crs_uid, len(entries)
            )

            if num_learners_in_course < self.min_learners_per_course:
                logger.info(
                    "CollaborativeRoleContributionEngine: 课程 %s 学习者数 %d < 聚类数 %d，跳过课程级聚类。",
                    crs_uid,
                    num_learners_in_course,
                    self.role_cluster_k,
                )
                continue

            logger.info(
                "CollaborativeRoleContributionEngine: 开始对课程 %s 做协作角色聚类，学习者数: %d",
                crs_uid,
                num_learners_in_course,
            )

            contrib_shares = [
                metrics.get("avg_share_contribution", 0.0) for _, metrics in entries
            ]
            partic_shares = [
                metrics.get("avg_share_participation", 0.0) for _, metrics in entries
            ]
            trans_shares = [
                metrics.get("avg_share_transactivity", 0.0) for _, metrics in entries
            ]

            z_contrib = self._z_score(contrib_shares)
            z_partic = self._z_score(partic_shares)
            z_trans = self._z_score(trans_shares)

            r_index = []
            for zc, zp, zt in zip(z_contrib, z_partic, z_trans):
                r = (zc + zp + zt) / math.sqrt(3.0)
                r_index.append(r)

            r_norm = self._min_max_norm(r_index)

            # 课程内部一维 KMeans 聚类
            k = min(self.role_cluster_k, num_learners_in_course)
            cluster_ids = self._kmeans_1d(r_norm, k=k, max_iter=50)

            # cluster 中心 → role_code（0: 无协作数据, 1~5 逐级增强）
            cluster_centers = self._compute_cluster_centers(cluster_ids, r_norm, k)
            ordered_clusters = sorted(
                range(k), key=lambda cid: cluster_centers.get(cid, 0.0)
            )
            role_scale = [1, 2, 3, 4, 5]
            cluster_to_role_code: Dict[int, int] = {}
            for rank, cid in enumerate(ordered_clusters):
                role_code = role_scale[min(rank, len(role_scale) - 1)]
                cluster_to_role_code[cid] = role_code

            # 为课程内每个学习者生成 (lrn, crs) 的课程级结果
            for idx, (lrn_uid, metrics) in enumerate(entries):
                cid = cluster_ids[idx]
                role_code = cluster_to_role_code.get(cid, 0)

                contrib_code, contrib_metrics = self._compute_contribution_type_for_course(
                    metrics
                )

                per_lc_result[(lrn_uid, crs_uid)] = {
                    "course_uid": crs_uid,
                    "learner_uid": lrn_uid,
                    "role_code": role_code,
                    "contribution_type_code": contrib_code,
                    "r_index_norm": r_norm[idx],
                    "avg_share_contribution": contrib_shares[idx],
                    "avg_share_participation": partic_shares[idx],
                    "avg_share_transactivity": trans_shares[idx],
                    "contribution_metrics": contrib_metrics,
                    "sessions_count": metrics.get("sessions_count", 0),
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
                "insufficient_reason": "该学习者在协作相关课程中的样本量不足，无法进行聚类分析。",
                "role": None,
                "contribution_type": None,
            }

        # 课程级 role / contribution_type 数据
        role_codes: Dict[str, int] = {}
        contrib_codes: Dict[str, int] = {}
        role_course_metrics: Dict[str, Dict[str, float]] = {}
        contrib_course_metrics: Dict[str, Dict[str, float]] = {}

        for (_, crs_uid) in course_keys:
            item = per_lc_result[(learner_uid, crs_uid)]

            # 角色
            role_codes[crs_uid] = int(item.get("role_code", 0))
            role_course_metrics[crs_uid] = {
                "r_index_norm": float(item.get("r_index_norm", 0.0)),
                "avg_share_contribution": float(
                    item.get("avg_share_contribution", 0.0)
                ),
                "avg_share_participation": float(
                    item.get("avg_share_participation", 0.0)
                ),
                "avg_share_transactivity": float(
                    item.get("avg_share_transactivity", 0.0)
                ),
                "sessions_count": int(item.get("sessions_count", 0)),
            }

            # 贡献类型
            contrib_codes[crs_uid] = int(item.get("contribution_type_code", 0))
            contrib_course_metrics[crs_uid] = {
                **item.get("contribution_metrics", {}),
                "sessions_count": int(item.get("sessions_count", 0)),
            }

        # 整体协作角色：出现次数最多；并列时 code 大的优先（与 profiles_labels 定义一致）
        final_role_code = self._choose_final_code_with_priority(
            role_codes.values(),
            priority_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        )

        # 整体贡献类型：出现次数最多；并列时“更活跃”的类型优先
        final_contrib_code = self._choose_final_code_with_priority(
            contrib_codes.values(),
            priority_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        )

        role_overall_metrics = self._aggregate_overall_role_metrics(
            role_course_metrics
        )
        contrib_overall_metrics = self._aggregate_overall_contribution_metrics(
            contrib_course_metrics
        )

        role_courses_dict = {
            crs_uid: {
                "code": role_codes[crs_uid],
                "metrics": role_course_metrics[crs_uid],
            }
            for crs_uid in role_codes
        }
        contrib_courses_dict = {
            crs_uid: {
                "code": contrib_codes[crs_uid],
                "metrics": contrib_course_metrics[crs_uid],
            }
            for crs_uid in contrib_codes
        }

        return {
            "insufficient_data": False,
            "insufficient_reason": None,
            "role": {
                "final_code": final_role_code,
                "overall_metrics": role_overall_metrics,
                "courses": role_courses_dict,
            },
            "contribution_type": {
                "final_code": final_contrib_code,
                "overall_metrics": contrib_overall_metrics,
                "courses": contrib_courses_dict,
            },
        }

    # ------------------------------------------------------------------
    # 工具：贡献类型计算 / kmeans / 聚合
    # ------------------------------------------------------------------
    def _compute_contribution_type_for_course(
        self, metrics: Dict[str, Any]
    ) -> Tuple[int, Dict[str, float]]:
        """
        根据课程级构成（create/modify/resource/discuss）判定贡献类型 code。

        对应 profiles_labels 中 collaborative_role_contribution.contribution_type：
            0: 无协作数据（一般用于整体无数据的情况）
            1: 无有效贡献
            2: 讨论参与型
            3: 资源提供型
            4: 修改完善型
            5: 内容创作型
        """
        create_cnt = float(metrics.get("create_count", 0.0))
        modify_cnt = float(metrics.get("modify_count", 0.0))
        resource_cnt = float(metrics.get("resource_count", 0.0))
        discuss_cnt = float(metrics.get("discuss_count", 0.0))

        total = create_cnt + modify_cnt + resource_cnt + discuss_cnt
        if total <= 0:
            # 有协作会话但没有具体贡献行为 → “无有效贡献”
            code = 1
            contrib_metrics = {
                "create_share": 0.0,
                "modify_share": 0.0,
                "resource_share": 0.0,
                "discuss_share": 0.0,
                "total_actions": 0.0,
            }
            return code, contrib_metrics

        create_share = create_cnt / total
        modify_share = modify_cnt / total
        resource_share = resource_cnt / total
        discuss_share = discuss_cnt / total

        # 贡献类型由占比最高的行为类型决定
        max_val = max(create_share, modify_share, resource_share, discuss_share)
        if max_val == create_share:
            code = 5  # 内容创作型
        elif max_val == modify_share:
            code = 4  # 修改完善型
        elif max_val == resource_share:
            code = 3  # 资源提供型
        else:
            code = 2  # 讨论参与型

        contrib_metrics = {
            "create_share": create_share,
            "modify_share": modify_share,
            "resource_share": resource_share,
            "discuss_share": discuss_share,
            "total_actions": total,
        }
        return code, contrib_metrics

    @staticmethod
    def _z_score(vals: List[float]) -> List[float]:
        if not vals:
            return []
        mean_v = sum(vals) / float(len(vals))
        var = sum((v - mean_v) ** 2 for v in vals) / float(len(vals))
        if var <= 1e-9:
            return [0.0 for _ in vals]
        std_v = math.sqrt(var)
        return [(v - mean_v) / std_v for v in vals]

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
        简单的一维 KMeans 实现，用于课程内协作强度指数聚类。
        返回：每个样本的 cluster id（0 ~ k-1），cluster id 越大代表中心越大。
        """
        n = len(values)
        if n == 0:
            return []
        if k <= 1:
            return [0] * n
        k = min(k, n)

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

        # 重排 cluster 编号，使得 id 越大中心越大
        order = sorted(range(k), key=lambda cid: centers[cid])
        cid_remap = {old: new for new, old in enumerate(order)}
        final_assignments = [cid_remap[cid] for cid in assignments]
        return final_assignments

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
        centers = {}
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

        # 次数并列时，优先级大的优先；再并列就选 code 更大的
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
        role_course_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        if not role_course_metrics:
            return {}

        r_vals = [m.get("r_index_norm", 0.0) for m in role_course_metrics.values()]
        sc_vals = [
            m.get("avg_share_contribution", 0.0)
            for m in role_course_metrics.values()
        ]
        sp_vals = [
            m.get("avg_share_participation", 0.0)
            for m in role_course_metrics.values()
        ]
        st_vals = [
            m.get("avg_share_transactivity", 0.0)
            for m in role_course_metrics.values()
        ]

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        return {
            "r_index_norm_mean": mean(r_vals),
            "share_contribution_mean": mean(sc_vals),
            "share_participation_mean": mean(sp_vals),
            "share_transactivity_mean": mean(st_vals),
            "courses_count": len(role_course_metrics),
        }

    @staticmethod
    def _aggregate_overall_contribution_metrics(
        contrib_course_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        if not contrib_course_metrics:
            return {}

        cs = [
            m.get("create_share", 0.0)
            for m in contrib_course_metrics.values()
        ]
        ms = [
            m.get("modify_share", 0.0)
            for m in contrib_course_metrics.values()
        ]
        rs = [
            m.get("resource_share", 0.0)
            for m in contrib_course_metrics.values()
        ]
        ds = [
            m.get("discuss_share", 0.0)
            for m in contrib_course_metrics.values()
        ]

        def mean(arr: List[float]) -> float:
            return sum(arr) / float(len(arr)) if arr else 0.0

        return {
            "create_share_mean": mean(cs),
            "modify_share_mean": mean(ms),
            "resource_share_mean": mean(rs),
            "discuss_share_mean": mean(ds),
            "courses_count": len(contrib_course_metrics),
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

    engine = CollaborativeRoleContributionEngine()

    test_learners = [
        "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
        "lrn_004a9c3f5bf246faab3d390ce716e658",
    ]

    print("=" * 80)
    print("1) 数值型原始结果（结构示意，仅打印顶层结构）：")
    numeric_result = engine.analyze(test_learners)
    pprint.pprint(
        {uid: list(numeric_result.get(uid, {}).keys()) for uid in test_learners},
        width=120,
    )

    print("\n" + "=" * 80)
    print("2) 带文本标签的整体结果（逐个学习者打印）：\n")

    dim_key = CollaborativeRoleContributionEngine.DIMENSION_KEY

    for uid in test_learners:
        dim_data = numeric_result.get(uid, {}).get(dim_key)
        print(f"\n>>> 学习者 {uid}")
        if not dim_data or dim_data.get("insufficient_data"):
            print("  - 数据不足，无法进行协作角色与贡献类型分析。")
            continue

        role_info = dim_data.get("role") or {}
        contrib_info = dim_data.get("contribution_type") or {}

        role_code = role_info.get("final_code")
        contrib_code = contrib_info.get("final_code")

        role_label = get_label(dim_key, "role", role_code)
        contrib_label = get_label(dim_key, "contribution_type", contrib_code)

        print(f"  - 协作角色（role_code={role_code}）: {role_label}")
        print(f"  - 贡献类型（contribution_type_code={contrib_code}）: {contrib_label}")

        print("  - 角色整体指标:")
        pprint.pprint(role_info.get("overall_metrics"), indent=4, width=120)

        print("  - 贡献类型整体指标:")
        pprint.pprint(contrib_info.get("overall_metrics"), indent=4, width=120)

        # 顺便展示一下每门课程的标签（确认课程内聚类 & 贡献类型是否合理）
        print("  - 课程级标签与指标（部分字段）：")
        for crs_uid, c in (contrib_info.get("courses") or {}).items():
            r_code = (role_info.get("courses") or {}).get(crs_uid, {}).get("code")
            r_label = get_label(dim_key, "role", r_code)
            ct_code = c.get("code")
            ct_label = get_label(dim_key, "contribution_type", ct_code)
            print(f"    · 课程 {crs_uid}: role={r_code}({r_label}), contribution_type={ct_code}({ct_label})")
