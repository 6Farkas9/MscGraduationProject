# BackEnd/app/engine/contribution_reputation_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from math import sqrt
from collections import defaultdict

from app.repositories.contribution_reputation_repository import (
    contribution_reputation_repository,
)

logger = logging.getLogger(__name__)

# 与分析脚本保持一致的扩展字段
EXT_BASE = "https://legend-meta.com/xapi/ext/"
EXT_VALUE_CHANGE = EXT_BASE + "value-change"       # 单次价值变动（正：获得，负：支出）


class ContributionReputationEngine:
    """
    元宇宙价值贡献与声望分析引擎

    功能：
    - 给定一个或多个学习者 UID，从 Repository 读取细粒度 xAPI 行为（价值交换 + 贡献行为）；
    - 以 (学习者, 课程) 为单位，计算：
        * token_gain / token_cost / token_net
        * 各类贡献行为次数（资源贡献 / 协作编辑 / 协同活动）
        * 课程内标准化后的综合指数 C（原始）与 C_norm（归一化）
        * 离散价值贡献标签 value_label + cluster_rank（0/1/2 = 低/中/高）
    - 对单个学习者的多门课程结果做聚合：
        * 数值：多课程 C_norm 的均值 overall_score；
        * 分类：多课程 cluster_rank 的众数，若并列则选择“更好”的那一档（rank 较大）。
    """

    def __init__(self):
        logger.info("ContributionReputationEngine 初始化完成")

    # ------------------------------------------------------------------
    # 工具函数：均值 / 标准差 / 一维 k-means
    # ------------------------------------------------------------------

    @staticmethod
    def compute_mean_std(values: List[float]) -> Tuple[float, float]:
        """
        计算一组数的均值和总体标准差：
        - 列表为空 -> (0.0, 0.0)
        - 仅一个元素 -> 标准差视为 0.0
        """
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        mean_v = sum(values) / float(n)
        if n == 1:
            return mean_v, 0.0
        var = sum((v - mean_v) ** 2 for v in values) / float(n)
        return mean_v, sqrt(var)

    @staticmethod
    def kmeans_1d(
        values: List[float], k: int = 3, max_iter: int = 50
    ) -> Tuple[List[float], List[int]]:
        """
        一维 k-means 聚类（Lloyd 算法），用于基于 C_norm 自动划分三档价值贡献类型。

        返回：
        - centers: 聚类中心列表
        - assignments: 与 values 一一对应的簇编号（0 ~ k-1）
        """
        import random

        n = len(values)
        if n == 0 or k <= 0:
            return [], []
        if n <= k:
            centers = values[:]
            while len(centers) < k:
                centers.append(values[-1])
            assignments = list(range(n))
            return centers, assignments

        centers = random.sample(values, k)
        assignments = [0] * n

        for _ in range(max_iter):
            changed = False
            for i, v in enumerate(values):
                dists = [abs(v - c) for c in centers]
                min_idx = dists.index(min(dists))
                if assignments[i] != min_idx:
                    assignments[i] = min_idx
                    changed = True
            if not changed:
                break

            for cluster_idx in range(k):
                cluster_vals = [v for v, a in zip(values, assignments) if a == cluster_idx]
                if cluster_vals:
                    centers[cluster_idx] = sum(cluster_vals) / float(len(cluster_vals))

        return centers, assignments

    # ------------------------------------------------------------------
    # 核心内部步骤：按 (学习者, 课程) 聚合价值贡献统计
    # ------------------------------------------------------------------

    def _aggregate_value_contribution(
        self, events: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        根据传入的 xAPI 事件列表，按 (学习者, 课程) 聚合价值贡献相关统计。

        返回：
        value_stats[(lrn_uid, crs_uid)] = {
            "token_gain": float,
            "token_cost": float,
            "token_net": float,
            "value_events": int,
            "resource_contrib_count": int,
            "coedit_count": int,
            "collab_count": int,
        }
        """
        from app.repositories.contribution_reputation_repository import (
            ContributionReputationRepository,
        )

        # 直接复用 Repository 中的 VERBS 定义，确保一致
        verb_dict = ContributionReputationRepository.VERBS

        value_stats: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "token_gain": 0.0,
                "token_cost": 0.0,
                "token_net": 0.0,
                "value_events": 0,
                "resource_contrib_count": 0,
                "coedit_count": 0,
                "collab_count": 0,
            }
        )

        used_events = 0

        for doc in events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            verb = (doc.get("verb") or {}).get("id")
            if not verb:
                continue

            key = (lrn_uid, crs_uid)
            stats = value_stats[key]

            if verb == verb_dict["exchanged_value"]:
                ctx = doc.get("context") or {}
                exts = ctx.get("extensions") or {}
                delta = exts.get(EXT_VALUE_CHANGE)

                try:
                    delta_val = float(delta)
                except (TypeError, ValueError):
                    continue

                if delta_val > 0:
                    stats["token_gain"] += delta_val
                elif delta_val < 0:
                    stats["token_cost"] += abs(delta_val)

                stats["token_net"] += delta_val
                stats["value_events"] += 1
                used_events += 1

            elif verb == verb_dict["contributed_resource"]:
                stats["resource_contrib_count"] += 1
                used_events += 1

            elif verb == verb_dict["co_edited_artifact"]:
                stats["coedit_count"] += 1
                used_events += 1

            elif verb == verb_dict["collaborated_on_activity"]:
                stats["collab_count"] += 1
                used_events += 1

        logger.info(
            f"[ContributionReputationEngine] 有效价值/贡献事件数: {used_events}, "
            f"(学习者, 课程) 对数量: {len(value_stats)}"
        )

        filtered_stats = {
            k: v
            for k, v in value_stats.items()
            if (
                v["value_events"] > 0
                or v["resource_contrib_count"] > 0
                or v["coedit_count"] > 0
                or v["collab_count"] > 0
            )
        }

        return filtered_stats

    # ------------------------------------------------------------------
    # 指数计算：课程内标准化 + 全局归一化
    # ------------------------------------------------------------------

    def _compute_value_contribution_index(
        self, value_stats: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在已经聚合的 value_stats 基础上，计算课程内标准化后的价值贡献指数。

        输出字段（追加到每个 stats 中）：
        - contrib_total
        - z_token_gain
        - z_contrib
        - C            # 原始指数
        - C_norm       # 全局 min-max 归一化后的指数
        """
        course_to_token: Dict[str, List[float]] = defaultdict(list)
        course_to_contrib: Dict[str, List[float]] = defaultdict(list)

        # 先构建 contrib_total，并收集每门课的 token_gain & contrib_total
        for (lrn_uid, crs_uid), stats in value_stats.items():
            contrib_total = (
                stats["resource_contrib_count"]
                + stats["coedit_count"]
                + stats["collab_count"]
            )
            stats["contrib_total"] = contrib_total

            token_gain = float(stats["token_gain"])

            if token_gain <= 0.0 and contrib_total <= 0:
                continue

            course_to_token[crs_uid].append(token_gain)
            course_to_contrib[crs_uid].append(float(contrib_total))

        # 课程内 mean / std
        course_token_stats: Dict[str, Tuple[float, float]] = {}
        course_contrib_stats: Dict[str, Tuple[float, float]] = {}

        for crs_uid, vals in course_to_token.items():
            course_token_stats[crs_uid] = self.compute_mean_std(vals)

        for crs_uid, vals in course_to_contrib.items():
            course_contrib_stats[crs_uid] = self.compute_mean_std(vals)

        C_values: List[float] = []

        for (lrn_uid, crs_uid), stats in value_stats.items():
            token_gain = float(stats["token_gain"])
            contrib_total = float(stats.get("contrib_total", 0.0))

            if token_gain <= 0.0 and contrib_total <= 0.0:
                continue

            mean_token, std_token = course_token_stats.get(crs_uid, (0.0, 0.0))
            if std_token > 1e-6:
                z_token = (token_gain - mean_token) / std_token
            else:
                z_token = 0.0

            mean_contrib, std_contrib = course_contrib_stats.get(crs_uid, (0.0, 0.0))
            if std_contrib > 1e-6:
                z_contrib = (contrib_total - mean_contrib) / std_contrib
            else:
                z_contrib = 0.0

            stats["z_token_gain"] = z_token
            stats["z_contrib"] = z_contrib

            C = (z_token + z_contrib) / sqrt(2.0)
            stats["C"] = C
            C_values.append(C)

        if not C_values:
            logger.warning("[ContributionReputationEngine] 无可用样本计算价值贡献指数")
            return {}

        min_C = min(C_values)
        max_C = max(C_values)
        span = max_C - min_C

        if span < 1e-6:
            for stats in value_stats.values():
                if "C" in stats:
                    stats["C_norm"] = 0.5
            logger.info(
                "[ContributionReputationEngine] 所有 C 完全相同，统一设置 C_norm=0.5"
            )
            return value_stats

        for stats in value_stats.values():
            if "C" not in stats:
                continue
            C = float(stats["C"])
            stats["C_norm"] = (C - min_C) / span

        logger.info("[ContributionReputationEngine] 已完成价值贡献指数计算")
        return value_stats

    # ------------------------------------------------------------------
    # 聚类 + 标签
    # ------------------------------------------------------------------

    def _assign_value_labels(
        self, value_results: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> None:
        """
        基于 C_norm 对 (学习者, 课程) 进行聚类并赋予语义标签：
        - cluster_index: k-means 的原始簇编号
        - cluster_rank: 按中心高低排序后的等级（0: 低, 1: 中, 2: 高）
        - value_label: 文字标签
        """
        C_norm_list: List[float] = []
        keys_list: List[Tuple[str, str]] = []

        for key, stats in value_results.items():
            C_norm = stats.get("C_norm")
            if C_norm is None:
                continue
            C_norm_list.append(float(C_norm))
            keys_list.append(key)

        if not C_norm_list:
            logger.warning("[ContributionReputationEngine] 无 C_norm 可用于聚类")
            return

        centers, assignments = self.kmeans_1d(C_norm_list, k=3, max_iter=50)
        if not centers or not assignments:
            logger.warning("[ContributionReputationEngine] k-means 聚类失败，跳过标签生成")
            return

        # 中心从小到大排序 -> rank 0/1/2
        center_with_idx = list(enumerate(centers))
        center_with_idx.sort(key=lambda x: x[1])
        cluster_to_rank = {cluster_idx: rank for rank, (cluster_idx, _) in enumerate(center_with_idx)}

        base_rank_to_label = {
            0: "低价值贡献型学习者（在本课程中几乎没有价值 token 流入，贡献行为也较少）",
            1: "中等价值贡献型学习者（在本课程中具有一定价值 token 流入与贡献行为）",
            2: "高价值贡献 & 高声望型学习者（在本课程中获得较多价值 token 奖励并频繁贡献）",
        }

        for key, cluster_idx in zip(keys_list, assignments):
            stats = value_results[key]
            rank = cluster_to_rank.get(cluster_idx, 1)
            base_label = base_rank_to_label.get(rank, "中等价值贡献型学习者（默认）")

            resource_count = stats.get("resource_contrib_count", 0)
            coedit_count = stats.get("coedit_count", 0)
            collab_count = stats.get("collab_count", 0)
            contrib_total = stats.get("contrib_total", 0)

            extra_desc = ""
            if contrib_total > 0:
                collab_like = coedit_count + collab_count
                collab_ratio = collab_like / float(contrib_total)
                resource_ratio = resource_count / float(contrib_total)

                if collab_ratio >= 0.6 and collab_like >= 2:
                    extra_desc = "，且在贡献行为中以协作与共同编辑为主（偏协作驱动型贡献者）"
                elif resource_ratio >= 0.6 and resource_count >= 2:
                    extra_desc = "，且在贡献行为中以上传 / 分享资源为主（偏资源驱动型贡献者）"
                else:
                    extra_desc = "，贡献行为在协作与资源分享之间较为均衡"

            stats["cluster_index"] = int(cluster_idx)
            stats["cluster_rank"] = int(rank)
            stats["value_label"] = base_label + extra_desc

        # 控制台简单统计
        label_counts: Dict[str, int] = defaultdict(int)
        for stats in value_results.values():
            label = stats.get("value_label")
            if label:
                label_counts[label] += 1

        for label, cnt in label_counts.items():
            logger.info(f"[ContributionReputationEngine] 标签分布: {label} -> {cnt}")

    # ------------------------------------------------------------------
    # 聚合到学习者级别
    # ------------------------------------------------------------------

    def _build_learner_summaries(
        self, value_results: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (lrn, crs) 级别结果聚合为按学习者的结果：

        返回结构：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,          # 多课程 C_norm 均值
                "overall_label": str 或 None,            # 综合价值贡献标签（按 cluster_rank 众数）
                "overall_cluster_rank": int 或 None,     # 0/1/2
                "per_course_results": [...],             # 每门课程详情
            },
            ...
        }
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), stats in value_results.items():
            if "C_norm" not in stats:
                continue

            if lrn_uid not in learner_data:
                learner_data[lrn_uid] = {
                    "learner_uid": lrn_uid,
                    "has_data": True,
                    "overall_score": None,
                    "overall_label": None,
                    "overall_cluster_rank": None,
                    "per_course_results": [],
                }

            per_course_item = {
                "course_uid": crs_uid,
                "token_gain": float(stats.get("token_gain", 0.0)),
                "token_cost": float(stats.get("token_cost", 0.0)),
                "token_net": float(stats.get("token_net", 0.0)),
                "value_events": int(stats.get("value_events", 0)),
                "resource_contrib_count": int(stats.get("resource_contrib_count", 0)),
                "coedit_count": int(stats.get("coedit_count", 0)),
                "collab_count": int(stats.get("collab_count", 0)),
                "contrib_total": int(stats.get("contrib_total", 0)),
                "z_token_gain": float(stats.get("z_token_gain", 0.0)),
                "z_contrib": float(stats.get("z_contrib", 0.0)),
                "value_index": float(stats.get("C", 0.0)),
                "value_normalized": float(stats.get("C_norm", 0.0)),
                "value_label": stats.get("value_label"),
                "cluster_rank": int(stats.get("cluster_rank", 1)),
            }

            learner_data[lrn_uid]["per_course_results"].append(per_course_item)

        # 聚合 overall_score + overall_label
        base_rank_to_overall_label = {
            0: "整体价值贡献偏低（在所参与课程中价值 token 流入和贡献行为整体偏少）",
            1: "整体价值贡献中等（在所参与课程中具有一定价值 token 流入和贡献行为）",
            2: "整体价值贡献与声望较高（在所参与课程中经常获得价值 token 奖励并有较多贡献）",
        }

        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            scores = [it["value_normalized"] for it in pcs]
            info["overall_score"] = sum(scores) / float(len(scores))

            rank_counts: Dict[int, int] = {}
            for it in pcs:
                r = int(it["cluster_rank"])
                rank_counts[r] = rank_counts.get(r, 0) + 1

            if rank_counts:
                max_count = max(rank_counts.values())
                candidate_ranks = [r for r, c in rank_counts.items() if c == max_count]
                best_rank = max(candidate_ranks)  # 并列时选“更好”的一档（数值更大）

                info["overall_cluster_rank"] = best_rank
                info["overall_label"] = base_rank_to_overall_label.get(
                    best_rank,
                    "整体价值贡献中等（默认）",
                )
            else:
                info["overall_cluster_rank"] = None
                info["overall_label"] = None

        return learner_data

    # ------------------------------------------------------------------
    # 对外公开接口：多学习者 / 单学习者
    # ------------------------------------------------------------------

    def analyze_multiple_learners(
        self, learner_uids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        对多个学习者进行“价值贡献与声望”分析。

        返回：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,
                "overall_label": str 或 None,
                "overall_cluster_rank": int 或 None,
                "per_course_results": [...],
                # 如出错，还会包含 "error": str
            },
            ...
        }
        """
        if not learner_uids:
            return {}

        try:
            # 1) 从 Repository 获取原始事件
            events = contribution_reputation_repository.get_value_and_contribution_events(
                learner_uids
            )

            if not events:
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            # 2) 按 (学习者, 课程) 聚合统计
            value_stats = self._aggregate_value_contribution(events)
            if not value_stats:
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            # 3) 计算价值贡献指数
            value_results = self._compute_value_contribution_index(value_stats)
            if not value_results:
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            # 4) 聚类 + 标签
            self._assign_value_labels(value_results)

            # 5) 聚合为学习者级别结果
            learner_summaries = self._build_learner_summaries(value_results)

            # 6) 对于传入但没有任何结果的学习者，也返回结构化空结果
            for uid in learner_uids:
                if uid not in learner_summaries:
                    learner_summaries[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }

            return learner_summaries

        except Exception as e:
            logger.error(f"多学习者价值贡献分析失败: {e}", exc_info=True)
            result: Dict[str, Dict[str, Any]] = {}
            for uid in learner_uids:
                result[uid] = {
                    "learner_uid": uid,
                    "has_data": False,
                    "overall_score": None,
                    "overall_label": None,
                    "overall_cluster_rank": None,
                    "per_course_results": [],
                    "error": str(e),
                }
            return result

    def analyze_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """
        单学习者便捷接口：返回结构等同于 analyze_multiple_learners()[learner_uid]
        """
        results = self.analyze_multiple_learners([learner_uid])
        return results.get(
            learner_uid,
            {
                "learner_uid": learner_uid,
                "has_data": False,
                "overall_score": None,
                "overall_label": None,
                "overall_cluster_rank": None,
                "per_course_results": [],
            },
        )


# 全局引擎实例 + 便捷函数（与 attention_allocation_engine 风格一致）
_contribution_engine_instance: Optional[ContributionReputationEngine] = None


def get_contribution_reputation_engine() -> ContributionReputationEngine:
    global _contribution_engine_instance
    if _contribution_engine_instance is None:
        _contribution_engine_instance = ContributionReputationEngine()
    return _contribution_engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_contribution_reputation_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(
    learner_uids: List[str],
) -> Dict[str, Dict[str, Any]]:
    engine = get_contribution_reputation_engine()
    return engine.analyze_multiple_learners(learner_uids)


# 简单本地测试（使用与 attention_allocation_engine 相同的测试 UID）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = ContributionReputationEngine()
    test_learner_uids = [
        "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
        "lrn_004a9c3f5bf246faab3d390ce716e658",
    ]

    print("=== 单学习者测试 ===")
    res_single = engine.analyze_single_learner(test_learner_uids[0])
    print(res_single)

    print("=== 多学习者测试 ===")
    res_multi = engine.analyze_multiple_learners(test_learner_uids)
    print(res_multi)
