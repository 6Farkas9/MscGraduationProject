# BackEnd/app/engine/task_efficiency_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from math import sqrt
import re
from collections import defaultdict

from app.repositories.task_efficiency_repository import task_efficiency_repository

logger = logging.getLogger(__name__)

# 预编译 duration 的正则，避免每次 re.compile
DURATION_RE = re.compile(r"^PT(\d+)S$")


class TaskEfficiencyEngine:
    """
    任务效率（Task Efficiency）分析引擎

    功能：
    - 给定一个或多个学习者 UID，从 Repository 读取细粒度 xAPI 行为；
    - 以 (学习者, 课程) 为单位，计算：
        * P_mean: 任务成功率（基于 success/completion）
        * T_mean: 平均任务耗时（秒）
        * z_P / z_T: 课程内 z 标准化后的表现与耗时
        * E: 认知效率指数 (z_P - z_T) / sqrt(2)
        * E_norm: 在课程内做 min-max 归一化后的效率指数 ∈ [0,1]
        * efficiency_label: 基于 k-means 的三档标签（低/中/高效率）
        * cluster_rank: 0=低效率, 1=中等效率, 2=高效率
    - 对外暴露两种接口：
        1）analyze_single_learner(learner_uid)
        2）analyze_multiple_learners(learner_uids)
    - 同时对多门课程结果做学习者层面的聚合：
        * 数值结果 overall_score：各课程 task_efficiency_normalized 的均值；
        * 分类结果 overall_efficiency_label：各课程 cluster_rank 的众数，
          若存在并列则选择语义上“更好”的（rank 较大的那个）。
    """

    def __init__(self):
        logger.info("TaskEfficiencyEngine 初始化完成")

    # ------------------------------------------------------------------
    # 通用工具函数
    # ------------------------------------------------------------------

    @staticmethod
    def parse_iso8601_duration(duration_str: Optional[str]) -> Optional[int]:
        """
        解析简单形式的 ISO8601 时长字符串，例如："PT120S"
        若为空或格式不符，返回 None。

        与分析脚本保持一致，只支持“PT{秒数}S”的整数秒格式。
        """
        if not duration_str:
            return None
        m = DURATION_RE.match(duration_str)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def compute_mean_std(values: List[float]) -> Tuple[float, float]:
        """
        计算一组数的均值和标准差（总体标准差）：
        - 列表为空 -> (0, 0)
        - 仅一个元素 -> 标准差视为 0
        """
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        mean_v = sum(values) / float(n)
        if n == 1:
            return mean_v, 0.0
        var = sum((v - mean_v) ** 2 for v in values) / float(n)
        std = sqrt(var)
        return mean_v, std

    @staticmethod
    def kmeans_1d(values: List[float], k: int = 3, max_iter: int = 50) -> Tuple[List[float], List[int]]:
        """
        一维 k-means 聚类（Lloyd 算法实现），用于基于效率指数 E_norm 自动划分学习者类型。
        """
        n = len(values)
        if n == 0 or k <= 0:
            return [], []

        if n < k:
            k = n

        v_min, v_max = min(values), max(values)
        if abs(v_max - v_min) < 1e-6:
            centers = [v_min for _ in range(k)]
            assignments = [0 for _ in range(n)]
            return centers, assignments

        centers = [
            v_min + (v_max - v_min) * (i + 0.5) / float(k)
            for i in range(k)
        ]

        for _ in range(max_iter):
            clusters: List[List[int]] = [[] for _ in range(k)]
            for idx, v in enumerate(values):
                best_c = 0
                best_dist = abs(v - centers[0])
                for ci in range(1, k):
                    d = abs(v - centers[ci])
                    if d < best_dist:
                        best_dist = d
                        best_c = ci
                clusters[best_c].append(idx)

            new_centers: List[float] = centers[:]
            for ci in range(k):
                if clusters[ci]:
                    new_centers[ci] = sum(values[i] for i in clusters[ci]) / float(len(clusters[ci]))
                else:
                    new_centers[ci] = centers[ci]

            max_shift = max(abs(new_centers[ci] - centers[ci]) for ci in range(k))
            centers = new_centers
            if max_shift < 1e-4:
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

    # ------------------------------------------------------------------
    # 核心内部步骤：从原始事件构造 (lrn, crs) 级别的 P_mean / T_mean
    # ------------------------------------------------------------------

    def _aggregate_task_stats(
        self, task_events: List[Dict[str, Any]]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, float]],
        Dict[str, List[Tuple[str, float, float]]],
    ]:
        """
        聚合任务级事件，得到每个 (学习者, 课程) 的任务成功率 / 平均耗时。

        返回：
        learner_course_metrics[(lrn_uid, crs_uid)] = {
            "P_mean": ...,
            "T_mean": ...,
        }

        course_to_entries[course_uid] = [
            (learner_uid, P_mean, T_mean),
            ...
        ]
        """
        from typing import Tuple as _Tuple  # 避免与返回类型别名冲突，仅在实现中使用

        learner_course_stats: Dict[_Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"sum_P": 0.0, "sum_T": 0.0, "count": 0}
        )

        used_events = 0

        for doc in task_events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            result = doc.get("result") or {}
            duration_str = result.get("duration")
            duration_sec = self.parse_iso8601_duration(duration_str)
            if duration_sec is None or duration_sec <= 0:
                # 无法用于效率分析的事件（缺少时长或时长为 0）
                continue

            success = result.get("success")
            completion = result.get("completion")

            if success is None and completion is None:
                # 若既没有 success 也没有 completion，则无法判断表现，跳过
                continue
            elif success is None:
                # 只有 completion 的情况：completion=True 视为成功完成一次任务
                P_task = 1.0 if completion else 0.0
            else:
                # 有 success 的情况：True 为成功，False/None 为失败
                P_task = 1.0 if bool(success) else 0.0

            key: _Tuple[str, str] = (lrn_uid, crs_uid)
            stat = learner_course_stats[key]
            stat["sum_P"] += P_task
            stat["sum_T"] += float(duration_sec)
            stat["count"] += 1
            used_events += 1

        logger.info(
            f"[TaskEfficiencyEngine] 参与任务效率统计的有效事件数: {used_events}, "
            f"(learner, course) 组合数: {len(learner_course_stats)}"
        )

        learner_course_metrics: Dict[_Tuple[str, str], Dict[str, float]] = {}
        course_to_entries: Dict[str, List[_Tuple[str, float, float]]] = defaultdict(list)

        for (lrn_uid, crs_uid), stat in learner_course_stats.items():
            c = stat["count"]
            if c <= 0:
                continue
            P_mean = stat["sum_P"] / float(c)
            T_mean = stat["sum_T"] / float(c)
            learner_course_metrics[(lrn_uid, crs_uid)] = {
                "P_mean": P_mean,
                "T_mean": T_mean,
            }
            course_to_entries[crs_uid].append((lrn_uid, P_mean, T_mean))

        return learner_course_metrics, course_to_entries

    # ------------------------------------------------------------------
    # 任务效率指数计算 + 归一化 + 聚类标签
    # ------------------------------------------------------------------

    def _compute_efficiency_indices(
        self,
        learner_course_metrics: Dict[Tuple[str, str], Dict[str, float]],
        course_to_entries: Dict[str, List[Tuple[str, float, float]]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        基于每个 (学习者, 课程) 的 P_mean / T_mean 计算：
        - z_P / z_T
        - 认知效率指数 E
        - 课程内 min-max 归一化 E_norm
        - 全局 k-means 聚类标签
        """
        if not learner_course_metrics:
            return {}

        efficiency_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # 1) 课程内部标准化 + 认知效率 E
        for crs_uid, entries in course_to_entries.items():
            if not entries:
                continue

            P_vals = [e[1] for e in entries]
            T_vals = [e[2] for e in entries]
            mean_P, std_P = self.compute_mean_std(P_vals)
            mean_T, std_T = self.compute_mean_std(T_vals)

            E_vals: List[float] = []

            for (lrn_uid, P_mean, T_mean) in entries:
                z_P = (P_mean - mean_P) / std_P if std_P > 1e-6 else 0.0
                z_T = (T_mean - mean_T) / std_T if std_T > 1e-6 else 0.0
                E = (z_P - z_T) / sqrt(2.0)

                key = (lrn_uid, crs_uid)
                efficiency_results[key] = {
                    "P_mean": P_mean,
                    "T_mean": T_mean,
                    "z_P": z_P,
                    "z_T": z_T,
                    "E": E,
                }
                E_vals.append(E)

            # 2) 在当前课程内部对 E 做 [0,1] 的 min-max 归一化
            if E_vals:
                E_min = min(E_vals)
                E_max = max(E_vals)
                span = E_max - E_min if E_max > E_min else 0.0

                for (lrn_uid, P_mean, T_mean) in entries:
                    key = (lrn_uid, crs_uid)
                    E = efficiency_results[key]["E"]
                    if span > 1e-6:
                        E_norm = (E - E_min) / span
                    else:
                        E_norm = 0.5  # 无法区分效率高低时给中间值
                    efficiency_results[key]["E_norm"] = E_norm

        if not efficiency_results:
            return {}

        # 3) 基于所有 E_norm 的一维 k-means 聚类，得到低/中/高三档
        all_results_items = list(efficiency_results.items())
        all_E_norm = [res["E_norm"] for _, res in all_results_items]

        centers, assignments = self.kmeans_1d(all_E_norm, k=3, max_iter=50)
        if centers:
            sorted_idx = sorted(range(len(centers)), key=lambda i: centers[i])
            cluster_to_rank = {cluster_idx: rank for rank, cluster_idx in enumerate(sorted_idx)}

            rank_to_label = {
                0: "低效率型学习者（在本课程中任务成功率相对较低、耗时相对较长 / 认知效率指数较低）",
                1: "中等效率型学习者（在本课程中任务成功率与耗时均处于中间水平）",
                2: "高效率型学习者（在本课程中任务成功率相对较高、耗时相对较短 / 认知效率指数较高）",
            }

            for ((key, res), cluster_idx) in zip(all_results_items, assignments):
                rank = cluster_to_rank.get(cluster_idx, 1)
                label = rank_to_label.get(rank, "中等效率型学习者（默认）")
                res["cluster_index"] = int(cluster_idx)
                res["cluster_rank"] = int(rank)
                res["efficiency_label"] = label

        return efficiency_results

    # ------------------------------------------------------------------
    # 按学习者聚合结果
    # ------------------------------------------------------------------

    def _build_learner_summaries(
        self,
        efficiency_results: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (lrn, crs) 级别的结果聚合为按学习者的结果：

        返回结构：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,           # 多课程 E_norm 的均值
                "overall_efficiency_label": str 或 None, # 多课程分类的众数（并列时选更“好”的那档）
                "overall_cluster_rank": int 或 None,
                "per_course_results": [  # 逐课程结果
                    {
                        "course_uid": "...",
                        "task_success_rate": ...,
                        "task_average_duration": ...,
                        "z_performance": ...,
                        "z_time": ...,
                        "task_efficiency_index": ...,
                        "task_efficiency_normalized": ...,
                        "efficiency_label": ...,
                        "cluster_rank": ...,
                    },
                    ...
                ],
            },
            ...
        }
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), res in efficiency_results.items():
            if lrn_uid not in learner_data:
                learner_data[lrn_uid] = {
                    "learner_uid": lrn_uid,
                    "has_data": True,
                    "overall_score": None,
                    "overall_efficiency_label": None,
                    "overall_cluster_rank": None,
                    "per_course_results": [],
                }

            per_course_item = {
                "course_uid": crs_uid,
                "task_success_rate": res.get("P_mean"),
                "task_average_duration": res.get("T_mean"),
                "z_performance": res.get("z_P"),
                "z_time": res.get("z_T"),
                "task_efficiency_index": res.get("E"),
                "task_efficiency_normalized": res.get("E_norm"),
                "efficiency_label": res.get("efficiency_label"),
                "cluster_rank": res.get("cluster_rank"),
            }
            learner_data[lrn_uid]["per_course_results"].append(per_course_item)

        # 聚合 overall_score + overall_efficiency_label
        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            scores = [
                it["task_efficiency_normalized"]
                for it in pcs
                if it.get("task_efficiency_normalized") is not None
            ]
            if scores:
                info["overall_score"] = sum(scores) / float(len(scores))
            else:
                info["overall_score"] = None

            rank_counts: Dict[int, int] = {}
            for it in pcs:
                r = it.get("cluster_rank")
                if r is None:
                    continue
                r_int = int(r)
                rank_counts[r_int] = rank_counts.get(r_int, 0) + 1

            if rank_counts:
                max_count = max(rank_counts.values())
                candidate_ranks = [r for r, c in rank_counts.items() if c == max_count]
                best_rank = max(candidate_ranks)  # 并列时选更高一档（更“好”的结果）

                if best_rank == 0:
                    eff_label = (
                        "低效率型学习者（在整体上任务成功率相对较低、耗时相对较长 / 认知效率指数较低）"
                    )
                elif best_rank == 2:
                    eff_label = (
                        "高效率型学习者（在整体上任务成功率相对较高、耗时相对较短 / 认知效率指数较高）"
                    )
                else:
                    eff_label = (
                        "中等效率型学习者（在整体上任务成功率与耗时均处于中间水平）"
                    )

                info["overall_cluster_rank"] = best_rank
                info["overall_efficiency_label"] = eff_label
            else:
                info["overall_cluster_rank"] = None
                info["overall_efficiency_label"] = None

        return learner_data

    # ------------------------------------------------------------------
    # 对外公开接口：单个 / 多个学习者
    # ------------------------------------------------------------------

    def analyze_multiple_learners(
        self, learner_uids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        对多个学习者进行任务效率分析。

        返回：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,
                "overall_efficiency_label": str 或 None,
                "overall_cluster_rank": int 或 None,
                "per_course_results": [...],
            },
            ...
        }
        """
        if not learner_uids:
            return {}

        try:
            raw = task_efficiency_repository.get_task_efficiency_raw_data_for_learners(
                learner_uids
            )
            task_events = raw.get("task_events", [])

            if not task_events:
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_efficiency_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            learner_course_metrics, course_to_entries = self._aggregate_task_stats(task_events)
            if not learner_course_metrics:
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_efficiency_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            efficiency_results = self._compute_efficiency_indices(
                learner_course_metrics, course_to_entries
            )
            if not efficiency_results:
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_efficiency_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            learner_summaries = self._build_learner_summaries(efficiency_results)

            # 对于传入但没有任何数据的学习者，也返回空结果
            for uid in learner_uids:
                if uid not in learner_summaries:
                    learner_summaries[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_efficiency_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }

            return learner_summaries

        except Exception as e:
            logger.error(f"多学习者任务效率分析失败: {e}", exc_info=True)
            result: Dict[str, Dict[str, Any]] = {}
            for uid in learner_uids:
                result[uid] = {
                    "learner_uid": uid,
                    "has_data": False,
                    "overall_score": None,
                    "overall_efficiency_label": None,
                    "overall_cluster_rank": None,
                    "per_course_results": [],
                    "error": str(e),
                }
            return result

    def analyze_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """
        单学习者便捷接口：
        返回结构同 analyze_multiple_learners()[learner_uid]
        """
        results = self.analyze_multiple_learners([learner_uid])
        return results.get(
            learner_uid,
            {
                "learner_uid": learner_uid,
                "has_data": False,
                "overall_score": None,
                "overall_efficiency_label": None,
                "overall_cluster_rank": None,
                "per_course_results": [],
            },
        )


# 全局引擎实例 + 便捷函数，风格与 AttentionAllocationEngine 对齐
_task_efficiency_engine_instance: Optional[TaskEfficiencyEngine] = None


def get_task_efficiency_engine() -> TaskEfficiencyEngine:
    global _task_efficiency_engine_instance
    if _task_efficiency_engine_instance is None:
        _task_efficiency_engine_instance = TaskEfficiencyEngine()
    return _task_efficiency_engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_task_efficiency_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
    engine = get_task_efficiency_engine()
    return engine.analyze_multiple_learners(learner_uids)


# 简单本地测试（可选）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = TaskEfficiencyEngine()
    # 真实存在的学习者UID（沿用注意力分配引擎的测试 UID）
    test_learner_uids = [
        "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
        "lrn_004a9c3f5bf246faab3d390ce716e658",
    ]

    print("=== 任务效率：单学习者测试 ===")
    res_single = engine.analyze_single_learner(test_learner_uids[0])
    print(res_single)

    print("=== 任务效率：多学习者测试 ===")
    res_multi = engine.analyze_multiple_learners(test_learner_uids)
    print(res_multi)
