# BackEnd/app/engine/spatial_exploration_orientation_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from math import sqrt
from collections import defaultdict

from app.repositories.spatial_exploration_orientation_repository import (
    spatial_exploration_orientation_repository,
)

logger = logging.getLogger(__name__)

# 与分析脚本保持一致的扩展字段 URL
EXT_SPACE_ID = "https://legend-meta.com/xapi/ext/space-id"
EXT_NAV_MODE = "https://legend-meta.com/xapi/ext/navigation-mode"
EXT_FOCUS_TARGET_ID = "https://legend-meta.com/xapi/ext/focus-target-id"


class SpatialExplorationOrientationEngine:
    """
    空间与资源探索倾向分析引擎

    功能：
    - 给定一个或多个学习者 UID，从 Repository 读取细粒度 xAPI 行为；
    - 以 (学习者, 课程) 为单位，计算：
        * unique_spaces      ：访问到的不同 space-id 数量（空间探索广度）
        * unique_resources   ：聚焦过的不同资源类别数（资源探索广度）
        * has_extension      ：是否参与可选拓展单元（0/1）
        * path_jump          ：是否存在回访型路径（离开后又回到同一空间）
        * teleport_ratio     ：teleport 导航占全部导航的比例
        * z_*                ：课程内 z 标准化后的特征
        * exploration_index  ：探索指数 E
        * exploration_normalized：课程内 min-max 归一化后的指数 E_norm ∈ [0,1]
        * exploration_label  ：基于 k-means 的三档标签（低/中/高探索）
        * cluster_rank       ：0=低探索，1=中等探索，2=高探索
    - 对外暴露两种接口：
        1）analyze_single_learner(learner_uid)
        2）analyze_multiple_learners(learner_uids)
    - 同时对多门课程结果做学习者层面的聚合：
        * 数值结果 overall_score：各课程 exploration_normalized 的均值；
        * 分类结果 overall_exploration_label：各课程 cluster_rank 的众数，
          若存在并列则选择语义上“更好”的（rank 较大的那个）。
    """

    def __init__(self):
        logger.info("SpatialExplorationOrientationEngine 初始化完成")

    # ------------------------------------------------------------------
    # 工具函数：统计 & 聚类
    # ------------------------------------------------------------------

    @staticmethod
    def compute_mean_std(values: List[float]) -> Tuple[float, float]:
        """
        计算均值和总体标准差：
        - 空列表 => (0.0, 0.0)
        - 单元素 => (value, 0.0)
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
    def compute_path_jump_flag(space_seq: List[str]) -> int:
        """
        判断空间访问序列中是否存在“回访型路径”，并给出二值标记。

        简化规则：
        1）先做相邻去重，例如 [A, A, B, B, A] -> [A, B, A]；
        2）在去重后的序列中，若某 space-id 出现两次及以上，
           且中间至少经过了其他 space-id，则视为存在回访；
        3）有回访 -> 1；否则 0。
        """
        if not space_seq:
            return 0

        reduced: List[str] = []
        for sid in space_seq:
            if not reduced or reduced[-1] != sid:
                reduced.append(sid)

        seen = set()
        for sid in reduced:
            if sid in seen:
                return 1
            seen.add(sid)
        return 0

    @staticmethod
    def kmeans_1d(values: List[float], k: int = 3, max_iter: int = 50) -> Tuple[List[float], List[int]]:
        """
        一维 k-means 聚类（Lloyd 算法简化版）。
        返回：
        - centers: 每个簇的中心值
        - assignments: 与 values 对应的簇编号列表（0..k-1）
        """
        n = len(values)
        if n == 0:
            return [], []

        if n < k:
            k = n

        v_min, v_max = min(values), max(values)
        if abs(v_max - v_min) < 1e-6:
            centers = [v_min for _ in range(k)]
            assignments = [0 for _ in range(n)]
            return centers, assignments

        # 均匀初始化
        centers = [
            v_min + (v_max - v_min) * (i + 0.5) / float(k)
            for i in range(k)
        ]

        for _ in range(max_iter):
            clusters: List[List[int]] = [[] for _ in range(k)]
            # 1) 分配
            for idx, v in enumerate(values):
                best_c = 0
                best_dist = abs(v - centers[0])
                for ci in range(1, k):
                    d = abs(v - centers[ci])
                    if d < best_dist:
                        best_dist = d
                        best_c = ci
                clusters[best_c].append(idx)

            # 2) 更新
            new_centers: List[float] = []
            for ci in range(k):
                if not clusters[ci]:
                    new_centers.append(centers[ci])
                else:
                    mean_v = sum(values[idx] for idx in clusters[ci]) / float(len(clusters[ci]))
                    new_centers.append(mean_v)

            shift = sum(abs(a - b) for a, b in zip(centers, new_centers))
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

    # ------------------------------------------------------------------
    # 核心内部步骤：从原始事件构造 (lrn, course) 级别的基础特征
    # ------------------------------------------------------------------

    def _aggregate_raw_events(
        self,
        navigation_events: List[Dict[str, Any]],
        extension_events: List[Dict[str, Any]],
        focus_events: List[Dict[str, Any]],
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, Dict[str, List[float]]],
    ]:
        """
        按 (learner_uid, course_uid) 聚合原始 xAPI 事件，生成基础计数特征：

        返回：
        base_metrics[(lrn, crs)] = {
            "unique_spaces": int,
            "unique_resources": int,
            "has_extension": int(0/1),
            "path_jump": int(0/1),
            "teleport_ratio": float,
        }

        per_course_values[course_uid] = {
            "space_counts": [...],
            "resource_counts": [...],
            "extension_flags": [...],
            "path_flags": [...],
            "teleport_ratios": [...],
        }
        """
        per_lrn_course: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # 1) 导航事件：构造空间序列 & teleport 统计
        # 为了保证路径顺序，按 timestamp 排序
        try:
            navigation_events_sorted = sorted(
                navigation_events,
                key=lambda d: d.get("timestamp") or ""
            )
        except Exception:
            navigation_events_sorted = navigation_events

        for doc in navigation_events_sorted:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            key = (lrn_uid, crs_uid)
            if key not in per_lrn_course:
                per_lrn_course[key] = {
                    "nav_spaces": set(),
                    "nav_sequence": [],
                    "nav_walk_count": 0,
                    "nav_teleport_count": 0,
                    "extension_count": 0,
                    "focus_targets": set(),
                    "focus_count": 0,
                }

            rec = per_lrn_course[key]
            context = doc.get("context") or {}
            ext = context.get("extensions") or {}
            verb = (doc.get("verb") or {}).get("id")

            space_id = ext.get(EXT_SPACE_ID)
            if space_id:
                rec["nav_spaces"].add(space_id)
                rec["nav_sequence"].append(space_id)

            nav_mode = ext.get(EXT_NAV_MODE)
            if verb and (
                "teleported-to-space" in verb or nav_mode == "teleport"
            ):
                rec["nav_teleport_count"] += 1
            else:
                rec["nav_walk_count"] += 1

        # 2) explored-extension 事件
        for doc in extension_events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            key = (lrn_uid, crs_uid)
            if key not in per_lrn_course:
                per_lrn_course[key] = {
                    "nav_spaces": set(),
                    "nav_sequence": [],
                    "nav_walk_count": 0,
                    "nav_teleport_count": 0,
                    "extension_count": 0,
                    "focus_targets": set(),
                    "focus_count": 0,
                }

            rec = per_lrn_course[key]
            rec["extension_count"] = rec.get("extension_count", 0) + 1

        # 3) focused-on-resource 事件
        for doc in focus_events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            key = (lrn_uid, crs_uid)
            if key not in per_lrn_course:
                per_lrn_course[key] = {
                    "nav_spaces": set(),
                    "nav_sequence": [],
                    "nav_walk_count": 0,
                    "nav_teleport_count": 0,
                    "extension_count": 0,
                    "focus_targets": set(),
                    "focus_count": 0,
                }

            rec = per_lrn_course[key]
            context = doc.get("context") or {}
            ext = context.get("extensions") or {}
            target_id = ext.get(EXT_FOCUS_TARGET_ID)
            if target_id:
                rec["focus_targets"].add(target_id)
                rec["focus_count"] = rec.get("focus_count", 0) + 1

        # 4) 汇总为基础特征 + 汇总课程内列表
        base_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
        per_course_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {
            "space_counts": [],
            "resource_counts": [],
            "extension_flags": [],
            "path_flags": [],
            "teleport_ratios": [],
        })

        for (lrn_uid, crs_uid), rec in per_lrn_course.items():
            unique_spaces = len(rec["nav_spaces"])
            unique_resources = len(rec["focus_targets"])
            has_extension = 1 if rec.get("extension_count", 0) > 0 else 0
            path_jump = self.compute_path_jump_flag(rec["nav_sequence"])

            total_nav = rec.get("nav_walk_count", 0) + rec.get("nav_teleport_count", 0)
            if total_nav > 0:
                teleport_ratio = rec.get("nav_teleport_count", 0) / float(total_nav)
            else:
                teleport_ratio = 0.0

            base_metrics[(lrn_uid, crs_uid)] = {
                "unique_spaces": unique_spaces,
                "unique_resources": unique_resources,
                "has_extension": has_extension,
                "path_jump": path_jump,
                "teleport_ratio": teleport_ratio,
            }

            stats = per_course_values[crs_uid]
            stats["space_counts"].append(unique_spaces)
            stats["resource_counts"].append(unique_resources)
            stats["extension_flags"].append(has_extension)
            stats["path_flags"].append(path_jump)
            stats["teleport_ratios"].append(teleport_ratio)

        return base_metrics, per_course_values

    # ------------------------------------------------------------------
    # 探索指数计算 + 归一化 + 聚类标签
    # ------------------------------------------------------------------

    def _compute_exploration_indices(
        self,
        base_metrics: Dict[Tuple[str, str], Dict[str, Any]],
        per_course_values: Dict[str, Dict[str, List[float]]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        基于基础特征计算：
        - 各特征课程内 z 分数
        - 探索指数 E
        - 课程内 min-max 归一化 E_norm
        - 全局 k-means 聚类标签（低/中/高探索）
        """
        if not base_metrics:
            return {}

        exploration_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        course_E_values: Dict[str, List[float]] = defaultdict(list)

        # 权重设计与原分析脚本一致
        w_space = 0.35
        w_ext = 0.30
        w_res = 0.20
        w_path = 0.10
        w_tp = 0.05
        w_norm = sqrt(w_space ** 2 + w_ext ** 2 + w_res ** 2 + w_path ** 2 + w_tp ** 2)

        # 1) 课程内部 z 标准化 + 探索指数 E
        for crs_uid, stats in per_course_values.items():
            mean_space, std_space = self.compute_mean_std(stats["space_counts"])
            mean_res, std_res = self.compute_mean_std(stats["resource_counts"])
            mean_ext, std_ext = self.compute_mean_std(stats["extension_flags"])
            mean_path, std_path = self.compute_mean_std(stats["path_flags"])
            mean_tp, std_tp = self.compute_mean_std(stats["teleport_ratios"])

            for (lrn_uid, c_uid), m in base_metrics.items():
                if c_uid != crs_uid:
                    continue

                unique_spaces = m["unique_spaces"]
                unique_resources = m["unique_resources"]
                has_extension = m["has_extension"]
                path_jump = m["path_jump"]
                teleport_ratio = m["teleport_ratio"]

                z_space = (unique_spaces - mean_space) / std_space if std_space > 1e-6 else 0.0
                z_res = (unique_resources - mean_res) / std_res if std_res > 1e-6 else 0.0
                z_ext = (has_extension - mean_ext) / std_ext if std_ext > 1e-6 else 0.0
                z_path = (path_jump - mean_path) / std_path if std_path > 1e-6 else 0.0
                z_tp = (teleport_ratio - mean_tp) / std_tp if std_tp > 1e-6 else 0.0

                E = (
                    w_space * z_space +
                    w_ext * z_ext +
                    w_res * z_res +
                    w_path * z_path +
                    w_tp * z_tp
                ) / (w_norm if w_norm > 1e-6 else 1.0)

                exploration_results[(lrn_uid, crs_uid)] = {
                    "unique_spaces": unique_spaces,
                    "unique_resources": unique_resources,
                    "has_extension": has_extension,
                    "path_jump": path_jump,
                    "teleport_ratio": teleport_ratio,
                    "z_space_breadth": z_space,
                    "z_extension_flag": z_ext,
                    "z_resource_breadth": z_res,
                    "z_path_pattern": z_path,
                    "z_teleport_ratio": z_tp,
                    "exploration_index": E,
                }
                course_E_values[crs_uid].append(E)

        # 2) 课程内部对 E 做 min-max 归一化
        for crs_uid, E_list in course_E_values.items():
            if not E_list:
                continue
            E_min = min(E_list)
            E_max = max(E_list)
            span = E_max - E_min

            for (lrn_uid, c_uid), res in exploration_results.items():
                if c_uid != crs_uid:
                    continue
                E = res["exploration_index"]
                if span < 1e-6:
                    E_norm = 0.5
                else:
                    E_norm = (E - E_min) / span
                res["exploration_normalized"] = E_norm

        # 3) 全局基于 exploration_normalized 做 k-means 三档聚类并打标签
        all_records = list(exploration_results.items())
        E_norm_values = [res["exploration_normalized"] for _, res in all_records if "exploration_normalized" in res]

        if not E_norm_values:
            return exploration_results

        centers, assignments = self.kmeans_1d(E_norm_values, k=3, max_iter=50)

        cluster_with_center = list(enumerate(centers))
        cluster_with_center.sort(key=lambda x: x[1])
        cluster_to_rank: Dict[int, int] = {
            cluster_idx: rank for rank, (cluster_idx, _) in enumerate(cluster_with_center)
        }

        rank_to_label: Dict[int, str] = {
            0: "到点即学型（低探索）",
            1: "平衡探索型（中等探索）",
            2: "高探索型探索者",
        }

        for ((key, res), cluster_idx) in zip(all_records, assignments):
            rank = cluster_to_rank.get(cluster_idx, 1)
            label = rank_to_label.get(rank, "平衡探索型（中等探索）")
            res["cluster_index"] = int(cluster_idx)
            res["cluster_rank"] = int(rank)
            res["exploration_label"] = label

        return exploration_results

    # ------------------------------------------------------------------
    # 按学习者聚合结果
    # ------------------------------------------------------------------

    def _build_learner_summaries(
        self,
        exploration_results: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (lrn, crs) 级别的结果聚合为按学习者的结果：

        返回结构：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,          # 多课程 exploration_normalized 的均值
                "overall_exploration_label": str 或 None,# 多课程分类的众数（并列时选更“好”的那档）
                "overall_cluster_rank": int 或 None,
                "per_course_results": [  # 逐课程结果
                    {
                        "course_uid": "...",
                        "unique_spaces": ...,
                        "unique_resources": ...,
                        "has_extension": 0/1,
                        "path_jump": 0/1,
                        "teleport_ratio": ...,
                        "z_space_breadth": ...,
                        "z_extension_flag": ...,
                        "z_resource_breadth": ...,
                        "z_path_pattern": ...,
                        "z_teleport_ratio": ...,
                        "exploration_index": ...,
                        "exploration_normalized": ...,
                        "exploration_label": ...,
                        "cluster_rank": ...,
                    },
                    ...
                ],
            },
            ...
        }
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), res in exploration_results.items():
            if lrn_uid not in learner_data:
                learner_data[lrn_uid] = {
                    "learner_uid": lrn_uid,
                    "has_data": True,
                    "overall_score": None,
                    "overall_exploration_label": None,
                    "overall_cluster_rank": None,
                    "per_course_results": [],
                }

            item = {
                "course_uid": crs_uid,
                "unique_spaces": res.get("unique_spaces"),
                "unique_resources": res.get("unique_resources"),
                "has_extension": res.get("has_extension"),
                "path_jump": res.get("path_jump"),
                "teleport_ratio": res.get("teleport_ratio"),
                "z_space_breadth": res.get("z_space_breadth"),
                "z_extension_flag": res.get("z_extension_flag"),
                "z_resource_breadth": res.get("z_resource_breadth"),
                "z_path_pattern": res.get("z_path_pattern"),
                "z_teleport_ratio": res.get("z_teleport_ratio"),
                "exploration_index": res.get("exploration_index"),
                "exploration_normalized": res.get("exploration_normalized"),
                "exploration_label": res.get("exploration_label"),
                "cluster_rank": res.get("cluster_rank"),
            }
            learner_data[lrn_uid]["per_course_results"].append(item)

        # 聚合 overall_score + overall_exploration_label
        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            scores = [
                it["exploration_normalized"]
                for it in pcs
                if it.get("exploration_normalized") is not None
            ]
            if scores:
                info["overall_score"] = sum(scores) / float(len(scores))
            else:
                info["overall_score"] = None

            # 分类：按 cluster_rank 众数，若并列则选 rank 较大者（语义上“更好”的那档）
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
                    overall_label = "到点即学型（低探索）"
                elif best_rank == 2:
                    overall_label = "高探索型探索者"
                else:
                    overall_label = "平衡探索型（中等探索）"

                info["overall_cluster_rank"] = best_rank
                info["overall_exploration_label"] = overall_label
            else:
                info["overall_cluster_rank"] = None
                info["overall_exploration_label"] = None

        return learner_data

    # ------------------------------------------------------------------
    # 对外公开接口：单个 / 多个学习者
    # ------------------------------------------------------------------

    def analyze_multiple_learners(
        self, learner_uids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        对多个学习者进行空间与资源探索倾向分析。

        返回：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,
                "overall_exploration_label": str 或 None,
                "overall_cluster_rank": int 或 None,
                "per_course_results": [...],
            },
            ...
        }
        """
        if not learner_uids:
            return {}

        try:
            # 1) 从 Repository 获取原始事件
            raw = spatial_exploration_orientation_repository.get_spatial_exploration_raw_data_for_learners(
                learner_uids
            )
            navigation_events = raw.get("navigation_events", [])
            extension_events = raw.get("extension_events", [])
            focus_events = raw.get("focus_events", [])

            # 如果完全没有相关事件，则直接返回空结果结构
            if not (navigation_events or extension_events or focus_events):
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_exploration_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            # 2) 聚合基础特征
            base_metrics, per_course_values = self._aggregate_raw_events(
                navigation_events, extension_events, focus_events
            )

            if not base_metrics:
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_exploration_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            # 3) 计算探索指数 + 归一化 + 聚类标签
            exploration_results = self._compute_exploration_indices(
                base_metrics, per_course_values
            )

            # 4) 按学习者聚合结果
            learner_summaries = self._build_learner_summaries(exploration_results)

            # 5) 对于传入但没有任何数据的学习者，也返回空结果
            for uid in learner_uids:
                if uid not in learner_summaries:
                    learner_summaries[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_exploration_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }

            return learner_summaries

        except Exception as e:
            logger.error(f"多学习者空间探索分析失败: {e}", exc_info=True)
            # 出错时，也保证返回结构是按 learner_uid 的 dict
            result: Dict[str, Dict[str, Any]] = {}
            for uid in learner_uids:
                result[uid] = {
                    "learner_uid": uid,
                    "has_data": False,
                    "overall_score": None,
                    "overall_exploration_label": None,
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
                "overall_exploration_label": None,
                "overall_cluster_rank": None,
                "per_course_results": [],
            },
        )


# 全局引擎实例 + 便捷函数，风格与 AttentionAllocationEngine 对齐
_spatial_exploration_engine_instance: Optional[SpatialExplorationOrientationEngine] = None


def get_spatial_exploration_orientation_engine() -> SpatialExplorationOrientationEngine:
    global _spatial_exploration_engine_instance
    if _spatial_exploration_engine_instance is None:
        _spatial_exploration_engine_instance = SpatialExplorationOrientationEngine()
    return _spatial_exploration_engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_spatial_exploration_orientation_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
    engine = get_spatial_exploration_orientation_engine()
    return engine.analyze_multiple_learners(learner_uids)


# 简单本地测试（可选）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = SpatialExplorationOrientationEngine()
    # 真实存在的学习者UID（沿用注意力引擎的测试 UID）
    test_learner_uids = [
        "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
        "lrn_004a9c3f5bf246faab3d390ce716e658",
    ]

    print("=== 空间探索：单学习者测试 ===")
    res_single = engine.analyze_single_learner(test_learner_uids[0])
    print(res_single)

    print("=== 空间探索：多学习者测试 ===")
    res_multi = engine.analyze_multiple_learners(test_learner_uids)
    print(res_multi)
