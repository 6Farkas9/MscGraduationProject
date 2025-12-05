# app/repositories/spatial_exploration_orientation_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class SpatialExplorationOrientationRepository(BaseRepository):
    """
    空间与资源探索倾向（Spatial & Resource Exploration Orientation）维度的数据仓库。

    职责：
    - 仅负责从 MongoDB.MLS.Interaction 读取与该维度相关的 xAPI 行为事件；
    - 利用现有复合索引：
        * idx_lrn_verb_course: { _lrn_uid, 'verb.id', _course_uid }
        * idx_course_verb_lrn: { _course_uid, 'verb.id', _lrn_uid }
      进行课程发现和按课程分批扫描；
    - 为每个 (lrn_uid, crs_uid) 计算：
        · unique_spaces        : 访问到的 space-id 个数
        · unique_resources     : 聚焦的资源类别数
        · has_extension        : 是否参与可选拓展（0/1）
        · path_jump            : 是否存在回访型路径（0/1）
        · teleport_ratio       : teleport 导航在全部导航中的比例
        · 各特征在课程内的 z 分数
        · 探索指数 exploration_index
        · 课程内归一化 exploration_index_norm ∈ [0, 1]
    - 不进行聚类或标签判定（交给 engine）。
    """

    DB_NAME = "MLS"
    INTERACTION_COLLECTION = "Interaction"

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "navigated_to_space": VERB_BASE + "navigated-to-space",
        "teleported_to_space": VERB_BASE + "teleported-to-space",
        "explored_extension": VERB_BASE + "explored-extension",
        "focused_on_resource": VERB_BASE + "focused-on-resource",
    }

    EXT_SPACE_ID = "https://legend-meta.com/xapi/ext/space-id"
    EXT_NAV_MODE = "https://legend-meta.com/xapi/ext/navigation-mode"
    EXT_FOCUS_TARGET_ID = "https://legend-meta.com/xapi/ext/focus-target-id"

    def __init__(
        self,
        batch_size: int = 5000,
        course_chunk_size: int = 200,
    ):
        """
        Args:
            batch_size: Mongo 游标 batch_size，控制单次从服务器端取回的文档数量。
            course_chunk_size: 每次查询 _course_uid 的分片大小，避免 $in 过大。
        """
        super().__init__()
        self.batch_size = batch_size
        self.course_chunk_size = course_chunk_size

    # ------------------------------------------------------------------
    # 对外公共接口
    # ------------------------------------------------------------------
    def load_metrics_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
        Dict[str, Set[str]],
    ]:
        """
        为若干学习者准备“空间与资源探索倾向”分析所需的课程级特征。

        注意：为了做“课程内聚类”，本仓库会在相关课程中统计**所有**学习者，
        不仅仅是目标学习者。但只返回与目标学习者出现在同一课程里的 (lrn, crs)。

        返回:
            metrics_by_lc:
                (lrn_uid, crs_uid) -> {
                    "unique_spaces": int,
                    "unique_resources": int,
                    "has_extension": int,
                    "path_jump": int,
                    "teleport_ratio": float,
                    # 课程内 z 分数：
                    "z_space_breadth": float,
                    "z_extension_flag": float,
                    "z_resource_breadth": float,
                    "z_path_pattern": float,
                    "z_teleport_ratio": float,
                    # 探索指数及归一化：
                    "exploration_index": float,
                    "exploration_index_norm": float,
                }

            learners_per_course:
                crs_uid -> 该课程内参与相关事件的去重学习者数量（用于课程内聚类）

            learner_courses_map:
                lrn_uid -> set(course_uid)
                仅包含 metrics_by_lc 中存在数据的课程。
        """
        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        if not learner_uids:
            logger.info(
                "SpatialExplorationOrientationRepository.load_metrics_for_learners: 空学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "SpatialExplorationOrientationRepository: 开始准备空间与资源探索原始数据，目标学习者数: %d",
            len(learner_uids),
        )

        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "SpatialExplorationOrientationRepository: 与目标学习者相关的课程数: %d",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "SpatialExplorationOrientationRepository: 目标学习者在空间/资源相关事件上没有任何课程记录。"
            )
            return {}, {}, learner_courses_map

        spatial_stats, learners_per_course = self._aggregate_spatial_stats_for_courses(
            all_courses
        )

        logger.info(
            "SpatialExplorationOrientationRepository: 已完成 (lrn, crs) 粗粒度空间/资源统计，条目数: %d",
            len(spatial_stats),
        )

        metrics_by_lc = self._compute_exploration_index_for_courses(spatial_stats)

        logger.info(
            "SpatialExplorationOrientationRepository: exploration_index_norm 计算完成，(lrn, crs) 有效条目数: %d",
            len(metrics_by_lc),
        )

        # 过滤 learner_courses_map，仅保留有指标的课程
        filtered_map: Dict[str, Set[str]] = {}
        for lrn_uid, courses in learner_courses_map.items():
            valid = {crs for crs in courses if (lrn_uid, crs) in metrics_by_lc}
            if valid:
                filtered_map[lrn_uid] = valid

        return metrics_by_lc, learners_per_course, filtered_map

    # ------------------------------------------------------------------
    # 内部：Mongo 访问与课程发现
    # ------------------------------------------------------------------
    def _get_interaction_collection(self):
        return self.mongodb_operator.get_collection(self.INTERACTION_COLLECTION)

    def _get_courses_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        使用 idx_lrn_verb_course 复合索引：
            key: { _lrn_uid: 1, 'verb.id': 1, _course_uid: 1 }

        pipeline:
            match _lrn_uid in learner_uids
                  AND verb.id in 空间/资源探索相关 verbs
            group by (lrn_uid, course_uid)
        """
        col = self._get_interaction_collection()
        verb_list = list(self.VERBS.values())

        pipeline = [
            {
                "$match": {
                    "_lrn_uid": {"$in": learner_uids},
                    "verb.id": {"$in": verb_list},
                }
            },
            {
                "$group": {
                    "_id": {
                        "lrn_uid": "$_lrn_uid",
                        "course_uid": "$_course_uid",
                    }
                }
            },
        ]

        learner_courses_map: Dict[str, Set[str]] = defaultdict(set)
        all_courses: Set[str] = set()

        for doc in col.aggregate(pipeline, allowDiskUse=True):
            _id = doc.get("_id") or {}
            lrn_uid = _id.get("lrn_uid")
            crs_uid = _id.get("course_uid")
            if not lrn_uid or not crs_uid:
                continue
            learner_courses_map[lrn_uid].add(crs_uid)
            all_courses.add(crs_uid)

        return learner_courses_map, all_courses

    # ------------------------------------------------------------------
    # 内部：通用迭代器
    # ------------------------------------------------------------------
    def _iterate_events(
        self,
        match_query: Dict[str, Any],
        projection: Dict[str, int],
        batch_size: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        通用迭代器：按指定查询和投影，使用服务器端游标分批返回文档。

        为了利用 idx_course_verb_lrn 复合索引，match_query 中应包含：
            - "_course_uid": 某值或 {"$in": [...]}；
            - "verb.id": 某值或 {"$in": [...]}。
        """
        col = self._get_interaction_collection()
        cursor = (
            col.find(match_query, projection=projection, no_cursor_timeout=True)
            .batch_size(batch_size or self.batch_size)
        )
        try:
            for doc in cursor:
                yield doc
        finally:
            cursor.close()

    # ------------------------------------------------------------------
    # 工具：均值/标准差 & 路径回访标记
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_mean_std(values: List[float]) -> Tuple[float, float]:
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        mean_v = sum(values) / float(n)
        if n == 1:
            return mean_v, 0.0
        var = sum((v - mean_v) ** 2 for v in values) / float(n)
        return mean_v, math.sqrt(var)

    @staticmethod
    def _compute_path_jump_flag(space_seq: List[str]) -> int:
        """
        简化版“回访型路径”检测：
        - 对 space-id 序列做相邻去重；
        - 若某 space-id 在去重后出现 >=2 次，则认为存在回访型路径，返回 1，否则 0。
        """
        if not space_seq:
            return 0

        reduced: List[str] = []
        for sid in space_seq:
            if not reduced or reduced[-1] != sid:
                reduced.append(sid)

        seen: Set[str] = set()
        for sid in reduced:
            if sid in seen:
                return 1
            seen.add(sid)
        return 0

    # ------------------------------------------------------------------
    # 内部：按课程分片聚合 (lrn, crs) 空间/资源统计
    # ------------------------------------------------------------------
    def _aggregate_spatial_stats_for_courses(
        self, course_uids: Set[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
    ]:
        """
        对一批课程的空间/资源相关事件进行聚合，生成：

        spatial_stats[(lrn_uid, crs_uid)] = {
            "nav_spaces": set(),
            "nav_sequence": list(),
            "nav_walk_count": int,
            "nav_teleport_count": int,
            "extension_count": int,
            "focus_targets": set(),
            "focus_count": int,
        }

        learners_per_course[crs_uid] = 去重学习者数量。
        """
        if not course_uids:
            return {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        spatial_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
        learners_per_course_sets: Dict[str, Set[str]] = defaultdict(set)

        projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "context": 1,
            "timestamp": 1,
        }
        verb_list = list(self.VERBS.values())

        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1)
                * self.course_chunk_size
            ]

            logger.info(
                "SpatialExplorationOrientationRepository: 读取事件，课程分片 %d/%d，课程数: %d",
                chunk_idx + 1,
                total_chunks,
                len(sub_courses),
            )

            match_query = {
                "_course_uid": {"$in": sub_courses},
                "verb.id": {"$in": verb_list},
            }

            events: List[Dict[str, Any]] = []
            for doc in self._iterate_events(match_query, projection):
                events.append(doc)

            logger.info(
                "SpatialExplorationOrientationRepository: 完成 Mongo 读取，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                len(events),
            )

            # 分片内按 timestamp 排序，保证空间序列时序正确
            events.sort(key=lambda d: str(d.get("timestamp") or ""))

            for doc in events:
                lrn_uid = doc.get("_lrn_uid")
                crs_uid = doc.get("_course_uid")
                if not lrn_uid or not crs_uid:
                    continue

                learners_per_course_sets[crs_uid].add(lrn_uid)

                key_lc = (lrn_uid, crs_uid)
                if key_lc not in spatial_stats:
                    spatial_stats[key_lc] = {
                        "nav_spaces": set(),
                        "nav_sequence": [],
                        "nav_walk_count": 0,
                        "nav_teleport_count": 0,
                        "extension_count": 0,
                        "focus_targets": set(),
                        "focus_count": 0,
                    }

                rec = spatial_stats[key_lc]

                verb_id = (doc.get("verb") or {}).get("id")
                context = doc.get("context") or {}
                ext = context.get("extensions") or {}

                if verb_id in (
                    self.VERBS["navigated_to_space"],
                    self.VERBS["teleported_to_space"],
                ):
                    # 这里修正了之前的 AttributeError：
                    # 使用已定义的 EXT_SPACE_ID，且增加对简单键 'space-id' 的兜底。
                    space_id = (
                        ext.get(self.EXT_SPACE_ID)
                        or ext.get("space-id")
                    )
                    if space_id:
                        rec["nav_spaces"].add(space_id)
                        rec["nav_sequence"].append(space_id)

                    nav_mode = ext.get(self.EXT_NAV_MODE)
                    if (
                        verb_id == self.VERBS["teleported_to_space"]
                        or nav_mode == "teleport"
                    ):
                        rec["nav_teleport_count"] += 1
                    else:
                        rec["nav_walk_count"] += 1

                elif verb_id == self.VERBS["explored_extension"]:
                    rec["extension_count"] += 1

                elif verb_id == self.VERBS["focused_on_resource"]:
                    target_id = ext.get(self.EXT_FOCUS_TARGET_ID)
                    if target_id:
                        rec["focus_targets"].add(target_id)
                        rec["focus_count"] += 1

        learners_per_course = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course_sets.items()
        }

        logger.info(
            "SpatialExplorationOrientationRepository: 粗粒度空间/资源统计聚合完成，课程数: %d，(lrn, crs) 组合数: %d",
            len(learners_per_course),
            len(spatial_stats),
        )

        return spatial_stats, learners_per_course

    # ------------------------------------------------------------------
    # 内部：从粗粒度统计计算 exploration_index & exploration_index_norm
    # ------------------------------------------------------------------
    def _compute_exploration_index_for_courses(
        self,
        spatial_stats: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        将粗粒度空间/资源统计转换为课程级探索指数。

        逻辑基于原 analyze_spatial_exploration_orientation.py：
        - 在课程内部对 unique_spaces / unique_resources / has_extension /
          path_jump / teleport_ratio 做 z 标准化；
        - 用加权组合得到 exploration_index；
        - 再在课程内部对 exploration_index 做 min-max 归一化为 exploration_index_norm ∈ [0,1]。
        """
        if not spatial_stats:
            return {}

        # 先拆成按课程的列表
        course_to_entries: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(
            list
        )
        for (lrn_uid, crs_uid), stat in spatial_stats.items():
            course_to_entries[crs_uid].append((lrn_uid, stat))

        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # 权重与原脚本保持一致
        w_space = 0.35
        w_ext = 0.30
        w_res = 0.20
        w_path = 0.10
        w_tp = 0.05
        w_norm = math.sqrt(
            w_space ** 2 + w_ext ** 2 + w_res ** 2 + w_path ** 2 + w_tp ** 2
        )

        for crs_uid, entries in course_to_entries.items():
            if not entries:
                continue

            space_counts: List[float] = []
            resource_counts: List[float] = []
            ext_flags: List[float] = []
            path_flags: List[float] = []
            tp_ratios: List[float] = []

            # 第一次遍历：构造基础特征
            base_metrics: Dict[str, Dict[str, Any]] = {}
            for lrn_uid, stat in entries:
                unique_spaces = len(stat["nav_spaces"])
                unique_resources = len(stat["focus_targets"])
                has_extension = 1 if stat["extension_count"] > 0 else 0
                path_jump = self._compute_path_jump_flag(stat["nav_sequence"])

                total_nav = stat["nav_walk_count"] + stat["nav_teleport_count"]
                teleport_ratio = (
                    stat["nav_teleport_count"] / float(total_nav)
                    if total_nav > 0
                    else 0.0
                )

                base_metrics[lrn_uid] = {
                    "unique_spaces": unique_spaces,
                    "unique_resources": unique_resources,
                    "has_extension": has_extension,
                    "path_jump": path_jump,
                    "teleport_ratio": teleport_ratio,
                }

                space_counts.append(float(unique_spaces))
                resource_counts.append(float(unique_resources))
                ext_flags.append(float(has_extension))
                path_flags.append(float(path_jump))
                tp_ratios.append(float(teleport_ratio))

            mean_space, std_space = self._compute_mean_std(space_counts)
            mean_res, std_res = self._compute_mean_std(resource_counts)
            mean_ext, std_ext = self._compute_mean_std(ext_flags)
            mean_path, std_path = self._compute_mean_std(path_flags)
            mean_tp, std_tp = self._compute_mean_std(tp_ratios)

            E_values: List[float] = []

            # 第二次遍历：计算 z 分数和 exploration_index
            for lrn_uid, _stat in entries:
                m = base_metrics[lrn_uid]

                unique_spaces = m["unique_spaces"]
                unique_resources = m["unique_resources"]
                has_extension = m["has_extension"]
                path_jump = m["path_jump"]
                teleport_ratio = m["teleport_ratio"]

                z_space = (
                    (unique_spaces - mean_space) / std_space
                    if std_space > 1e-6
                    else 0.0
                )
                z_res = (
                    (unique_resources - mean_res) / std_res
                    if std_res > 1e-6
                    else 0.0
                )
                z_ext = (
                    (has_extension - mean_ext) / std_ext
                    if std_ext > 1e-6
                    else 0.0
                )
                z_path = (
                    (path_jump - mean_path) / std_path
                    if std_path > 1e-6
                    else 0.0
                )
                z_tp = (
                    (teleport_ratio - mean_tp) / std_tp
                    if std_tp > 1e-6
                    else 0.0
                )

                E = (
                    w_space * z_space
                    + w_ext * z_ext
                    + w_res * z_res
                    + w_path * z_path
                    + w_tp * z_tp
                ) / (w_norm if w_norm > 1e-6 else 1.0)

                key = (lrn_uid, crs_uid)
                metrics_by_lc[key] = {
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
                E_values.append(E)

            # 第三步：在课程内部对 E 做 [0,1] 的 min-max 归一化
            if not E_values:
                continue
            E_min = min(E_values)
            E_max = max(E_values)
            span = E_max - E_min if E_max > E_min else 0.0

            for lrn_uid, _stat in entries:
                key = (lrn_uid, crs_uid)
                E = metrics_by_lc[key]["exploration_index"]
                if span > 1e-6:
                    E_norm = (E - E_min) / span
                else:
                    E_norm = 0.5  # 所有人完全一致时，统一给中间值
                metrics_by_lc[key]["exploration_index_norm"] = E_norm

        logger.info(
            "SpatialExplorationOrientationRepository: 已为 %d 个 (lrn, crs) 计算 exploration_index_norm。",
            len(metrics_by_lc),
        )

        return metrics_by_lc
