# app/repositories/interaction_style_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class InteractionStyleRepository(BaseRepository):
    """
    交互与操作熟练度 / 风格（interaction_style）维度的数据仓库。

    职责：
    - 仅负责从 MongoDB.MLS.Interaction 读取与该维度相关的 xAPI 行为事件；
    - 利用现有复合索引：
        * idx_lrn_verb_course: {_lrn_uid, 'verb.id', _course_uid}
        * idx_course_verb_lrn: {_course_uid, 'verb.id', _lrn_uid}
      进行课程发现和按课程分批扫描；
    - 输出课程级的基础数值特征，不进行聚类和标签判定。
    """

    DB_NAME = "MLS"
    INTERACTION_COLLECTION = "Interaction"

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "manipulated_object": VERB_BASE + "manipulated-object",
        "performed_procedure_step": VERB_BASE + "performed-procedure-step",
        "completed": VERB_BASE + "completed",
    }

    # 仅关注 VR/AR/交互型单元
    UNIT_TYPES_FOR_STYLE = {"vr", "ar", "interact"}

    DURATION_RE = re.compile(r"^PT(\d+)S$")

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
        为若干学习者准备“交互与操作熟练度 / 风格”分析所需的课程级特征。

        返回:
            metrics_by_lc:
                (lrn_uid, crs_uid) -> {
                    "freq_per_minute": float,
                    "step_success_rate": float,
                    "unit_success_rate": float,
                    "performance_score": float,
                    "x": float,   # log(1 + freq_per_minute)
                    "y": float,   # 1 - performance_score
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
                "InteractionStyleRepository.load_metrics_for_learners: 空的学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "InteractionStyleRepository: 开始准备交互风格原始数据，目标学习者数: %d",
            len(learner_uids),
        )

        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "InteractionStyleRepository: 与目标学习者相关的课程数: %d",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "InteractionStyleRepository: 目标学习者在交互相关事件上没有任何课程记录。"
            )
            return {}, {}, learner_courses_map

        raw_stats_by_lc, learners_per_course = self._aggregate_raw_stats_for_courses(
            all_courses
        )

        logger.info(
            "InteractionStyleRepository: 已完成原始统计聚合，(lrn, crs) 粗粒度条目数: %d",
            len(raw_stats_by_lc),
        )

        metrics_by_lc = self._compute_metrics_from_raw(raw_stats_by_lc)

        logger.info(
            "InteractionStyleRepository: 指标计算完成，(lrn, crs) 有效条目数: %d",
            len(metrics_by_lc),
        )

        # 按有数据的课程过滤 learner_courses_map
        filtered_map: Dict[str, Set[str]] = {}
        for lrn_uid, courses in learner_courses_map.items():
            valid = {crs for crs in courses if (lrn_uid, crs) in metrics_by_lc}
            if valid:
                filtered_map[lrn_uid] = valid

        return metrics_by_lc, learners_per_course, filtered_map

    # ------------------------------------------------------------------
    # 内部: Mongo 访问与课程发现
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
                  AND verb.id in [manipulated_object, performed_procedure_step, completed]
                  AND _type in UNIT_TYPES_FOR_STYLE
            group by (lrn_uid, course_uid)

        返回:
            learner_courses_map: learner_uid -> set(course_uid)
            all_courses: 所有相关课程 uid 集合
        """
        col = self._get_interaction_collection()
        verb_list = list(self.VERBS.values())

        pipeline = [
            {
                "$match": {
                    "_lrn_uid": {"$in": learner_uids},
                    "verb.id": {"$in": verb_list},
                    "_type": {"$in": list(self.UNIT_TYPES_FOR_STYLE)},
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
    # 内部: 通用迭代器与原始统计聚合
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

    @classmethod
    def _parse_iso8601_duration(cls, duration_str: Optional[str]) -> Optional[int]:
        """
        解析简单形式的 ISO8601 时长字符串，例如："PT120S"。
        若为空或格式不符，返回 None。
        """
        if not duration_str or not isinstance(duration_str, str):
            return None
        m = cls.DURATION_RE.match(duration_str)
        if not m:
            return None
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            return None

    def _aggregate_raw_stats_for_courses(
        self, course_uids: Set[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
    ]:
        """
        对一批课程的交互相关事件进行原始统计聚合，生成课程级基础统计。

        返回:
            raw_stats_by_lc[(lrn_uid, crs_uid)] = {
                "manip_count": int,
                "step_total": int,
                "step_success": int,
                "unit_total": int,
                "unit_success": int,
                "total_interact_duration": float,  # 秒
            }

            learners_per_course[crs_uid] = 课程内去重学习者数量
        """
        if not course_uids:
            return {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        raw_stats_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "manip_count": 0,
                "step_total": 0,
                "step_success": 0,
                "unit_total": 0,
                "unit_success": 0,
                "total_interact_duration": 0.0,
            }
        )
        learners_per_course_sets: Dict[str, Set[str]] = defaultdict(set)

        projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "result": 1,
            "_type": 1,
        }
        verb_list = list(self.VERBS.values())

        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1) * self.course_chunk_size
            ]
            logger.info(
                "InteractionStyleRepository: 读取交互事件，课程分片 %d/%d，课程数: %d",
                chunk_idx + 1,
                total_chunks,
                len(sub_courses),
            )

            match_query = {
                "_course_uid": {"$in": sub_courses},
                "verb.id": {"$in": verb_list},
                "_type": {"$in": list(self.UNIT_TYPES_FOR_STYLE)},
            }

            event_cnt = 0
            for doc in self._iterate_events(match_query, projection):
                event_cnt += 1

                lrn_uid = doc.get("_lrn_uid")
                crs_uid = doc.get("_course_uid")
                if not lrn_uid or not crs_uid:
                    continue

                learners_per_course_sets[crs_uid].add(lrn_uid)

                verb_id = (doc.get("verb") or {}).get("id") or doc.get("verb.id")
                result = doc.get("result") or {}

                key_lc = (lrn_uid, crs_uid)
                st = raw_stats_by_lc[key_lc]

                if verb_id == self.VERBS["manipulated_object"]:
                    st["manip_count"] += 1

                elif verb_id == self.VERBS["performed_procedure_step"]:
                    st["step_total"] += 1
                    if result.get("success") is True:
                        st["step_success"] += 1

                elif verb_id == self.VERBS["completed"]:
                    st["unit_total"] += 1
                    if result.get("success") is True:
                        st["unit_success"] += 1
                    dur_sec = self._parse_iso8601_duration(result.get("duration"))
                    if dur_sec is not None:
                        st["total_interact_duration"] += float(dur_sec)

            logger.info(
                "InteractionStyleRepository: 完成交互事件读取，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        learners_per_course = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course_sets.items()
        }

        logger.info(
            "InteractionStyleRepository: 原始统计聚合完成，课程数: %d",
            len(learners_per_course),
        )

        return dict(raw_stats_by_lc), learners_per_course

    # ------------------------------------------------------------------
    # 内部: 从原始统计派生课程级指标
    # ------------------------------------------------------------------
    def _compute_metrics_from_raw(
        self,
        raw_stats_by_lc: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        将原始统计转换为课程级交互指标：
            freq_per_minute / step_success_rate / unit_success_rate /
            performance_score / x / y
        """
        from math import log

        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for key, st in raw_stats_by_lc.items():
            manip = int(st.get("manip_count", 0))
            total_dur = float(st.get("total_interact_duration", 0.0))
            step_total = int(st.get("step_total", 0))
            step_success = int(st.get("step_success", 0))
            unit_total = int(st.get("unit_total", 0))
            unit_success = int(st.get("unit_success", 0))

            # 完全没有任何行为数据则跳过
            if manip == 0 and step_total == 0 and unit_total == 0:
                continue

            # 交互强度：每分钟操作次数，若时长为 0 则按 1 分钟计算下限
            minutes = max(total_dur / 60.0, 1.0)
            freq_per_minute = manip / minutes if minutes > 0 else 0.0

            # 步骤成功率
            if step_total > 0:
                step_success_rate = step_success / float(step_total)
            else:
                step_success_rate = 0.5  # 中性值

            # 单元成功率
            if unit_total > 0:
                unit_success_rate = unit_success / float(unit_total)
            else:
                unit_success_rate = 0.5  # 中性值

            performance_score = 0.5 * step_success_rate + 0.5 * unit_success_rate

            x = log(1.0 + freq_per_minute)
            y = 1.0 - performance_score

            metrics_by_lc[key] = {
                "freq_per_minute": float(freq_per_minute),
                "step_success_rate": float(step_success_rate),
                "unit_success_rate": float(unit_success_rate),
                "performance_score": float(performance_score),
                "x": float(x),
                "y": float(y),
            }

        return metrics_by_lc
