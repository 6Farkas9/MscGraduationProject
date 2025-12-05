# app/repositories/contribution_reputation_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ContributionReputationRepository(BaseRepository):
    """
    元宇宙价值贡献与声望（contribution_reputation）维度的数据仓库。

    职责：
    - 仅负责从 MongoDB.MLS.Interaction 读取与“价值贡献与声望”相关的 xAPI 事件；
    - 利用现有复合索引（_lrn_uid + verb.id + _course_uid、
      _course_uid + verb.id + _lrn_uid）按课程/学习者聚合基础统计；
    - 不做任何聚类或离散标签判定，只产出分析所需的数值特征。
    """

    DB_NAME = "MLS"
    INTERACTION_COLLECTION = "Interaction"

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        # 价值交换行为：token 流入/支出
        "exchanged_value": VERB_BASE + "exchanged-value",
        # 贡献行为：资源、共同编辑、协作活动
        "contributed_resource": VERB_BASE + "contributed-resource",
        "co_edited_artifact": VERB_BASE + "co-edited-artifact",
        "collaborated_on_activity": VERB_BASE + "collaborated-on-activity",
    }

    EXT_BASE = "https://legend-meta.com/xapi/ext/"
    EXT_VALUE_CHANGE = EXT_BASE + "value-change"

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
    # 公共接口
    # ------------------------------------------------------------------
    def load_metrics_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
        Dict[str, Set[str]],
    ]:
        """
        为一批学习者准备“价值贡献与声望”分析所需的课程级基础数据。

        返回:
            value_stats_by_lc:
                (lrn_uid, crs_uid) -> {
                    "token_gain": float,
                    "token_cost": float,
                    "token_net": float,
                    "value_events": int,
                    "resource_contrib_count": int,
                    "coedit_count": int,
                    "collab_count": int,
                    "contrib_total": int,
                }

            learners_per_course:
                crs_uid -> 该课程内参与过价值相关事件的去重学习者数量

            learner_courses_map:
                learner_uid -> set(course_uid)
                仅包含 value_stats_by_lc 中存在数据的课程。
        """
        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        if not learner_uids:
            logger.info(
                "ContributionReputationRepository.load_metrics_for_learners: 空的学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "ContributionReputationRepository: 开始准备原始价值贡献数据，目标学习者数: %d",
            len(learner_uids),
        )

        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "ContributionReputationRepository: 与目标学习者相关的课程数: %d",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "ContributionReputationRepository: 目标学习者在价值贡献相关事件上没有任何记录。"
            )
            return {}, {}, learner_courses_map

        value_stats_by_lc, learners_per_course = self._aggregate_value_metrics_for_courses(
            all_courses
        )

        # 按有数据的课程过滤 learner_courses_map
        filtered_map: Dict[str, Set[str]] = {}
        for lrn_uid, courses in learner_courses_map.items():
            valid = {
                crs for crs in courses if (lrn_uid, crs) in value_stats_by_lc
            }
            if valid:
                filtered_map[lrn_uid] = valid

        logger.info(
            "ContributionReputationRepository: 课程级基础统计整理完成，(lrn, crs) 条目: %d",
            len(value_stats_by_lc),
        )

        return value_stats_by_lc, learners_per_course, filtered_map

    # ------------------------------------------------------------------
    # 内部：Mongo 访问 & 课程发现
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
            match _lrn_uid in learner_uids AND verb.id in [相关 verb]
            group by (lrn_uid, course_uid)

        返回:
            learner_courses_map: learner_uid -> set(course_uid)
            all_courses: 所有课程 uid 集合
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
    # 内部：迭代器与课程级聚合
    # ------------------------------------------------------------------
    def _iterate_events(
        self,
        match_query: Dict,
        projection: Dict,
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

    def _aggregate_value_metrics_for_courses(
        self, course_uids: Set[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
    ]:
        """
        对一批课程的价值相关事件进行聚合，生成课程级基础统计：

        返回:
            value_stats_by_lc[(lrn_uid, crs_uid)] = {
                "token_gain": float,
                "token_cost": float,
                "token_net": float,
                "value_events": int,
                "resource_contrib_count": int,
                "coedit_count": int,
                "collab_count": int,
                "contrib_total": int,
            }

            learners_per_course[crs_uid] = 课程内去重学习者数量
        """
        if not course_uids:
            return {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        value_stats_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "token_gain": 0.0,
                "token_cost": 0.0,
                "token_net": 0.0,
                "value_events": 0,
                "resource_contrib_count": 0,
                "coedit_count": 0,
                "collab_count": 0,
                "contrib_total": 0,
            }
        )
        learners_per_course: Dict[str, Set[str]] = defaultdict(set)

        projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "context": 1,
        }

        verb_list = list(self.VERBS.values())

        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1) * self.course_chunk_size
            ]
            logger.info(
                "ContributionReputationRepository: 读取价值相关事件，课程分片 %d/%d，课程数: %d",
                chunk_idx + 1,
                total_chunks,
                len(sub_courses),
            )

            match_query = {
                "_course_uid": {"$in": sub_courses},
                "verb.id": {"$in": verb_list},
            }

            event_cnt = 0
            for doc in self._iterate_events(match_query, projection):
                event_cnt += 1
                lrn_uid = doc.get("_lrn_uid")
                crs_uid = doc.get("_course_uid")
                if not lrn_uid or not crs_uid:
                    continue

                learners_per_course[crs_uid].add(lrn_uid)

                verb_id = (doc.get("verb") or {}).get("id")
                key = (lrn_uid, crs_uid)
                stats = value_stats_by_lc[key]

                if verb_id == self.VERBS["exchanged_value"]:
                    ctx = doc.get("context") or {}
                    exts = ctx.get("extensions") or {}
                    delta = exts.get(self.EXT_VALUE_CHANGE)
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

                elif verb_id == self.VERBS["contributed_resource"]:
                    stats["resource_contrib_count"] += 1
                elif verb_id == self.VERBS["co_edited_artifact"]:
                    stats["coedit_count"] += 1
                elif verb_id == self.VERBS["collaborated_on_activity"]:
                    stats["collab_count"] += 1

            logger.info(
                "ContributionReputationRepository: 完成价值相关事件读取，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        # 完成聚合后，填充 contrib_total
        for key, stats in value_stats_by_lc.items():
            stats["contrib_total"] = int(
                stats["resource_contrib_count"]
                + stats["coedit_count"]
                + stats["collab_count"]
            )

        learners_per_course_count = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course.items()
        }

        return dict(value_stats_by_lc), learners_per_course_count
