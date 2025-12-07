# BackEnd/app/data_access/profiling/attention_allocation_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository
from app.shared.utils.repository_mixins import parse_iso8601_duration_seconds

logger = logging.getLogger(__name__)


class AttentionAllocationRepository(MongoDBBaseRepository):
    """
    注意力分配画像的仓库层（MongoDB 版）。

    职责边界:
    --------
    - 只负责在 MongoDB 层利用现有索引，筛选指定学习者相关的课程；
    - 在 Interaction 集合中按 (学习者, 课程) 聚合出分析所需的基础统计量；
    - 不做任何标签、聚类和高层分析逻辑（这些由 Engine 完成）。

    返回的数据由上层 AttentionAllocationEngine 进一步计算比例、E_att、聚类结果等。
    """

    DB_NAME = "MLS"
    INTERACTION_COLLECTION = "Interaction"

    VERB_BASE = "https://legend-meta.com/xapi/verb/"
    VERBS = {
        "focused_on_resource": VERB_BASE + "focused-on-resource",
        "observed_peer": VERB_BASE + "observed-peer",
        "answered": VERB_BASE + "answered",
        "passed": VERB_BASE + "passed",
        "completed": VERB_BASE + "completed",
    }

    # 扩展字段 key
    EXT_UNIT_TYPE = "https://legend-meta.com/xapi/ext/unit-type"
    EXT_FOCUS_TARGET = "https://legend-meta.com/xapi/ext/focus-target-id"

    def __init__(
        self,
        batch_size: int = 5000,
        course_chunk_size: int = 200,
        mongo_operator: Optional[Any] = None,
    ) -> None:
        """
        Args:
            batch_size: 单次从 MongoDB 服务器取回的文档数量（游标 batch_size）
            course_chunk_size: 每次查询时 _course_uid 的分片大小，避免 $in 太长
            mongo_operator: 可注入的 MongoDBOperator 实例；不传则由基类内部创建
        """
        super().__init__(mongo_operator=mongo_operator)
        self.batch_size = batch_size
        self.course_chunk_size = course_chunk_size

    # ==================================================================
    # 公共接口
    # ==================================================================
    def load_metrics_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, object]],
        Dict[str, int],
        Dict[str, Set[str]],
    ]:
        """
        为一批学习者准备注意力分配分析所需的 **原始聚合数据**。

        Returns
        -------
        raw_by_lc:
            (lrn_uid, crs_uid) -> {
                "durations": {
                    "text": float, "visual": float,
                    "example": float, "ui_other": float,
                },
                "first_counts": {
                    "text": int, "visual": int, "example": int,
                },
                "perf_sum": float,     # performance 之和
                "perf_cnt": int,       # performance 样本数
            }

        learners_per_course:
            crs_uid -> 该课程参与行为的去重学习者数量
            （包含所有在该课程有 focused/observed/answered/... 行为的学习者）

        learner_courses_map:
            learner_uid -> 该学习者涉及到的课程集合
            （只保留出现在 raw_by_lc 中的课程）
        """
        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        if not learner_uids:
            logger.info(
                "AttentionAllocationRepository.load_metrics_for_learners: 空的学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "AttentionAllocationRepository: 开始准备原始数据，目标学习者数: %d",
            len(learner_uids),
        )

        # 1. 找出这些学习者参与过的课程
        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "AttentionAllocationRepository: 找到关联课程数: %d（仅包含与目标学习者有关的课程）",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "AttentionAllocationRepository: 目标学习者在 Interaction 中没有任何行为记录。"
            )
            return {}, {}, learner_courses_map

        # 2. 对这些课程在 Interaction 中聚合基础统计量
        raw_by_lc, learners_per_course = self._aggregate_raw_data_for_courses(all_courses)
        logger.info(
            "AttentionAllocationRepository: 完成原始数据聚合，(learner, course) 对数: %d, 课程数: %d",
            len(raw_by_lc),
            len(learners_per_course),
        )

        # 3. 过滤 learner_courses_map，只保留真正有数据的课程
        filtered_map: Dict[str, Set[str]] = {}
        for lrn_uid, crs_set in learner_courses_map.items():
            valid_courses = {
                crs_uid
                for crs_uid in crs_set
                if (lrn_uid, crs_uid) in raw_by_lc
            }
            if valid_courses:
                filtered_map[lrn_uid] = valid_courses

        return raw_by_lc, learners_per_course, filtered_map

    # ==================================================================
    # 内部工具：AOI 分类
    # ==================================================================
    def _categorize_aoi(self, target_id: Optional[str]) -> str:
        """
        根据 focus-target-id 粗略划分 AOI 类型。
        返回值:
            - text / visual / example / ui_other
        """
        if not target_id:
            return "ui_other"

        tid = target_id.lower()

        # 文本区域
        if any(key in tid for key in ("subtitle", "caption", "text", "label", "title")):
            return "text"

        # 图像 / 模型 / 主屏等
        if any(
            key in tid
            for key in ("diagram", "image", "picture", "screen", "model")
        ) or tid.startswith("vr-object") or tid.startswith("ar-object"):
            return "visual"

        # 提示 / 示例 / 解答
        if any(
            key in tid
            for key in ("hint", "tip", "example", "demo", "solution", "explanation")
        ):
            return "example"

        # 其他界面元素
        return "ui_other"

    # ==================================================================
    # 内部工具：通过基类 aggregate 访问 Mongo
    # ==================================================================
    def _iterate_events(
        self,
        match_query: Dict[str, Any],
        projection: Dict[str, Any],
    ) -> Iterable[Dict[str, Any]]:
        """
        通用迭代器：基于 MongoDBBaseRepository.aggregate 封装获取文档。

        为了利用已有索引（例如 idx_course_verb_lrn），match_query 中一般至少包含:
            - "_course_uid": 某值或 {"$in": [...]}
            - "verb.id": 某值或 {"$in": [...]}
        """
        pipeline = [
            {"$match": match_query},
            {"$project": projection},
        ]
        # 这里直接使用基类提供的 aggregate 方法，
        # 底层由 MongoDBOperator.aggregate(collection, pipeline) 实现。
        docs = self.aggregate(self.INTERACTION_COLLECTION, pipeline)
        for doc in docs:
            yield doc

    # ==================================================================
    # 内部工具：course 列表
    # ==================================================================
    def _get_courses_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        使用复合索引（例如 idx_lrn_verb_course）：
            key: { _lrn_uid: 1, 'verb.id': 1, _course_uid: 1 }

        pipeline:
            match _lrn_uid in learner_uids AND verb.id in [相关 verb]
            group by (lrn_uid, course_uid)

        返回:
            learner_courses_map: learner_uid -> set(course_uid)
            all_courses: 所有课程 uid 集合
        """
        verb_list = [
            self.VERBS["focused_on_resource"],
            self.VERBS["observed_peer"],
            self.VERBS["answered"],
            self.VERBS["passed"],
            self.VERBS["completed"],
        ]

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

        # 使用基类的 aggregate 包装，而不是直接拿 collection
        docs = self.aggregate(self.INTERACTION_COLLECTION, pipeline)
        for doc in docs:
            _id = doc.get("_id") or {}
            lrn_uid = _id.get("lrn_uid")
            crs_uid = _id.get("course_uid")
            if not lrn_uid or not crs_uid:
                continue
            learner_courses_map[lrn_uid].add(crs_uid)
            all_courses.add(crs_uid)

        return learner_courses_map, all_courses

    # ==================================================================
    # 内部工具：按课程聚合基础统计量（不做算法分析）
    # ==================================================================
    def _aggregate_raw_data_for_courses(
        self, course_uids: Set[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, object]],
        Dict[str, int],
    ]:
        """
        对若干课程的 Interaction 记录做一次性聚合，生成基础统计量：

        raw_by_lc[(lrn_uid, crs_uid)] = {
            "durations": {
                "text": float, "visual": float,
                "example": float, "ui_other": float,
            },
            "first_counts": {
                "text": int, "visual": int, "example": int,
            },
            "perf_sum": float,
            "perf_cnt": int,
        }

        learners_per_course[crs_uid] = 去重后的学习者数量
        """
        if not course_uids:
            return {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        # AOI 时长累积
        aoi_durations: Dict[
            Tuple[str, str],
            Dict[str, float],
        ] = defaultdict(
            lambda: {"text": 0.0, "visual": 0.0, "example": 0.0, "ui_other": 0.0}
        )

        # 首注视 AOI： (lrn_uid, crs_uid, unit_key) -> (timestamp, aoi_type)
        first_aoi: Dict[Tuple[str, str, str], Tuple[datetime, str]] = {}

        # performance 统计： (lrn_uid, crs_uid) -> {sum, cnt}
        perf_stats: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"sum": 0.0, "cnt": 0.0}
        )

        # 课程参与者统计
        learners_per_course: Dict[str, Set[str]] = defaultdict(set)

        focus_projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "result.duration": 1,
            "context.extensions": 1,
            "object.id": 1,
            "timestamp": 1,
        }
        observed_projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "result.duration": 1,
        }
        perf_projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "result.success": 1,
            "result.completion": 1,
        }

        # ---------- 1) focused-on-resource ----------
        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1) * self.course_chunk_size
            ]
            logger.info(
                "AttentionAllocationRepository: 读取 focused-on-resource 事件，课程分片 %d/%d，课程数: %d",
                chunk_idx + 1,
                total_chunks,
                len(sub_courses),
            )
            focus_query = {
                "_course_uid": {"$in": sub_courses},
                "verb.id": self.VERBS["focused_on_resource"],
            }

            event_cnt = 0
            for doc in self._iterate_events(focus_query, focus_projection):
                event_cnt += 1
                lrn_uid = doc.get("_lrn_uid")
                crs_uid = doc.get("_course_uid")
                if not lrn_uid or not crs_uid:
                    continue

                learners_per_course[crs_uid].add(lrn_uid)

                result = doc.get("result") or {}
                duration_str = result.get("duration")
                duration_sec = parse_iso8601_duration_seconds(duration_str)
                if duration_sec is None or duration_sec <= 0:
                    continue

                context = doc.get("context") or {}
                ctx_ext = context.get("extensions") or {}
                target_id = ctx_ext.get(self.EXT_FOCUS_TARGET)
                aoi_type = self._categorize_aoi(target_id)

                ts_str = doc.get("timestamp")
                try:
                    ts = (
                        datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts_str
                        else datetime.utcnow()
                    )
                except Exception:
                    ts = datetime.utcnow()

                obj = doc.get("object") or {}
                obj_id = obj.get("id")
                if obj_id:
                    unit_key = obj_id
                else:
                    unit_type = ctx_ext.get(self.EXT_UNIT_TYPE, "unknown")
                    unit_key = f"{crs_uid}:{unit_type}"

                lc_key = (lrn_uid, crs_uid)

                # 累加 AOI 时长
                aoi_durations[lc_key][aoi_type] += float(duration_sec)

                # 记录首注视 AOI
                fu_key = (lrn_uid, crs_uid, unit_key)
                prev = first_aoi.get(fu_key)
                if prev is None or ts < prev[0]:
                    first_aoi[fu_key] = (ts, aoi_type)

            logger.info(
                "AttentionAllocationRepository: 完成 focused-on-resource 事件处理，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        # ---------- 2) observed-peer 视为 example ----------
        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1) * self.course_chunk_size
            ]
            logger.info(
                "AttentionAllocationRepository: 读取 observed-peer 事件，课程分片 %d/%d，课程数: %d",
                chunk_idx + 1,
                total_chunks,
                len(sub_courses),
            )
            observed_query = {
                "_course_uid": {"$in": sub_courses},
                "verb.id": self.VERBS["observed_peer"],
            }

            event_cnt = 0
            for doc in self._iterate_events(observed_query, observed_projection):
                event_cnt += 1
                lrn_uid = doc.get("_lrn_uid")
                crs_uid = doc.get("_course_uid")
                if not lrn_uid or not crs_uid:
                    continue
                learners_per_course[crs_uid].add(lrn_uid)

                result = doc.get("result") or {}
                duration_str = result.get("duration")
                duration_sec = parse_iso8601_duration_seconds(duration_str)
                if duration_sec is None or duration_sec <= 0:
                    continue

                lc_key = (lrn_uid, crs_uid)
                # 观摩同伴视作示例类
                aoi_durations[lc_key]["example"] += float(duration_sec)

            logger.info(
                "AttentionAllocationRepository: 完成 observed-peer 事件处理，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        # ---------- 3) performance 相关事件 ----------
        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1) * self.course_chunk_size
            ]
            logger.info(
                "AttentionAllocationRepository: 读取 performance 事件(answered/passed/completed)，课程分片 %d/%d，课程数: %d",
                chunk_idx + 1,
                total_chunks,
                len(sub_courses),
            )
            perf_query = {
                "_course_uid": {"$in": sub_courses},
                "verb.id": {
                    "$in": [
                        self.VERBS["answered"],
                        self.VERBS["passed"],
                        self.VERBS["completed"],
                    ]
                },
            }

            event_cnt = 0
            for doc in self._iterate_events(perf_query, perf_projection):
                event_cnt += 1
                lrn_uid = doc.get("_lrn_uid")
                crs_uid = doc.get("_course_uid")
                if not lrn_uid or not crs_uid:
                    continue
                learners_per_course[crs_uid].add(lrn_uid)

                result = doc.get("result") or {}
                success = result.get("success")
                completion = result.get("completion")
                if success is None and completion is None:
                    continue

                if success is None:
                    val = 1.0 if completion else 0.0
                else:
                    val = 1.0 if bool(success) else 0.0

                lc_key = (lrn_uid, crs_uid)
                perf_stats[lc_key]["sum"] += val
                perf_stats[lc_key]["cnt"] += 1.0

            logger.info(
                "AttentionAllocationRepository: 完成 performance 事件处理，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        # ---------- 4) 汇总为 (lrn, crs) 的基础统计 ----------
        first_counts: Dict[
            Tuple[str, str],
            Dict[str, int],
        ] = defaultdict(lambda: {"text": 0, "visual": 0, "example": 0})

        for (lrn_uid, crs_uid, unit_key), (ts, aoi_type) in first_aoi.items():
            lc_key = (lrn_uid, crs_uid)
            if aoi_type in ("text", "visual", "example"):
                first_counts[lc_key][aoi_type] += 1

        raw_by_lc: Dict[Tuple[str, str], Dict[str, object]] = {}

        for lc_key, dur_dict in aoi_durations.items():
            lrn_uid, crs_uid = lc_key
            if (
                dur_dict["text"]
                + dur_dict["visual"]
                + dur_dict["example"]
                + dur_dict["ui_other"]
            ) <= 0:
                continue

            fc = first_counts.get(lc_key, {"text": 0, "visual": 0, "example": 0})
            perf = perf_stats.get(lc_key, {"sum": 0.0, "cnt": 0.0})

            raw_by_lc[lc_key] = {
                "durations": {
                    "text": float(dur_dict["text"]),
                    "visual": float(dur_dict["visual"]),
                    "example": float(dur_dict["example"]),
                    "ui_other": float(dur_dict["ui_other"]),
                },
                "first_counts": {
                    "text": int(fc.get("text", 0)),
                    "visual": int(fc.get("visual", 0)),
                    "example": int(fc.get("example", 0)),
                },
                "perf_sum": float(perf["sum"]),
                "perf_cnt": int(perf["cnt"]),
            }

        learners_per_course_count = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course.items()
        }

        logger.info(
            "AttentionAllocationRepository: 原始统计整理完成，(lrn, crs) 有效条目: %d",
            len(raw_by_lc),
        )

        return raw_by_lc, learners_per_course_count
