# BackEnd/app/data_access/profiling/engagement_persistence_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository
from app.shared.utils.repository_mixins import parse_iso8601_duration_seconds

logger = logging.getLogger(__name__)


class EngagementPersistenceRepository(MongoDBBaseRepository):
    """
    行为投入度与坚持性（engagement_persistence）维度的数据仓库。

    职责：
    - 仅负责从 MongoDB.MLS.Interaction 读取与该维度相关的 xAPI 行为事件；
    - 利用现有复合索引：
        * idx_lrn_verb_course: {_lrn_uid, 'verb.id', _course_uid}
        * idx_course_verb_lrn: {_course_uid, 'verb.id', _lrn_uid}
      进行课程发现和按课程分批扫描；
    - 只输出课程级的基础数值特征，不进行聚类和标签判定。
    """

    DB_NAME = "MLS"
    INTERACTION_COLLECTION = "Interaction"

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "initialized": VERB_BASE + "initialized",
        "completed": VERB_BASE + "completed",
        "answered": VERB_BASE + "answered",
        "performed_procedure_step": VERB_BASE + "performed-procedure-step",
        "explored_extension": VERB_BASE + "explored-extension",
        "remained_idle": VERB_BASE + "remained-idle",
        "exchanged_value": VERB_BASE + "exchanged-value",
    }

    CTX_EXT_BASE = "https://legend-meta.com/xapi/ext/"
    EXT_STEP_ID = CTX_EXT_BASE + "step-id"
    EXT_VALUE_CHANGE = CTX_EXT_BASE + "value-change"

    def __init__(
        self,
        batch_size: int = 5000,
        course_chunk_size: int = 200,
        mongo_operator: Optional[Any] = None,
    ) -> None:
        """
        Args:
            batch_size: Mongo 游标 batch_size（逻辑参数，目前通过 aggregate 读取）。
            course_chunk_size: 每次查询 _course_uid 的分片大小，避免 $in 过大。
            mongo_operator: 可注入的 MongoDBOperator 实例；不传则由基类内部创建。
        """
        super().__init__(mongo_operator=mongo_operator)
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
        为若干学习者准备“行为投入度与坚持性”分析所需的课程级数值特征。

        返回:
            metrics_by_lc:
                (lrn_uid, crs_uid) -> {
                    "completion_rate": float,
                    "interaction_per_unit": float,
                    "retry_rate": float,
                    "extension_rate": float,
                    "idle_ratio": float,
                    "value_rate": float,
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
                "EngagementPersistenceRepository.load_metrics_for_learners: 空的学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "EngagementPersistenceRepository: 开始准备原始行为投入度数据，目标学习者数: %d",
            len(learner_uids),
        )

        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "EngagementPersistenceRepository: 与目标学习者相关的课程数: %d",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "EngagementPersistenceRepository: 目标学习者在相关事件上没有任何课程记录。"
            )
            return {}, {}, learner_courses_map

        raw_stats_by_lc, learners_per_course = self._aggregate_raw_stats_for_courses(
            all_courses
        )

        logger.info(
            "EngagementPersistenceRepository: 已完成原始统计聚合，(lrn, crs) 粗粒度条目数: %d",
            len(raw_stats_by_lc),
        )

        metrics_by_lc = self._compute_metrics_from_raw(raw_stats_by_lc)

        logger.info(
            "EngagementPersistenceRepository: 指标计算完成，(lrn, crs) 有效条目数: %d",
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
    # 内部: 课程发现（基于 MongoDBBaseRepository.aggregate）
    # ------------------------------------------------------------------
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
            all_courses: 所有相关课程 uid 集合
        """
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
        通用迭代器：按指定查询和投影，使用 aggregate 管道返回文档。

        为了利用 idx_course_verb_lrn 复合索引，match_query 中应包含：
            - "_course_uid": 某值或 {"$in": [...]}；
            - "verb.id": 某值或 {"$in": [...]}。
        """
        pipeline = [
            {"$match": match_query},
            {"$project": projection},
        ]
        docs = self.aggregate(self.INTERACTION_COLLECTION, pipeline)
        for doc in docs:
            yield doc

    def _aggregate_raw_stats_for_courses(
        self, course_uids: Set[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
    ]:
        """
        对一批课程的相关事件进行原始统计聚合，生成课程级行为投入度/坚持性基础统计。

        返回:
            raw_stats_by_lc[(lrn_uid, crs_uid)] = {
                "units_started": set(),
                "units_completed": set(),
                "event_count": int,
                "active_time": float,
                "idle_time": float,
                "extension_count": int,
                "value_events": int,
                "value_change_sum": float,
                "q_fail_count": int,
                "q_fail_then_success": int,
                "step_fail_count": int,
                "step_fail_then_success": int,
            }

            learners_per_course[crs_uid] = 课程内去重学习者数量
        """
        if not course_uids:
            return {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        raw_stats_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "units_started": set(),
                "units_completed": set(),
                "event_count": 0,
                "active_time": 0.0,
                "idle_time": 0.0,
                "extension_count": 0,
                "value_events": 0,
                "value_change_sum": 0.0,
                "q_fail_count": 0,
                "q_fail_then_success": 0,
                "step_fail_count": 0,
                "step_fail_then_success": 0,
            }
        )
        learners_per_course_sets: Dict[str, Set[str]] = defaultdict(set)

        # 题目与步骤级事件序列（用于后续重试统计）
        question_events: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        step_events: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

        projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "result": 1,
            "context": 1,
            "object.id": 1,
            "_unt_uid": 1,
            "_type": 1,
            "timestamp": 1,
        }
        verb_list = list(self.VERBS.values())

        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1) * self.course_chunk_size
            ]
            logger.info(
                "EngagementPersistenceRepository: 读取事件，课程分片 %d/%d，课程数: %d",
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

                learners_per_course_sets[crs_uid].add(lrn_uid)

                verb_id = (doc.get("verb") or {}).get("id") or doc.get("verb.id")
                result = doc.get("result") or {}
                context = doc.get("context") or {}
                extensions = context.get("extensions") or {}
                obj_id = (doc.get("object") or {}).get("id") or doc.get("object.id")
                unt_uid = doc.get("_unt_uid")
                utype = doc.get("_type")
                timestamp = doc.get("timestamp") or ""

                key_lc = (lrn_uid, crs_uid)
                stat = raw_stats_by_lc[key_lc]

                # 标记单元参与（question / course-level 不计入单元集合）
                if unt_uid and utype and utype not in ("question", "course-level"):
                    stat["units_started"].add(unt_uid)

                # 事件计数（行为交互量）
                stat["event_count"] += 1

                # completed：完成率 + active_time
                if verb_id == self.VERBS["completed"]:
                    if unt_uid and utype and utype not in ("question", "course-level"):
                        if result.get("completion") is True:
                            stat["units_completed"].add(unt_uid)

                    dur_sec = parse_iso8601_duration_seconds(result.get("duration"))
                    if dur_sec is not None and dur_sec > 0:
                        stat["active_time"] += float(dur_sec)

                # performed-procedure-step：步骤级 active_time + retry 序列
                elif verb_id == self.VERBS["performed_procedure_step"]:
                    dur_sec = parse_iso8601_duration_seconds(result.get("duration"))
                    if dur_sec is not None and dur_sec > 0:
                        stat["active_time"] += float(dur_sec)

                    step_id = extensions.get(self.EXT_STEP_ID)
                    if step_id:
                        success_flag = bool(result.get("success"))
                        step_events[(lrn_uid, crs_uid, step_id)].append(
                            {"t": timestamp, "success": success_flag}
                        )

                # answered：题目级 retry 序列
                elif verb_id == self.VERBS["answered"]:
                    success_flag = bool(result.get("success"))
                    if obj_id:
                        question_events[(lrn_uid, crs_uid, obj_id)].append(
                            {"t": timestamp, "success": success_flag}
                        )

                # explored-extension：失败后的额外练习
                elif verb_id == self.VERBS["explored_extension"]:
                    stat["extension_count"] += 1

                # remained-idle：空闲时长
                elif verb_id == self.VERBS["remained_idle"]:
                    dur_sec = parse_iso8601_duration_seconds(result.get("duration"))
                    if dur_sec is not None and dur_sec > 0:
                        stat["idle_time"] += float(dur_sec)

                # exchanged-value：价值交换行为
                elif verb_id == self.VERBS["exchanged_value"]:
                    stat["value_events"] += 1
                    value_change = extensions.get(self.EXT_VALUE_CHANGE)
                    try:
                        if value_change is not None:
                            stat["value_change_sum"] += float(value_change)
                    except (TypeError, ValueError):
                        pass

            logger.info(
                "EngagementPersistenceRepository: 完成事件读取，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        # ---------- 处理题目与步骤级重试行为 ----------
        logger.info("EngagementPersistenceRepository: 开始统计题目级重试行为。")
        for (lrn_uid, crs_uid, qid), seq in question_events.items():
            if not seq:
                continue
            seq_sorted = sorted(seq, key=lambda x: x["t"])
            had_wrong = False
            had_wrong_then_success = False
            for ev in seq_sorted:
                if not ev["success"]:
                    had_wrong = True
                elif ev["success"] and had_wrong:
                    had_wrong_then_success = True
                    break
            if had_wrong:
                raw_stats_by_lc[(lrn_uid, crs_uid)]["q_fail_count"] += 1
            if had_wrong_then_success:
                raw_stats_by_lc[(lrn_uid, crs_uid)]["q_fail_then_success"] += 1

        logger.info("EngagementPersistenceRepository: 开始统计步骤级重试行为。")
        for (lrn_uid, crs_uid, step_id), seq in step_events.items():
            if not seq:
                continue
            seq_sorted = sorted(seq, key=lambda x: x["t"])
            had_wrong = False
            had_wrong_then_success = False
            for ev in seq_sorted:
                if not ev["success"]:
                    had_wrong = True
                elif ev["success"] and had_wrong:
                    had_wrong_then_success = True
                    break
            if had_wrong:
                raw_stats_by_lc[(lrn_uid, crs_uid)]["step_fail_count"] += 1
            if had_wrong_then_success:
                raw_stats_by_lc[(lrn_uid, crs_uid)]["step_fail_then_success"] += 1

        learners_per_course = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course_sets.items()
        }

        logger.info(
            "EngagementPersistenceRepository: 重试行为统计完成，课程数: %d",
            len(learners_per_course),
        )
        return dict(raw_stats_by_lc), learners_per_course

    # ------------------------------------------------------------------
    # 内部: 从原始统计派生课程级指标
    # ------------------------------------------------------------------
    def _compute_metrics_from_raw(
        self,
        raw_stats_by_lc: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        将原始统计转换为课程级行为指标：
            completion_rate / interaction_per_unit /
            retry_rate / extension_rate / idle_ratio / value_rate
        """
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, float]] = {}

        for key, stat in raw_stats_by_lc.items():
            units_started = len(stat["units_started"])
            units_completed = len(stat["units_completed"])
            event_count = stat["event_count"]
            active_time = stat["active_time"]
            idle_time = stat["idle_time"]
            extension_count = stat["extension_count"]
            value_events = stat["value_events"]
            q_fail = stat["q_fail_count"]
            q_retry = stat["q_fail_then_success"]
            step_fail = stat["step_fail_count"]
            step_retry = stat["step_fail_then_success"]

            # 完全没有有效行为数据的窗口直接跳过
            if units_started == 0 and event_count == 0:
                continue

            # 完成率
            if units_started > 0:
                completion_rate = units_completed / float(units_started)
            else:
                completion_rate = 0.0

            # 单位单元交互量
            denom_units = float(units_started) if units_started > 0 else 1.0
            interaction_per_unit = event_count / denom_units

            # 重试率：题目+步骤维度 “先错后对”的比例
            total_fail = q_fail + step_fail
            total_retry = q_retry + step_retry
            if total_fail > 0:
                retry_rate = total_retry / float(total_fail)
            else:
                retry_rate = 0.5  # 没有失败，视为中性值

            # 失败后额外练习比例
            if q_fail > 0:
                extension_rate = extension_count / float(q_fail)
            else:
                extension_rate = extension_count / float(units_started + 1)

            # idle 比例
            total_time_for_idle = idle_time + active_time
            if total_time_for_idle > 0:
                idle_ratio = idle_time / float(total_time_for_idle)
            else:
                idle_ratio = 0.0

            # value 率（事件频次归一化到单元数）
            value_rate = value_events / denom_units

            metrics_by_lc[key] = {
                "completion_rate": float(completion_rate),
                "interaction_per_unit": float(interaction_per_unit),
                "retry_rate": float(retry_rate),
                "extension_rate": float(extension_rate),
                "idle_ratio": float(idle_ratio),
                "value_rate": float(value_rate),
            }

        return metrics_by_lc
