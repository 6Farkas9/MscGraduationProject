# BackEnd/app/data_access/profiling/feedback_orientation_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository

logger = logging.getLogger(__name__)


class FeedbackOrientationRepository(MongoDBBaseRepository):
    """
    反馈敏感度与数据使用能力（feedback_orientation）维度的数据仓库。

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
        "reviewed_feedback": VERB_BASE + "reviewed-feedback",
        "requested_support": VERB_BASE + "requested-support",
        "answered": VERB_BASE + "answered",
        "completed": VERB_BASE + "completed",
        "performed_procedure_step": VERB_BASE + "performed-procedure-step",
    }

    # 与原脚本保持一致的窗口参数
    FEEDBACK_WINDOW_MINUTES = 10   # 错误后多久内查看反馈算“使用反馈”
    POST_FEEDBACK_K = 3            # 反馈后取多少次任务结果来比较正确率

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
        为若干学习者准备“反馈敏感度与数据使用能力”分析所需的课程级特征。

        返回:
            metrics_by_lc:
                (lrn_uid, crs_uid) -> {
                    "feedback_view_count": int,
                    "feedback_view_rate": float,
                    "feedback_view_type_dist": Dict[str, float],
                    "support_view_count": int,
                    "support_view_rate": float,
                    "improvement_after_feedback": float,
                    "opportunity_count": int,
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
                "FeedbackOrientationRepository.load_metrics_for_learners: 空的学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "FeedbackOrientationRepository: 开始准备反馈敏感度原始数据，目标学习者数: %d",
            len(learner_uids),
        )

        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "FeedbackOrientationRepository: 与目标学习者相关的课程数: %d",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "FeedbackOrientationRepository: 目标学习者在反馈相关事件上没有任何课程记录。"
            )
            return {}, {}, learner_courses_map

        raw_stats_by_lc, learners_per_course = self._aggregate_raw_stats_for_courses(
            all_courses
        )

        logger.info(
            "FeedbackOrientationRepository: 已完成原始统计聚合，(lrn, crs) 粗粒度条目数: %d",
            len(raw_stats_by_lc),
        )

        metrics_by_lc = self._compute_metrics_from_raw(raw_stats_by_lc)

        logger.info(
            "FeedbackOrientationRepository: 指标计算完成，(lrn, crs) 有效条目数: %d",
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
        对一批课程的相关事件进行原始统计聚合，生成课程级基础统计。

        返回:
            raw_stats_by_lc[(lrn_uid, crs_uid)] = {
                "feedback_view_count": int,
                "support_view_count": int,
                "feedback_view_type_counts": Dict[str, int],
                "opportunity_count": int,
                "timeline": List[Tuple[str, datetime, Optional[bool]]],
            }

            learners_per_course[crs_uid] = 课程内去重学习者数量
        """
        if not course_uids:
            return {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        raw_stats_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "feedback_view_count": 0,
                "support_view_count": 0,
                "feedback_view_type_counts": defaultdict(int),
                "opportunity_count": 0,
                "timeline": [],  # (typ, ts, success: Optional[bool])
            }
        )
        learners_per_course_sets: Dict[str, Set[str]] = defaultdict(set)

        projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "timestamp": 1,
            "result": 1,
            "context": 1,
        }
        verb_list = list(self.VERBS.values())

        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1) * self.course_chunk_size
            ]
            logger.info(
                "FeedbackOrientationRepository: 读取事件，课程分片 %d/%d，课程数: %d",
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
                timestamp = doc.get("timestamp")
                ts: Optional[datetime]
                if isinstance(timestamp, datetime):
                    ts = timestamp
                elif isinstance(timestamp, str):
                    try:
                        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    except Exception:
                        ts = None
                else:
                    ts = None

                key_lc = (lrn_uid, crs_uid)
                stat = raw_stats_by_lc[key_lc]

                # 反馈查看
                if verb_id == self.VERBS["reviewed_feedback"]:
                    stat["feedback_view_count"] += 1

                    # 反馈查看类型（unit/course/group/...）
                    ctx = doc.get("context") or {}
                    res = doc.get("result") or {}
                    ext_ctx = (ctx.get("extensions") or {}) if isinstance(ctx, dict) else {}
                    ext_res = res.get("extensions") or {}
                    view_type = (
                        ext_res.get("feedback-view-type")
                        or ext_ctx.get("feedback-view-type")
                        or "unknown"
                    )
                    view_type = str(view_type)
                    stat["feedback_view_type_counts"][view_type] += 1

                    if ts:
                        stat["timeline"].append(("feedback", ts, None))

                # 即时支持使用（requested-support）
                elif verb_id == self.VERBS["requested_support"]:
                    stat["support_view_count"] += 1
                    # support 不计入“反馈后正确率提升”的时间线，所以这里不加入 timeline

                # 任务事件：产生反馈机会 + 用于错误→反馈→提升 序列
                elif verb_id in (
                    self.VERBS["answered"],
                    self.VERBS["completed"],
                    self.VERBS["performed_procedure_step"],
                ):
                    result = doc.get("result") or {}
                    success = result.get("success")
                    completion = result.get("completion")
                    if success is None and completion is None:
                        continue
                    ok = bool(success) if success is not None else bool(completion)

                    stat["opportunity_count"] += 1

                    if ts:
                        stat["timeline"].append(("task", ts, ok))

            logger.info(
                "FeedbackOrientationRepository: 完成事件读取，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        learners_per_course = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course_sets.items()
        }

        logger.info(
            "FeedbackOrientationRepository: 原始统计聚合完成，课程数: %d",
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
        将原始统计转换为课程级反馈指标：
            feedback_view_rate / support_view_rate /
            improvement_after_feedback / feedback_view_type_dist
        """
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = {}

        window_delta = timedelta(minutes=self.FEEDBACK_WINDOW_MINUTES)

        for key, stat in raw_stats_by_lc.items():
            feedback_view_count = int(stat["feedback_view_count"])
            support_view_count = int(stat["support_view_count"])
            type_counts: Dict[str, int] = stat["feedback_view_type_counts"]
            opportunity_count = int(stat["opportunity_count"])
            timeline = stat["timeline"]

            # 没有任何机会与行为则忽略该 (lrn, crs)
            if (
                opportunity_count <= 0
                and feedback_view_count <= 0
                and support_view_count <= 0
            ):
                continue

            # 1) 查看类型分布
            total_type = sum(type_counts.values()) or 1
            feedback_view_type_dist = {
                t: c / float(total_type) for t, c in type_counts.items()
            }

            # 2) 机会归一化频率
            if opportunity_count > 0:
                feedback_view_rate = feedback_view_count / float(opportunity_count)
                support_view_rate = support_view_count / float(opportunity_count)
            else:
                feedback_view_rate = 0.0
                support_view_rate = 0.0

            # 3) 反馈后正确率提升（与原脚本相同逻辑，仅在 Repository 里计算数值）
            improvements: List[float] = []

            # timeline: List[(typ, ts, ok)]
            # 先排序
            timeline_sorted = sorted(timeline, key=lambda x: x[1])

            for i, (typ, ts, ok) in enumerate(timeline_sorted):
                if typ != "task" or ok is True:
                    continue  # 只关注错误任务

                # 找错误后窗口内的最近 feedback
                fb_idx: Optional[int] = None
                for j in range(i + 1, len(timeline_sorted)):
                    typ2, ts2, _ = timeline_sorted[j]
                    if ts2 - ts > window_delta:
                        break
                    if typ2 == "feedback":
                        fb_idx = j
                        break

                if fb_idx is None:
                    continue

                # 取反馈前后各 K 个 task 结果
                pre: List[bool] = []
                post: List[bool] = []

                # pre
                k = i - 1
                while k >= 0 and len(pre) < self.POST_FEEDBACK_K:
                    typ2, _ts2, ok2 = timeline_sorted[k]
                    if typ2 == "task":
                        pre.append(bool(ok2))
                    k -= 1

                # post
                k = fb_idx + 1
                while k < len(timeline_sorted) and len(post) < self.POST_FEEDBACK_K:
                    typ2, _ts2, ok2 = timeline_sorted[k]
                    if typ2 == "task":
                        post.append(bool(ok2))
                    k += 1

                if pre and post:
                    pre_acc = sum(1 for x in pre if x) / float(len(pre))
                    post_acc = sum(1 for x in post if x) / float(len(post))
                    improvements.append(post_acc - pre_acc)

            if improvements:
                improvement_after_feedback = sum(improvements) / float(len(improvements))
            else:
                improvement_after_feedback = 0.0

            metrics_by_lc[key] = {
                "feedback_view_count": feedback_view_count,
                "feedback_view_rate": float(feedback_view_rate),
                "feedback_view_type_dist": feedback_view_type_dist,
                "support_view_count": support_view_count,
                "support_view_rate": float(support_view_rate),
                "improvement_after_feedback": float(improvement_after_feedback),
                "opportunity_count": opportunity_count,
            }

        return metrics_by_lc
