# BackEnd/app/data_access/profiling/slr_helpseeking_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository

logger = logging.getLogger(__name__)


class SRLHelpseekingRepository(MongoDBBaseRepository):
    """
    自我调节与求助策略（Self-Regulated Learning & Help-Seeking）维度的数据仓库。

    职责：
    - 仅负责从 MongoDB.MLS.Interaction 读取与该维度相关的 xAPI 行为事件；
    - 利用现有复合索引：
        * idx_lrn_verb_course: {_lrn_uid, 'verb.id', _course_uid}
        * idx_course_verb_lrn: {_course_uid, 'verb.id', _lrn_uid}
      进行课程发现和按课程分批扫描；
    - 以 (学习者, 课程, 题目) 粒度统计：
        错误/正确/求助（错误后求助 / 首答前求助）/反馈/扩展/反思；
    - 为每个 (lrn_uid, crs_uid) 计算 SRL_help_index 及其子指标；
    - 不进行聚类与标签判定（交给 engine）。
    """

    DB_NAME = "MLS"
    INTERACTION_COLLECTION = "Interaction"

    VERB_BASE = "https://legend-meta.com/xapi/verb/"
    ACTIVITY_TYPE_BASE = "https://legend-meta.com/xapi/activity-type/"

    VERBS = {
        "answered": VERB_BASE + "answered",
        "requested_support": VERB_BASE + "requested-support",
        "reviewed_feedback": VERB_BASE + "reviewed-feedback",
        "explored_extension": VERB_BASE + "explored-extension",
        "reflected_on_activity": VERB_BASE + "reflected-on-activity",
    }

    QUESTION_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "item"
    COURSE_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "course"

    def __init__(
        self,
        batch_size: int = 5000,
        course_chunk_size: int = 200,
        mongo_operator: Optional[Any] = None,
    ) -> None:
        """
        Args:
            batch_size: Mongo 游标 batch_size，控制单次从服务器端取回的文档数量。
            course_chunk_size: 每次查询 _course_uid 的分片大小，避免 $in 过大。
            mongo_operator: 可注入的 MongoDBOperator 实例；若为 None 则由基类内部创建。
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
        为若干学习者准备“自我调节与求助策略”分析所需的课程级特征。

        返回:
            metrics_by_lc:
                (lrn_uid, crs_uid) -> {
                    "srl_help_index": float in [0, 1],
                    # 下面是子指标：
                    "help_need_rate": float,
                    "early_help_ratio": float,
                    "no_help_success_ratio": float,
                    "feedback_density": float,
                    "extension_flag": int (0/1),
                    "reflection_flag": int (0/1),
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
                "SRLHelpseekingRepository.load_metrics_for_learners: 空学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "SRLHelpseekingRepository: 开始准备自我调节与求助策略原始数据，目标学习者数: %d",
            len(learner_uids),
        )

        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "SRLHelpseekingRepository: 与目标学习者相关的课程数: %d",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "SRLHelpseekingRepository: 目标学习者在相关事件上没有任何课程记录。"
            )
            return {}, {}, learner_courses_map

        course_stats, learners_per_course = self._aggregate_course_stats_for_courses(
            all_courses
        )

        logger.info(
            "SRLHelpseekingRepository: 已完成 (lrn, crs) 粗粒度统计，条目数: %d",
            len(course_stats),
        )

        metrics_by_lc = self._compute_srl_index_for_courses(course_stats)

        logger.info(
            "SRLHelpseekingRepository: SRL_help_index 计算完成，(lrn, crs) 有效条目数: %d",
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
    # 内部：课程发现（基于 MongoDBBaseRepository.aggregate）
    # ------------------------------------------------------------------
    def _get_courses_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        使用 idx_lrn_verb_course 复合索引：
            key: { _lrn_uid: 1, 'verb.id': 1, _course_uid: 1 }

        pipeline:
            match _lrn_uid in learner_uids
                  AND verb.id in 自我调节相关 verbs
            group by (lrn_uid, course_uid)
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
    # 内部：通用迭代器（基于 aggregate 的 $match+$project）
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

    # ------------------------------------------------------------------
    # 内部：xAPI 对象类型判断 & 比例计算
    # ------------------------------------------------------------------
    def _is_question_object(self, obj: Dict[str, Any]) -> bool:
        """
        判断 xAPI 对象是否为题目级活动。
        - id 形如: https://legend-meta.com/item/{question_uid}
        - 或 definition.type 为 QUESTION_ACTIVITY_TYPE
        """
        if not obj:
            return False
        obj_id = obj.get("id") or ""
        if obj_id.startswith("https://legend-meta.com/item/"):
            return True
        definition = obj.get("definition") or {}
        tp = definition.get("type")
        if tp == self.QUESTION_ACTIVITY_TYPE:
            return True
        return False

    def _is_course_object(self, obj: Dict[str, Any]) -> bool:
        """
        判断 xAPI 对象是否为课程级活动。
        - id 形如 https://legend-meta.com/course/{course_uid}
        - 或 definition.type 为 COURSE_ACTIVITY_TYPE
        """
        if not obj:
            return False
        obj_id = obj.get("id") or ""
        if obj_id.startswith("https://legend-meta.com/course/"):
            return True
        definition = obj.get("definition") or {}
        tp = definition.get("type")
        if tp == self.COURSE_ACTIVITY_TYPE:
            return True
        return False

    @staticmethod
    def _safe_ratio(num: float, den: float) -> float:
        if den is None or den == 0:
            return 0.0
        return float(num) / float(den)

    # ------------------------------------------------------------------
    # 内部：按课程分片聚合 (lrn, crs) 统计
    # ------------------------------------------------------------------
    def _aggregate_course_stats_for_courses(
        self, course_uids: Set[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
    ]:
        """
        对一批课程的自我调节/求助相关事件进行聚合，生成：

        course_stats[(lrn_uid, crs_uid)] = {
            "questions": {
                q_id: {
                    "attempts": int,
                    "wrong": int,
                    "correct": int,
                    "help_total": int,
                    "help_after_error": int,
                    "help_before_attempt": int,
                },
                ...
            },
            "feedback_question": int,
            "feedback_course": int,
            "extension_cnt": int,
            "reflection_cnt": int,
        }

        learners_per_course[crs_uid] = 去重学习者数量。
        """
        if not course_uids:
            return {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        course_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
        learners_per_course_sets: Dict[str, Set[str]] = defaultdict(set)

        projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "object": 1,
            "result": 1,
            "timestamp": 1,
        }
        verb_list = list(self.VERBS.values())

        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1) * self.course_chunk_size
            ]

            logger.info(
                "SRLHelpseekingRepository: 读取事件，课程分片 %d/%d，课程数: %d",
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
                "SRLHelpseekingRepository: 完成 Mongo 读取，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                len(events),
            )

            # 为保持与原脚本逻辑一致，在分片内按 timestamp 排序后再处理
            events.sort(key=lambda d: str(d.get("timestamp") or ""))

            for doc in events:
                lrn_uid = doc.get("_lrn_uid")
                crs_uid = doc.get("_course_uid")
                if not lrn_uid or not crs_uid:
                    continue

                learners_per_course_sets[crs_uid].add(lrn_uid)

                key_lc = (lrn_uid, crs_uid)
                if key_lc not in course_stats:
                    course_stats[key_lc] = {
                        "questions": defaultdict(
                            lambda: {
                                "attempts": 0,
                                "wrong": 0,
                                "correct": 0,
                                "help_total": 0,
                                "help_after_error": 0,
                                "help_before_attempt": 0,
                            }
                        ),
                        "feedback_question": 0,
                        "feedback_course": 0,
                        "extension_cnt": 0,
                        "reflection_cnt": 0,
                    }

                stat = course_stats[key_lc]

                verb_id = (doc.get("verb") or {}).get("id")
                obj = doc.get("object") or {}
                result = doc.get("result") or {}

                is_q = self._is_question_object(obj)
                is_c = self._is_course_object(obj)
                obj_id = obj.get("id") or ""

                # 1) 题目作答
                if verb_id == self.VERBS["answered"] and is_q:
                    qstat = stat["questions"][obj_id]
                    qstat["attempts"] += 1
                    success = result.get("success")
                    if success is True:
                        qstat["correct"] += 1
                    else:
                        qstat["wrong"] += 1

                # 2) 求助
                elif verb_id == self.VERBS["requested_support"] and is_q:
                    qstat = stat["questions"][obj_id]
                    qstat["help_total"] += 1
                    if qstat["attempts"] == 0:
                        qstat["help_before_attempt"] += 1
                    else:
                        if qstat["wrong"] > 0 and qstat["correct"] == 0:
                            qstat["help_after_error"] += 1

                # 3) 查看反馈
                elif verb_id == self.VERBS["reviewed_feedback"]:
                    if is_q:
                        stat["feedback_question"] += 1
                    elif is_c:
                        stat["feedback_course"] += 1
                    else:
                        stat["feedback_course"] += 1

                # 4) 使用扩展/补救单元
                elif verb_id == self.VERBS["explored_extension"]:
                    stat["extension_cnt"] += 1

                # 5) 提交反思
                elif verb_id == self.VERBS["reflected_on_activity"]:
                    stat["reflection_cnt"] += 1

        learners_per_course = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course_sets.items()
        }

        logger.info(
            "SRLHelpseekingRepository: 粗粒度统计聚合完成，课程数: %d，(lrn, crs) 组合数: %d",
            len(learners_per_course),
            len(course_stats),
        )

        return course_stats, learners_per_course

    # ------------------------------------------------------------------
    # 内部：单课程 (lrn, crs) → SRL_help_index 计算（移植自原脚本）
    # ------------------------------------------------------------------
    def _compute_srl_help_index_for_course(
        self, course_stat: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        从单个 (学习者, 课程) 的题目级统计中计算 SRL_help_index 及各子特征。

        逻辑完全参考原 analyze_slr_helpseeking.py 中的
        compute_srl_help_index_for_course。
        """
        questions: Dict[str, Dict[str, Any]] = course_stat.get("questions") or {}
        feedback_question = course_stat.get("feedback_question", 0)
        feedback_course = course_stat.get("feedback_course", 0)
        extension_cnt = course_stat.get("extension_cnt", 0)
        reflection_cnt = course_stat.get("reflection_cnt", 0)

        if not questions:
            # 没有任何题目行为时，很难评估求助策略，返回中性偏低值
            srl_index = 0.4
            feature_summary = {
                "help_need_rate": 0.0,
                "early_help_ratio": 0.0,
                "no_help_success_ratio": 0.0,
                "feedback_density": 0.0,
                "extension_flag": 1 if extension_cnt > 0 else 0,
                "reflection_flag": 1 if reflection_cnt > 0 else 0,
            }
            return srl_index, feature_summary

        num_questions = 0
        num_error_questions = 0
        total_wrong_attempts = 0
        total_help_events = 0
        total_help_after_error = 0
        total_help_before_attempt = 0

        num_success_questions = 0
        num_success_no_help_questions = 0

        for q_id, qstat in questions.items():
            attempts = qstat.get("attempts", 0)
            wrong = qstat.get("wrong", 0)
            correct = qstat.get("correct", 0)
            help_total = qstat.get("help_total", 0)
            help_after_error = qstat.get("help_after_error", 0)
            help_before_attempt = qstat.get("help_before_attempt", 0)

            if attempts <= 0:
                continue

            num_questions += 1
            total_wrong_attempts += wrong
            total_help_events += help_total
            total_help_after_error += help_after_error
            total_help_before_attempt += help_before_attempt

            if wrong > 0:
                num_error_questions += 1

            if correct > 0:
                num_success_questions += 1
                if help_total == 0:
                    num_success_no_help_questions += 1

        # 1) 错误后的求助比例
        if num_error_questions > 0:
            help_need_rate = self._safe_ratio(total_help_after_error, num_error_questions)
        else:
            help_need_rate = 0.5  # 中性值

        # 2) 过早求助比例
        if total_help_events > 0:
            early_help_ratio = self._safe_ratio(
                total_help_before_attempt, total_help_events
            )
        else:
            early_help_ratio = 0.0

        # 3) 无帮助成功比例
        if num_success_questions > 0:
            no_help_success_ratio = self._safe_ratio(
                num_success_no_help_questions, num_success_questions
            )
        else:
            no_help_success_ratio = 0.0

        # 4) 反馈密度（题目数为基础）
        total_feedback = feedback_question + feedback_course
        if num_questions > 0:
            feedback_density = self._safe_ratio(total_feedback, num_questions)
        else:
            feedback_density = 0.0

        extension_flag = 1 if extension_cnt > 0 else 0
        reflection_flag = 1 if reflection_cnt > 0 else 0

        # ---------- 子分数映射到 [0,1] ----------
        from math import exp

        # 1) 适应性求助得分：help_need_rate 理想 ~0.6
        x = help_need_rate
        adaptivity_score = exp(-((x - 0.6) ** 2) / (2 * (0.3 ** 2)))
        adaptivity_score = max(0.0, min(1.0, adaptivity_score))

        # 2) 过早求助惩罚
        early_score = 1.0 - early_help_ratio
        early_score = max(0.0, min(1.0, early_score))

        # 3) 独立解决得分：向 0.5 轻微收缩
        independent_score = 0.5 + 0.5 * (no_help_success_ratio - 0.5)
        independent_score = max(0.0, min(1.0, independent_score))

        # 4) 反馈得分：以 0.5 次/题为上限
        feedback_score = min(feedback_density / 0.5, 1.0)

        extension_score = float(extension_flag)
        reflection_score = float(reflection_flag)

        # ---------- 综合加权 ----------
        w1 = 0.25  # 适应性求助
        w2 = 0.15  # 不过早依赖
        w3 = 0.25  # 独立解决
        w4 = 0.15  # 反馈使用
        w5 = 0.10  # 补救资源
        w6 = 0.10  # 反思行为

        total_weight = w1 + w2 + w3 + w4 + w5 + w6
        srl_index = (
            w1 * adaptivity_score
            + w2 * early_score
            + w3 * independent_score
            + w4 * feedback_score
            + w5 * extension_score
            + w6 * reflection_score
        ) / float(total_weight)

        srl_index = max(0.0, min(1.0, srl_index))

        feature_summary = {
            "help_need_rate": float(help_need_rate),
            "early_help_ratio": float(early_help_ratio),
            "no_help_success_ratio": float(no_help_success_ratio),
            "feedback_density": float(feedback_density),
            "extension_flag": int(extension_flag),
            "reflection_flag": int(reflection_flag),
        }

        return srl_index, feature_summary

    def _compute_srl_index_for_courses(
        self,
        course_stats: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        将粗粒度统计转换为最终的课程级 SRL_help_index 指标。
        """
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for key_lc, stat in course_stats.items():
            srl_index, feat = self._compute_srl_help_index_for_course(stat)
            metrics_by_lc[key_lc] = {
                "srl_help_index": float(srl_index),
                **feat,
            }

        return metrics_by_lc
