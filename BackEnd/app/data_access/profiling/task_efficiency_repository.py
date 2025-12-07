# BackEnd/app/data_access/profiling/task_efficiency_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository
from app.shared.utils.repository_mixins import parse_iso8601_duration_seconds

logger = logging.getLogger(__name__)


class TaskEfficiencyRepository(MongoDBBaseRepository):
    """
    任务效率与认知负荷代理（Task Efficiency & Cognitive Load Proxy）维度的数据仓库。

    职责：
    - 仅负责从 MongoDB.MLS.Interaction 读取与任务相关的 xAPI 事件；
    - 利用现有复合索引：
        * idx_lrn_verb_course: {_lrn_uid, 'verb.id', _course_uid}
        * idx_course_verb_lrn: {_course_uid, 'verb.id', _lrn_uid}
      来发现课程并在课程内按批读取；
    - 为每个 (lrn_uid, crs_uid) 计算：
        P_mean: 任务成功率（基于 success / completion）
        T_mean: 平均任务耗时（秒）
        z_P, z_T: 课程内 z 分数
        efficiency_index: 认知效率 E = (z_P - z_T) / sqrt(2)
        efficiency_index_norm: 在课程内部对 E 做 [0,1] 归一化
        task_count: 参与统计的任务数
    - 不做任何聚类或文本标签映射（交给 engine）。
    """

    DB_NAME = "MLS"
    INTERACTION_COLLECTION = "Interaction"

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "completed": VERB_BASE + "completed",
        "performed_procedure_step": VERB_BASE + "performed-procedure-step",
    }

    def __init__(
        self,
        batch_size: int = 5000,
        course_chunk_size: int = 200,
        mongo_operator: Optional[Any] = None,
    ) -> None:
        """
        Args:
            batch_size: Mongo 游标 batch_size，逻辑参数（当前通过 aggregate 读取）。
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
        为若干学习者准备“任务效率”分析所需的课程级指标。

        注意：
        - 为了做“课程内聚类”，本仓库会在相关课程中统计**所有**学习者；
        - 但 learner_courses_map 中只保留调用方传入的学习者与它们相关的课程。

        返回:
            metrics_by_lc:
                (lrn_uid, crs_uid) -> {
                    "P_mean": float,                    # 成功率
                    "T_mean": float,                    # 平均耗时（秒）
                    "z_P": float,
                    "z_T": float,
                    "efficiency_index": float,          # E
                    "efficiency_index_norm": float,     # E_norm
                    "task_count": int,
                }

            learners_per_course:
                crs_uid -> 该课程内有任务效率数据的去重学习者数量
                （用于课程内部的聚类健壮性检查）

            learner_courses_map:
                lrn_uid -> set(course_uid)
                仅包含 metrics_by_lc 中存在数据的课程。
        """
        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        if not learner_uids:
            logger.info(
                "TaskEfficiencyRepository.load_metrics_for_learners: 空学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "TaskEfficiencyRepository: 开始准备任务效率原始数据，目标学习者数: %d",
            len(learner_uids),
        )

        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "TaskEfficiencyRepository: 与目标学习者相关的课程数: %d",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "TaskEfficiencyRepository: 目标学习者在任务相关事件上没有任何课程记录。"
            )
            return {}, {}, learner_courses_map

        task_stats, learners_per_course = self._aggregate_task_stats_for_courses(
            all_courses
        )
        logger.info(
            "TaskEfficiencyRepository: 已完成 (lrn, crs) 粗粒度任务统计，条目数: %d",
            len(task_stats),
        )

        metrics_by_lc = self._compute_efficiency_for_courses(task_stats)
        logger.info(
            "TaskEfficiencyRepository: efficiency_index_norm 计算完成，(lrn, crs) 有效条目数: %d",
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
    # 内部：Mongo 访问与课程发现（基于 aggregate）
    # ------------------------------------------------------------------
    def _get_courses_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        使用 idx_lrn_verb_course 复合索引：
            key: { _lrn_uid: 1, 'verb.id': 1, _course_uid: 1 }

        pipeline:
            match _lrn_uid in learner_uids
                  AND verb.id in [completed, performed-procedure-step]
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
    # 工具：均值/标准差
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

    # ------------------------------------------------------------------
    # 内部：按课程分片聚合 (lrn, crs) 任务统计
    # ------------------------------------------------------------------
    def _aggregate_task_stats_for_courses(
        self, course_uids: Set[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
    ]:
        """
        对一批课程的任务事件进行聚合，生成：

        task_stats[(lrn_uid, crs_uid)] = {
            "sum_P": float,   # 成功次数
            "sum_T": float,   # 总耗时
            "count": int,     # 任务数量
        }

        learners_per_course[crs_uid] = 去重学习者数量。
        """
        if not course_uids:
            return {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        task_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
        learners_per_course_sets: Dict[str, Set[str]] = defaultdict(set)

        projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "result": 1,
        }
        verb_list = list(self.VERBS.values())

        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1)
                * self.course_chunk_size
            ]

            logger.info(
                "TaskEfficiencyRepository: 读取事件，课程分片 %d/%d，课程数: %d",
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

                result = doc.get("result") or {}
                duration_str = result.get("duration")
                duration_sec = parse_iso8601_duration_seconds(duration_str)
                if duration_sec is None:
                    # 无有效时长的事件不纳入效率统计
                    continue

                success = result.get("success")
                completion = result.get("completion")

                if success is None and completion is None:
                    # 没有任何表现信息，跳过
                    continue
                elif success is None:
                    P_task = 1.0 if completion else 0.0
                else:
                    P_task = 1.0 if bool(success) else 0.0

                key_lc = (lrn_uid, crs_uid)
                if key_lc not in task_stats:
                    task_stats[key_lc] = {
                        "sum_P": 0.0,
                        "sum_T": 0.0,
                        "count": 0,
                    }

                stat = task_stats[key_lc]
                stat["sum_P"] += P_task
                stat["sum_T"] += float(duration_sec)
                stat["count"] += 1

            logger.info(
                "TaskEfficiencyRepository: 完成事件读取，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        learners_per_course = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course_sets.items()
        }

        logger.info(
            "TaskEfficiencyRepository: 任务统计聚合完成，课程数: %d，(lrn, crs) 组合数: %d",
            len(learners_per_course),
            len(task_stats),
        )

        return task_stats, learners_per_course

    # ------------------------------------------------------------------
    # 内部：从粗粒度统计计算 efficiency_index & efficiency_index_norm
    # ------------------------------------------------------------------
    def _compute_efficiency_for_courses(
        self,
        task_stats: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        将粗粒度任务统计转换为课程级任务效率指标。

        逻辑与原 analyze_task_efficiency.py 一致：
        - 在课程内部对 P_mean、T_mean 做 z 标准化；
        - 计算 E = (z_P - z_T) / sqrt(2)，z_T 越大表示耗时越长；
        - 在课程内部对 E 做 min-max 归一化为 efficiency_index_norm ∈ [0,1]。
        """
        if not task_stats:
            return {}

        # 先拆成按课程的列表
        course_to_entries: Dict[str, List[Tuple[str, float, float, int]]] = defaultdict(
            list
        )
        for (lrn_uid, crs_uid), stat in task_stats.items():
            c = int(stat.get("count", 0))
            if c <= 0:
                continue
            sum_P = float(stat.get("sum_P", 0.0))
            sum_T = float(stat.get("sum_T", 0.0))
            P_mean = sum_P / float(c)
            T_mean = sum_T / float(c)
            course_to_entries[crs_uid].append((lrn_uid, P_mean, T_mean, c))

        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_to_entries.items():
            if not entries:
                continue

            P_vals = [e[1] for e in entries]
            T_vals = [e[2] for e in entries]
            mean_P, std_P = self._compute_mean_std(P_vals)
            mean_T, std_T = self._compute_mean_std(T_vals)

            E_vals: List[float] = []

            # 先计算每个学习者的 E 值
            for (lrn_uid, P_mean, T_mean, task_count) in entries:
                if std_P > 1e-6:
                    z_P = (P_mean - mean_P) / std_P
                else:
                    z_P = 0.0

                if std_T > 1e-6:
                    z_T = (T_mean - mean_T) / std_T
                else:
                    z_T = 0.0

                E = (z_P - z_T) / math.sqrt(2.0)

                key = (lrn_uid, crs_uid)
                metrics_by_lc[key] = {
                    "P_mean": P_mean,
                    "T_mean": T_mean,
                    "z_P": z_P,
                    "z_T": z_T,
                    "efficiency_index": E,
                    "task_count": int(task_count),
                }
                E_vals.append(E)

            # 在当前课程内部对 E 做 [0,1] 的 min-max 归一化
            if not E_vals:
                continue

            E_min = min(E_vals)
            E_max = max(E_vals)
            span = E_max - E_min if E_max > E_min else 0.0

            for (lrn_uid, _, _, _) in entries:
                key = (lrn_uid, crs_uid)
                E = metrics_by_lc[key]["efficiency_index"]
                if span > 1e-6:
                    E_norm = (E - E_min) / span
                else:
                    E_norm = 0.5  # 所有人完全一致时，统一给中间值
                metrics_by_lc[key]["efficiency_index_norm"] = E_norm

        logger.info(
            "TaskEfficiencyRepository: 已为 %d 个 (lrn, crs) 计算 efficiency_index_norm。",
            len(metrics_by_lc),
        )

        return metrics_by_lc
