# BackEnd/app/data_access/profiling/social_learning_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository
from app.shared.utils.repository_mixins import parse_iso8601_duration_seconds

logger = logging.getLogger(__name__)


class SocialLearningRepository(MongoDBBaseRepository):
    """
    社会性学习与同伴取向（Social Learning & Peer Orientation）维度的数据仓库。

    职责：
    - 仅负责从 MongoDB.MLS.Interaction 读取与该维度相关的 xAPI 行为事件；
    - 利用现有复合索引：
        * idx_lrn_verb_course: {_lrn_uid, 'verb.id', _course_uid}
        * idx_course_verb_lrn: {_course_uid, 'verb.id', _lrn_uid}
      进行课程发现和按课程分批扫描；
    - 为每个 (lrn_uid, crs_uid) 计算：
        obs/collab 次数与总时长、观摩同伴人数、
        课程内 z 标准化后的 z_obs / z_collab、
        社会性学习指数 social_index 及其在课程内的归一化 social_index_norm；
    - 不进行聚类或角色标签判定（交给 engine）。
    """

    DB_NAME = "MLS"
    INTERACTION_COLLECTION = "Interaction"

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "observed_peer": VERB_BASE + "observed-peer",
        "collaborated_on_activity": VERB_BASE + "collaborated-on-activity",
    }

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
        为若干学习者准备“社会性学习与同伴取向”分析所需的课程级特征。

        返回:
            metrics_by_lc:
                (lrn_uid, crs_uid) -> {
                    "obs_count": int,
                    "obs_total_time": float,
                    "obs_unique_peers": int,
                    "collab_count": int,
                    "collab_total_time": float,
                    "z_obs": float,
                    "z_collab": float,
                    "social_index": float,
                    "social_index_norm": float,
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
                "SocialLearningRepository.load_metrics_for_learners: 空学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "SocialLearningRepository: 开始准备社会性学习原始数据，目标学习者数: %d",
            len(learner_uids),
        )

        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "SocialLearningRepository: 与目标学习者相关的课程数: %d",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "SocialLearningRepository: 目标学习者在社会性相关事件上没有任何课程记录。"
            )
            return {}, {}, learner_courses_map

        social_stats, learners_per_course = self._aggregate_social_stats_for_courses(
            all_courses
        )

        logger.info(
            "SocialLearningRepository: 已完成 (lrn, crs) 粗粒度社会性统计，条目数: %d",
            len(social_stats),
        )

        metrics_by_lc = self._compute_social_index_for_courses(social_stats)

        logger.info(
            "SocialLearningRepository: social_index_norm 计算完成，(lrn, crs) 有效条目数: %d",
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
                  AND verb.id in [observed-peer, collaborated-on-activity]
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
    # 工具：平均 / 标准差
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
    # 内部：按课程分片聚合 (lrn, crs) 社会性统计
    # ------------------------------------------------------------------
    def _aggregate_social_stats_for_courses(
        self, course_uids: Set[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
    ]:
        """
        对一批课程的社会性事件进行聚合，生成：

        social_stats[(lrn_uid, crs_uid)] = {
            "obs_count": int,
            "obs_total_time": float,
            "obs_peers": set(),
            "collab_count": int,
            "collab_total_time": float,
        }

        learners_per_course[crs_uid] = 去重学习者数量。
        """
        if not course_uids:
            return {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        social_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
        learners_per_course_sets: Dict[str, Set[str]] = defaultdict(set)

        projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "result": 1,
            "context": 1,
        }
        verb_list = list(self.VERBS.values())

        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1)
                * self.course_chunk_size
            ]

            logger.info(
                "SocialLearningRepository: 读取事件，课程分片 %d/%d，课程数: %d",
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

                key_lc = (lrn_uid, crs_uid)
                if key_lc not in social_stats:
                    social_stats[key_lc] = {
                        "obs_count": 0,
                        "obs_total_time": 0.0,
                        "obs_peers": set(),
                        "collab_count": 0,
                        "collab_total_time": 0.0,
                    }

                stat = social_stats[key_lc]

                verb_id = (doc.get("verb") or {}).get("id")
                result = doc.get("result") or {}
                duration_str = result.get("duration")
                duration_sec = parse_iso8601_duration_seconds(duration_str)

                # 若 parse 出来是负数（理论上不会），则忽略
                if duration_sec is not None and duration_sec < 0:
                    continue

                if verb_id == self.VERBS["observed_peer"]:
                    stat["obs_count"] += 1
                    if duration_sec is not None:
                        stat["obs_total_time"] += float(duration_sec)
                    context = doc.get("context") or {}
                    ext = context.get("extensions") or {}
                    peer_id = ext.get(
                        "https://legend-meta.com/xapi/ext/observed-learner-id"
                    )
                    if peer_id:
                        stat["obs_peers"].add(peer_id)

                elif verb_id == self.VERBS["collaborated_on_activity"]:
                    stat["collab_count"] += 1
                    if duration_sec is not None:
                        stat["collab_total_time"] += float(duration_sec)

            logger.info(
                "SocialLearningRepository: 完成事件读取，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        learners_per_course = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course_sets.items()
        }

        logger.info(
            "SocialLearningRepository: 粗粒度社会性统计聚合完成，课程数: %d，(lrn, crs) 组合数: %d",
            len(learners_per_course),
            len(social_stats),
        )

        return social_stats, learners_per_course

    # ------------------------------------------------------------------
    # 内部：从粗粒度统计计算 social_index & social_index_norm
    # ------------------------------------------------------------------
    def _compute_social_index_for_courses(
        self,
        social_stats: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        将粗粒度社会性统计转换为课程级社会性学习指数。

        逻辑与原脚本一致：
        - 在课程内部对 obs_total_time / collab_total_time 做 z 标准化；
        - 构造 S = (z_obs + z_collab) / sqrt(2)；
        - 再在课程内部对 S 做 min-max 归一化为 social_index_norm ∈ [0,1]。
        """
        if not social_stats:
            return {}

        course_to_entries: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(
            list
        )
        for (lrn_uid, crs_uid), stat in social_stats.items():
            course_to_entries[crs_uid].append((lrn_uid, stat))

        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_to_entries.items():
            if not entries:
                continue

            obs_vals = [float(s["obs_total_time"]) for _, s in entries]
            collab_vals = [float(s["collab_total_time"]) for _, s in entries]
            mean_obs, std_obs = self._compute_mean_std(obs_vals)
            mean_collab, std_collab = self._compute_mean_std(collab_vals)

            S_vals: List[float] = []

            # 第一次遍历：计算 z_obs / z_collab / S
            for (lrn_uid, stat) in entries:
                obs_total = float(stat["obs_total_time"])
                collab_total = float(stat["collab_total_time"])
                obs_count = int(stat["obs_count"])
                collab_count = int(stat["collab_count"])
                obs_peers_count = len(stat["obs_peers"]) if stat["obs_peers"] else 0

                if std_obs > 1e-6:
                    z_obs = (obs_total - mean_obs) / std_obs
                else:
                    z_obs = 0.0

                if std_collab > 1e-6:
                    z_collab = (collab_total - mean_collab) / std_collab
                else:
                    z_collab = 0.0

                S = (z_obs + z_collab) / math.sqrt(2.0)

                key = (lrn_uid, crs_uid)
                metrics_by_lc[key] = {
                    "obs_count": obs_count,
                    "obs_total_time": obs_total,
                    "obs_unique_peers": obs_peers_count,
                    "collab_count": collab_count,
                    "collab_total_time": collab_total,
                    "z_obs": z_obs,
                    "z_collab": z_collab,
                    "social_index": S,
                }
                S_vals.append(S)

            # 第二次：在课程内对 S 做 [0,1] 的 min-max 归一化
            if not S_vals:
                continue

            S_min = min(S_vals)
            S_max = max(S_vals)
            span = S_max - S_min if S_max > S_min else 0.0

            for (lrn_uid, _stat) in entries:
                key = (lrn_uid, crs_uid)
                S = metrics_by_lc[key]["social_index"]
                if span > 1e-6:
                    S_norm = (S - S_min) / span
                else:
                    S_norm = 0.5  # 所有人完全一致时，统一给中间值
                metrics_by_lc[key]["social_index_norm"] = S_norm

        logger.info(
            "SocialLearningRepository: 已为 %d 个 (lrn, crs) 计算 social_index_norm。",
            len(metrics_by_lc),
        )

        return metrics_by_lc
