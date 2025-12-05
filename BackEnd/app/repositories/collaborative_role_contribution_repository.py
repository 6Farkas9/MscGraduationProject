# app/repositories/collaborative_role_contribution_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any

from app.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class CollaborativeRoleContributionRepository(BaseRepository):
    """
    协作角色与贡献类型（Collaborative Role & Contribution Type）维度的数据仓库。

    职责：
    - 仅负责从 MongoDB.MLS.Interaction 中读取“协作相关”事件；
    - 利用已有复合索引（_lrn_uid + verb.id + _course_uid、
      _course_uid + verb.id + _lrn_uid）筛选相关课程与事件；
    - 按 (会话, 课程, 学习者) 聚合基础行为计数；
    - 再按 (学习者, 课程) 聚合为课程级统计量；
    - 不做任何标签判定或聚类分析，所有“角色/贡献风格”逻辑由 engine 完成。
    """

    DB_NAME = "MLS"
    INTERACTION_COLLECTION = "Interaction"

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "collaborated_on_activity": VERB_BASE + "collaborated-on-activity",
        "co_edited_artifact": VERB_BASE + "co-edited-artifact",
        "contributed_resource": VERB_BASE + "contributed-resource",
        "responded": VERB_BASE + "responded",
        "referred": VERB_BASE + "referred",
        "followed": VERB_BASE + "followed",
        "managed_resource": VERB_BASE + "managed-resource",
        "took_turn": VERB_BASE + "took-turn",
    }

    # 简单 ISO8601 秒级 duration，例如 "PT120S"
    DURATION_RE = re.compile(r"^PT(\d+)S$")

    # context 扩展字段键
    EXT_SESSION_ID_1 = "https://legend-meta.com/xapi/ext/sessionId"
    EXT_SESSION_ID_2 = "https://legend-meta.com/xapi/ext/session-id"
    EXT_PARTICIPANTS = "https://legend-meta.com/xapi/ext/participants"

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
    # 对外主接口
    # ------------------------------------------------------------------
    def load_metrics_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
        Dict[str, Set[str]],
    ]:
        """
        为一批学习者准备协作角色与贡献类型分析所需的 **课程级基础数据**。

        返回:
            course_metrics_by_lc:
                (lrn_uid, crs_uid) -> {
                    "avg_share_contribution": float,
                    "avg_share_participation": float,
                    "avg_share_transactivity": float,
                    "sessions_count": int,

                    # 用于贡献风格分析的原始构成（课程内总量）：
                    "create_count": float,
                    "modify_count": float,
                    "resource_count": float,
                    "discuss_count": float,
                }

            learners_per_course:
                crs_uid -> 该课程内参与过协作相关事件的去重学习者数量

            learner_courses_map:
                learner_uid -> set(course_uid)
                仅包含 course_metrics_by_lc 中存在数据的课程。
        """
        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        if not learner_uids:
            logger.info(
                "CollaborativeRoleContributionRepository.load_metrics_for_learners: 空的学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "CollaborativeRoleContributionRepository: 开始准备原始协作数据，目标学习者数: %d",
            len(learner_uids),
        )

        # 1) 找到这些学习者参与过协作相关事件的课程集合
        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "CollaborativeRoleContributionRepository: 与目标学习者相关的课程数: %d",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "CollaborativeRoleContributionRepository: 目标学习者在协作相关事件上没有任何记录。"
            )
            return {}, {}, learner_courses_map

        # 2) 对这些课程的协作事件按会话与课程聚合，再汇总到课程级
        course_metrics_by_lc, learners_per_course = (
            self._aggregate_session_and_course_metrics_for_courses(all_courses)
        )

        logger.info(
            "CollaborativeRoleContributionRepository: 聚合完成，(learner, course) 条目数: %d，课程数: %d",
            len(course_metrics_by_lc),
            len(learners_per_course),
        )

        # 3) 过滤 learner_courses_map，仅保留真正有数据的课程
        filtered_map: Dict[str, Set[str]] = {}
        for lrn_uid, crs_set in learner_courses_map.items():
            valid_courses = {
                crs_uid
                for crs_uid in crs_set
                if (lrn_uid, crs_uid) in course_metrics_by_lc
            }
            if valid_courses:
                filtered_map[lrn_uid] = valid_courses

        return course_metrics_by_lc, learners_per_course, filtered_map

    # ------------------------------------------------------------------
    # 内部工具：Mongo 访问与基本解析
    # ------------------------------------------------------------------
    def _get_interaction_collection(self):
        """通过 mongodb_operator 拿到底层 Interaction 集合。"""
        return self.mongodb_operator.get_collection(self.INTERACTION_COLLECTION)

    def _parse_iso8601_duration(self, duration_str: Optional[str]) -> Optional[int]:
        """解析 'PT120S' -> 120，若失败返回 None。"""
        if not duration_str:
            return None
        m = self.DURATION_RE.match(duration_str)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        if b <= 1e-9:
            return 0.0
        return float(a) / float(b)

    def _get_session_id(self, doc: Dict) -> str:
        """
        取得协作会话标识：
        - 优先 context.extensions.sessionId / session-id；
        - 其次 context.registration；
        - 否则取 timestamp 的日期作为近似会话。
        """
        ctx = doc.get("context") or {}
        exts = ctx.get("extensions") or {}

        sid = (
            exts.get(self.EXT_SESSION_ID_1)
            or exts.get(self.EXT_SESSION_ID_2)
            or exts.get("sessionId")
        )
        if sid:
            return str(sid)

        reg = ctx.get("registration")
        if reg:
            return str(reg)

        ts = doc.get("timestamp")
        if ts:
            return str(ts)[:10]

        return "unknown-session"

    # ------------------------------------------------------------------
    # 内部工具：按课程查找 + 分批迭代
    # ------------------------------------------------------------------
    def _get_courses_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        使用 idx_lrn_verb_course 复合索引：
            key: { _lrn_uid: 1, 'verb.id': 1, _course_uid: 1 }

        pipeline:
            match _lrn_uid in learner_uids AND verb.id in [协作相关 verb]
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

    def _iterate_events(
        self,
        match_query: Dict,
        projection: Dict,
        batch_size: Optional[int] = None,
    ) -> Iterable[Dict]:
        """
        通用迭代器：按指定查询和投影，使用服务器端游标分批返回文档。

        为了利用 idx_course_verb_lrn 复合索引，match_query 中至少应包含：
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
    # 内部工具：按会话聚合，再汇总到课程级
    # ------------------------------------------------------------------
    def _aggregate_session_and_course_metrics_for_courses(
        self, course_uids: Set[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
    ]:
        """
        对若干课程的协作事件进行聚合，生成课程级基础统计：

        返回:
            course_metrics_by_lc[(lrn_uid, crs_uid)] = {
                "avg_share_contribution": float,
                "avg_share_participation": float,
                "avg_share_transactivity": float,
                "sessions_count": int,
                "create_count": float,
                "modify_count": float,
                "resource_count": float,
                "discuss_count": float,
            }

            learners_per_course[crs_uid] = 课程内去重学习者数量
        """
        if not course_uids:
            return {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        # (session_id, course_uid, learner_uid) -> 原始计数
        per_session_metrics: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(
            lambda: {
                "create_edits": 0.0,
                "update_edits": 0.0,
                "delete_edits": 0.0,
                "resources_contributed": 0.0,
                "collaborated_count": 0.0,
                "collaborated_duration": 0.0,
                "responded": 0.0,
                "referred": 0.0,
                "followed": 0.0,
                "managed_resource": 0.0,
                "took_turn": 0.0,
            }
        )

        # 课程参与者集合
        learners_per_course: Dict[str, Set[str]] = defaultdict(set)

        projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "result": 1,
            "context": 1,
            "timestamp": 1,
        }

        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1) * self.course_chunk_size
            ]
            logger.info(
                "CollaborativeRoleContributionRepository: 读取协作事件，课程分片 %d/%d，课程数: %d",
                chunk_idx + 1,
                total_chunks,
                len(sub_courses),
            )

            match_query = {
                "_course_uid": {"$in": sub_courses},
                "verb.id": {"$in": list(self.VERBS.values())},
            }

            event_cnt = 0
            for doc in self._iterate_events(match_query, projection):
                event_cnt += 1
                lrn_uid = doc.get("_lrn_uid")
                crs_uid = doc.get("_course_uid")
                if not lrn_uid or not crs_uid:
                    continue

                learners_per_course[crs_uid].add(lrn_uid)

                sid = self._get_session_id(doc)
                key = (sid, crs_uid, lrn_uid)

                verb_id = (doc.get("verb") or {}).get("id")
                result = doc.get("result") or {}

                m = per_session_metrics[key]

                if verb_id == self.VERBS["co_edited_artifact"]:
                    exts = result.get("extensions") or {}
                    etype = (
                        exts.get("edit-type")
                        or result.get("edit-type")
                        or result.get("edit_type")
                    )
                    etype = str(etype).lower() if etype else "update"

                    if etype == "create":
                        m["create_edits"] += 1
                    elif etype == "delete":
                        m["delete_edits"] += 1
                    else:
                        m["update_edits"] += 1

                elif verb_id == self.VERBS["contributed_resource"]:
                    m["resources_contributed"] += 1

                elif verb_id == self.VERBS["collaborated_on_activity"]:
                    m["collaborated_count"] += 1
                    dur = self._parse_iso8601_duration(result.get("duration"))
                    if dur:
                        m["collaborated_duration"] += float(dur)

                elif verb_id == self.VERBS["responded"]:
                    m["responded"] += 1
                elif verb_id == self.VERBS["referred"]:
                    m["referred"] += 1
                elif verb_id == self.VERBS["followed"]:
                    m["followed"] += 1
                elif verb_id == self.VERBS["managed_resource"]:
                    m["managed_resource"] += 1
                elif verb_id == self.VERBS["took_turn"]:
                    m["took_turn"] += 1

            logger.info(
                "CollaborativeRoleContributionRepository: 完成协作事件读取，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        logger.info(
            "CollaborativeRoleContributionRepository: 会话级原始计数聚合完成，会话条目数: %d",
            len(per_session_metrics),
        )

        # 2) 计算每个会话的总量，用于份额计算
        session_totals: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"contribution": 0.0, "participation": 0.0, "transactivity": 0.0}
        )

        for (sid, crs_uid, _lrn_uid), m in per_session_metrics.items():
            contrib = (
                m["create_edits"]
                + m["update_edits"]
                + m["delete_edits"]
                + m["resources_contributed"]
            )
            partic = m["collaborated_count"] + self._safe_div(
                m["collaborated_duration"], 60.0
            )
            trans = (
                m["responded"]
                + m["referred"]
                + m["followed"]
                + m["managed_resource"]
                + m["took_turn"]
            )

            st = session_totals[(sid, crs_uid)]
            st["contribution"] += contrib
            st["participation"] += partic
            st["transactivity"] += trans

        logger.info(
            "CollaborativeRoleContributionRepository: 会话级总量统计完成，会话数: %d",
            len(session_totals),
        )

        # 3) 按课程与学习者聚合（课程级）
        course_agg: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "share_contribution": [],
                "share_participation": [],
                "share_transactivity": [],
                "create_count": 0.0,
                "modify_count": 0.0,
                "resource_count": 0.0,
                "discuss_count": 0.0,
                "sessions": 0,
            }
        )

        processed_sessions = 0
        total_sessions = len(session_totals)

        for (sid, crs_uid), totals in session_totals.items():
            processed_sessions += 1
            contrib_total = totals["contribution"]
            partic_total = totals["participation"]
            trans_total = totals["transactivity"]

            # 找出本会话中的所有学习者
            learners_in_session: List[str] = [
                lrn_uid
                for (sid2, crs2, lrn_uid) in per_session_metrics.keys()
                if sid2 == sid and crs2 == crs_uid
            ]

            for lrn_uid in learners_in_session:
                m = per_session_metrics[(sid, crs_uid, lrn_uid)]

                contrib = (
                    m["create_edits"]
                    + m["update_edits"]
                    + m["delete_edits"]
                    + m["resources_contributed"]
                )
                partic = m["collaborated_count"] + self._safe_div(
                    m["collaborated_duration"], 60.0
                )
                trans = (
                    m["responded"]
                    + m["referred"]
                    + m["followed"]
                    + m["managed_resource"]
                    + m["took_turn"]
                )

                share_contrib = self._safe_div(contrib, contrib_total)
                share_partic = self._safe_div(partic, partic_total)
                share_trans = self._safe_div(trans, trans_total)

                create_cnt = m["create_edits"]
                modify_cnt = m["update_edits"] + m["delete_edits"]
                resource_cnt = m["resources_contributed"]
                discuss_cnt = (
                    m["responded"]
                    + m["referred"]
                    + m["followed"]
                    + m["managed_resource"]
                    + m["took_turn"]
                )

                ca = course_agg[(lrn_uid, crs_uid)]
                ca["share_contribution"].append(share_contrib)
                ca["share_participation"].append(share_partic)
                ca["share_transactivity"].append(share_trans)
                ca["create_count"] += create_cnt
                ca["modify_count"] += modify_cnt
                ca["resource_count"] += resource_cnt
                ca["discuss_count"] += discuss_cnt
                ca["sessions"] += 1

            if processed_sessions % 100 == 0 or processed_sessions == total_sessions:
                logger.info(
                    "CollaborativeRoleContributionRepository: 已处理会话 %d/%d",
                    processed_sessions,
                    total_sessions,
                )

        # 4) 计算课程级平均值，形成最终返回结构
        course_metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), ca in course_agg.items():
            sc = ca["share_contribution"]
            sp = ca["share_participation"]
            st = ca["share_transactivity"]
            sessions = int(ca["sessions"])

            avg_sc = sum(sc) / float(len(sc)) if sc else 0.0
            avg_sp = sum(sp) / float(len(sp)) if sp else 0.0
            avg_st = sum(st) / float(len(st)) if st else 0.0

            course_metrics_by_lc[(lrn_uid, crs_uid)] = {
                "avg_share_contribution": float(avg_sc),
                "avg_share_participation": float(avg_sp),
                "avg_share_transactivity": float(avg_st),
                "sessions_count": sessions,
                "create_count": float(ca["create_count"]),
                "modify_count": float(ca["modify_count"]),
                "resource_count": float(ca["resource_count"]),
                "discuss_count": float(ca["discuss_count"]),
            }

        learners_per_course_count = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course.items()
        }

        logger.info(
            "CollaborativeRoleContributionRepository: 课程级基础统计整理完成，(lrn, crs) 条目: %d",
            len(course_metrics_by_lc),
        )

        return course_metrics_by_lc, learners_per_course_count
