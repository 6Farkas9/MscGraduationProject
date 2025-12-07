# BackEnd/app/data_access/prediction/hgc_repository.py
import logging
from typing import List, Dict, Tuple, Set, Optional, Any

from app.data_access.base.mysql_base_repository import MySQLBaseRepository

logger = logging.getLogger(__name__)


class HGCRepository(MySQLBaseRepository):
    """HGC 模型数据仓库"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # ---- 学习者-实体列表 ----

    def get_learners_interacted_units(self, learner_uids: List[str]) -> Dict[str, List[str]]:
        if not learner_uids:
            return {}

        placeholders = ", ".join(["%s"] * len(learner_uids))
        sql = f"""
        SELECT lrn_uid, unt_uid
        FROM Interaction
        WHERE lrn_uid IN ({placeholders})
        """
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(learner_uids))
            result: Dict[str, List[str]] = {}
            for row in rows:
                uid = row["lrn_uid"]
                unt = row["unt_uid"]
                bucket = result.setdefault(uid, [])
                if unt not in bucket:
                    bucket.append(unt)
            return result
        except Exception as exc:
            logger.error("get_learners_interacted_units failed: %s", exc)
            return {}

    def get_learners_topics(self, learner_uids: List[str]) -> Dict[str, List[str]]:
        if not learner_uids:
            return {}

        placeholders = ", ".join(["%s"] * len(learner_uids))
        sql = f"""
        SELECT lrn_uid, tpc_uid
        FROM Learner_Topic
        WHERE lrn_uid IN ({placeholders})
        """
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(learner_uids))
            result: Dict[str, List[str]] = {}
            for row in rows:
                uid = row["lrn_uid"]
                tpc = row["tpc_uid"]
                bucket = result.setdefault(uid, [])
                if tpc not in bucket:
                    bucket.append(tpc)
            return result
        except Exception as exc:
            logger.error("get_learners_topics failed: %s", exc)
            return {}

    def get_learners_courses(self, learner_uids: List[str]) -> Dict[str, List[str]]:
        if not learner_uids:
            return {}

        placeholders = ", ".join(["%s"] * len(learner_uids))
        sql = f"""
        SELECT lrn_uid, crs_uid
        FROM Learner_Course
        WHERE lrn_uid IN ({placeholders})
        """
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(learner_uids))
            result: Dict[str, List[str]] = {}
            for row in rows:
                uid = row["lrn_uid"]
                crs = row["crs_uid"]
                bucket = result.setdefault(uid, [])
                if crs not in bucket:
                    bucket.append(crs)
            return result
        except Exception as exc:
            logger.error("get_learners_courses failed: %s", exc)
            return {}

    # ---- 根据实体找其他学习者 ----

    def get_other_learners_by_units(self, unit_uids: List[str]) -> Set[str]:
        if not unit_uids:
            return set()

        placeholders = ", ".join(["%s"] * len(unit_uids))
        sql = f"""
        SELECT DISTINCT lrn_uid
        FROM Interaction
        WHERE unt_uid IN ({placeholders})
        """
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(unit_uids))
            return {row["lrn_uid"] for row in rows}
        except Exception as exc:
            logger.error("get_other_learners_by_units failed: %s", exc)
            return set()

    def get_other_learners_by_topics(self, topic_uids: List[str]) -> Set[str]:
        if not topic_uids:
            return set()

        placeholders = ", ".join(["%s"] * len(topic_uids))
        sql = f"""
        SELECT DISTINCT lrn_uid
        FROM Learner_Topic
        WHERE tpc_uid IN ({placeholders})
        """
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(topic_uids))
            return {row["lrn_uid"] for row in rows}
        except Exception as exc:
            logger.error("get_other_learners_by_topics failed: %s", exc)
            return set()

    def get_other_learners_by_courses(self, course_uids: List[str]) -> Set[str]:
        if not course_uids:
            return set()

        placeholders = ", ".join(["%s"] * len(course_uids))
        sql = f"""
        SELECT DISTINCT lrn_uid
        FROM Learner_Course
        WHERE crs_uid IN ({placeholders})
        """
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(course_uids))
            return {row["lrn_uid"] for row in rows}
        except Exception as exc:
            logger.error("get_other_learners_by_courses failed: %s", exc)
            return set()

    def get_optimized_related_learners(
        self,
        target_learner_uids: List[str],
        all_interacted_units: List[str],
        all_learner_topics: List[str],
        all_learner_courses: List[str],
    ) -> Set[str]:
        """
        优化策略：先尝试按单元/主题/课程交集找相关学习者，若交集为空再取并集
        """
        from_units = self.get_other_learners_by_units(all_interacted_units)
        from_topics = self.get_other_learners_by_topics(all_learner_topics)
        from_courses = self.get_other_learners_by_courses(all_learner_courses)

        logger.info(
            "meta-path learner counts - units=%d, topics=%d, courses=%d",
            len(from_units),
            len(from_topics),
            len(from_courses),
        )

        intersection = from_units & from_topics & from_courses
        for uid in target_learner_uids:
            intersection.discard(uid)

        if intersection:
            logger.info("use intersection strategy, related_learners=%d", len(intersection))
            return intersection

        union = from_units | from_topics | from_courses
        for uid in target_learner_uids:
            union.discard(uid)

        logger.info("use union strategy, related_learners=%d", len(union))
        return union

    # ---- 精确交互记录 ----

    def get_unit_interaction_records(
        self, learner_uids: List[str], unit_uids: List[str]
    ) -> List[Tuple[str, str, str]]:
        if not learner_uids or not unit_uids:
            return []

        learner_ph = ", ".join(["%s"] * len(learner_uids))
        unit_ph = ", ".join(["%s"] * len(unit_uids))
        sql = f"""
        SELECT lrn_uid, unt_uid
        FROM Interaction
        WHERE lrn_uid IN ({learner_ph})
          AND unt_uid IN ({unit_ph})
        """
        try:
            params = tuple(learner_uids + unit_uids)
            rows = self.execute_custom_mysql_query(sql, params)
            return [(row["lrn_uid"], row["unt_uid"], "unit") for row in rows]
        except Exception as exc:
            logger.error("get_unit_interaction_records failed: %s", exc)
            return []

    def get_topic_interaction_records(
        self, learner_uids: List[str], topic_uids: List[str]
    ) -> List[Tuple[str, str, str]]:
        if not learner_uids or not topic_uids:
            return []

        learner_ph = ", ".join(["%s"] * len(learner_uids))
        topic_ph = ", ".join(["%s"] * len(topic_uids))
        sql = f"""
        SELECT lrn_uid, tpc_uid
        FROM Learner_Topic
        WHERE lrn_uid IN ({learner_ph})
          AND tpc_uid IN ({topic_ph})
        """
        try:
            params = tuple(learner_uids + topic_uids)
            rows = self.execute_custom_mysql_query(sql, params)
            return [(row["lrn_uid"], row["tpc_uid"], "topic") for row in rows]
        except Exception as exc:
            logger.error("get_topic_interaction_records failed: %s", exc)
            return []

    def get_course_interaction_records(
        self, learner_uids: List[str], course_uids: List[str]
    ) -> List[Tuple[str, str, str]]:
        if not learner_uids or not course_uids:
            return []

        learner_ph = ", ".join(["%s"] * len(learner_uids))
        course_ph = ", ".join(["%s"] * len(course_uids))
        sql = f"""
        SELECT lrn_uid, crs_uid
        FROM Learner_Course
        WHERE lrn_uid IN ({learner_ph})
          AND crs_uid IN ({course_ph})
        """
        try:
            params = tuple(learner_uids + course_uids)
            rows = self.execute_custom_mysql_query(sql, params)
            return [(row["lrn_uid"], row["crs_uid"], "course") for row in rows]
        except Exception as exc:
            logger.error("get_course_interaction_records failed: %s", exc)
            return []

    # ---- HGC 总数据 ----

    def get_data_for_multiple_learners(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        为多个学习者构建 HGC 所需数据
        """
        try:
            units_by_learner = self.get_learners_interacted_units(learner_uids)
            topics_by_learner = self.get_learners_topics(learner_uids)
            courses_by_learner = self.get_learners_courses(learner_uids)

            all_units = list(
                {u for units in units_by_learner.values() for u in units}
            )
            all_topics = list(
                {t for topics in topics_by_learner.values() for t in topics}
            )
            all_courses = list(
                {c for courses in courses_by_learner.values() for c in courses}
            )

            logger.info(
                "HGC multi learners: learners=%d, units=%d, topics=%d, courses=%d",
                len(learner_uids),
                len(all_units),
                len(all_topics),
                len(all_courses),
            )

            related_learners = self.get_optimized_related_learners(
                learner_uids, all_units, all_topics, all_courses
            )
            logger.info("related_learners=%d", len(related_learners))

            all_learners = learner_uids + list(related_learners)

            interaction_records: List[Tuple[str, str, str]] = []
            interaction_records.extend(
                self.get_unit_interaction_records(all_learners, all_units)
            )
            interaction_records.extend(
                self.get_topic_interaction_records(all_learners, all_topics)
            )
            interaction_records.extend(
                self.get_course_interaction_records(all_learners, all_courses)
            )

            logger.info("interaction_records=%d", len(interaction_records))

            return {
                "target_learner_uids": learner_uids,
                "learner_entities": {
                    "units": units_by_learner,
                    "topics": topics_by_learner,
                    "courses": courses_by_learner,
                },
                "all_entities": {
                    "units": all_units,
                    "topics": all_topics,
                    "courses": all_courses,
                },
                "related_learners": list(related_learners),
                "interaction_records": interaction_records,
                # 粗略策略标记：如果相关学习者规模明显小于“所有-目标”，视为 intersection
                "strategy_used": "intersection"
                if len(related_learners)
                < len(set(all_learners) - set(learner_uids))
                else "union",
            }
        except Exception as exc:
            logger.error("get_data_for_multiple_learners failed: %s", exc)
            raise

    def get_hgc_data_for_learner(self, learner_uid: str) -> Dict[str, Any]:
        """
        单个学习者版本：包装在多学习者接口之上
        """
        try:
            multi = self.get_data_for_multiple_learners([learner_uid])
            return {
                "target_learner_uid": learner_uid,
                "interacted_units": multi["learner_entities"]["units"].get(learner_uid, []),
                "learner_topics": multi["learner_entities"]["topics"].get(learner_uid, []),
                "learner_courses": multi["learner_entities"]["courses"].get(learner_uid, []),
                "related_learners": multi["related_learners"],
                "interaction_records": multi["interaction_records"],
                "strategy_used": multi["strategy_used"],
            }
        except Exception as exc:
            logger.error("get_hgc_data_for_learner failed for %s: %s", learner_uid, exc)
            raise

    # 保留原有别名
    def get_data_for_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        return self.get_hgc_data_for_learner(learner_uid)
