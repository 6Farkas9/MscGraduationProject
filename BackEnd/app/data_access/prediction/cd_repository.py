# BackEnd/app/data_access/prediction/cd_repository.py
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from app.data_access.base.mysql_base_repository import MySQLBaseRepository

logger = logging.getLogger(__name__)


class CDRepository(MySQLBaseRepository):
    """认知诊断(CD)模型数据仓库"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # ---- 交互明细 ----

    def get_learner_question_interactions(
        self, learner_uid: str, max_records: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取单个学习者的题目交互记录（按时间排序）
        """
        sql = """
        SELECT lrn_uid, qus_uid, correct, create_time
        FROM Learner_Question
        WHERE lrn_uid = %s
        ORDER BY create_time ASC
        LIMIT %s
        """
        try:
            return self.execute_custom_mysql_query(sql, (learner_uid, max_records))
        except Exception as exc:
            logger.error("get_learner_question_interactions failed for %s: %s", learner_uid, exc)
            return []

    def get_question_interactions_batch(
        self, learner_uids: List[str], max_records_per_learner: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量获取多个学习者的题目交互记录（按时间排序）
        """
        if not learner_uids:
            return {}

        placeholders = ", ".join(["%s"] * len(learner_uids))
        sql = f"""
        SELECT lrn_uid, qus_uid, correct, create_time
        FROM Learner_Question
        WHERE lrn_uid IN ({placeholders})
        ORDER BY lrn_uid, create_time ASC
        """
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(learner_uids))
            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                uid = row["lrn_uid"]
                if len(grouped[uid]) < max_records_per_learner:
                    grouped[uid].append(row)
            return dict(grouped)
        except Exception as exc:
            logger.error("get_question_interactions_batch failed: %s", exc)
            return {}

    # ---- 序列构建 ----

    def build_question_sequences(
        self, learner_uids: List[str], max_seq_len: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        为多个学习者构建题目序列（保持时间顺序）
        """
        if not learner_uids:
            return {
                "sequences": {},
                "all_question_uids": [],
                "max_seq_len": max_seq_len or 0,
                "actual_max_seq_len": 0,
                "statistics": {
                    "total_learners": 0,
                    "total_interactions": 0,
                    "average_sequence_length": 0,
                    "unique_questions": 0,
                },
            }

        try:
            limit = max_seq_len if max_seq_len is not None else 1000
            batch = self.get_question_interactions_batch(learner_uids, limit)

            sequences: Dict[str, Dict[str, Any]] = {}
            all_question_uids: set[str] = set()
            actual_max_seq_len = 0

            for uid in learner_uids:
                interactions = batch.get(uid, [])
                qus_seq = [row["qus_uid"] for row in interactions]

                if max_seq_len is not None and len(qus_seq) > max_seq_len:
                    qus_seq = qus_seq[-max_seq_len:]

                seq_len = len(qus_seq)
                all_question_uids.update(qus_seq)
                actual_max_seq_len = max(actual_max_seq_len, seq_len)

                sequences[uid] = {
                    "qus_seq": qus_seq,
                    "seq_len": seq_len,
                    "interaction_count": len(interactions),
                }

            if max_seq_len is None:
                max_seq_len = actual_max_seq_len

            total_interactions = sum(v["interaction_count"] for v in sequences.values())
            avg_seq_len = total_interactions / len(learner_uids) if learner_uids else 0

            return {
                "sequences": sequences,
                "all_question_uids": list(all_question_uids),
                "max_seq_len": max_seq_len,
                "actual_max_seq_len": actual_max_seq_len,
                "statistics": {
                    "total_learners": len(learner_uids),
                    "total_interactions": total_interactions,
                    "average_sequence_length": round(avg_seq_len, 2),
                    "unique_questions": len(all_question_uids),
                },
            }
        except Exception as exc:
            logger.error("build_question_sequences failed: %s", exc)
            return {
                "sequences": {},
                "all_question_uids": [],
                "max_seq_len": max_seq_len or 0,
                "actual_max_seq_len": 0,
                "statistics": {
                    "total_learners": len(learner_uids),
                    "total_interactions": 0,
                    "average_sequence_length": 0,
                    "unique_questions": 0,
                },
            }

    def get_question_sequence_for_learner(self, learner_uid: str, max_seq_len: int = 50) -> Dict[str, Any]:
        """
        获取单个学习者的题目序列
        """
        try:
            interactions = self.get_learner_question_interactions(learner_uid, max_seq_len)
            qus_seq = [row["qus_uid"] for row in interactions]
            return {
                "learner_uid": learner_uid,
                "qus_seq": qus_seq,
                "seq_len": len(qus_seq),
                "interaction_count": len(interactions),
                "unique_questions": len(set(qus_seq)),
            }
        except Exception as exc:
            logger.error("get_question_sequence_for_learner failed for %s: %s", learner_uid, exc)
            return {
                "learner_uid": learner_uid,
                "qus_seq": [],
                "seq_len": 0,
                "interaction_count": 0,
                "unique_questions": 0,
            }

    def get_involved_questions(self, learner_uids: List[str]) -> List[str]:
        """
        获取多个学习者涉及的所有题目 UID
        """
        try:
            data = self.build_question_sequences(learner_uids)
            return data["all_question_uids"]
        except Exception as exc:
            logger.error("get_involved_questions failed: %s", exc)
            return []

    # ---- 统计 ----

    def get_interaction_statistics(self, learner_uid: str) -> Dict[str, Any]:
        """
        获取单个学习者交互统计信息
        """
        sql = """
        SELECT
            COUNT(*) AS total_interactions,
            COUNT(DISTINCT qus_uid) AS unique_questions,
            AVG(correct) AS accuracy_rate,
            MIN(create_time) AS start_time,
            MAX(create_time) AS end_time
        FROM Learner_Question
        WHERE lrn_uid = %s
        """
        try:
            row = self.execute_custom_single_query(sql, (learner_uid,))
            if row:
                return {
                    "learner_uid": learner_uid,
                    "total_interactions": row.get("total_interactions") or 0,
                    "unique_questions": row.get("unique_questions") or 0,
                    "accuracy_rate": float(row.get("accuracy_rate") or 0),
                    "time_range": {
                        "start": row.get("start_time"),
                        "end": row.get("end_time"),
                    },
                }
        except Exception as exc:
            logger.error("get_interaction_statistics failed for %s: %s", learner_uid, exc)

        return {
            "learner_uid": learner_uid,
            "total_interactions": 0,
            "unique_questions": 0,
            "accuracy_rate": 0.0,
            "time_range": {"start": None, "end": None},
        }

    def get_batch_interaction_statistics(self, learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取学习者交互统计信息
        """
        if not learner_uids:
            return {}

        placeholders = ", ".join(["%s"] * len(learner_uids))
        sql = f"""
        SELECT
            lrn_uid,
            COUNT(*) AS total_interactions,
            COUNT(DISTINCT qus_uid) AS unique_questions,
            AVG(correct) AS accuracy_rate,
            MIN(create_time) AS start_time,
            MAX(create_time) AS end_time
        FROM Learner_Question
        WHERE lrn_uid IN ({placeholders})
        GROUP BY lrn_uid
        """
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(learner_uids))
            stats: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                uid = row["lrn_uid"]
                stats[uid] = {
                    "total_interactions": row.get("total_interactions") or 0,
                    "unique_questions": row.get("unique_questions") or 0,
                    "accuracy_rate": float(row.get("accuracy_rate") or 0),
                    "time_range": {
                        "start": row.get("start_time"),
                        "end": row.get("end_time"),
                    },
                }

            # 补齐没有交互的学习者
            for uid in learner_uids:
                if uid not in stats:
                    stats[uid] = {
                        "total_interactions": 0,
                        "unique_questions": 0,
                        "accuracy_rate": 0.0,
                        "time_range": {"start": None, "end": None},
                    }
            return stats
        except Exception as exc:
            logger.error("get_batch_interaction_statistics failed: %s", exc)
            return {
                uid: {
                    "total_interactions": 0,
                    "unique_questions": 0,
                    "accuracy_rate": 0.0,
                    "time_range": {"start": None, "end": None},
                }
                for uid in learner_uids
            }

    # ---- 校验 / 最近交互 ----

    def validate_learner_has_interactions(self, learner_uid: str) -> bool:
        sql = "SELECT COUNT(*) AS count FROM Learner_Question WHERE lrn_uid = %s"
        try:
            row = self.execute_custom_single_query(sql, (learner_uid,))
            return bool(row and row.get("count", 0) > 0)
        except Exception as exc:
            logger.error("validate_learner_has_interactions failed for %s: %s", learner_uid, exc)
            return False

    def get_recent_interactions(self, limit: int = 128) -> List[Dict[str, Any]]:
        sql = """
        SELECT lrn_uid, qus_uid, correct, create_time
        FROM Learner_Question
        ORDER BY create_time DESC
        LIMIT %s
        """
        try:
            return self.execute_custom_mysql_query(sql, (limit,))
        except Exception as exc:
            logger.error("get_recent_interactions failed: %s", exc)
            return []
