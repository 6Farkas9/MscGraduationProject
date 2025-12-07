# BackEnd/app/data_access/prediction/kt_repository.py
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from app.data_access.base.mysql_base_repository import MySQLBaseRepository

logger = logging.getLogger(__name__)


class KTRepository(MySQLBaseRepository):
    """知识追踪(KT)模型数据仓库 - 输出纯数据结构"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # ---- 核心序列接口 ----

    def get_learner_interaction_sequence(
        self, learner_uid: str, max_seq_len: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        获取单个学习者的完整交互序列（包含学习单元 & 题目）
        返回：7 元素列表 [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]
        """
        sql = """
        SELECT lrn_uid, unt_uid, additioninfo1, additioninfo2, create_time
        FROM Interaction
        WHERE lrn_uid = %s
        ORDER BY create_time ASC
        """
        params = (learner_uid,)
        if max_seq_len:
            sql += f" LIMIT {max_seq_len}"

        try:
            interactions = self.execute_custom_mysql_query(sql, params)
            if not interactions:
                return {
                    "learner_uid": learner_uid,
                    "sequence": [[], [], [], [], [], [], []],
                    "seq_len": 0,
                    "interaction_count": 0,
                    "question_count": 0,
                    "valid_prediction_count": 0,
                }

            unit_uids = list({row["unt_uid"] for row in interactions})
            unit_types = self._get_unit_types_batch(unit_uids)
            unit_concepts = self._get_unit_concepts_batch(unit_uids)

            seq_data = self._build_sequence_data(interactions, unit_types, unit_concepts)
            seq_len = len(seq_data[0])

            return {
                "learner_uid": learner_uid,
                "sequence": seq_data,
                "seq_len": seq_len,
                "interaction_count": len(interactions),
                "question_count": sum(seq_data[3]),
                "valid_prediction_count": sum(seq_data[5]),
            }
        except Exception as exc:
            logger.error("get_learner_interaction_sequence failed for %s: %s", learner_uid, exc)
            return {
                "learner_uid": learner_uid,
                "sequence": [[], [], [], [], [], [], []],
                "seq_len": 0,
                "interaction_count": 0,
                "question_count": 0,
                "valid_prediction_count": 0,
            }

    def get_learner_interaction_sequences_batch(
        self, learner_uids: List[str], max_seq_len: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量获取多个学习者交互序列
        """
        if not learner_uids:
            return {}

        placeholders = ", ".join(["%s"] * len(learner_uids))
        sql = f"""
        SELECT lrn_uid, unt_uid, additioninfo1, additioninfo2, create_time
        FROM Interaction
        WHERE lrn_uid IN ({placeholders})
        ORDER BY lrn_uid, create_time ASC
        """
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(learner_uids))

            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[row["lrn_uid"]].append(row)

            all_units: set[str] = set()
            for lst in grouped.values():
                for row in lst:
                    all_units.add(row["unt_uid"])

            unit_types = self._get_unit_types_batch(list(all_units))
            unit_concepts = self._get_unit_concepts_batch(list(all_units))

            result: Dict[str, Dict[str, Any]] = {}
            for uid in learner_uids:
                interactions = grouped.get(uid, [])
                if max_seq_len and len(interactions) > max_seq_len:
                    interactions = interactions[-max_seq_len:]

                if not interactions:
                    result[uid] = {
                        "learner_uid": uid,
                        "sequence": [[], [], [], [], [], [], []],
                        "seq_len": 0,
                        "interaction_count": 0,
                        "question_count": 0,
                        "valid_prediction_count": 0,
                    }
                    continue

                seq_data = self._build_sequence_data(interactions, unit_types, unit_concepts)
                result[uid] = {
                    "learner_uid": uid,
                    "sequence": seq_data,
                    "seq_len": len(seq_data[0]),
                    "interaction_count": len(interactions),
                    "question_count": sum(seq_data[3]),
                    "valid_prediction_count": sum(seq_data[5]),
                }

            return result
        except Exception as exc:
            logger.error("get_learner_interaction_sequences_batch failed: %s", exc)
            return {
                uid: {
                    "learner_uid": uid,
                    "sequence": [[], [], [], [], [], [], []],
                    "seq_len": 0,
                    "interaction_count": 0,
                    "question_count": 0,
                    "valid_prediction_count": 0,
                }
                for uid in learner_uids
            }

    # ---- 辅助查询：单位类型 & 单位-知识点 ----

    def _get_unit_types_batch(self, unit_uids: List[str]) -> Dict[str, str]:
        if not unit_uids:
            return {}

        placeholders = ", ".join(["%s"] * len(unit_uids))
        sql = f"SELECT uid, type FROM Units WHERE uid IN ({placeholders})"
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(unit_uids))
            return {row["uid"]: row.get("type", "unknown") for row in rows}
        except Exception as exc:
            logger.error("_get_unit_types_batch failed: %s", exc)
            return {}

    def _get_unit_concepts_batch(self, unit_uids: List[str]) -> Dict[str, List[str]]:
        if not unit_uids:
            return {}

        placeholders = ", ".join(["%s"] * len(unit_uids))
        sql = f"""
        SELECT unt_uid, cpt_uid
        FROM Unit_Concept
        WHERE unt_uid IN ({placeholders})
        """
        try:
            rows = self.execute_custom_mysql_query(sql, tuple(unit_uids))
            mapping: Dict[str, List[str]] = defaultdict(list)
            for row in rows:
                mapping[row["unt_uid"]].append(row["cpt_uid"])
            return dict(mapping)
        except Exception as exc:
            logger.error("_get_unit_concepts_batch failed: %s", exc)
            return {}

    # ---- 序列构建逻辑：保留原 7-元素结构 ----

    def _build_sequence_data(
        self,
        interactions: List[Dict[str, Any]],
        unit_types: Dict[str, str],
        unit_concepts: Dict[str, List[str]],
    ) -> List[List[Any]]:
        """
        构建 7 元素序列：
        [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]
        """

        unt_uids: List[str] = []
        add1s: List[float] = []
        add2s: List[float] = []
        is_questions: List[int] = []
        results: List[float] = []
        prediction_masks: List[int] = []
        next_results: List[float] = []

        def _is_question(unit_type: str) -> bool:
            return unit_type == "question"

        for idx, row in enumerate(interactions):
            unt_uid = row["unt_uid"]
            utype = unit_types.get(unt_uid, "unknown")
            is_q = 1 if _is_question(utype) else 0

            # 对应原逻辑：题目交互时 additioninfo2 为正确性
            if is_q:
                result_val = float(row.get("additioninfo2") or 0.0)
            else:
                result_val = -1.0

            unt_uids.append(unt_uid)
            add1s.append(float(row.get("additioninfo1") or 0.0))
            add2s.append(float(row.get("additioninfo2") or 0.0))
            is_questions.append(is_q)
            results.append(result_val)

            if idx < len(interactions) - 1:
                nxt = interactions[idx + 1]
                nxt_type = unit_types.get(nxt["unt_uid"], "unknown")
                nxt_is_q = 1 if _is_question(nxt_type) else 0
                prediction_masks.append(1 if nxt_is_q else 0)
                next_results.append(float(nxt.get("additioninfo2") or 0.0) if nxt_is_q else 0.0)
            else:
                prediction_masks.append(0)
                next_results.append(0.0)

        return [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]

    # ---- 统计 & 校验 ----

    def get_interaction_statistics(self, learner_uid: str) -> Dict[str, Any]:
        sql = """
        SELECT
            COUNT(*) AS total_interactions,
            COUNT(DISTINCT unt_uid) AS unique_units,
            AVG(additioninfo1) AS avg_info1,
            AVG(additioninfo2) AS avg_info2,
            MIN(create_time) AS start_time,
            MAX(create_time) AS end_time
        FROM Interaction
        WHERE lrn_uid = %s
        """
        try:
            row = self.execute_custom_single_query(sql, (learner_uid,))
            if row:
                return {
                    "learner_uid": learner_uid,
                    "total_interactions": row.get("total_interactions") or 0,
                    "unique_units": row.get("unique_units") or 0,
                    "avg_additioninfo1": float(row.get("avg_info1") or 0.0),
                    "avg_additioninfo2": float(row.get("avg_info2") or 0.0),
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
            "unique_units": 0,
            "avg_additioninfo1": 0.0,
            "avg_additioninfo2": 0.0,
            "time_range": {"start": None, "end": None},
        }

    def validate_learner_has_interactions(self, learner_uid: str) -> bool:
        sql = "SELECT COUNT(*) AS count FROM Interaction WHERE lrn_uid = %s"
        try:
            row = self.execute_custom_single_query(sql, (learner_uid,))
            return bool(row and row.get("count", 0) > 0)
        except Exception as exc:
            logger.error("validate_learner_has_interactions failed for %s: %s", learner_uid, exc)
            return False
