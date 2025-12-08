# resource_planning_repository.py
# -*- coding: utf-8 -*-
from typing import Dict, List

from app.data_access.base.mysql_base_repository import MySQLBaseRepository


class ResourcePlanningRepository(MySQLBaseRepository):
    """
    资源规划引擎专用 Repository：
    - 从 Concepts 表读取知识点名称
    - 禁止让大模型看到知识点 uid
    - 不直接操作 MySQLOperator，只复用 MySQLBaseRepository 的通用查询方法
    """

    def __init__(self, *args, **kwargs):
        """
        沿用 MySQLBaseRepository 的构造逻辑：
        - 若外部需要指定 mysql_operator，可仍然通过参数透传进来；
        - 若不传，则由 MySQLBaseRepository 内部自行创建默认 MySQLOperator。
        """
        super().__init__(*args, **kwargs)

    # ----------------------------------------------------------------------
    # 返回 uid → name 映射（仅 name 会被用于大模型输入）
    # ----------------------------------------------------------------------
    def get_concepts_by_uids(self, concept_uids: List[str]) -> Dict[str, str]:
        if not concept_uids:
            return {}

        # 使用基类中的 execute_custom_mysql_query（内部已经封装 Operator）
        placeholders = ",".join(["%s"] * len(concept_uids))
        sql = f"""
            SELECT uid, name FROM Concepts
            WHERE uid IN ({placeholders})
        """
        rows = self.execute_custom_mysql_query(sql, tuple(concept_uids))

        mapping: Dict[str, str] = {}
        for r in rows:
            uid = r.get("uid")
            name = r.get("name")
            if uid and name:
                mapping[uid] = name
        return mapping

    # ----------------------------------------------------------------------
    # 返回全部知识点名称（供大模型做“可用知识点全集”约束）
    # ----------------------------------------------------------------------
    def get_all_concept_names(self) -> List[str]:
        sql = "SELECT name FROM Concepts"
        rows = self.execute_custom_mysql_query(sql)
        return [r["name"] for r in rows if r.get("name")]
