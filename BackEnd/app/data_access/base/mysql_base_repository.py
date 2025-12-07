# BackEnd/app/data_access/base/mysql_base_repository.py
import logging
from typing import List, Dict, Any, Optional, Tuple

from app.infrastructure.db.mysql_operator import MySQLOperator
from app.shared.utils.repository_mixins import UIDRepositoryMixin, MappingRepositoryMixin

logger = logging.getLogger(__name__)


class MySQLBaseRepository(UIDRepositoryMixin, MappingRepositoryMixin):
    """
    所有基于 MySQL 的仓库公共基类
    - 提供通用的自定义 SQL 执行
    - 提供按 UID 获取实体等基础方法
    - 提供知识点 uid->id 映射等通用能力
    """

    def __init__(self, mysql_operator: Optional[MySQLOperator] = None) -> None:
        self._mysql = mysql_operator or MySQLOperator()

    # ---- 通用 SQL ----

    def execute_custom_mysql_query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        try:
            return self._mysql.execute_custom_query(sql, params)
        except Exception as exc:
            logger.error("execute_custom_mysql_query failed: %s, sql=%s", exc, sql)
            return []

    def execute_custom_single_query(self, sql: str, params: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        try:
            return self._mysql.execute_custom_single_query(sql, params)
        except Exception as exc:
            logger.error("execute_custom_single_query failed: %s, sql=%s", exc, sql)
            return None

    # ---- 通用实体访问 ----

    def get_learner_basic_info(self, learner_uid: str) -> Optional[Dict[str, Any]]:
        """
        获取学习者基础信息
        - 默认从 BasicLearners 表查询
        """
        try:
            return self._mysql.fetch_by_uid("BasicLearners", learner_uid)
        except Exception as exc:
            logger.error("get_learner_basic_info failed for %s: %s", learner_uid, exc)
            return None

    def get_entity_info(self, table: str, entity_uid: str) -> Optional[Dict[str, Any]]:
        """
        从任意表按 UID 获取实体
        """
        try:
            return self._mysql.fetch_by_uid(table, entity_uid)
        except Exception as exc:
            logger.error("get_entity_info failed: table=%s, uid=%s, err=%s", table, entity_uid, exc)
            return None

    def get_entities_info(self, table: str, entity_uids: List[str]) -> List[Dict[str, Any]]:
        """
        从任意表按 UID 列表批量获取实体
        """
        if not entity_uids:
            return []
        try:
            return self._mysql.fetch_in_list(table, "uid", entity_uids)
        except Exception as exc:
            logger.error("get_entities_info failed: table=%s, err=%s", table, exc)
            return []

    def get_concept_uid_to_id_mapping(self) -> Dict[str, int]:
        """
        获取知识点 UID -> ID 映射（按 id 升序）
        """
        sql = "SELECT uid, id FROM Concepts ORDER BY id"
        try:
            rows = self._mysql.execute_custom_query(sql)
            # 使用通用 mapping 工具函数构建映射
            raw_mapping = self.build_mapping_from_rows(rows, key_field="uid", value_field="id")
            # 确保所有 value 为 int
            return {k: int(v) for k, v in raw_mapping.items()}
        except Exception as exc:
            logger.error("get_concept_uid_to_id_mapping failed: %s", exc)
            return {}
