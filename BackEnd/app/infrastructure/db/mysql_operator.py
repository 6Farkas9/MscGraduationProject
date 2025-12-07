# BackEnd/app/infrastructure/db/mysql_operator.py
import logging
from typing import List, Dict, Any, Optional, Tuple

from app.infrastructure.db.mysql_client import MySQLClient

logger = logging.getLogger(__name__)


class MySQLOperator:
    """
    MySQL 通用操作类
    - 仅包含基础的 CRUD / 通用查询逻辑
    - 不包含任何领域逻辑
    """

    def __init__(self, client: Optional[MySQLClient] = None) -> None:
        self._client = client or MySQLClient()

    # 基础查询
    def execute_query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        return self._client.query_many(sql, params)

    def execute_single_query(self, sql: str, params: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        return self._client.query_one(sql, params)

    # 批量执行
    def execute_many(self, sql: str, params_list: List[Tuple]) -> int:
        conn = self._client.get_connection()
        try:
            with conn.cursor() as cursor:
                affected_rows = cursor.executemany(sql, params_list)
                return int(affected_rows)
        except Exception as exc:
            logger.error("MySQL execute_many failed: %s, sql=%s", exc, sql)
            raise

    # 一些通用访问封装
    def fetch_by_uid(self, table: str, uid: str) -> Optional[Dict[str, Any]]:
        sql = f"SELECT * FROM {table} WHERE uid = %s"
        return self.execute_single_query(sql, (uid,))

    def fetch_all(self, table: str, columns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if columns:
            cols = ", ".join(columns)
            sql = f"SELECT {cols} FROM {table}"
        else:
            sql = f"SELECT * FROM {table}"
        return self.execute_query(sql)

    def fetch_by_field(self, table: str, field: str, value: Any) -> List[Dict[str, Any]]:
        sql = f"SELECT * FROM {table} WHERE {field} = %s"
        return self.execute_query(sql, (value,))

    def fetch_by_fields(self, table: str, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not conditions:
            return self.fetch_all(table)

        where_clause = " AND ".join(f"{key} = %s" for key in conditions.keys())
        sql = f"SELECT * FROM {table} WHERE {where_clause}"
        return self.execute_query(sql, tuple(conditions.values()))

    def fetch_in_list(self, table: str, field: str, values: List[Any]) -> List[Dict[str, Any]]:
        if not values:
            return []
        placeholders = ", ".join(["%s"] * len(values))
        sql = f"SELECT * FROM {table} WHERE {field} IN ({placeholders})"
        return self.execute_query(sql, tuple(values))

    def fetch_with_order(self, table: str, order_by: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        sql = f"SELECT * FROM {table} ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {limit}"
        return self.execute_query(sql)

    def fetch_by_field_with_order(
        self, table: str, field: str, value: Any, order_by: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        sql = f"SELECT * FROM {table} WHERE {field} = %s ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {limit}"
        return self.execute_query(sql, (value,))

    def count_records(self, table: str, conditions: Optional[Dict[str, Any]] = None) -> int:
        if conditions:
            where_clause = " AND ".join(f"{key} = %s" for key in conditions.keys())
            sql = f"SELECT COUNT(*) AS count FROM {table} WHERE {where_clause}"
            row = self.execute_single_query(sql, tuple(conditions.values()))
        else:
            sql = f"SELECT COUNT(*) AS count FROM {table}"
            row = self.execute_single_query(sql)

        return int(row["count"]) if row and "count" in row else 0

    # 自定义 SQL（留给 Repository 使用）
    def execute_custom_query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        return self.execute_query(sql, params)

    def execute_custom_single_query(self, sql: str, params: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        return self.execute_single_query(sql, params)
