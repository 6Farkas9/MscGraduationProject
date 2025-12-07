# BackEnd/app/infrastructure/db/mysql_client.py
import logging
from typing import Optional, Dict, Any, List
import pymysql
from pymysql.connections import Connection
from pymysql.cursors import DictCursor

from app.core.settings import db_settings

logger = logging.getLogger(__name__)


class MySQLClient:
    """
    MySQL 连接管理器
    - 使用单例保证进程内共享一个底层连接
    - 不包含业务逻辑，仅负责连接与基础查询
    """

    _instance: Optional["MySQLClient"] = None
    _connection: Optional[Connection] = None

    def __new__(cls) -> "MySQLClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _create_connection(self) -> Connection:
        cfg: Dict[str, Any] = db_settings.mysql_config
        # autocommit 单独设置
        autocommit = cfg.pop("autocommit", True)
        conn = pymysql.connect(**cfg, cursorclass=DictCursor)
        conn.autocommit(autocommit)
        logger.info("MySQL connection established")
        return conn

    def get_connection(self) -> Connection:
        if self._connection is None or not self._connection.open:
            self._connection = self._create_connection()
        return self._connection

    def close(self) -> None:
        if self._connection and self._connection.open:
            self._connection.close()
            self._connection = None
            logger.info("MySQL connection closed")

    # 基础查询封装（供 Operator 使用）
    def query_many(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params or ())
                return list(cursor.fetchall())
        except Exception as exc:
            logger.error("MySQL query_many failed: %s, sql=%s", exc, sql)
            raise

    def query_one(self, sql: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params or ())
                return cursor.fetchone()
        except Exception as exc:
            logger.error("MySQL query_one failed: %s, sql=%s", exc, sql)
            raise
