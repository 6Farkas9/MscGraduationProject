# BackEnd/app/infrastructure/db/mongodb_client.py
import logging
from typing import Optional

import pymongo
from pymongo.database import Database as MongoDatabase
from pymongo.errors import ConnectionFailure

from app.core.settings import db_settings

logger = logging.getLogger(__name__)


class MongoDBClient:
    """
    MongoDB 连接管理器
    - 使用单例共享 pymongo.MongoClient
    - 不包含业务逻辑
    """

    _instance: Optional["MongoDBClient"] = None
    _client: Optional[pymongo.MongoClient] = None
    _database: Optional[MongoDatabase] = None

    def __new__(cls) -> "MongoDBClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _create_database(self) -> MongoDatabase:
        uri = db_settings.build_mongodb_uri()
        client = pymongo.MongoClient(uri)
        cfg = db_settings.mongodb_config
        db = client[cfg["database"]]
        try:
            client.admin.command("ismaster")
        except ConnectionFailure as exc:
            logger.error("MongoDB connection failed: %s", exc)
            raise
        self._client = client
        self._database = db
        logger.info("MongoDB connection established")
        return db

    def get_database(self) -> MongoDatabase:
        if self._database is None:
            return self._create_database()
        return self._database

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("MongoDB connection closed")
