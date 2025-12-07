# BackEnd/app/data_access/base/mongodb_base_repository.py
import logging
from typing import List, Dict, Any, Optional

from app.infrastructure.db.mongodb_operator import MongoDBOperator
from app.shared.utils.repository_mixins import UIDRepositoryMixin

logger = logging.getLogger(__name__)


class MongoDBBaseRepository(UIDRepositoryMixin):
    """
    所有基于 MongoDB 的仓库公共基类
    - 目前你的领域仓库都只用 MySQL，这里预留给未来的 Mongo 仓库使用
    """

    def __init__(self, mongo_operator: Optional[MongoDBOperator] = None) -> None:
        self._mongo = mongo_operator or MongoDBOperator()

    def get_document(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._mongo.find_by_id(collection, doc_id)
        except Exception as exc:
            logger.error("get_document failed: collection=%s, id=%s, err=%s", collection, doc_id, exc)
            return None

    def get_documents(self, collection: str, query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            return self._mongo.find_by_fields(collection, query or {})
        except Exception as exc:
            logger.error("get_documents failed: collection=%s, err=%s", collection, exc)
            return []

    def insert_document(self, collection: str, document: Dict[str, Any]) -> Optional[str]:
        try:
            return self._mongo.insert_one(collection, document)
        except Exception as exc:
            logger.error("insert_document failed: collection=%s, err=%s", collection, exc)
            return None

    def update_document(self, collection: str, query: Dict[str, Any], update_data: Dict[str, Any]) -> bool:
        try:
            return self._mongo.update_one(collection, query, update_data)
        except Exception as exc:
            logger.error("update_document failed: collection=%s, err=%s", collection, exc)
            return False

    def aggregate(self, collection: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            return self._mongo.aggregate(collection, pipeline)
        except Exception as exc:
            logger.error("aggregate failed: collection=%s, err=%s", collection, exc)
            return []
