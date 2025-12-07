# BackEnd/app/infrastructure/db/mongodb_operator.py
import logging
from typing import List, Dict, Any, Optional

from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError, DuplicateKeyError

from app.infrastructure.db.mongodb_client import MongoDBClient
from app.shared.utils.mongo_utils import (
    is_valid_object_id,
    convert_id_in_query,
    normalize_mongo_document,
    normalize_mongo_documents,
)

logger = logging.getLogger(__name__)


class MongoDBOperator:
    """
    MongoDB 通用操作类（依赖 MongoDBClient）
    - 所有与 ObjectId / 文档规范化相关的逻辑统一放在 shared.utils.mongo_utils 中
    - 这里只保留“调用 + 组合”逻辑，不再重复粘贴转换代码
    """

    def __init__(self, client: Optional[MongoDBClient] = None) -> None:
        self._client = client or MongoDBClient()

    # --- 基础工具 ---

    def _db(self) -> Database:
        return self._client.get_database()

    def _collection(self, name: str) -> Collection:
        return self._db()[name]

    # --- 插入 ---

    def insert_one(self, collection: str, document: Dict[str, Any]) -> Optional[str]:
        try:
            col = self._collection(collection)
            doc = convert_id_in_query(document)
            result = col.insert_one(doc)
            return str(result.inserted_id)
        except DuplicateKeyError as exc:
            logger.error("Mongo insert_one duplicate key: %s, collection=%s", exc, collection)
            raise
        except PyMongoError as exc:
            logger.error("Mongo insert_one failed: %s, collection=%s", exc, collection)
            raise

    def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
        try:
            col = self._collection(collection)
            docs = [convert_id_in_query(doc) for doc in documents]
            result = col.insert_many(docs)
            return [str(_id) for _id in result.inserted_ids]
        except PyMongoError as exc:
            logger.error("Mongo insert_many failed: %s, collection=%s", exc, collection)
            raise

    # --- 查询 ---

    def find_by_id(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            col = self._collection(collection)
            if is_valid_object_id(doc_id):
                query = {"_id": convert_id_in_query({"_id": doc_id})["_id"]}
            else:
                query = {"_id": doc_id}
            doc = col.find_one(query)
            return normalize_mongo_document(doc) if doc else None
        except PyMongoError as exc:
            logger.error("Mongo find_by_id failed: %s, collection=%s, id=%s", exc, collection, doc_id)
            return None

    def find_by_field(self, collection: str, field: str, value: Any) -> List[Dict[str, Any]]:
        return self.find_by_fields(collection, {field: value})

    def find_by_fields(self, collection: str, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            col = self._collection(collection)
            cursor = col.find(conditions or {})
            return normalize_mongo_documents(cursor)
        except PyMongoError as exc:
            logger.error("Mongo find_by_fields failed: %s, collection=%s", exc, collection)
            return []

    def find_all(self, collection: str, limit: int = 0) -> List[Dict[str, Any]]:
        try:
            col = self._collection(collection)
            cursor = col.find()
            if limit > 0:
                cursor = cursor.limit(limit)
            return normalize_mongo_documents(cursor)
        except PyMongoError as exc:
            logger.error("Mongo find_all failed: %s, collection=%s", exc, collection)
            return []

    def find_with_projection(
        self,
        collection: str,
        query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            col = self._collection(collection)
            cursor = col.find(query or {}, projection or {})
            return normalize_mongo_documents(cursor)
        except PyMongoError as exc:
            logger.error("Mongo find_with_projection failed: %s, collection=%s", exc, collection)
            return []

    # --- 排序 / 分页 ---

    def find_with_sort(
        self, collection: str, sort_field: str, sort_order: str = "asc", limit: int = 0
    ) -> List[Dict[str, Any]]:
        try:
            col = self._collection(collection)
            direction = ASCENDING if sort_order.lower() == "asc" else DESCENDING
            cursor = col.find().sort(sort_field, direction)
            if limit > 0:
                cursor = cursor.limit(limit)
            return normalize_mongo_documents(cursor)
        except PyMongoError as exc:
            logger.error("Mongo find_with_sort failed: %s, collection=%s", exc, collection)
            return []

    def find_by_field_with_sort(
        self,
        collection: str,
        field: str,
        value: Any,
        sort_field: str,
        sort_order: str = "asc",
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        try:
            col = self._collection(collection)
            direction = ASCENDING if sort_order.lower() == "asc" else DESCENDING
            cursor = col.find({field: value}).sort(sort_field, direction)
            if limit > 0:
                cursor = cursor.limit(limit)
            return normalize_mongo_documents(cursor)
        except PyMongoError as exc:
            logger.error("Mongo find_by_field_with_sort failed: %s, collection=%s", exc, collection)
            return []

    def find_with_pagination(
        self,
        collection: str,
        query: Optional[Dict[str, Any]] = None,
        sort_field: Optional[str] = None,
        sort_order: str = "desc",
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        try:
            col = self._collection(collection)
            q = query or {}
            skip = (max(page, 1) - 1) * page_size
            cursor = col.find(q)
            if sort_field:
                direction = ASCENDING if sort_order.lower() == "asc" else DESCENDING
                cursor = cursor.sort(sort_field, direction)
            cursor = cursor.skip(skip).limit(page_size)

            docs = normalize_mongo_documents(cursor)
            total = col.count_documents(q)
            total_pages = (total + page_size - 1) // page_size if page_size > 0 else 1
            return {
                "documents": docs,
                "total_count": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
            }
        except PyMongoError as exc:
            logger.error("Mongo find_with_pagination failed: %s, collection=%s", exc, collection)
            return {
                "documents": [],
                "total_count": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0,
            }

    # --- 更新 / 删除 ---

    def update_one(self, collection: str, query: Dict[str, Any], update_data: Dict[str, Any]) -> bool:
        try:
            col = self._collection(collection)
            q = convert_id_in_query(query)
            result = col.update_one(q, {"$set": update_data})
            return result.modified_count > 0
        except PyMongoError as exc:
            logger.error("Mongo update_one failed: %s, collection=%s", exc, collection)
            return False

    def update_many(self, collection: str, query: Dict[str, Any], update_data: Dict[str, Any]) -> int:
        try:
            col = self._collection(collection)
            q = convert_id_in_query(query)
            result = col.update_many(q, {"$set": update_data})
            return int(result.modified_count)
        except PyMongoError as exc:
            logger.error("Mongo update_many failed: %s, collection=%s", exc, collection)
            return 0

    def upsert_one(self, collection: str, query: Dict[str, Any], update_data: Dict[str, Any]) -> bool:
        try:
            col = self._collection(collection)
            q = convert_id_in_query(query)
            result = col.update_one(q, {"$set": update_data}, upsert=True)
            return bool(result.upserted_id or result.modified_count)
        except PyMongoError as exc:
            logger.error("Mongo upsert_one failed: %s, collection=%s", exc, collection)
            return False

    def delete_one(self, collection: str, query: Dict[str, Any]) -> bool:
        try:
            col = self._collection(collection)
            q = convert_id_in_query(query)
            result = col.delete_one(q)
            return result.deleted_count > 0
        except PyMongoError as exc:
            logger.error("Mongo delete_one failed: %s, collection=%s", exc, collection)
            return False

    def delete_many(self, collection: str, query: Dict[str, Any]) -> int:
        try:
            col = self._collection(collection)
            q = convert_id_in_query(query)
            result = col.delete_many(q)
            return int(result.deleted_count)
        except PyMongoError as exc:
            logger.error("Mongo delete_many failed: %s, collection=%s", exc, collection)
            return 0

    # --- 聚合 / 统计 ---

    def aggregate(self, collection: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            col = self._collection(collection)
            cursor = col.aggregate(pipeline)
            return normalize_mongo_documents(cursor)
        except PyMongoError as exc:
            logger.error("Mongo aggregate failed: %s, collection=%s", exc, collection)
            return []

    def count_documents(self, collection: str, query: Optional[Dict[str, Any]] = None) -> int:
        try:
            col = self._collection(collection)
            q = convert_id_in_query(query or {})
            return int(col.count_documents(q))
        except PyMongoError as exc:
            logger.error("Mongo count_documents failed: %s, collection=%s", exc, collection)
            return 0

    def distinct_values(self, collection: str, field: str, query: Optional[Dict[str, Any]] = None) -> List[Any]:
        try:
            col = self._collection(collection)
            q = convert_id_in_query(query or {})
            return list(col.distinct(field, q))
        except PyMongoError as exc:
            logger.error("Mongo distinct_values failed: %s, collection=%s", exc, collection)
            return []

    # --- 索引 / 集合管理 ---

    def create_index(self, collection: str, keys: List[tuple], unique: bool = False, background: bool = True) -> str:
        try:
            col = self._collection(collection)
            name = col.create_index(keys, unique=unique, background=background)
            logger.info("Mongo index created: %s on %s", name, collection)
            return name
        except PyMongoError as exc:
            logger.error("Mongo create_index failed: %s, collection=%s", exc, collection)
            raise

    def list_indexes(self, collection: str) -> List[Dict[str, Any]]:
        try:
            col = self._collection(collection)
            return list(col.list_indexes())
        except PyMongoError as exc:
            logger.error("Mongo list_indexes failed: %s, collection=%s", exc, collection)
            return []

    def collection_exists(self, collection: str) -> bool:
        try:
            db = self._db()
            return collection in db.list_collection_names()
        except PyMongoError as exc:
            logger.error("Mongo collection_exists failed: %s, collection=%s", exc, collection)
            return False

    def drop_collection(self, collection: str) -> bool:
        try:
            col = self._collection(collection)
            col.drop()
            logger.info("Mongo collection dropped: %s", collection)
            return True
        except PyMongoError as exc:
            logger.error("Mongo drop_collection failed: %s, collection=%s", exc, collection)
            return False
