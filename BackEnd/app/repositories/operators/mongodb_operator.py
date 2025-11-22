# mongodb_operator.py
import logging
from typing import List, Dict, Any, Optional, Union
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError, DuplicateKeyError

# 修改为相对导入
from ...core.database import MongoDBManager

logger = logging.getLogger(__name__)

class MongoDBOperator:
    """MongoDB通用操作类 - 提供基础的MongoDB CRUD操作"""
    
    def __init__(self):
        self.mongodb_manager = MongoDBManager()
    
    def get_database(self) -> Database:
        """获取数据库实例"""
        return self.mongodb_manager.get_database()
    
    def get_collection(self, collection_name: str) -> Collection:
        """获取集合对象"""
        database = self.get_database()
        return database[collection_name]
    
    def is_valid_objectid(self, objectid_str: str) -> bool:
        """检查字符串是否为有效的ObjectId"""
        try:
            ObjectId(objectid_str)
            return True
        except (InvalidId, TypeError):
            return False
    
    def convert_to_objectid(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """将文档中的_id字段转换为ObjectId（如果有效）"""
        if '_id' in document and isinstance(document['_id'], str) and self.is_valid_objectid(document['_id']):
            document['_id'] = ObjectId(document['_id'])
        return document
    
    # 基础CRUD操作
    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """插入单个文档"""
        try:
            collection = self.get_collection(collection_name)
            document = self.convert_to_objectid(document)
            result = collection.insert_one(document)
            return str(result.inserted_id)
        except DuplicateKeyError as e:
            logger.error(f"插入文档失败 - 重复键: {e}, 集合: {collection_name}")
            raise
        except PyMongoError as e:
            logger.error(f"插入文档失败: {e}, 集合: {collection_name}")
            raise
    
    def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """批量插入文档"""
        try:
            collection = self.get_collection(collection_name)
            processed_docs = [self.convert_to_objectid(doc) for doc in documents]
            result = collection.insert_many(processed_docs)
            return [str(inserted_id) for inserted_id in result.inserted_ids]
        except PyMongoError as e:
            logger.error(f"批量插入文档失败: {e}, 集合: {collection_name}")
            raise
    
    def find_by_id(self, collection_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """根据ID查找文档"""
        try:
            collection = self.get_collection(collection_name)
            
            # 根据ID类型构建查询条件
            if self.is_valid_objectid(document_id):
                query = {'_id': ObjectId(document_id)}
            else:
                query = {'_id': document_id}
            
            document = collection.find_one(query)
            
            # 将ObjectId转换为字符串以便于使用
            if document and '_id' in document:
                document['_id'] = str(document['_id'])
            
            return document
        except PyMongoError as e:
            logger.error(f"根据ID查找文档失败: {e}, 集合: {collection_name}, ID: {document_id}")
            return None
    
    def find_by_field(self, collection_name: str, field: str, value: Any) -> List[Dict[str, Any]]:
        """根据字段值查找文档"""
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find({field: value})
            
            documents = []
            for doc in cursor:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            return documents
        except PyMongoError as e:
            logger.error(f"根据字段查找文档失败: {e}, 集合: {collection_name}, 字段: {field}")
            return []
    
    def find_by_fields(self, collection_name: str, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据多个字段条件查找文档"""
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(conditions)
            
            documents = []
            for doc in cursor:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            return documents
        except PyMongoError as e:
            logger.error(f"根据多字段查找文档失败: {e}, 集合: {collection_name}, 条件: {conditions}")
            return []
    
    def find_all(self, collection_name: str, limit: int = 0) -> List[Dict[str, Any]]:
        """获取集合中的所有文档"""
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find().limit(limit) if limit > 0 else collection.find()
            
            documents = []
            for doc in cursor:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            return documents
        except PyMongoError as e:
            logger.error(f"获取所有文档失败: {e}, 集合: {collection_name}")
            return []
    
    def find_with_projection(self, collection_name: str, query: Dict[str, Any] = None, 
                           projection: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """带投影的查询"""
        try:
            collection = self.get_collection(collection_name)
            query = query or {}
            projection = projection or {}
            
            cursor = collection.find(query, projection)
            
            documents = []
            for doc in cursor:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            return documents
        except PyMongoError as e:
            logger.error(f"带投影查询失败: {e}, 集合: {collection_name}")
            return []
    
    # 高级查询操作
    def find_with_sort(self, collection_name: str, sort_field: str, 
                      sort_order: str = 'asc', limit: int = 0) -> List[Dict[str, Any]]:
        """带排序的查询"""
        try:
            collection = self.get_collection(collection_name)
            sort_direction = ASCENDING if sort_order.lower() == 'asc' else DESCENDING
            
            cursor = collection.find().sort(sort_field, sort_direction)
            if limit > 0:
                cursor = cursor.limit(limit)
            
            documents = []
            for doc in cursor:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            return documents
        except PyMongoError as e:
            logger.error(f"带排序查询失败: {e}, 集合: {collection_name}")
            return []
    
    def find_by_field_with_sort(self, collection_name: str, field: str, value: Any,
                               sort_field: str, sort_order: str = 'asc', 
                               limit: int = 0) -> List[Dict[str, Any]]:
        """根据字段值查询并排序"""
        try:
            collection = self.get_collection(collection_name)
            sort_direction = ASCENDING if sort_order.lower() == 'asc' else DESCENDING
            
            cursor = collection.find({field: value}).sort(sort_field, sort_direction)
            if limit > 0:
                cursor = cursor.limit(limit)
            
            documents = []
            for doc in cursor:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            return documents
        except PyMongoError as e:
            logger.error(f"字段查询带排序失败: {e}, 集合: {collection_name}")
            return []
    
    def find_with_pagination(self, collection_name: str, query: Dict[str, Any] = None,
                           sort_field: str = None, sort_order: str = 'desc',
                           page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """分页查询"""
        try:
            collection = self.get_collection(collection_name)
            query = query or {}
            
            # 计算跳过的文档数量
            skip_count = (page - 1) * page_size
            
            # 构建查询
            cursor = collection.find(query)
            
            # 应用排序
            if sort_field:
                sort_direction = ASCENDING if sort_order.lower() == 'asc' else DESCENDING
                cursor = cursor.sort(sort_field, sort_direction)
            
            # 应用分页
            cursor = cursor.skip(skip_count).limit(page_size)
            
            # 获取总数量
            total_count = collection.count_documents(query)
            
            # 处理结果
            documents = []
            for doc in cursor:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                documents.append(doc)
            
            return {
                'documents': documents,
                'total_count': total_count,
                'page': page,
                'page_size': page_size,
                'total_pages': (total_count + page_size - 1) // page_size
            }
        except PyMongoError as e:
            logger.error(f"分页查询失败: {e}, 集合: {collection_name}")
            return {'documents': [], 'total_count': 0, 'page': page, 'page_size': page_size, 'total_pages': 0}
    
    # 更新操作
    def update_one(self, collection_name: str, query: Dict[str, Any], 
                  update_data: Dict[str, Any]) -> bool:
        """更新单个文档"""
        try:
            collection = self.get_collection(collection_name)
            query = self.convert_to_objectid(query)
            
            result = collection.update_one(query, {'$set': update_data})
            return result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"更新文档失败: {e}, 集合: {collection_name}, 查询: {query}")
            return False
    
    def update_many(self, collection_name: str, query: Dict[str, Any], 
                   update_data: Dict[str, Any]) -> int:
        """批量更新文档"""
        try:
            collection = self.get_collection(collection_name)
            query = self.convert_to_objectid(query)
            
            result = collection.update_many(query, {'$set': update_data})
            return result.modified_count
        except PyMongoError as e:
            logger.error(f"批量更新文档失败: {e}, 集合: {collection_name}, 查询: {query}")
            return 0
    
    def upsert_one(self, collection_name: str, query: Dict[str, Any], 
                  update_data: Dict[str, Any]) -> bool:
        """更新或插入文档（upsert）"""
        try:
            collection = self.get_collection(collection_name)
            query = self.convert_to_objectid(query)
            
            result = collection.update_one(query, {'$set': update_data}, upsert=True)
            return result.upserted_id is not None or result.modified_count > 0
        except PyMongoError as e:
            logger.error(f"upsert文档失败: {e}, 集合: {collection_name}, 查询: {query}")
            return False
    
    # 删除操作
    def delete_one(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """删除单个文档"""
        try:
            collection = self.get_collection(collection_name)
            query = self.convert_to_objectid(query)
            
            result = collection.delete_one(query)
            return result.deleted_count > 0
        except PyMongoError as e:
            logger.error(f"删除文档失败: {e}, 集合: {collection_name}, 查询: {query}")
            return False
    
    def delete_many(self, collection_name: str, query: Dict[str, Any]) -> int:
        """批量删除文档"""
        try:
            collection = self.get_collection(collection_name)
            query = self.convert_to_objectid(query)
            
            result = collection.delete_many(query)
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"批量删除文档失败: {e}, 集合: {collection_name}, 查询: {query}")
            return 0
    
    # 聚合操作
    def aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行聚合管道"""
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.aggregate(pipeline)
            
            results = []
            for doc in cursor:
                # 处理ObjectId转换
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
                results.append(doc)
            
            return results
        except PyMongoError as e:
            logger.error(f"聚合查询失败: {e}, 集合: {collection_name}, 管道: {pipeline}")
            return []
    
    # 统计操作
    def count_documents(self, collection_name: str, query: Dict[str, Any] = None) -> int:
        """统计文档数量"""
        try:
            collection = self.get_collection(collection_name)
            query = query or {}
            query = self.convert_to_objectid(query)
            
            return collection.count_documents(query)
        except PyMongoError as e:
            logger.error(f"统计文档数量失败: {e}, 集合: {collection_name}, 查询: {query}")
            return 0
    
    def distinct_values(self, collection_name: str, field: str, 
                       query: Dict[str, Any] = None) -> List[Any]:
        """获取字段的唯一值"""
        try:
            collection = self.get_collection(collection_name)
            query = query or {}
            query = self.convert_to_objectid(query)
            
            return collection.distinct(field, query)
        except PyMongoError as e:
            logger.error(f"获取唯一值失败: {e}, 集合: {collection_name}, 字段: {field}")
            return []
    
    # 索引操作
    def create_index(self, collection_name: str, keys: List[tuple], 
                    unique: bool = False, background: bool = True) -> str:
        """创建索引"""
        try:
            collection = self.get_collection(collection_name)
            index_name = collection.create_index(keys, unique=unique, background=background)
            logger.info(f"创建索引成功: {index_name}, 集合: {collection_name}")
            return index_name
        except PyMongoError as e:
            logger.error(f"创建索引失败: {e}, 集合: {collection_name}, 键: {keys}")
            raise
    
    def list_indexes(self, collection_name: str) -> List[Dict[str, Any]]:
        """列出集合的所有索引"""
        try:
            collection = self.get_collection(collection_name)
            return list(collection.list_indexes())
        except PyMongoError as e:
            logger.error(f"列出索引失败: {e}, 集合: {collection_name}")
            return []
    
    # 集合操作
    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        try:
            database = self.get_database()
            return collection_name in database.list_collection_names()
        except PyMongoError as e:
            logger.error(f"检查集合存在失败: {e}, 集合: {collection_name}")
            return False
    
    def drop_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            collection = self.get_collection(collection_name)
            collection.drop()
            logger.info(f"删除集合成功: {collection_name}")
            return True
        except PyMongoError as e:
            logger.error(f"删除集合失败: {e}, 集合: {collection_name}")
            return False

# 全局操作器实例
mongodb_operator = MongoDBOperator()