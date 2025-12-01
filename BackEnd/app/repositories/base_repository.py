# base_repository.py
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.core.database import db_manager
from .operators.mysql_operator import mysql_operator
from .operators.mongodb_operator import mongodb_operator

logger = logging.getLogger(__name__)

class BaseRepository:
    """基础仓库类，提供通用的数据访问方法和多数据库支持"""
    
    def __init__(self):
        self.mysql_operator = mysql_operator
        self.mongodb_operator = mongodb_operator
    
    # 通用工具方法
    def extract_uids_from_results(self, results: List[Dict[str, Any]], uid_field: str = 'uid') -> List[str]:
        """从结果列表中提取UID列表"""
        return [result[uid_field] for result in results if uid_field in result]
    
    def filter_unique_uids(self, uid_lists: List[List[str]]) -> List[str]:
        """从多个UID列表中过滤出唯一UID"""
        unique_uids = set()
        for uid_list in uid_lists:
            unique_uids.update(uid_list)
        return list(unique_uids)
    
    def remove_uid_from_list(self, uid_list: List[str], uid_to_remove: str) -> List[str]:
        """从UID列表中移除指定的UID"""
        return [uid for uid in uid_list if uid != uid_to_remove]
    
    # MySQL数据访问方法
    def execute_custom_mysql_query(self, query: str, params: Tuple = None) -> List[Dict[str, Any]]:
        """执行自定义MySQL查询"""
        try:
            return self.mysql_operator.execute_custom_query(query, params)
        except Exception as e:
            logger.error(f"执行自定义查询失败: {e}, SQL: {query}")
            return []
    
    def execute_custom_single_query(self, query: str, params: Tuple = None) -> Optional[Dict[str, Any]]:
        """执行自定义MySQL查询返回单条记录"""
        try:
            return self.mysql_operator.execute_custom_single_query(query, params)
        except Exception as e:
            logger.error(f"执行自定义单条查询失败: {e}, SQL: {query}")
            return None
    
    # 通用数据访问方法
    def get_learner_basic_info(self, learner_uid: str) -> Optional[Dict[str, Any]]:
        """获取学习者基本信息"""
        try:
            return self.mysql_operator.fetch_by_uid('BasicLearners', learner_uid)
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 基本信息失败: {e}")
            return None
    
    def get_entity_info(self, table: str, entity_uid: str) -> Optional[Dict[str, Any]]:
        """获取实体信息"""
        try:
            return self.mysql_operator.fetch_by_uid(table, entity_uid)
        except Exception as e:
            logger.error(f"获取实体 {entity_uid} 从表 {table} 失败: {e}")
            return None
    
    def get_entities_info(self, table: str, entity_uids: List[str]) -> List[Dict[str, Any]]:
        """批量获取实体信息"""
        if not entity_uids:
            return []
        
        try:
            return self.mysql_operator.fetch_in_list(table, 'uid', entity_uids)
        except Exception as e:
            logger.error(f"批量获取实体从表 {table} 失败: {e}")
            return []
    
    # MongoDB数据访问方法
    def get_mongodb_document(self, collection_name: str, document_id: str) -> Optional[Dict[str, Any]]:
        """获取MongoDB文档"""
        try:
            return self.mongodb_operator.find_by_id(collection_name, document_id)
        except Exception as e:
            logger.error(f"获取MongoDB文档失败: {e}, 集合: {collection_name}, ID: {document_id}")
            return None
    
    def get_mongodb_documents(self, collection_name: str, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """获取MongoDB文档列表"""
        try:
            return self.mongodb_operator.find_by_fields(collection_name, query or {})
        except Exception as e:
            logger.error(f"获取MongoDB文档列表失败: {e}, 集合: {collection_name}")
            return []
    
    def insert_mongodb_document(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """插入MongoDB文档"""
        try:
            return self.mongodb_operator.insert_one(collection_name, document)
        except Exception as e:
            logger.error(f"插入MongoDB文档失败: {e}, 集合: {collection_name}")
            return None
    
    def update_mongodb_document(self, collection_name: str, query: Dict[str, Any], 
                               update_data: Dict[str, Any]) -> bool:
        """更新MongoDB文档"""
        try:
            return self.mongodb_operator.update_one(collection_name, query, update_data)
        except Exception as e:
            logger.error(f"更新MongoDB文档失败: {e}, 集合: {collection_name}")
            return False
    
    def execute_mongodb_aggregation(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行MongoDB聚合查询"""
        try:
            return self.mongodb_operator.aggregate(collection_name, pipeline)
        except Exception as e:
            logger.error(f"执行MongoDB聚合失败: {e}, 集合: {collection_name}")
            return []
        
    def get_concept_uid_to_id_mapping(self) -> Dict[str, int]:
        """获取知识点UID到ID的映射（按id顺序）"""
        try:
            query = "SELECT uid, id FROM Concepts ORDER BY id"
            results = self.mysql_operator.execute_custom_query(query)
            
            # 构建映射字典 {uid: id}
            uid_to_id_map = {}
            for result in results:
                uid_to_id_map[result['uid']] = result['id']
                
            return uid_to_id_map
        except Exception as e:
            logger.error(f"获取知识点UID到ID映射失败: {e}")
            return {}