# base_repository.py
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Set
from app.core.database import db_manager
from .operators.mysql_operator import mysql_operator

logger = logging.getLogger(__name__)

class BaseRepository(ABC):
    """基础仓库类，提供通用的数据访问方法和多数据库支持"""
    
    def __init__(self):
        self.mysql_operator = mysql_operator
        # 预留MongoDB操作器，未来扩展
        self.mongodb_operator = None
    
    # 通用工具方法
    def build_uid_to_idx_mapping(self, uids: List[str]) -> Dict[str, int]:
        """构建UID到索引的映射"""
        return {uid: idx for idx, uid in enumerate(uids)}
    
    def build_idx_to_uid_mapping(self, uids: List[str]) -> Dict[int, str]:
        """构建索引到UID的映射"""
        return {idx: uid for idx, uid in enumerate(uids)}
    
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
    
    # 抽象方法 - 子类需要实现
    @abstractmethod
    def get_data_for_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """为单个学习者获取模型数据"""
        pass
    
    @abstractmethod
    def get_data_for_multiple_learners(self, learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
        """为多个学习者获取模型数据"""
        pass
    
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
    
    def execute_custom_mysql_query(self, query: str, params: Tuple = None) -> List[Dict[str, Any]]:
        """执行自定义MySQL查询"""
        try:
            return self.mysql_operator.execute_custom_query(query, params)
        except Exception as e:
            logger.error(f"执行自定义查询失败: {e}, SQL: {query}")
            return []
    
    # 预留MongoDB方法 - 未来扩展
    def get_mongodb_collection(self, collection_name: str):
        """获取MongoDB集合（预留）"""
        if self.mongodb_operator is None:
            logger.warning("MongoDB操作器未初始化")
            return None
        # 未来实现MongoDB操作器后，这里可以返回集合对象
        return None
    
    def execute_mongodb_query(self, collection_name: str, query: Dict[str, Any]):
        """执行MongoDB查询（预留）"""
        logger.warning("MongoDB功能尚未实现")
        return []