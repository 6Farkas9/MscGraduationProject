# EmbeddingRepository.py 修复版
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Dict, List, Optional, Tuple, Any
from bson import Binary

from Data.MongoDBOperator import mongodb_op

class EmbeddingRepository:
    """嵌入向量存储仓库 - 修复保存格式"""
    
    def __init__(self):
        self.collection_name = "Embeddings"
        self.db_op = mongodb_op
        
    def save_embeddings(self, embeddings_dict: Dict[str, Dict]) -> Dict[str, Any]:
        """
        保存嵌入向量 - 使用列表格式
        
        Args:
            embeddings_dict: {
                'lrn': {uid1: tensor1, uid2: tensor2, ...},
                'unt': {uid1: tensor1, uid2: tensor2, ...}, 
                'cpt': {uid1: tensor1, uid2: tensor2, ...}
            }
        
        Returns:
            保存统计信息
        """
        documents = []
        
        for entity_type, uid_tensor_dict in embeddings_dict.items():
            for uid, tensor in uid_tensor_dict.items():
                document = {
                    'uid': uid,
                    'entity_type': entity_type,  # 'lrn', 'unt', 'cpt'
                    'embedding': self.db_op.tensor_to_list(tensor),  # 使用列表格式
                    'updated_time': self._get_current_time()
                }
                documents.append(document)
        
        # 使用upsert逻辑
        result = self.db_op.upsert_many(
            self.collection_name, 
            documents, 
            key_fields=['uid', 'entity_type']
        )
        
        return result
    
    def save_single_embedding(self, uid: str, entity_type: str, embedding: torch.Tensor) -> str:
        """
        保存单个嵌入向量 - 使用列表格式
        """
        document = {
            'uid': uid,
            'entity_type': entity_type,
            'embedding': self.db_op.tensor_to_list(embedding),  # 使用列表格式
            'updated_time': self._get_current_time()
        }
        
        # 使用upsert逻辑
        return self.db_op.upsert_one(
            self.collection_name, 
            {'uid': uid, 'entity_type': entity_type}, 
            document
        )
    
    def get_embedding_by_uid(self, uid: str, entity_type: str = None, 
                           device: str = 'cpu') -> Optional[torch.Tensor]:
        """根据UID获取嵌入向量"""
        query = {'uid': uid}
        if entity_type:
            query['entity_type'] = entity_type
            
        document = self.db_op.find_one(self.collection_name, query)
        
        if document and 'embedding' in document:
            return self.db_op.list_to_tensor(document['embedding'], device)
        return None
    
    def get_embeddings_by_type(self, entity_type: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """获取指定类型的所有嵌入向量"""
        query = {'entity_type': entity_type}
        documents = self.db_op.find_many(self.collection_name, query)
        
        result = {}
        for doc in documents:
            if 'embedding' in doc and doc['embedding']:
                result[doc['uid']] = self.db_op.list_to_tensor(doc['embedding'], device)
        
        return result
    
    def get_all_embeddings(self, device: str = 'cpu') -> Dict[str, Dict[str, torch.Tensor]]:
        """获取所有嵌入向量，按类型分组"""
        documents = self.db_op.find_many(self.collection_name)
        
        result = {
            'lrn': {},
            'unt': {},
            'cpt': {}
        }
        
        for doc in documents:
            entity_type = doc.get('entity_type')
            uid = doc.get('uid')
            embedding_list = doc.get('embedding')
            
            if entity_type in result and uid and embedding_list:
                result[entity_type][uid] = self.db_op.list_to_tensor(embedding_list, device)
        
        return result
    
    def update_embedding(self, uid: str, entity_type: str, new_embedding: torch.Tensor) -> int:
        """更新嵌入向量"""
        query = {'uid': uid, 'entity_type': entity_type}
        update = {
            'embedding': self.db_op.tensor_to_list(new_embedding),  # 使用列表格式
            'updated_time': self._get_current_time()
        }
        
        return self.db_op.update_one(self.collection_name, query, update)
    
    def delete_embeddings_by_uid(self, uid: str, entity_type: str = None) -> int:
        """删除指定UID的嵌入向量"""
        query = {'uid': uid}
        if entity_type:
            query['entity_type'] = entity_type
            
        return self.db_op.delete_many(self.collection_name, query)
    
    def delete_embeddings_by_type(self, entity_type: str) -> int:
        """删除指定类型的所有嵌入向量"""
        query = {'entity_type': entity_type}
        return self.db_op.delete_many(self.collection_name, query)
    
    def get_embedding_count(self, entity_type: str = None) -> int:
        """获取嵌入向量数量"""
        query = {}
        if entity_type:
            query['entity_type'] = entity_type
            
        return self.db_op.count_documents(self.collection_name, query)
    
    def create_indexes(self):
        """创建索引"""
        # 创建复合索引，提高查询性能
        self.db_op.create_index(self.collection_name, [('uid', 1), ('entity_type', 1)], unique=True)
        self.db_op.create_index(self.collection_name, [('entity_type', 1)])
    
    def _get_current_time(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now()

# 创建全局实例
embedding_repo = EmbeddingRepository()