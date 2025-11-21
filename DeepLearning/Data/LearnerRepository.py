# LearnerRepository.py 修复版
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Dict, List, Optional, Any

from Data.MongoDBOperator import mongodb_op

class LearnerRepository:
    """学习者知识追踪结果存储仓库 - 修复保存格式"""
    
    def __init__(self):
        self.collection_name = "Learners"
        self.db_op = mongodb_op
        
    def save_learner_knowledge_state(self, learner_id: str, 
                                   concept_mastery: torch.Tensor,
                                   model_type: str = 'KT',
                                   sequence_data: Dict[str, Any] = None,
                                   metadata: Dict[str, Any] = None) -> str:
        """
        保存学习者知识点掌握状态 - 修复格式
        
        Args:
            learner_id: 学习者ID
            concept_mastery: 知识点掌握状态tensor
            model_type: 模型类型 'KT' | 'CD'
            sequence_data: 序列数据信息
            metadata: 其他元数据
        """
        document = {
            'uid': learner_id,  # 使用uid字段
            'KT': metadata.get('concept_mastery_dict', {}) if metadata else {},  # 直接保存KT结果字典
            'updated_time': self._get_current_time()
        }
        
        # 如果学习者已存在，则更新
        existing = self.db_op.find_one(self.collection_name, {'uid': learner_id})
        if existing:
            return self.db_op.update_one(self.collection_name, 
                                       {'uid': learner_id}, 
                                       document)
        
        return self.db_op.insert_one(self.collection_name, document)
    
    def save_batch_learner_states(self, learner_states: List[Dict]) -> List[str]:
        """
        批量保存学习者状态 - 修复格式
        """
        documents = []
        
        for state in learner_states:
            learner_id = state['learner_id']
            concept_mastery_dict = state.get('concept_mastery', {})
            metadata = state.get('metadata', {})
            
            document = {
                'uid': learner_id,
                'KT': concept_mastery_dict,  # 直接保存KT结果字典
                'updated_time': self._get_current_time()
            }
                
            documents.append(document)
        
        # 使用upsert逻辑
        result = self.db_op.upsert_many(
            self.collection_name, 
            documents, 
            key_fields=['uid']
        )
        
        return result.get('upserted_count', 0) + result.get('modified_count', 0)
    
    def get_learner_knowledge_state(self, learner_id: str, device: str = 'cpu') -> Optional[Dict[str, Any]]:
        """获取学习者知识点掌握状态"""
        document = self.db_op.find_one(self.collection_name, {'uid': learner_id})
        
        if not document:
            return None
            
        result = {
            'uid': document['uid'],
            'KT': document.get('KT', {}),
            'updated_time': document.get('updated_time')
        }
            
        return result
    
    def get_all_learners_states(self, device: str = 'cpu') -> Dict[str, Dict[str, Any]]:
        """获取所有学习者的状态"""
        documents = self.db_op.find_many(self.collection_name)
        
        result = {}
        for doc in documents:
            learner_id = doc['uid']
            result[learner_id] = {
                'KT': doc.get('KT', {}),
                'updated_time': doc.get('updated_time')
            }
                
        return result
    
    def update_learner_state(self, learner_id: str, 
                           concept_mastery: torch.Tensor = None,
                           model_type: str = None,
                           sequence_data: Dict[str, Any] = None,
                           metadata: Dict[str, Any] = None) -> int:
        """更新学习者状态"""
        update_fields = {
            'updated_time': self._get_current_time()
        }
        
        if metadata and 'concept_mastery_dict' in metadata:
            update_fields['KT'] = metadata['concept_mastery_dict']
            
        return self.db_op.update_one(self.collection_name, 
                                   {'uid': learner_id}, 
                                   update_fields)
    
    def delete_learner_state(self, learner_id: str) -> int:
        """删除学习者状态"""
        return self.db_op.delete_one(self.collection_name, {'uid': learner_id})
    
    def get_learner_count(self, model_type: str = None) -> int:
        """获取学习者数量"""
        return self.db_op.count_documents(self.collection_name, {})
    
    def create_indexes(self):
        """创建索引"""
        self.db_op.create_index(self.collection_name, [('uid', 1)], unique=True)
    
    def _get_current_time(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now()

# 创建全局实例
learner_repo = LearnerRepository()