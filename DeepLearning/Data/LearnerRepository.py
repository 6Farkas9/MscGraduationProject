import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Dict, List, Optional, Any
from bson import Binary

from Data.MongoDBOperator import mongodb_op

class LearnerRepository:
    """学习者知识追踪结果存储仓库 - 增强版"""
    
    def __init__(self):
        self.collection_name = "Learners"
        self.db_op = mongodb_op
        
    def save_learner_knowledge_state(self, learner_id: str, 
                                   concept_mastery: torch.Tensor,
                                   model_type: str = 'KT',  # 新增：区分CD/KT
                                   sequence_data: Dict[str, Any] = None,
                                   metadata: Dict[str, Any] = None) -> str:
        """
        保存学习者知识点掌握状态
        
        Args:
            learner_id: 学习者ID
            concept_mastery: 知识点掌握状态tensor
            model_type: 模型类型 'KT' | 'CD'  # 新增参数
            sequence_data: 序列数据信息
            metadata: 其他元数据
        """
        document = {
            'learner_id': learner_id,
            'concept_mastery': self.db_op.tensor_to_binary(concept_mastery),
            'concept_num': concept_mastery.shape[-1] if concept_mastery is not None else 0,
            'sequence_length': concept_mastery.shape[0] if len(concept_mastery.shape) > 1 else 1,
            'model_type': model_type,  # 新增字段
            'created_time': self._get_current_time()
        }
        
        if sequence_data:
            document['sequence_data'] = sequence_data
            
        if metadata:
            document['metadata'] = metadata
            
        # 如果学习者已存在，则更新
        existing = self.db_op.find_one(self.collection_name, {'learner_id': learner_id})
        if existing:
            return self.db_op.update_one(self.collection_name, 
                                       {'learner_id': learner_id}, 
                                       document)
        
        return self.db_op.insert_one(self.collection_name, document)
    
    def save_batch_learner_states(self, learner_states: List[Dict]) -> List[str]:
        """
        批量保存学习者状态 - 增强参数支持
        
        Args:
            learner_states: [
                {
                    'learner_id': 'lrn001',
                    'concept_mastery': tensor,
                    'model_type': 'KT',  # 可选
                    'sequence_data': {...},
                    'metadata': {...}
                },
                ...
            ]
        """
        documents = []
        
        for state in learner_states:
            concept_mastery = state.get('concept_mastery')
            document = {
                'learner_id': state['learner_id'],
                'concept_mastery': self.db_op.tensor_to_binary(concept_mastery),
                'concept_num': concept_mastery.shape[-1] if concept_mastery is not None else 0,
                'sequence_length': concept_mastery.shape[0] if len(concept_mastery.shape) > 1 else 1,
                'model_type': state.get('model_type', 'KT'),  # 默认KT
                'created_time': self._get_current_time()
            }
            
            if 'sequence_data' in state:
                document['sequence_data'] = state['sequence_data']
                
            if 'metadata' in state:
                document['metadata'] = state['metadata']
                
            documents.append(document)
        
        return self.db_op.insert_many(self.collection_name, documents)
    
    def get_learners_by_model_type(self, model_type: str, device: str = 'cpu') -> Dict[str, Dict[str, Any]]:
        """
        根据模型类型获取学习者状态 - 新增方法
        
        Args:
            model_type: 模型类型 'KT' | 'CD'
            device: 返回tensor的设备
        """
        query = {'model_type': model_type}
        documents = self.db_op.find_many(self.collection_name, query)
        
        result = {}
        for doc in documents:
            learner_id = doc['learner_id']
            result[learner_id] = {
                'concept_mastery': self.db_op.binary_to_tensor(doc.get('concept_mastery'), device),
                'concept_num': doc.get('concept_num', 0),
                'sequence_length': doc.get('sequence_length', 1),
                'model_type': doc.get('model_type', 'KT'),
                'created_time': doc.get('created_time')
            }
            
            if 'sequence_data' in doc:
                result[learner_id]['sequence_data'] = doc['sequence_data']
                
            if 'metadata' in doc:
                result[learner_id]['metadata'] = doc['metadata']
                
        return result
    
    # 以下方法保持原有实现，只需在document中添加model_type字段
    def get_learner_knowledge_state(self, learner_id: str, device: str = 'cpu') -> Optional[Dict[str, Any]]:
        """获取学习者知识点掌握状态"""
        document = self.db_op.find_one(self.collection_name, {'learner_id': learner_id})
        
        if not document:
            return None
            
        result = {
            'learner_id': document['learner_id'],
            'concept_mastery': self.db_op.binary_to_tensor(document.get('concept_mastery'), device),
            'concept_num': document.get('concept_num', 0),
            'sequence_length': document.get('sequence_length', 1),
            'model_type': document.get('model_type', 'KT'),  # 新增字段
            'created_time': document.get('created_time')
        }
        
        if 'sequence_data' in document:
            result['sequence_data'] = document['sequence_data']
            
        if 'metadata' in document:
            result['metadata'] = document['metadata']
            
        return result
    
    def get_all_learners_states(self, device: str = 'cpu') -> Dict[str, Dict[str, Any]]:
        """获取所有学习者的状态"""
        documents = self.db_op.find_many(self.collection_name)
        
        result = {}
        for doc in documents:
            learner_id = doc['learner_id']
            result[learner_id] = {
                'concept_mastery': self.db_op.binary_to_tensor(doc.get('concept_mastery'), device),
                'concept_num': doc.get('concept_num', 0),
                'sequence_length': doc.get('sequence_length', 1),
                'model_type': doc.get('model_type', 'KT'),  # 新增字段
                'created_time': doc.get('created_time')
            }
            
            if 'sequence_data' in doc:
                result[learner_id]['sequence_data'] = doc['sequence_data']
                
            if 'metadata' in doc:
                result[learner_id]['metadata'] = doc['metadata']
                
        return result
    
    # 其他方法保持不变...
    def get_learners_by_concept_mastery(self, concept_index: int, 
                                      min_mastery: float = 0.5,
                                      device: str = 'cpu') -> List[Dict[str, Any]]:
        """根据知识点掌握程度筛选学习者"""
        # 实现保持不变，但返回的数据中会包含model_type
        
    def update_learner_state(self, learner_id: str, 
                           concept_mastery: torch.Tensor = None,
                           model_type: str = None,  # 新增参数
                           sequence_data: Dict[str, Any] = None,
                           metadata: Dict[str, Any] = None) -> int:
        """更新学习者状态"""
        update_fields = {
            'updated_time': self._get_current_time()
        }
        
        if concept_mastery is not None:
            update_fields['concept_mastery'] = self.db_op.tensor_to_binary(concept_mastery)
            update_fields['concept_num'] = concept_mastery.shape[-1] if concept_mastery is not None else 0
            update_fields['sequence_length'] = concept_mastery.shape[0] if len(concept_mastery.shape) > 1 else 1
            
        if model_type is not None:  # 新增
            update_fields['model_type'] = model_type
            
        if sequence_data is not None:
            update_fields['sequence_data'] = sequence_data
            
        if metadata is not None:
            update_fields['metadata'] = metadata
        
        return self.db_op.update_one(self.collection_name, 
                                   {'learner_id': learner_id}, 
                                   update_fields)
    
    # 其余方法保持原样...
    def delete_learner_state(self, learner_id: str) -> int:
        """删除学习者状态"""
        
    def get_learner_count(self, model_type: str = None) -> int:  # 增强：支持按类型统计
        """获取学习者数量"""
        query = {}
        if model_type:
            query['model_type'] = model_type
        return self.db_op.count_documents(self.collection_name, query)
    
    def get_concept_statistics(self, model_type: str = None, device: str = 'cpu') -> Dict[str, Any]:
        """获取知识点统计信息 - 增强：支持按模型类型统计"""
        query = {}
        if model_type:
            query['model_type'] = model_type
            
        documents = self.db_op.find_many(self.collection_name, query)
        # ... 其余实现保持不变
    
    def create_indexes(self):
        """创建索引 - 增强：为model_type创建索引"""
        self.db_op.create_index(self.collection_name, [('learner_id', 1)], unique=True)
        self.db_op.create_index(self.collection_name, [('model_type', 1)])  # 新增索引
    
    def _get_current_time(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now()

# 创建全局实例
learner_repo = LearnerRepository()