# EmbeddingDataService.py 简化版
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from Data.EmbeddingRepository import embedding_repo

@dataclass
class EmbeddingStats:
    """嵌入向量统计信息"""
    total_count: int
    learner_count: int
    unit_count: int
    concept_count: int

class EmbeddingDataService:
    """嵌入向量数据服务层 - 简化版"""
    
    def __init__(self):
        self.repo = embedding_repo
        
    def save_embeddings_dict(self, embeddings_dict: Dict[str, Dict[str, torch.Tensor]]) -> EmbeddingStats:
        """
        保存嵌入字典
        
        Args:
            embeddings_dict: {
                'lrn': {uid1: tensor1, uid2: tensor2, ...},
                'unt': {uid1: tensor1, uid2: tensor2, ...},
                'cpt': {uid1: tensor1, uid2: tensor2, ...}
            }
        """
        print("💾 保存嵌入向量...")
        
        # 保存到Repository
        result = self.repo.save_embeddings(embeddings_dict)
        
        # 生成统计信息
        stats = EmbeddingStats(
            total_count=result.get('upserted_count', 0) + result.get('modified_count', 0),
            learner_count=len(embeddings_dict.get('lrn', {})),
            unit_count=len(embeddings_dict.get('unt', {})),
            concept_count=len(embeddings_dict.get('cpt', {}))
        )
        
        print(f"✅ 嵌入向量保存完成: {stats}")
        return stats

# 创建全局实例
embedding_data_service = EmbeddingDataService()