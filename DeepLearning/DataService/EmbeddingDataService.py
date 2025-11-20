import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from Data.EmbeddingRepository import embedding_repo

@dataclass
class EmbeddingStats:
    """嵌入向量统计信息"""
    total_count: int
    learner_count: int
    unit_count: int
    concept_count: int
    embedding_dim: int
    creation_time: str

@dataclass
class EntityEmbeddings:
    """实体嵌入数据类"""
    entity_type: str
    embeddings: Dict[str, torch.Tensor]  # {uid: tensor}
    embedding_dim: int
    metadata: Dict[str, Any] = None

class EmbeddingDataService:
    """嵌入向量数据服务层 - 提供业务化的嵌入数据操作接口"""
    
    def __init__(self):
        self.repo = embedding_repo
        self._cache = {}  # 简单的内存缓存
        
    def save_training_embeddings(self, 
                               lrn_embeddings: torch.Tensor,
                               unit_embeddings: torch.Tensor, 
                               concept_embeddings: torch.Tensor,
                               training_epoch: int = None,
                               model_version: str = "v1.0") -> EmbeddingStats:
        """
        保存训练产生的嵌入向量（业务化接口）
        
        Args:
            lrn_embeddings: 学习者嵌入 [num_learners, dim]
            unit_embeddings: 学习单元嵌入 [num_units, dim]
            concept_embeddings: 知识点嵌入 [num_concepts, dim]
            training_epoch: 训练轮次
            model_version: 模型版本
            
        Returns:
            嵌入统计信息
        """
        print("💾 保存训练嵌入向量...")
        
        # 数据验证
        self._validate_embeddings(lrn_embeddings, unit_embeddings, concept_embeddings)
        
        # 构建嵌入字典
        embeddings_data = {
            'learner': self._create_entity_embeddings('learner', lrn_embeddings, 
                                                     training_epoch, model_version),
            'unit': self._create_entity_embeddings('unit', unit_embeddings,
                                                  training_epoch, model_version),
            'concept': self._create_entity_embeddings('concept', concept_embeddings,
                                                     training_epoch, model_version)
        }
        
        # 保存到Repository
        saved_ids = self.repo.save_embeddings(embeddings_data)
        
        # 清除缓存
        self._cache.clear()
        
        # 生成统计信息
        stats = EmbeddingStats(
            total_count=len(saved_ids),
            learner_count=len(embeddings_data['learner']),
            unit_count=len(embeddings_data['unit']),
            concept_count=len(embeddings_data['concept']),
            embedding_dim=lrn_embeddings.shape[1],
            creation_time=self._get_current_time()
        )
        
        print(f"✅ 训练嵌入向量保存完成: {stats}")
        return stats
    
    def get_embeddings_for_inference(self, 
                                   entity_types: List[str] = None,
                                   device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        获取推理所需的嵌入向量（批量获取，优化性能）
        
        Args:
            entity_types: 需要的实体类型，None表示全部
            device: 返回tensor的设备
            
        Returns:
            {entity_type: tensor} 其中tensor形状为 [num_entities, embedding_dim]
        """
        if entity_types is None:
            entity_types = ['learner', 'unit', 'concept']
        
        # 检查缓存
        cache_key = f"{'_'.join(entity_types)}_{device}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        embeddings_dict = {}
        
        for entity_type in entity_types:
            # 获取该类型的所有嵌入
            uid_embeddings = self.repo.get_embeddings_by_type(entity_type, device)
            
            if uid_embeddings:
                # 转换为tensor矩阵 [num_entities, embedding_dim]
                uids = sorted(uid_embeddings.keys())
                embedding_tensors = [uid_embeddings[uid] for uid in uids]
                embedding_matrix = torch.stack(embedding_tensors)
                
                embeddings_dict[entity_type] = {
                    'embeddings': embedding_matrix,
                    'uid_mapping': uids,  # 保持UID顺序映射
                    'uid_to_index': {uid: idx for idx, uid in enumerate(uids)}
                }
        
        # 缓存结果
        self._cache[cache_key] = embeddings_dict
        
        print(f"📥 加载 {len(entity_types)} 类实体嵌入，共 {sum(len(v['uid_mapping']) for v in embeddings_dict.values())} 个实体")
        return embeddings_dict
    
    def get_embedding_by_uid(self, 
                           uid: str, 
                           entity_type: str,
                           device: str = 'cpu') -> Optional[torch.Tensor]:
        """
        根据UID获取单个嵌入向量（带缓存）
        
        Args:
            uid: 实体ID
            entity_type: 实体类型
            device: 返回tensor的设备
            
        Returns:
            嵌入向量tensor
        """
        cache_key = f"{entity_type}_{uid}_{device}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        embedding = self.repo.get_embedding_by_uid(uid, entity_type, device)
        
        if embedding is not None:
            self._cache[cache_key] = embedding
            
        return embedding
    
    def get_similar_entities(self, 
                           query_embedding: torch.Tensor,
                           entity_type: str,
                           top_k: int = 10,
                           device: str = 'cpu') -> List[Tuple[str, float]]:
        """
        查找相似实体（基于余弦相似度）
        
        Args:
            query_embedding: 查询嵌入向量
            entity_type: 实体类型
            top_k: 返回最相似的K个
            device: 计算设备
            
        Returns:
            [(uid, 相似度分数), ...]
        """
        # 获取所有该类型的嵌入
        embeddings_data = self.get_embeddings_for_inference([entity_type], device)
        
        if entity_type not in embeddings_data:
            return []
        
        entity_embeddings = embeddings_data[entity_type]['embeddings']
        uid_mapping = embeddings_data[entity_type]['uid_mapping']
        
        # 计算余弦相似度
        query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        entity_norm = torch.nn.functional.normalize(entity_embeddings, p=2, dim=1)
        
        similarities = torch.matmul(query_norm, entity_norm.T).squeeze(0)
        
        # 获取top-k
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append((uid_mapping[idx], score.item()))
        
        return results
    
    def update_embeddings_batch(self,
                              updates_dict: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, int]:
        """
        批量更新嵌入向量
        
        Args:
            updates_dict: {
                'learner': {uid1: new_emb1, uid2: new_emb2, ...},
                'unit': {...},
                'concept': {...}
            }
            
        Returns:
            各类型更新数量的统计
        """
        update_stats = {}
        
        for entity_type, uid_updates in updates_dict.items():
            update_count = 0
            for uid, new_embedding in uid_updates.items():
                try:
                    updated = self.repo.update_embedding(uid, entity_type, new_embedding)
                    if updated > 0:
                        update_count += 1
                        # 清除相关缓存
                        self._clear_entity_cache(uid, entity_type)
                except Exception as e:
                    print(f"⚠ 更新嵌入失败 {entity_type}_{uid}: {e}")
                    continue
            
            update_stats[entity_type] = update_count
        
        print(f"🔄 嵌入向量批量更新完成: {update_stats}")
        return update_stats
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        获取详细的嵌入向量统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'summary': {
                'total_embeddings': self.repo.get_embedding_count(),
                'learner_embeddings': self.repo.get_embedding_count('learner'),
                'unit_embeddings': self.repo.get_embedding_count('unit'),
                'concept_embeddings': self.repo.get_embedding_count('concept')
            },
            'dimension_info': {},
            'quality_metrics': self._calculate_quality_metrics(),
            'timestamp': self._get_current_time()
        }
        
        # 获取维度信息
        for entity_type in ['learner', 'unit', 'concept']:
            sample = self.repo.get_embedding_by_uid('0', entity_type)
            if sample is not None:
                stats['dimension_info'][entity_type] = {
                    'embedding_dim': sample.shape[0],
                    'dtype': str(sample.dtype)
                }
        
        return stats
    
    def export_embeddings_to_file(self, 
                                file_path: str,
                                format_type: str = 'numpy') -> bool:
        """
        导出嵌入向量到文件
        
        Args:
            file_path: 文件路径
            format_type: 导出格式 'numpy' | 'json'
            
        Returns:
            是否导出成功
        """
        try:
            embeddings_data = self.get_embeddings_for_inference()
            
            if format_type == 'numpy':
                self._export_to_numpy(file_path, embeddings_data)
            elif format_type == 'json':
                self._export_to_json(file_path, embeddings_data)
            else:
                raise ValueError(f"不支持的格式: {format_type}")
                
            print(f"📤 嵌入向量导出成功: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ 嵌入向量导出失败: {e}")
            return False
    
    def _validate_embeddings(self, lrn_emb, unit_emb, concept_emb):
        """验证嵌入向量数据"""
        assert lrn_emb.dim() == 2, "学习者嵌入必须是2维"
        assert unit_emb.dim() == 2, "学习单元嵌入必须是2维" 
        assert concept_emb.dim() == 2, "知识点嵌入必须是2维"
        
        embedding_dim = lrn_emb.shape[1]
        assert unit_emb.shape[1] == embedding_dim, "嵌入维度不一致"
        assert concept_emb.shape[1] == embedding_dim, "嵌入维度不一致"
    
    def _create_entity_embeddings(self, entity_type, embeddings, epoch, version):
        """创建实体嵌入字典"""
        uid_embeddings = {}
        for i in range(len(embeddings)):
            uid = str(i)  # 使用索引作为UID
            uid_embeddings[uid] = embeddings[i]
        return uid_embeddings
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """计算嵌入质量指标"""
        # 这里可以实现各种质量评估指标
        # 例如：方差、稀疏度、聚类质量等
        return {
            'calculated_at': self._get_current_time(),
            'metrics_available': False  # 预留接口
        }
    
    def _export_to_numpy(self, file_path, embeddings_data):
        """导出为numpy格式"""
        import numpy as np
        
        export_data = {}
        for entity_type, data in embeddings_data.items():
            export_data[entity_type] = {
                'embeddings': data['embeddings'].cpu().numpy(),
                'uid_mapping': data['uid_mapping']
            }
        
        np.savez(file_path, **export_data)
    
    def _export_to_json(self, file_path, embeddings_data):
        """导出为JSON格式"""
        import json
        
        export_data = {}
        for entity_type, data in embeddings_data.items():
            # 只导出前几个作为示例（避免文件过大）
            sample_embeddings = {}
            for i, uid in enumerate(data['uid_mapping'][:10]):  # 限制数量
                sample_embeddings[uid] = data['embeddings'][i].cpu().numpy().tolist()
            
            export_data[entity_type] = sample_embeddings
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    def _clear_entity_cache(self, uid: str, entity_type: str):
        """清除实体相关缓存"""
        keys_to_remove = []
        for key in self._cache.keys():
            if uid in key or entity_type in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
    
    def _get_current_time(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        print("🗑️  嵌入数据缓存已清空")

# 创建全局实例
embedding_data_service = EmbeddingDataService()