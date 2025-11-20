import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import statistics

from Data.LearnerRepository import learner_repo

@dataclass
class KnowledgeState:
    """知识点掌握状态数据类"""
    learner_id: str
    concept_mastery: torch.Tensor  # [seq_len, concept_num] 或 [concept_num]
    sequence_length: int
    concept_num: int
    model_type: str  # 'KT' 或 'CD'
    timestamp: str
    metadata: Dict[str, Any] = None

@dataclass
class MasteryStatistics:
    """掌握程度统计信息"""
    total_learners: int
    concept_num: int
    avg_mastery_per_concept: List[float]
    mastery_distribution: Dict[str, int]  # 高、中、低掌握度分布
    learning_progress: Dict[str, float]  # 学习进度指标

@dataclass
class LearningTrajectory:
    """学习轨迹数据"""
    learner_id: str
    sequence_data: Dict[str, Any]
    concept_mastery_sequence: torch.Tensor  # 时间序列的掌握程度
    performance_metrics: Dict[str, float]

class LearnerDataService:
    """学习者数据服务层 - 提供业务化的学习者知识状态操作接口"""
    
    def __init__(self):
        self.repo = learner_repo
        self._cache = {}  # 内存缓存
    
    def save_kt_training_results(self,
                               kt_results: List[Dict[str, Any]],
                               training_epoch: int = None,
                               model_version: str = "v1.0") -> Dict[str, int]:
        """
        保存KT训练结果（业务化接口）
        
        Args:
            kt_results: KT计算结果列表
            training_epoch: 训练轮次
            model_version: 模型版本
            
        Returns:
            保存统计信息
        """
        print("💾 保存KT训练结果...")
        
        saved_count = 0
        failed_count = 0
        batch_data = []
        
        for i, result in enumerate(kt_results):
            try:
                learner_id = result['learner_id']
                concept_mastery = result['concept_mastery']
                
                # 构建元数据
                metadata = {
                    'model_type': 'KT',
                    'training_epoch': training_epoch,
                    'model_version': model_version,
                    'computation_timestamp': self._get_current_time(),
                    'result_index': i,
                    'source': 'training_pipeline'
                }
                
                # 合并额外元数据
                if 'metadata' in result:
                    metadata.update(result['metadata'])
                
                # 准备批量保存
                batch_data.append({
                    'learner_id': learner_id,
                    'concept_mastery': concept_mastery,
                    'sequence_data': result.get('sequence_data'),
                    'metadata': metadata
                })
                
            except Exception as e:
                print(f"⚠ 准备KT结果失败 {result.get('learner_id', 'unknown')}: {e}")
                failed_count += 1
                continue
        
        # 批量保存
        if batch_data:
            try:
                saved_ids = self.repo.save_batch_learner_states(batch_data)
                saved_count = len(saved_ids)
            except Exception as e:
                print(f"❌ 批量保存KT结果失败: {e}")
                # 回退到逐个保存
                saved_count = self._save_individual_states(batch_data)
        
        stats = {
            'total_processed': len(kt_results),
            'successfully_saved': saved_count,
            'failed': failed_count,
            'success_rate': saved_count / len(kt_results) if kt_results else 0
        }
        
        print(f"✅ KT训练结果保存完成: {stats}")
        return stats
    
    def save_cd_ability_results(self,
                              cd_results: List[Dict[str, Any]],
                              purpose: str = "kt_initialization") -> Dict[str, int]:
        """
        保存CD能力估计结果
        
        Args:
            cd_results: CD计算结果列表
            purpose: 用途描述
            
        Returns:
            保存统计信息
        """
        print("💾 保存CD能力估计结果...")
        
        saved_count = 0
        for result in cd_results:
            try:
                learner_id = result['learner_id']
                cd_ability = result['cd_ability']
                
                metadata = {
                    'model_type': 'CD',
                    'purpose': purpose,
                    'computation_timestamp': self._get_current_time(),
                    'is_intermediate': True  # 标记为中间结果
                }
                
                self.repo.save_learner_knowledge_state(
                    learner_id=f"{learner_id}_cd",
                    concept_mastery=cd_ability,
                    sequence_data=result.get('sequence_data'),
                    metadata=metadata
                )
                saved_count += 1
                
            except Exception as e:
                print(f"⚠ 保存CD结果失败 {result.get('learner_id', 'unknown')}: {e}")
                continue
        
        stats = {
            'cd_states_saved': saved_count,
            'purpose': purpose
        }
        
        print(f"✅ CD能力估计保存完成: {stats}")
        return stats
    
    def get_learner_knowledge_state(self, 
                                  learner_id: str,
                                  include_sequence: bool = False,
                                  device: str = 'cpu') -> Optional[KnowledgeState]:
        """
        获取学习者知识点掌握状态（带业务逻辑处理）
        
        Args:
            learner_id: 学习者ID
            include_sequence: 是否包含序列数据
            device: 返回tensor的设备
            
        Returns:
            知识点掌握状态对象
        """
        cache_key = f"learner_{learner_id}_{device}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        raw_data = self.repo.get_learner_knowledge_state(learner_id, device)
        
        if not raw_data:
            return None
        
        # 转换为业务对象
        knowledge_state = KnowledgeState(
            learner_id=raw_data['learner_id'],
            concept_mastery=raw_data['concept_mastery'],
            sequence_length=raw_data.get('sequence_length', 1),
            concept_num=raw_data.get('concept_num', 0),
            model_type=raw_data.get('metadata', {}).get('model_type', 'unknown'),
            timestamp=raw_data.get('created_time', ''),
            metadata=raw_data.get('metadata', {})
        )
        
        # 缓存结果
        self._cache[cache_key] = knowledge_state
        
        return knowledge_state
    
    def get_learners_by_mastery_level(self,
                                    concept_index: int,
                                    mastery_level: str = 'high',
                                    min_confidence: float = 0.0,
                                    device: str = 'cpu') -> List[KnowledgeState]:
        """
        根据掌握程度筛选学习者
        
        Args:
            concept_index: 知识点索引
            mastery_level: 掌握程度 'high'|'medium'|'low'
            min_confidence: 最小置信度阈值
            device: 计算设备
            
        Returns:
            符合条件的学习者状态列表
        """
        # 定义掌握程度阈值
        level_thresholds = {
            'high': 0.7,
            'medium': 0.3,
            'low': 0.0
        }
        
        min_mastery = level_thresholds.get(mastery_level, 0.0)
        raw_results = self.repo.get_learners_by_concept_mastery(
            concept_index, min_mastery, device
        )
        
        # 转换为业务对象
        knowledge_states = []
        for raw_data in raw_results:
            # 应用置信度过滤
            if raw_data['mastery_level'] >= min_confidence:
                state = KnowledgeState(
                    learner_id=raw_data['learner_id'],
                    concept_mastery=raw_data['concept_mastery'],
                    sequence_length=raw_data['concept_mastery'].shape[0] 
                                 if len(raw_data['concept_mastery'].shape) > 1 else 1,
                    concept_num=raw_data['concept_num'],
                    model_type='KT',  # 假设来自KT
                    timestamp=self._get_current_time(),
                    metadata={'mastery_level': raw_data['mastery_level']}
                )
                knowledge_states.append(state)
        
        print(f"🔍 找到 {len(knowledge_states)} 个{mastery_level}掌握程度的学习者")
        return knowledge_states
    
    def get_learning_trajectory(self,
                              learner_id: str,
                              device: str = 'cpu') -> Optional[LearningTrajectory]:
        """
        获取学习轨迹分析
        
        Args:
            learner_id: 学习者ID
            device: 计算设备
            
        Returns:
            学习轨迹对象
        """
        knowledge_state = self.get_learner_knowledge_state(learner_id, True, device)
        
        if not knowledge_state or knowledge_state.sequence_length <= 1:
            return None
        
        # 计算学习进度指标
        concept_mastery = knowledge_state.concept_mastery
        performance_metrics = self._calculate_learning_metrics(concept_mastery)
        
        trajectory = LearningTrajectory(
            learner_id=learner_id,
            sequence_data=knowledge_state.metadata.get('sequence_data', {}),
            concept_mastery_sequence=concept_mastery,
            performance_metrics=performance_metrics
        )
        
        return trajectory
    
    def get_mastery_statistics(self) -> MasteryStatistics:
        """
        获取全局掌握程度统计
        
        Returns:
            掌握程度统计信息
        """
        raw_stats = self.repo.get_concept_statistics()
        
        if not raw_stats:
            return MasteryStatistics(
                total_learners=0,
                concept_num=0,
                avg_mastery_per_concept=[],
                mastery_distribution={'high_mastery': 0, 'medium_mastery': 0, 'low_mastery': 0},
                learning_progress={}
            )
        
        # 计算学习进度指标
        learning_progress = self._calculate_learning_progress()
        
        stats = MasteryStatistics(
            total_learners=raw_stats['learner_count'],
            concept_num=raw_stats['concept_num'],
            avg_mastery_per_concept=raw_stats['average_mastery'],
            mastery_distribution=raw_stats['mastery_distribution'],
            learning_progress=learning_progress
        )
        
        return stats
    
    def update_learner_progress(self,
                              learner_id: str,
                              new_mastery: torch.Tensor,
                              progress_data: Dict[str, Any] = None) -> bool:
        """
        更新学习者进度
        
        Args:
            learner_id: 学习者ID
            new_mastery: 新的掌握程度
            progress_data: 进度数据
            
        Returns:
            是否更新成功
        """
        try:
            metadata = {
                'model_type': 'KT',
                'update_timestamp': self._get_current_time(),
                'update_type': 'progress_update'
            }
            
            if progress_data:
                metadata.update(progress_data)
            
            updated = self.repo.update_learner_state(
                learner_id=learner_id,
                concept_mastery=new_mastery,
                metadata=metadata
            )
            
            if updated > 0:
                # 清除缓存
                self._clear_learner_cache(learner_id)
                print(f"🔄 学习者 {learner_id} 进度更新成功")
                return True
            else:
                print(f"⚠ 学习者 {learner_id} 进度更新失败")
                return False
                
        except Exception as e:
            print(f"❌ 更新学习者进度失败 {learner_id}: {e}")
            return False
    
    def export_learner_data(self,
                          output_path: str,
                          learner_ids: List[str] = None,
                          export_format: str = 'json') -> bool:
        """
        导出学习者数据
        
        Args:
            output_path: 输出路径
            learner_ids: 要导出的学习者ID列表，None表示全部
            export_format: 导出格式 'json' | 'csv'
            
        Returns:
            是否导出成功
        """
        try:
            if learner_ids is None:
                all_states = self.repo.get_all_learners_states()
            else:
                all_states = {}
                for lid in learner_ids:
                    state = self.get_learner_knowledge_state(lid)
                    if state:
                        all_states[lid] = {
                            'concept_mastery': state.concept_mastery.cpu().numpy().tolist(),
                            'model_type': state.model_type,
                            'metadata': state.metadata
                        }
            
            if export_format == 'json':
                self._export_to_json(output_path, all_states)
            elif export_format == 'csv':
                self._export_to_csv(output_path, all_states)
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")
            
            print(f"📤 学习者数据导出成功: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ 学习者数据导出失败: {e}")
            return False
    
    def get_weak_concepts_recommendation(self,
                                      learner_id: str,
                                      top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        获取薄弱知识点推荐
        
        Args:
            learner_id: 学习者ID
            top_k: 返回前K个薄弱知识点
            
        Returns:
            [(概念索引, 掌握程度, 推荐理由), ...]
        """
        knowledge_state = self.get_learner_knowledge_state(learner_id)
        
        if not knowledge_state:
            return []
        
        concept_mastery = knowledge_state.concept_mastery
        
        # 如果是序列数据，取最后一个时间步
        if len(concept_mastery.shape) > 1:
            current_mastery = concept_mastery[-1]
        else:
            current_mastery = concept_mastery
        
        # 获取掌握程度最低的知识点
        weak_indices = torch.argsort(current_mastery)[:top_k]
        
        recommendations = []
        for idx in weak_indices:
            mastery_level = current_mastery[idx].item()
            reason = self._generate_recommendation_reason(mastery_level)
            recommendations.append((idx.item(), mastery_level, reason))
        
        return recommendations
    
    def _save_individual_states(self, batch_data: List[Dict]) -> int:
        """逐个保存学习者状态（回退方案）"""
        saved_count = 0
        for data in batch_data:
            try:
                self.repo.save_learner_knowledge_state(
                    learner_id=data['learner_id'],
                    concept_mastery=data['concept_mastery'],
                    sequence_data=data.get('sequence_data'),
                    metadata=data.get('metadata')
                )
                saved_count += 1
            except Exception as e:
                print(f"⚠ 单个保存失败 {data['learner_id']}: {e}")
                continue
        return saved_count
    
    def _calculate_learning_metrics(self, concept_mastery: torch.Tensor) -> Dict[str, float]:
        """计算学习进度指标"""
        if len(concept_mastery.shape) == 1:
            return {
                'current_mastery_avg': concept_mastery.mean().item(),
                'mastery_improvement': 0.0,
                'consistency': 1.0
            }
        
        # 时间序列数据
        initial_mastery = concept_mastery[0].mean().item()
        final_mastery = concept_mastery[-1].mean().item()
        
        # 计算一致性（掌握程度的稳定性）
        mastery_std = concept_mastery.std(dim=0).mean().item()
        consistency = 1.0 / (1.0 + mastery_std)  # 简单的稳定性指标
        
        return {
            'current_mastery_avg': final_mastery,
            'mastery_improvement': final_mastery - initial_mastery,
            'consistency': consistency,
            'learning_rate': (final_mastery - initial_mastery) / len(concept_mastery)
        }
    
    def _calculate_learning_progress(self) -> Dict[str, float]:
        """计算全局学习进度指标"""
        all_states = self.repo.get_all_learners_states()
        
        if not all_states:
            return {}
        
        total_improvement = 0
        total_consistency = 0
        learner_count = 0
        
        for learner_id, state in all_states.items():
            concept_mastery = state['concept_mastery']
            if len(concept_mastery.shape) > 1 and concept_mastery.shape[0] > 1:
                initial = concept_mastery[0].mean().item()
                final = concept_mastery[-1].mean().item()
                total_improvement += (final - initial)
                learner_count += 1
        
        avg_improvement = total_improvement / learner_count if learner_count > 0 else 0
        
        return {
            'average_improvement': avg_improvement,
            'active_learners': learner_count,
            'overall_progress': 'positive' if avg_improvement > 0 else 'stable'
        }
    
    def _generate_recommendation_reason(self, mastery_level: float) -> str:
        """生成推荐理由"""
        if mastery_level < 0.3:
            return "基础薄弱，建议从基础概念开始学习"
        elif mastery_level < 0.5:
            return "掌握程度一般，建议进行针对性练习"
        elif mastery_level < 0.7:
            return "掌握程度良好，建议进行巩固练习"
        else:
            return "掌握程度优秀，可以挑战更高难度"
    
    def _export_to_json(self, file_path: str, data: Dict):
        """导出为JSON格式"""
        import json
        
        # 简化数据，避免文件过大
        export_data = {}
        for learner_id, state in list(data.items())[:100]:  # 限制数量
            export_data[learner_id] = {
                'model_type': state.get('model_type', 'unknown'),
                'concept_mastery_avg': float(state['concept_mastery'].mean().item() 
                                           if hasattr(state['concept_mastery'], 'mean') 
                                           else statistics.mean(state['concept_mastery'])),
                'metadata': state.get('metadata', {})
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    def _export_to_csv(self, file_path: str, data: Dict):
        """导出为CSV格式"""
        import csv
        import numpy as np
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['LearnerID', 'AvgMastery', 'ModelType', 'ConceptsCount'])
            
            for learner_id, state in data.items():
                concept_mastery = state['concept_mastery']
                
                # 统一转换为numpy数组计算平均值
                if hasattr(concept_mastery, 'numpy'):
                    # PyTorch tensor
                    avg_mastery = concept_mastery.mean().item()
                elif hasattr(concept_mastery, 'mean'):
                    # numpy数组
                    avg_mastery = float(concept_mastery.mean())
                else:
                    # 其他类型，尝试转换
                    try:
                        array = np.array(concept_mastery)
                        avg_mastery = float(array.mean())
                    except:
                        avg_mastery = 0.0
                
                writer.writerow([
                    learner_id,
                    f"{avg_mastery:.4f}",
                    state.get('model_type', 'unknown'),
                    state.get('concept_num', 0)
                ])
    
    def _clear_learner_cache(self, learner_id: str):
        """清除学习者相关缓存"""
        keys_to_remove = [key for key in self._cache.keys() if learner_id in key]
        for key in keys_to_remove:
            del self._cache[key]
    
    def _get_current_time(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        print("🗑️  学习者数据缓存已清空")

# 创建全局实例
learner_data_service = LearnerDataService()