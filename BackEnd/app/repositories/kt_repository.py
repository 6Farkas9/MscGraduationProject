# kt_repository.py
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import defaultdict
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class KTRepository(BaseRepository):
    """知识追踪(KT)模型数据仓库 - 优化版本，输出纯数据格式"""
    
    def __init__(self):
        super().__init__()
    
    def get_learner_interaction_sequence(self, learner_uid: str, max_seq_len: int = None) -> Dict[str, Any]:
        """
        获取学习者的完整交互序列（包含学习单元和题目）
        
        Args:
            learner_uid: 学习者UID
            max_seq_len: 最大序列长度，None表示不限制
            
        Returns:
            交互序列数据字典，包含7个元素的数据结构
        """
        try:
            # 获取学习者的所有交互记录
            query = """
            SELECT lrn_uid, unt_uid, additioninfo1, additioninfo2, create_time 
            FROM Interaction 
            WHERE lrn_uid = %s 
            ORDER BY create_time ASC
            """
            if max_seq_len:
                query += f" LIMIT {max_seq_len}"
            
            interactions = self.execute_custom_mysql_query(query, (learner_uid,))
            
            if not interactions:
                return {
                    'learner_uid': learner_uid,
                    'sequence': [[], [], [], [], [], [], []],
                    'seq_len': 0,
                    'interaction_count': 0
                }
            
            # 获取涉及的学习单元UID
            unit_uids = list(set([interaction['unt_uid'] for interaction in interactions]))
            
            # 批量获取学习单元类型
            unit_types = self._get_unit_types_batch(unit_uids)
            
            # 批量获取学习单元-知识点映射
            unit_concepts = self._get_unit_concepts_batch(unit_uids)
            
            # 构建7元素数据结构
            sequence_data = self._build_sequence_data(interactions, unit_types, unit_concepts)
            
            return {
                'learner_uid': learner_uid,
                'sequence': sequence_data,
                'seq_len': len(sequence_data[0]),
                'interaction_count': len(interactions),
                'question_count': sum(sequence_data[3]),  # is_questions计数
                'valid_prediction_count': sum(sequence_data[5])  # prediction_masks计数
            }
            
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 交互序列失败: {e}")
            return {
                'learner_uid': learner_uid,
                'sequence': [[], [], [], [], [], [], []],
                'seq_len': 0,
                'interaction_count': 0,
                'question_count': 0,
                'valid_prediction_count': 0
            }
    
    def get_learner_interaction_sequences_batch(self, learner_uids: List[str], max_seq_len: int = None) -> Dict[str, Dict[str, Any]]:
        """
        批量获取多个学习者的交互序列
        
        Args:
            learner_uids: 学习者UID列表
            max_seq_len: 最大序列长度，None表示不限制
            
        Returns:
            按学习者UID分组的交互序列字典
        """
        if not learner_uids:
            return {}
        
        try:
            # 批量获取所有交互记录
            placeholders = ', '.join(['%s'] * len(learner_uids))
            query = f"""
            SELECT lrn_uid, unt_uid, additioninfo1, additioninfo2, create_time 
            FROM Interaction 
            WHERE lrn_uid IN ({placeholders}) 
            ORDER BY lrn_uid, create_time ASC
            """
            if max_seq_len:
                # 这里需要更复杂的逻辑来限制每个学习者的序列长度
                # 暂时返回所有数据，在内存中处理
                pass
            
            all_interactions = self.execute_custom_mysql_query(query, tuple(learner_uids))
            
            # 按学习者分组
            grouped_interactions = defaultdict(list)
            for interaction in all_interactions:
                learner_uid = interaction['lrn_uid']
                grouped_interactions[learner_uid].append(interaction)
            
            # 收集所有涉及的学习单元UID
            all_unit_uids = set()
            for interactions in grouped_interactions.values():
                for interaction in interactions:
                    all_unit_uids.add(interaction['unt_uid'])
            
            # 批量获取学习单元类型和知识点映射
            unit_types = self._get_unit_types_batch(list(all_unit_uids))
            unit_concepts = self._get_unit_concepts_batch(list(all_unit_uids))
            
            # 为每个学习者构建序列
            results = {}
            for learner_uid in learner_uids:
                interactions = grouped_interactions.get(learner_uid, [])
                
                if max_seq_len and len(interactions) > max_seq_len:
                    interactions = interactions[-max_seq_len:]  # 取最近的记录
                
                if not interactions:
                    results[learner_uid] = {
                        'learner_uid': learner_uid,
                        'sequence': [[], [], [], [], [], [], []],
                        'seq_len': 0,
                        'interaction_count': 0,
                        'question_count': 0,
                        'valid_prediction_count': 0
                    }
                    continue
                
                # 构建7元素数据结构
                sequence_data = self._build_sequence_data(interactions, unit_types, unit_concepts)
                
                results[learner_uid] = {
                    'learner_uid': learner_uid,
                    'sequence': sequence_data,
                    'seq_len': len(sequence_data[0]),
                    'interaction_count': len(interactions),
                    'question_count': sum(sequence_data[3]),
                    'valid_prediction_count': sum(sequence_data[5])
                }
            
            return results
            
        except Exception as e:
            logger.error(f"批量获取交互序列失败: {e}")
            return {uid: {
                'learner_uid': uid,
                'sequence': [[], [], [], [], [], [], []],
                'seq_len': 0,
                'interaction_count': 0,
                'question_count': 0,
                'valid_prediction_count': 0
            } for uid in learner_uids}
    
    def _get_unit_types_batch(self, unit_uids: List[str]) -> Dict[str, str]:
        """批量获取学习单元类型"""
        if not unit_uids:
            return {}
        
        try:
            placeholders = ', '.join(['%s'] * len(unit_uids))
            query = f"SELECT uid, type FROM Units WHERE uid IN ({placeholders})"
            results = self.execute_custom_mysql_query(query, tuple(unit_uids))
            
            unit_type_dict = {}
            for result in results:
                unit_type_dict[result['uid']] = result.get('type', 'unknown')
            
            return unit_type_dict
        except Exception as e:
            logger.error(f"批量获取学习单元类型失败: {e}")
            return {}
    
    def _get_unit_concepts_batch(self, unit_uids: List[str]) -> Dict[str, List[str]]:
        """批量获取学习单元-知识点映射"""
        if not unit_uids:
            return {}
        
        try:
            placeholders = ', '.join(['%s'] * len(unit_uids))
            query = f"""
            SELECT unt_uid, cpt_uid 
            FROM Unit_Concept 
            WHERE unt_uid IN ({placeholders})
            """
            results = self.execute_custom_mysql_query(query, tuple(unit_uids))
            
            unit_concepts_dict = defaultdict(list)
            for result in results:
                unt_uid = result['unt_uid']
                cpt_uid = result['cpt_uid']
                unit_concepts_dict[unt_uid].append(cpt_uid)
            
            return dict(unit_concepts_dict)
        except Exception as e:
            logger.error(f"批量获取学习单元-知识点映射失败: {e}")
            return {}
    
    def _build_sequence_data(self, interactions: List[Dict[str, Any]], 
                            unit_types: Dict[str, str],
                            unit_concepts: Dict[str, List[str]]) -> List[List[Any]]:
        """
        构建7元素序列数据结构
        
        数据结构: [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]
        """
        # 初始化7个列表
        unt_uids = []
        add1s = []
        add2s = []
        is_questions = []
        results = []
        prediction_masks = []
        next_results = []
        
        # 判断是否为题目的函数
        def is_question_unit(unit_type: str) -> bool:
            return unit_type == 'question'
        
        # 处理每个交互记录
        for i, interaction in enumerate(interactions):
            unt_uid = interaction['unt_uid']
            unit_type = unit_types.get(unt_uid, 'unknown')
            
            # 确定是否为题目
            is_question = 1 if is_question_unit(unit_type) else 0
            
            # 对于题目交互，additioninfo2表示正确性；对于其他交互，设置为-1
            if is_question:
                result = float(interaction['additioninfo2'] or 0)
            else:
                result = -1  # 表示无结果
            
            # 添加当前步骤数据
            unt_uids.append(unt_uid)
            add1s.append(float(interaction['additioninfo1'] or 0))
            add2s.append(float(interaction['additioninfo2'] or 0))
            is_questions.append(is_question)
            results.append(result)
            
            # 计算预测掩码和下一个结果
            if i < len(interactions) - 1:  # 不是最后一个元素
                next_unt_uid = interactions[i+1]['unt_uid']
                next_unit_type = unit_types.get(next_unt_uid, 'unknown')
                next_is_question = 1 if is_question_unit(next_unit_type) else 0
                
                # prediction_mask: 下一步骤是否是题目
                prediction_mask = 1 if next_is_question == 1 else 0
                
                # next_result: 下一步骤的结果（如果是题目）
                if next_is_question == 1:
                    next_result = float(interactions[i+1]['additioninfo2'] or 0)
                else:
                    next_result = 0
            else:
                # 序列末尾，没有下一步骤
                prediction_mask = 0
                next_result = 0
            
            prediction_masks.append(prediction_mask)
            next_results.append(next_result)
        
        return [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]
    
    def get_interaction_statistics(self, learner_uid: str) -> Dict[str, Any]:
        """
        获取学习者的交互统计信息
        
        Args:
            learner_uid: 学习者UID
            
        Returns:
            统计信息字典
        """
        try:
            query = """
            SELECT 
                COUNT(*) as total_interactions,
                COUNT(DISTINCT unt_uid) as unique_units,
                AVG(additioninfo1) as avg_info1,
                AVG(additioninfo2) as avg_info2,
                MIN(create_time) as start_time,
                MAX(create_time) as end_time
            FROM Interaction 
            WHERE lrn_uid = %s
            """
            result = self.execute_custom_single_query(query, (learner_uid,))
            
            if result:
                return {
                    'learner_uid': learner_uid,
                    'total_interactions': result['total_interactions'] or 0,
                    'unique_units': result['unique_units'] or 0,
                    'avg_additioninfo1': float(result['avg_info1'] or 0),
                    'avg_additioninfo2': float(result['avg_info2'] or 0),
                    'time_range': {
                        'start': result['start_time'],
                        'end': result['end_time']
                    }
                }
            return {
                'learner_uid': learner_uid,
                'total_interactions': 0,
                'unique_units': 0,
                'avg_additioninfo1': 0.0,
                'avg_additioninfo2': 0.0,
                'time_range': {'start': None, 'end': None}
            }
            
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 交互统计失败: {e}")
            return {
                'learner_uid': learner_uid,
                'total_interactions': 0,
                'unique_units': 0,
                'avg_additioninfo1': 0.0,
                'avg_additioninfo2': 0.0,
                'time_range': {'start': None, 'end': None}
            }
    
    def validate_learner_has_interactions(self, learner_uid: str) -> bool:
        """
        验证学习者是否有交互记录
        
        Args:
            learner_uid: 学习者UID
            
        Returns:
            是否有交互记录
        """
        try:
            query = "SELECT COUNT(*) as count FROM Interaction WHERE lrn_uid = %s"
            result = self.execute_custom_single_query(query, (learner_uid,))
            return result and result['count'] > 0
        except Exception as e:
            logger.error(f"验证学习者 {learner_uid} 交互记录失败: {e}")
            return False


# 全局仓库实例
kt_repository = KTRepository()