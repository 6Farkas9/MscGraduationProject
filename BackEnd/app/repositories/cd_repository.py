# cd_repository.py
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class CDRepository(BaseRepository):
    """认知诊断(CD)模型数据仓库 - 优化版本"""
    
    def __init__(self):
        super().__init__()
    
    def get_learner_question_interactions(self, learner_uid: str, max_records: int = 100) -> List[Dict[str, Any]]:
        """
        获取学习者的题目交互记录（按时间排序）
        
        Args:
            learner_uid: 学习者UID
            max_records: 最大记录数
            
        Returns:
            按时间排序的交互记录列表
        """
        try:
            query = """
            SELECT lrn_uid, qus_uid, correct, create_time 
            FROM Learner_Question 
            WHERE lrn_uid = %s 
            ORDER BY create_time ASC
            LIMIT %s
            """
            return self.execute_custom_mysql_query(query, (learner_uid, max_records))
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 题目交互记录失败: {e}")
            return []
    
    def get_question_interactions_batch(self, learner_uids: List[str], max_records_per_learner: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量获取多个学习者的题目交互记录（按时间排序）
        
        Args:
            learner_uids: 学习者UID列表
            max_records_per_learner: 每个学习者的最大记录数
            
        Returns:
            按学习者UID分组的交互记录字典
        """
        if not learner_uids:
            return {}
        
        try:
            placeholders = ', '.join(['%s'] * len(learner_uids))
            query = f"""
            SELECT lrn_uid, qus_uid, correct, create_time 
            FROM Learner_Question 
            WHERE lrn_uid IN ({placeholders}) 
            ORDER BY lrn_uid, create_time ASC
            """
            results = self.execute_custom_mysql_query(query, tuple(learner_uids))
            
            # 按学习者分组并限制每个学习者的记录数
            grouped_results = defaultdict(list)
            for result in results:
                learner_uid = result['lrn_uid']
                if len(grouped_results[learner_uid]) < max_records_per_learner:
                    grouped_results[learner_uid].append(result)
            
            return dict(grouped_results)
            
        except Exception as e:
            logger.error(f"批量获取题目交互记录失败: {e}")
            return {}
    
    def build_question_sequences(self, learner_uids: List[str], max_seq_len: int = None) -> Dict[str, Any]:
        """
        为多个学习者构建题目序列（保持时序）
        
        Args:
            learner_uids: 学习者UID列表
            max_seq_len: 最大序列长度，如果为None则使用每个学习者的实际长度
            
        Returns:
            包含序列数据的字典
        """
        try:
            # 批量获取交互记录，如果不指定max_seq_len，获取所有记录
            limit = max_seq_len if max_seq_len is not None else 1000  # 足够大的值
            batch_interactions = self.get_question_interactions_batch(learner_uids, limit)
            
            sequences = {}
            all_question_uids = set()
            actual_max_seq_len = 0
            
            for learner_uid in learner_uids:
                interactions = batch_interactions.get(learner_uid, [])
                
                # 构建题目序列（保持时序）
                qus_seq = [interaction['qus_uid'] for interaction in interactions]
                
                # 如果指定了max_seq_len，截取最近的max_seq_len条
                if max_seq_len is not None and len(qus_seq) > max_seq_len:
                    qus_seq = qus_seq[-max_seq_len:]  # 取最近的部分
                
                seq_len = len(qus_seq)
                
                # 收集所有涉及的题目UID
                all_question_uids.update(qus_seq)
                
                # 更新实际最大序列长度
                actual_max_seq_len = max(actual_max_seq_len, seq_len)
                
                sequences[learner_uid] = {
                    'qus_seq': qus_seq,
                    'seq_len': seq_len,  # 记录实际长度
                    'interaction_count': len(interactions)
                }
            
            # 如果未指定max_seq_len，使用实际最大长度
            if max_seq_len is None:
                max_seq_len = actual_max_seq_len
            
            # 统计信息
            total_interactions = sum(seq_data['interaction_count'] for seq_data in sequences.values())
            avg_seq_len = total_interactions / len(learner_uids) if learner_uids else 0
            
            return {
                'sequences': sequences,
                'all_question_uids': list(all_question_uids),
                'max_seq_len': max_seq_len,
                'actual_max_seq_len': actual_max_seq_len,
                'statistics': {
                    'total_learners': len(learner_uids),
                    'total_interactions': total_interactions,
                    'average_sequence_length': round(avg_seq_len, 2),
                    'unique_questions': len(all_question_uids)
                }
            }
            
        except Exception as e:
            logger.error(f"构建题目序列失败: {e}")
            return {
                'sequences': {},
                'all_question_uids': [],
                'max_seq_len': max_seq_len or 0,
                'actual_max_seq_len': 0,
                'statistics': {
                    'total_learners': len(learner_uids),
                    'total_interactions': 0,
                    'average_sequence_length': 0,
                    'unique_questions': 0
                }
            }
    
    def get_question_sequence_for_learner(self, learner_uid: str, max_seq_len: int = 50) -> Dict[str, Any]:
        """
        获取单个学习者的题目序列
        
        Args:
            learner_uid: 学习者UID
            max_seq_len: 最大序列长度
            
        Returns:
            序列数据字典
        """
        try:
            interactions = self.get_learner_question_interactions(learner_uid, max_seq_len)
            
            # 构建题目序列（保持时序）
            qus_seq = [interaction['qus_uid'] for interaction in interactions]
            
            return {
                'learner_uid': learner_uid,
                'qus_seq': qus_seq,
                'seq_len': len(qus_seq),
                'interaction_count': len(interactions),
                'unique_questions': len(set(qus_seq))
            }
            
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 题目序列失败: {e}")
            return {
                'learner_uid': learner_uid,
                'qus_seq': [],
                'seq_len': 0,
                'interaction_count': 0,
                'unique_questions': 0
            }
    
    def get_involved_questions(self, learner_uids: List[str]) -> List[str]:
        """
        获取学习者涉及的所有题目UID
        
        Args:
            learner_uids: 学习者UID列表
            
        Returns:
            题目UID列表
        """
        try:
            sequence_data = self.build_question_sequences(learner_uids)
            return sequence_data['all_question_uids']
        except Exception as e:
            logger.error(f"获取涉及题目失败: {e}")
            return []
    
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
                COUNT(DISTINCT qus_uid) as unique_questions,
                AVG(correct) as accuracy_rate,
                MIN(create_time) as start_time,
                MAX(create_time) as end_time
            FROM Learner_Question 
            WHERE lrn_uid = %s
            """
            result = self.execute_custom_single_query(query, (learner_uid,))
            
            if result:
                return {
                    'learner_uid': learner_uid,
                    'total_interactions': result['total_interactions'] or 0,
                    'unique_questions': result['unique_questions'] or 0,
                    'accuracy_rate': float(result['accuracy_rate'] or 0),
                    'time_range': {
                        'start': result['start_time'],
                        'end': result['end_time']
                    }
                }
            return {
                'learner_uid': learner_uid,
                'total_interactions': 0,
                'unique_questions': 0,
                'accuracy_rate': 0.0,
                'time_range': {'start': None, 'end': None}
            }
            
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 交互统计失败: {e}")
            return {
                'learner_uid': learner_uid,
                'total_interactions': 0,
                'unique_questions': 0,
                'accuracy_rate': 0.0,
                'time_range': {'start': None, 'end': None}
            }
    
    def get_batch_interaction_statistics(self, learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取学习者的交互统计信息
        
        Args:
            learner_uids: 学习者UID列表
            
        Returns:
            按学习者UID分组的统计信息字典
        """
        if not learner_uids:
            return {}
        
        try:
            placeholders = ', '.join(['%s'] * len(learner_uids))
            query = f"""
            SELECT 
                lrn_uid,
                COUNT(*) as total_interactions,
                COUNT(DISTINCT qus_uid) as unique_questions,
                AVG(correct) as accuracy_rate,
                MIN(create_time) as start_time,
                MAX(create_time) as end_time
            FROM Learner_Question 
            WHERE lrn_uid IN ({placeholders})
            GROUP BY lrn_uid
            """
            results = self.execute_custom_mysql_query(query, tuple(learner_uids))
            
            statistics = {}
            for result in results:
                learner_uid = result['lrn_uid']
                statistics[learner_uid] = {
                    'total_interactions': result['total_interactions'] or 0,
                    'unique_questions': result['unique_questions'] or 0,
                    'accuracy_rate': float(result['accuracy_rate'] or 0),
                    'time_range': {
                        'start': result['start_time'],
                        'end': result['end_time']
                    }
                }
            
            # 为没有交互记录的学习者添加默认统计
            for learner_uid in learner_uids:
                if learner_uid not in statistics:
                    statistics[learner_uid] = {
                        'total_interactions': 0,
                        'unique_questions': 0,
                        'accuracy_rate': 0.0,
                        'time_range': {'start': None, 'end': None}
                    }
            
            return statistics
            
        except Exception as e:
            logger.error(f"批量获取交互统计失败: {e}")
            return {uid: {
                'total_interactions': 0,
                'unique_questions': 0,
                'accuracy_rate': 0.0,
                'time_range': {'start': None, 'end': None}
            } for uid in learner_uids}
    
    def validate_learner_has_interactions(self, learner_uid: str) -> bool:
        """
        验证学习者是否有交互记录
        
        Args:
            learner_uid: 学习者UID
            
        Returns:
            是否有交互记录
        """
        try:
            query = "SELECT COUNT(*) as count FROM Learner_Question WHERE lrn_uid = %s"
            result = self.execute_custom_single_query(query, (learner_uid,))
            return result and result['count'] > 0
        except Exception as e:
            logger.error(f"验证学习者 {learner_uid} 交互记录失败: {e}")
            return False
    
    def get_recent_interactions(self, limit: int = 128) -> List[Dict[str, Any]]:
        """
        获取最近的交互记录
        
        Args:
            limit: 限制返回数量
            
        Returns:
            最近的交互记录列表
        """
        try:
            query = """
            SELECT lrn_uid, qus_uid, correct, create_time 
            FROM Learner_Question 
            ORDER BY create_time DESC 
            LIMIT %s
            """
            return self.execute_custom_mysql_query(query, (limit,))
        except Exception as e:
            logger.error(f"获取最近交互记录失败: {e}")
            return []


# 全局仓库实例
cd_repository = CDRepository()