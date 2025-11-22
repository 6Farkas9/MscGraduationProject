# cd_repository.py
import logging
from typing import List, Dict, Tuple, Optional, Any
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class CDRepository(BaseRepository):
    """认知诊断(CD)模型数据仓库 - 优化版本"""
    
    def __init__(self):
        super().__init__()
    
    def get_learner_question_interactions(self, learner_uid: str) -> List[Dict[str, Any]]:
        """获取学习者的题目交互记录 - 使用Learner_Question视图"""
        try:
            query = """
            SELECT lrn_uid, qus_uid, correct, create_time 
            FROM Learner_Question 
            WHERE lrn_uid = %s 
            ORDER BY create_time ASC
            """
            return self.execute_custom_mysql_query(query, (learner_uid,))
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 题目交互记录失败: {e}")
            return []
    
    def get_questions_from_interactions(self, interactions: List[Dict[str, Any]]) -> List[str]:
        """从交互记录中提取涉及的题目UID"""
        question_uids = set()
        for interaction in interactions:
            question_uids.add(interaction['qus_uid'])
        return list(question_uids)
    
    def get_question_interactions_batch(self, learner_uids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """批量获取多个学习者的题目交互记录 - 优化版本"""
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
            
            # 按学习者分组
            grouped_results = {}
            for result in results:
                learner_uid = result['lrn_uid']
                if learner_uid not in grouped_results:
                    grouped_results[learner_uid] = []
                grouped_results[learner_uid].append(result)
            
            return grouped_results
            
        except Exception as e:
            logger.error(f"批量获取题目交互记录失败: {e}")
            return {}
    
    def format_interaction_records(self, interactions: List[Dict[str, Any]]) -> List[Tuple[str, str, int, str]]:
        """格式化交互记录为(学习者UID, 题目UID, 是否正确, 时间戳)元组列表"""
        formatted_records = []
        for interaction in interactions:
            record = (
                interaction['lrn_uid'],
                interaction['qus_uid'],
                int(interaction['correct']),  # 转换为0/1
                interaction['create_time'].isoformat() if interaction['create_time'] else ''
            )
            formatted_records.append(record)
        return formatted_records
    
    def get_data_for_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """为单个学习者获取CD模型数据"""
        return self.get_cd_data_for_learner(learner_uid)
    
    def get_data_for_multiple_learners(self, learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
        """为多个学习者获取CD模型数据 - 优化版本使用批量查询"""
        try:
            # 批量获取交互记录
            batch_interactions = self.get_question_interactions_batch(learner_uids)
            
            results = {}
            for learner_uid in learner_uids:
                try:
                    interactions = batch_interactions.get(learner_uid, [])
                    
                    # 只使用实际涉及的题目
                    involved_questions = self.get_questions_from_interactions(interactions)
                    formatted_interactions = self.format_interaction_records(interactions)
                    
                    results[learner_uid] = {
                        'target_learner_uid': learner_uid,
                        'question_interactions': formatted_interactions,
                        'involved_questions': involved_questions,
                        'interaction_count': len(interactions)
                    }
                    
                    logger.info(f"学习者 {learner_uid} CD数据 - 交互记录: {len(interactions)}, 涉及题目: {len(involved_questions)}")
                    
                except Exception as e:
                    logger.error(f"为学习者 {learner_uid} 处理CD数据失败: {e}")
                    results[learner_uid] = None
            
            return results
            
        except Exception as e:
            logger.error(f"批量获取CD数据失败: {e}")
            return {}
    
    def get_cd_data_for_learner(self, learner_uid: str) -> Dict[str, Any]:
        """为单个学习者获取认知诊断模型所需的所有数据"""
        try:
            # 1. 获取学习者的题目交互记录
            interactions = self.get_learner_question_interactions(learner_uid)
            logger.info(f"学习者 {learner_uid} 题目交互记录数: {len(interactions)}")
            
            # 2. 只使用实际涉及的题目
            involved_questions = self.get_questions_from_interactions(interactions)
            
            # 3. 格式化交互记录
            formatted_interactions = self.format_interaction_records(interactions)
            
            logger.info(f"涉及题目数: {len(involved_questions)}")
            
            return {
                'target_learner_uid': learner_uid,
                'question_interactions': formatted_interactions,
                'involved_questions': involved_questions,
                'interaction_count': len(interactions)
            }
            
        except Exception as e:
            logger.error(f"为学习者 {learner_uid} 获取CD数据失败: {e}")
            raise
    
    def get_interaction_statistics(self, learner_uid: str) -> Dict[str, Any]:
        """获取学习者的交互统计信息 - 优化版本"""
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
                    'total_interactions': result['total_interactions'] or 0,
                    'unique_questions': result['unique_questions'] or 0,
                    'accuracy_rate': float(result['accuracy_rate'] or 0),
                    'time_range': {
                        'start': result['start_time'],
                        'end': result['end_time']
                    }
                }
            return {}
            
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 交互统计失败: {e}")
            return {}


# 全局仓库实例
cd_repository = CDRepository()