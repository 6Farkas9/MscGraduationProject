# kt_repository.py
import logging
from typing import List, Dict, Tuple, Optional, Any
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class KTRepository(BaseRepository):
    """知识追踪(KT)模型数据仓库 - 优化版本"""
    
    def __init__(self):
        super().__init__()
    
    def get_learner_unit_interactions(self, learner_uid: str) -> List[Dict[str, Any]]:
        """获取学习者的学习单元交互记录"""
        try:
            query = """
            SELECT lrn_uid, unt_uid, additioninfo1, additioninfo2, create_time 
            FROM Interaction 
            WHERE lrn_uid = %s 
            ORDER BY create_time ASC
            """
            return self.execute_custom_mysql_query(query, (learner_uid,))
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 学习单元交互记录失败: {e}")
            return []
    
    def get_units_from_interactions(self, interactions: List[Dict[str, Any]]) -> List[str]:
        """从交互记录中提取涉及的学习单元UID"""
        unit_uids = set()
        for interaction in interactions:
            unit_uids.add(interaction['unt_uid'])
        return list(unit_uids)
    
    def get_unit_types_from_uids(self, unit_uids: List[str]) -> Dict[str, str]:
        """批量获取学习单元类型"""
        if not unit_uids:
            return {}
        
        try:
            placeholders = ', '.join(['%s'] * len(unit_uids))
            query = f"SELECT uid, type FROM Units WHERE uid IN ({placeholders})"
            results = self.execute_custom_mysql_query(query, tuple(unit_uids))
            return {result['uid']: result.get('type', 'unknown') for result in results}
        except Exception as e:
            logger.error(f"批量获取学习单元类型失败: {e}")
            return {}
    
    def get_unit_interactions_batch(self, learner_uids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """批量获取多个学习者的学习单元交互记录 - 优化版本"""
        if not learner_uids:
            return {}
        
        try:
            placeholders = ', '.join(['%s'] * len(learner_uids))
            query = f"""
            SELECT lrn_uid, unt_uid, additioninfo1, additioninfo2, create_time 
            FROM Interaction 
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
            logger.error(f"批量获取学习单元交互记录失败: {e}")
            return {}
    
    def build_optimized_unit_mapping(self, unit_uids: List[str]) -> Dict[str, int]:
        """构建学习单元UID到索引的映射"""
        return {uid: idx for idx, uid in enumerate(unit_uids)}
    
    def build_optimized_learner_mapping(self, target_learner_uid: str, learner_uids: List[str]) -> Dict[str, int]:
        """构建优化的学习者UID映射：目标学习者放在索引0"""
        # 确保目标学习者在第一位
        all_learners = [target_learner_uid]
        
        # 添加其他学习者（排除重复）
        for learner in learner_uids:
            if learner != target_learner_uid and learner not in all_learners:
                all_learners.append(learner)
        
        return {uid: idx for idx, uid in enumerate(all_learners)}
    
    def format_unit_interaction_records(self, interactions: List[Dict[str, Any]]) -> List[Tuple[str, str, float, float, str]]:
        """格式化交互记录为(学习者UID, 单元UID, 附加信息1, 附加信息2, 时间戳)元组列表"""
        formatted_records = []
        for interaction in interactions:
            record = (
                interaction['lrn_uid'],
                interaction['unt_uid'],
                float(interaction['additioninfo1'] or 0.0),
                float(interaction['additioninfo2'] or 0.0),
                interaction['create_time'].isoformat() if interaction['create_time'] else ''
            )
            formatted_records.append(record)
        return formatted_records
    
    # 实现抽象方法
    def get_data_for_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """为单个学习者获取KT模型数据"""
        return self.get_kt_data_for_learner(learner_uid)
    
    def get_data_for_multiple_learners(self, learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
        """为多个学习者获取KT模型数据 - 优化版本使用批量查询"""
        try:
            # 批量获取交互记录
            batch_interactions = self.get_unit_interactions_batch(learner_uids)
            
            results = {}
            for learner_uid in learner_uids:
                try:
                    interactions = batch_interactions.get(learner_uid, [])
                    
                    # 只使用实际涉及的学习单元构建映射
                    involved_units = self.get_units_from_interactions(interactions)
                    unit_uid_mapping = self.build_optimized_unit_mapping(involved_units)
                    
                    # 批量获取单元类型
                    unit_types_mapping = self.get_unit_types_from_uids(involved_units)
                    
                    # 构建学习者映射（目标学习者在索引0）
                    involved_learners = list(set(interaction['lrn_uid'] for interaction in interactions))
                    learner_uid_mapping = self.build_optimized_learner_mapping(learner_uid, involved_learners)
                    
                    formatted_interactions = self.format_unit_interaction_records(interactions)
                    
                    results[learner_uid] = {
                        'target_learner_uid': learner_uid,
                        'unit_interactions': formatted_interactions,
                        'involved_units': involved_units,
                        'unit_uid_mapping': unit_uid_mapping,
                        'unit_types_mapping': unit_types_mapping,
                        'learner_uid_mapping': learner_uid_mapping,
                        'interaction_count': len(interactions)
                    }
                    
                    logger.info(f"学习者 {learner_uid} KT数据 - 交互记录: {len(interactions)}, 涉及单元: {len(involved_units)}")
                    
                except Exception as e:
                    logger.error(f"为学习者 {learner_uid} 处理KT数据失败: {e}")
                    results[learner_uid] = None
            
            return results
            
        except Exception as e:
            logger.error(f"批量获取KT数据失败: {e}")
            return {}
    
    def get_kt_data_for_learner(self, learner_uid: str) -> Dict[str, Any]:
        """为单个学习者获取知识追踪模型所需的所有数据"""
        try:
            # 1. 获取学习者的学习单元交互记录
            interactions = self.get_learner_unit_interactions(learner_uid)
            logger.info(f"学习者 {learner_uid} 学习单元交互记录数: {len(interactions)}")
            
            # 2. 只使用实际涉及的学习单元构建映射
            involved_units = self.get_units_from_interactions(interactions)
            
            # 3. 批量获取单元类型
            unit_types_mapping = self.get_unit_types_from_uids(involved_units)
            
            # 4. 构建优化的映射
            unit_uid_mapping = self.build_optimized_unit_mapping(involved_units)
            
            # 5. 构建学习者映射（目标学习者在索引0）
            involved_learners = list(set(interaction['lrn_uid'] for interaction in interactions))
            learner_uid_mapping = self.build_optimized_learner_mapping(learner_uid, involved_learners)
            
            # 6. 格式化交互记录
            formatted_interactions = self.format_unit_interaction_records(interactions)
            
            logger.info(f"涉及学习单元数: {len(involved_units)}, 涉及学习者数: {len(involved_learners)}")
            
            return {
                'target_learner_uid': learner_uid,
                'unit_interactions': formatted_interactions,
                'involved_units': involved_units,
                'unit_uid_mapping': unit_uid_mapping,
                'unit_types_mapping': unit_types_mapping,
                'learner_uid_mapping': learner_uid_mapping,
                'interaction_count': len(interactions)
            }
            
        except Exception as e:
            logger.error(f"为学习者 {learner_uid} 获取KT数据失败: {e}")
            raise
    
    def get_interaction_statistics(self, learner_uid: str) -> Dict[str, Any]:
        """获取学习者的交互统计信息 - 优化版本使用单次查询"""
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
                    'total_interactions': result['total_interactions'] or 0,
                    'unique_units': result['unique_units'] or 0,
                    'avg_additioninfo1': float(result['avg_info1'] or 0),
                    'avg_additioninfo2': float(result['avg_info2'] or 0),
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
kt_repository = KTRepository()