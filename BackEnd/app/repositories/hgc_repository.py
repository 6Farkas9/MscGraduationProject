# hgc_repository.py
import logging
from typing import List, Dict, Tuple, Set, Optional, Any
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class HGCRepository(BaseRepository):
    """HGC模型数据仓库 - 使用视图优化版本"""
    
    def __init__(self):
        super().__init__()
    
    def get_learner_interacted_units(self, learner_uid: str) -> List[str]:
        """获取学习者交互过的学习单元UID列表"""
        try:
            query = "SELECT DISTINCT unt_uid FROM Interaction WHERE lrn_uid = %s"
            results = self.execute_custom_mysql_query(query, (learner_uid,))
            return [result['unt_uid'] for result in results]
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 交互单元失败: {e}")
            return []
    
    def get_learner_topics(self, learner_uid: str) -> List[str]:
        """获取学习者涉及的主题UID列表 - 使用Learner_Topic视图"""
        try:
            query = "SELECT DISTINCT tpc_uid FROM Learner_Topic WHERE lrn_uid = %s"
            results = self.execute_custom_mysql_query(query, (learner_uid,))
            return [result['tpc_uid'] for result in results]
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 涉及主题失败: {e}")
            return []
    
    def get_learner_courses(self, learner_uid: str) -> List[str]:
        """获取学习者涉及的课程UID列表 - 使用Learner_Course视图"""
        try:
            query = "SELECT DISTINCT crs_uid FROM Learner_Course WHERE lrn_uid = %s"
            results = self.execute_custom_mysql_query(query, (learner_uid,))
            return [result['crs_uid'] for result in results]
        except Exception as e:
            logger.error(f"获取学习者 {learner_uid} 涉及课程失败: {e}")
            return []
    
    def get_other_learners_by_units(self, unit_uids: List[str]) -> Set[str]:
        """根据学习单元获取其他学习者UID集合"""
        if not unit_uids:
            return set()
        
        try:
            placeholders = ', '.join(['%s'] * len(unit_uids))
            query = f"""
            SELECT DISTINCT lrn_uid 
            FROM Interaction 
            WHERE unt_uid IN ({placeholders})
            """
            results = self.execute_custom_mysql_query(query, tuple(unit_uids))
            return set(result['lrn_uid'] for result in results)
        except Exception as e:
            logger.error(f"根据单元获取其他学习者失败: {e}")
            return set()
    
    def get_other_learners_by_topics(self, topic_uids: List[str]) -> Set[str]:
        """根据主题获取其他学习者UID集合 - 使用Learner_Topic视图"""
        if not topic_uids:
            return set()
        
        try:
            placeholders = ', '.join(['%s'] * len(topic_uids))
            query = f"""
            SELECT DISTINCT lrn_uid 
            FROM Learner_Topic 
            WHERE tpc_uid IN ({placeholders})
            """
            results = self.execute_custom_mysql_query(query, tuple(topic_uids))
            return set(result['lrn_uid'] for result in results)
        except Exception as e:
            logger.error(f"根据主题获取其他学习者失败: {e}")
            return set()
    
    def get_other_learners_by_courses(self, course_uids: List[str]) -> Set[str]:
        """根据课程获取其他学习者UID集合 - 使用Learner_Course视图"""
        if not course_uids:
            return set()
        
        try:
            placeholders = ', '.join(['%s'] * len(course_uids))
            query = f"""
            SELECT DISTINCT lrn_uid 
            FROM Learner_Course 
            WHERE crs_uid IN ({placeholders})
            """
            results = self.execute_custom_mysql_query(query, tuple(course_uids))
            return set(result['lrn_uid'] for result in results)
        except Exception as e:
            logger.error(f"根据课程获取其他学习者失败: {e}")
            return set()
    
    def get_optimized_related_learners(self, learner_uid: str, 
                                     interacted_units: List[str],
                                     learner_topics: List[str],
                                     learner_courses: List[str]) -> Set[str]:
        """优化策略获取相关学习者：先取交集，为空则取并集"""
        # 获取三个元路径的相关学习者
        learners_from_units = self.get_other_learners_by_units(interacted_units)
        learners_from_topics = self.get_other_learners_by_topics(learner_topics)
        learners_from_courses = self.get_other_learners_by_courses(learner_courses)
        
        logger.info(f"元路径学习者数量 - 单元: {len(learners_from_units)}, "
                   f"主题: {len(learners_from_topics)}, 课程: {len(learners_from_courses)}")
        
        # 先尝试取交集
        intersection_learners = learners_from_units & learners_from_topics & learners_from_courses
        
        # 移除目标学习者自己
        if learner_uid in intersection_learners:
            intersection_learners.remove(learner_uid)
        
        if intersection_learners:
            logger.info(f"使用交集策略，相关学习者数量: {len(intersection_learners)}")
            return intersection_learners
        else:
            # 交集为空，取并集
            union_learners = learners_from_units | learners_from_topics | learners_from_courses
            
            # 移除目标学习者自己
            if learner_uid in union_learners:
                union_learners.remove(learner_uid)
            
            logger.info(f"交集为空，使用并集策略，相关学习者数量: {len(union_learners)}")
            return union_learners
    
    def get_unit_interaction_records(self, learner_uids: List[str], unit_uids: List[str]) -> List[Tuple[str, str, str]]:
        """获取指定学习者和学习单元之间的交互记录"""
        if not learner_uids or not unit_uids:
            return []
        
        try:
            learner_placeholders = ', '.join(['%s'] * len(learner_uids))
            unit_placeholders = ', '.join(['%s'] * len(unit_uids))
            
            query = f"""
            SELECT lrn_uid, unt_uid 
            FROM Interaction 
            WHERE lrn_uid IN ({learner_placeholders}) 
            AND unt_uid IN ({unit_placeholders})
            """
            params = tuple(learner_uids + unit_uids)
            
            results = self.execute_custom_mysql_query(query, params)
            return [(result['lrn_uid'], result['unt_uid'], 'unit') for result in results]
            
        except Exception as e:
            logger.error(f"获取学习单元交互记录失败: {e}")
            return []
    
    def get_topic_interaction_records(self, learner_uids: List[str], topic_uids: List[str]) -> List[Tuple[str, str, str]]:
        """获取指定学习者和主题之间的交互记录 - 使用Learner_Topic视图"""
        if not learner_uids or not topic_uids:
            return []
        
        try:
            learner_placeholders = ', '.join(['%s'] * len(learner_uids))
            topic_placeholders = ', '.join(['%s'] * len(topic_uids))
            
            query = f"""
            SELECT lrn_uid, tpc_uid
            FROM Learner_Topic
            WHERE lrn_uid IN ({learner_placeholders})
            AND tpc_uid IN ({topic_placeholders})
            """
            params = tuple(learner_uids + topic_uids)
            
            results = self.execute_custom_mysql_query(query, params)
            return [(result['lrn_uid'], result['tpc_uid'], 'topic') for result in results]
            
        except Exception as e:
            logger.error(f"获取主题交互记录失败: {e}")
            return []
    
    def get_course_interaction_records(self, learner_uids: List[str], course_uids: List[str]) -> List[Tuple[str, str, str]]:
        """获取指定学习者和课程之间的交互记录 - 使用Learner_Course视图"""
        if not learner_uids or not course_uids:
            return []
        
        try:
            learner_placeholders = ', '.join(['%s'] * len(learner_uids))
            course_placeholders = ', '.join(['%s'] * len(course_uids))
            
            query = f"""
            SELECT lrn_uid, crs_uid
            FROM Learner_Course
            WHERE lrn_uid IN ({learner_placeholders})
            AND crs_uid IN ({course_placeholders})
            """
            params = tuple(learner_uids + course_uids)
            
            results = self.execute_custom_mysql_query(query, params)
            return [(result['lrn_uid'], result['crs_uid'], 'course') for result in results]
            
        except Exception as e:
            logger.error(f"获取课程交互记录失败: {e}")
            return []
    
    def build_optimized_learner_mapping(self, target_learner_uid: str, related_learners: Set[str]) -> Dict[str, int]:
        """构建优化的学习者UID映射：目标学习者放在索引0"""
        # 目标学习者放在第一位
        all_learners = [target_learner_uid]
        
        # 添加其他相关学习者
        for learner in related_learners:
            if learner != target_learner_uid:
                all_learners.append(learner)
        
        return {uid: idx for idx, uid in enumerate(all_learners)}
    
    def build_entity_mapping(self, entities: List[str]) -> Dict[str, int]:
        """构建实体UID映射"""
        return {uid: idx for idx, uid in enumerate(entities)}
    
    # 实现抽象方法
    def get_data_for_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """为单个学习者获取HGC模型数据"""
        return self.get_hgc_data_for_learner(learner_uid)
    
    def get_data_for_multiple_learners(self, learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
        """为多个学习者获取HGC模型数据"""
        results = {}
        for learner_uid in learner_uids:
            try:
                results[learner_uid] = self.get_hgc_data_for_learner(learner_uid)
            except Exception as e:
                logger.error(f"为学习者 {learner_uid} 获取HGC数据失败: {e}")
                results[learner_uid] = None
        return results
    
    def get_hgc_data_for_learner(self, learner_uid: str) -> Dict[str, Any]:
        """为单个学习者获取HGC模型所需的所有数据"""
        try:
            # 1-3. 获取学习者交互的单元、主题、课程
            interacted_units = self.get_learner_interacted_units(learner_uid)
            learner_topics = self.get_learner_topics(learner_uid)
            learner_courses = self.get_learner_courses(learner_uid)
            
            logger.info(f"学习者 {learner_uid} - 单元: {len(interacted_units)}, 主题: {len(learner_topics)}, 课程: {len(learner_courses)}")
            
            # 4-6. 使用优化策略获取相关学习者
            related_learners = self.get_optimized_related_learners(
                learner_uid, interacted_units, learner_topics, learner_courses
            )
            
            logger.info(f"相关学习者数量: {len(related_learners)}")
            
            # 7. 构建优化的学习者UID映射（目标学习者在索引0）
            learner_uid_mapping = self.build_optimized_learner_mapping(learner_uid, related_learners)
            
            # 8. 构建实体映射（只包含实际选取的实体）
            unit_uid_mapping = self.build_entity_mapping(interacted_units)
            topic_uid_mapping = self.build_entity_mapping(learner_topics)
            course_uid_mapping = self.build_entity_mapping(learner_courses)
            
            # 9. 构建精确的交互记录（只包含相关学习者和相关实体）
            all_learners = list(learner_uid_mapping.keys())
            all_interaction_records = []
            
            # 获取学习单元交互记录
            unit_records = self.get_unit_interaction_records(all_learners, interacted_units)
            all_interaction_records.extend(unit_records)
            
            # 获取主题交互记录
            topic_records = self.get_topic_interaction_records(all_learners, learner_topics)
            all_interaction_records.extend(topic_records)
            
            # 获取课程交互记录
            course_records = self.get_course_interaction_records(all_learners, learner_courses)
            all_interaction_records.extend(course_records)
            
            logger.info(f"精确交互记录数: {len(all_interaction_records)}")
            
            return {
                'target_learner_uid': learner_uid,
                'interacted_units': interacted_units,
                'learner_topics': learner_topics,
                'learner_courses': learner_courses,
                'related_learners': list(related_learners),
                'learner_uid_mapping': learner_uid_mapping,
                'unit_uid_mapping': unit_uid_mapping,
                'topic_uid_mapping': topic_uid_mapping,
                'course_uid_mapping': course_uid_mapping,
                'interaction_records': all_interaction_records,
                'strategy_used': 'intersection' if len(related_learners) < len(all_learners) - 1 else 'union'
            }
            
        except Exception as e:
            logger.error(f"为学习者 {learner_uid} 获取HGC数据失败: {e}")
            raise

# 全局仓库实例
hgc_repository = HGCRepository()