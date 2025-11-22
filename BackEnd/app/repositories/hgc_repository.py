# hgc_repository.py
import logging
from typing import List, Dict, Tuple, Set, Optional, Any
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class HGCRepository(BaseRepository):
    """HGC模型数据仓库 - 使用视图优化版本"""
    
    def __init__(self):
        super().__init__()
    
    def get_learners_interacted_units(self, learner_uids: List[str]) -> Dict[str, List[str]]:
        """批量获取多个学习者交互过的学习单元UID列表"""
        if not learner_uids:
            return {}
        
        try:
            placeholders = ', '.join(['%s'] * len(learner_uids))
            query = f"""
            SELECT lrn_uid, unt_uid 
            FROM Interaction 
            WHERE lrn_uid IN ({placeholders})
            """
            results = self.execute_custom_mysql_query(query, tuple(learner_uids))
            
            # 按学习者分组
            learner_units = {}
            for result in results:
                learner_uid = result['lrn_uid']
                unit_uid = result['unt_uid']
                if learner_uid not in learner_units:
                    learner_units[learner_uid] = []
                if unit_uid not in learner_units[learner_uid]:
                    learner_units[learner_uid].append(unit_uid)
            
            return learner_units
        except Exception as e:
            logger.error(f"批量获取学习者交互单元失败: {e}")
            return {}
    
    def get_learners_topics(self, learner_uids: List[str]) -> Dict[str, List[str]]:
        """批量获取多个学习者涉及的主题UID列表 - 使用Learner_Topic视图"""
        if not learner_uids:
            return {}
        
        try:
            placeholders = ', '.join(['%s'] * len(learner_uids))
            query = f"""
            SELECT lrn_uid, tpc_uid 
            FROM Learner_Topic 
            WHERE lrn_uid IN ({placeholders})
            """
            results = self.execute_custom_mysql_query(query, tuple(learner_uids))
            
            # 按学习者分组
            learner_topics = {}
            for result in results:
                learner_uid = result['lrn_uid']
                topic_uid = result['tpc_uid']
                if learner_uid not in learner_topics:
                    learner_topics[learner_uid] = []
                if topic_uid not in learner_topics[learner_uid]:
                    learner_topics[learner_uid].append(topic_uid)
            
            return learner_topics
        except Exception as e:
            logger.error(f"批量获取学习者涉及主题失败: {e}")
            return {}
    
    def get_learners_courses(self, learner_uids: List[str]) -> Dict[str, List[str]]:
        """批量获取多个学习者涉及的课程UID列表 - 使用Learner_Course视图"""
        if not learner_uids:
            return {}
        
        try:
            placeholders = ', '.join(['%s'] * len(learner_uids))
            query = f"""
            SELECT lrn_uid, crs_uid 
            FROM Learner_Course 
            WHERE lrn_uid IN ({placeholders})
            """
            results = self.execute_custom_mysql_query(query, tuple(learner_uids))
            
            # 按学习者分组
            learner_courses = {}
            for result in results:
                learner_uid = result['lrn_uid']
                course_uid = result['crs_uid']
                if learner_uid not in learner_courses:
                    learner_courses[learner_uid] = []
                if course_uid not in learner_courses[learner_uid]:
                    learner_courses[learner_uid].append(course_uid)
            
            return learner_courses
        except Exception as e:
            logger.error(f"批量获取学习者涉及课程失败: {e}")
            return {}
    
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
    
    def get_optimized_related_learners(self, target_learner_uids: List[str], 
                                     all_interacted_units: List[str],
                                     all_learner_topics: List[str],
                                     all_learner_courses: List[str]) -> Set[str]:
        """优化策略获取相关学习者：先取交集，为空则取并集"""
        # 获取三个元路径的相关学习者
        learners_from_units = self.get_other_learners_by_units(all_interacted_units)
        learners_from_topics = self.get_other_learners_by_topics(all_learner_topics)
        learners_from_courses = self.get_other_learners_by_courses(all_learner_courses)
        
        logger.info(f"元路径学习者数量 - 单元: {len(learners_from_units)}, "
                   f"主题: {len(learners_from_topics)}, 课程: {len(learners_from_courses)}")
        
        # 先尝试取交集
        intersection_learners = learners_from_units & learners_from_topics & learners_from_courses
        
        # 移除目标学习者自己
        for target_uid in target_learner_uids:
            if target_uid in intersection_learners:
                intersection_learners.remove(target_uid)
        
        if intersection_learners:
            logger.info(f"使用交集策略，相关学习者数量: {len(intersection_learners)}")
            return intersection_learners
        else:
            # 交集为空，取并集
            union_learners = learners_from_units | learners_from_topics | learners_from_courses
            
            # 移除目标学习者自己
            for target_uid in target_learner_uids:
                if target_uid in union_learners:
                    union_learners.remove(target_uid)
            
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
    
    def get_data_for_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """为单个学习者获取HGC模型数据"""
        return self.get_hgc_data_for_learner(learner_uid)
    
    def get_data_for_multiple_learners(self, learner_uids: List[str]) -> Dict[str, Any]:
        """为多个学习者获取HGC模型数据 - 优化版本"""
        try:
            # 1. 批量获取学习者交互的单元、主题、课程
            learner_units = self.get_learners_interacted_units(learner_uids)
            learner_topics = self.get_learners_topics(learner_uids)
            learner_courses = self.get_learners_courses(learner_uids)
            
            # 2. 收集所有涉及的实体
            all_interacted_units = list(set(
                unit for units in learner_units.values() for unit in units
            ))
            all_learner_topics = list(set(
                topic for topics in learner_topics.values() for topic in topics
            ))
            all_learner_courses = list(set(
                course for courses in learner_courses.values() for course in courses
            ))
            
            logger.info(f"批量处理 {len(learner_uids)} 个学习者 - "
                       f"单元: {len(all_interacted_units)}, "
                       f"主题: {len(all_learner_topics)}, "
                       f"课程: {len(all_learner_courses)}")
            
            # 3. 使用优化策略获取相关学习者
            related_learners = self.get_optimized_related_learners(
                learner_uids, all_interacted_units, all_learner_topics, all_learner_courses
            )
            
            logger.info(f"相关学习者数量: {len(related_learners)}")
            
            # 4. 构建精确的交互记录（包含所有目标学习者和相关学习者）
            all_learners = learner_uids + list(related_learners)
            all_interaction_records = []
            
            # 获取学习单元交互记录
            unit_records = self.get_unit_interaction_records(all_learners, all_interacted_units)
            all_interaction_records.extend(unit_records)
            
            # 获取主题交互记录
            topic_records = self.get_topic_interaction_records(all_learners, all_learner_topics)
            all_interaction_records.extend(topic_records)
            
            # 获取课程交互记录
            course_records = self.get_course_interaction_records(all_learners, all_learner_courses)
            all_interaction_records.extend(course_records)
            
            logger.info(f"精确交互记录数: {len(all_interaction_records)}")
            
            return {
                'target_learner_uids': learner_uids,
                'learner_entities': {
                    'units': learner_units,
                    'topics': learner_topics,
                    'courses': learner_courses
                },
                'all_entities': {
                    'units': all_interacted_units,
                    'topics': all_learner_topics,
                    'courses': all_learner_courses
                },
                'related_learners': list(related_learners),
                'interaction_records': all_interaction_records,
                'strategy_used': 'intersection' if len(related_learners) < len(all_learners) - len(learner_uids) else 'union'
            }
            
        except Exception as e:
            logger.error(f"为多个学习者获取HGC数据失败: {e}")
            raise
    
    def get_hgc_data_for_learner(self, learner_uid: str) -> Dict[str, Any]:
        """为单个学习者获取HGC模型所需的所有数据"""
        try:
            # 使用批量方法获取单个学习者的数据
            multi_result = self.get_data_for_multiple_learners([learner_uid])
            
            # 转换为单个学习者的格式
            return {
                'target_learner_uid': learner_uid,
                'interacted_units': multi_result['learner_entities']['units'].get(learner_uid, []),
                'learner_topics': multi_result['learner_entities']['topics'].get(learner_uid, []),
                'learner_courses': multi_result['learner_entities']['courses'].get(learner_uid, []),
                'related_learners': multi_result['related_learners'],
                'interaction_records': multi_result['interaction_records'],
                'strategy_used': multi_result['strategy_used']
            }
            
        except Exception as e:
            logger.error(f"为学习者 {learner_uid} 获取HGC数据失败: {e}")
            raise

# 全局仓库实例
hgc_repository = HGCRepository()