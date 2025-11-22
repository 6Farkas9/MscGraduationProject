# learner_repository.py
import logging
from typing import List, Dict, Any, Optional, Union
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class LearnerRepository(BaseRepository):
    """学习者数据仓库 - 专门处理Learner集合的操作"""
    
    def __init__(self):
        super().__init__()
        self.collection_name = "Learners"
    
    def get_kt_result_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        根据单个学习者的UID获取KT结果以及更新时间
        
        Args:
            uid: 学习者UID
            
        Returns:
            Dict包含KT结果和updated_time，如果未找到返回None
        """
        try:
            # 构建查询条件
            query = {"uid": uid}
            
            # 执行查询
            documents = self.mongodb_operator.find_by_fields(self.collection_name, query)
            
            if not documents:
                logger.warning(f"未找到UID为 {uid} 的学习者文档")
                return None
            
            if len(documents) > 1:
                logger.warning(f"找到多个UID为 {uid} 的文档，返回第一个")
            
            document = documents[0]
            
            # 提取所需字段并格式化
            result = {
                'uid': document.get('uid'),
                'KT': document.get('KT', {}),
                'updated_time': document.get('updated_time')
            }
            
            kt_result_count = len(result['KT'])
            logger.info(f"成功获取UID {uid} 的KT结果，包含 {kt_result_count} 个知识点掌握概率")
            return result
            
        except Exception as e:
            logger.error(f"获取UID {uid} 的KT结果失败: {e}")
            return None
    
    def get_kt_results_by_uids(self, uids: List[str], return_format: str = "list") -> Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        根据多个学习者的UID获取KT结果和更新时间
        
        Args:
            uids: 学习者UID列表
            return_format: 返回格式，"list" 或 "dict"
            
        Returns:
            根据return_format返回列表或字典格式的结果
        """
        if not uids:
            logger.warning("提供的UID列表为空")
            return [] if return_format == "list" else {}
        
        try:
            # 构建查询条件 - 使用$in操作符查询多个UID
            query = {"uid": {"$in": uids}}
            
            # 执行查询
            documents = self.mongodb_operator.find_by_fields(self.collection_name, query)
            
            if not documents:
                logger.warning(f"未找到任何UID在 {uids} 中的学习者文档")
                return [] if return_format == "list" else {}
            
            logger.info(f"成功获取 {len(documents)} 个学习者KT结果，查询UID数量: {len(uids)}")
            
            # 处理结果
            processed_docs = []
            uid_to_doc = {}
            
            for doc in documents:
                processed_doc = {
                    'uid': doc.get('uid'),
                    'KT': doc.get('KT', {}),
                    'updated_time': doc.get('updated_time')
                }
                processed_docs.append(processed_doc)
                uid_to_doc[doc.get('uid')] = processed_doc
            
            # 按照输入UID的顺序返回结果（列表格式）
            if return_format == "list":
                ordered_results = []
                for uid in uids:
                    if uid in uid_to_doc:
                        ordered_results.append(uid_to_doc[uid])
                    else:
                        # 对于未找到的UID，添加空结果
                        ordered_results.append({
                            'uid': uid,
                            'KT': {},
                            'updated_time': None
                        })
                        logger.warning(f"UID {uid} 未找到对应的KT结果")
                
                return ordered_results
            else:
                # 字典格式返回
                return uid_to_doc
            
        except Exception as e:
            logger.error(f"批量获取KT结果失败: {e}")
            return [] if return_format == "list" else {}
    
    def get_kt_statistics(self, uid: str = None) -> Dict[str, Any]:
        """
        获取KT结果的统计信息
        
        Args:
            uid: 可选的特定UID，如果为None则统计整个集合
            
        Returns:
            统计信息字典
        """
        try:
            pipeline = []
            
            # 如果有特定UID，添加匹配阶段
            if uid:
                pipeline.append({"$match": {"uid": uid}})
            
            # 添加统计阶段
            pipeline.extend([
                {
                    "$project": {
                        "uid": 1,
                        "kt_count": {"$size": {"$objectToArray": "$KT"}},
                        "kt_values": {"$objectToArray": "$KT"}
                    }
                },
                {
                    "$unwind": "$kt_values"
                },
                {
                    "$group": {
                        "_id": "$uid",
                        "total_concepts": {"$sum": 1},
                        "avg_mastery": {"$avg": "$kt_values.v"},
                        "min_mastery": {"$min": "$kt_values.v"},
                        "max_mastery": {"$max": "$kt_values.v"},
                        "high_mastery_count": {
                            "$sum": {
                                "$cond": [{"$gte": ["$kt_values.v", 0.8]}, 1, 0]
                            }
                        },
                        "medium_mastery_count": {
                            "$sum": {
                                "$cond": [
                                    {"$and": [
                                        {"$gte": ["$kt_values.v", 0.5]},
                                        {"$lt": ["$kt_values.v", 0.8]}
                                    ]}, 1, 0
                                ]
                            }
                        },
                        "low_mastery_count": {
                            "$sum": {
                                "$cond": [{"$lt": ["$kt_values.v", 0.5]}, 1, 0]
                            }
                        }
                    }
                },
                {
                    "$project": {
                        "uid": "$_id",
                        "total_concepts": 1,
                        "avg_mastery": 1,
                        "min_mastery": 1,
                        "max_mastery": 1,
                        "high_mastery_count": 1,
                        "medium_mastery_count": 1,
                        "low_mastery_count": 1,
                        "high_mastery_ratio": {
                            "$divide": ["$high_mastery_count", "$total_concepts"]
                        },
                        "_id": 0
                    }
                }
            ])
            
            stats = self.mongodb_operator.aggregate(self.collection_name, pipeline)
            
            # 计算总体统计
            total_learners = self.mongodb_operator.count_documents(
                self.collection_name, 
                {"uid": uid} if uid else {}
            )
            
            result = {
                "total_learners": total_learners,
                "statistics": list(stats)
            }
            
            if uid:
                result["query_uid"] = uid
            
            logger.info(f"成功获取KT统计信息，总学习者数: {total_learners}")
            return result
            
        except Exception as e:
            logger.error(f"获取KT统计信息失败: {e}")
            return {"total_learners": 0, "statistics": []}
    
    def get_concept_mastery(self, uid: str, concept_ids: List[str] = None) -> Dict[str, Any]:
        """
        获取特定学习者对指定知识点的掌握情况
        
        Args:
            uid: 学习者UID
            concept_ids: 知识点ID列表，如果为None则返回所有知识点
            
        Returns:
            知识点掌握情况字典
        """
        try:
            kt_data = self.get_kt_result_by_uid(uid)
            if not kt_data:
                return {}
            
            kt_results = kt_data['KT']
            
            if concept_ids:
                # 只返回指定的知识点
                result = {}
                for concept_id in concept_ids:
                    if concept_id in kt_results:
                        result[concept_id] = kt_results[concept_id]
                    else:
                        result[concept_id] = None  # 知识点不存在
                return result
            else:
                # 返回所有知识点
                return kt_results
            
        except Exception as e:
            logger.error(f"获取学习者 {uid} 知识点掌握情况失败: {e}")
            return {}
    
    def get_learners_by_mastery_level(self, concept_id: str, 
                                    min_mastery: float = 0.0, 
                                    max_mastery: float = 1.0,
                                    limit: int = 0) -> List[Dict[str, Any]]:
        """
        根据知识点掌握水平获取学习者
        
        Args:
            concept_id: 知识点ID
            min_mastery: 最小掌握水平
            max_mastery: 最大掌握水平
            limit: 限制返回数量
            
        Returns:
            学习者列表
        """
        try:
            # 构建查询条件
            query = {
                f"KT.{concept_id}": {
                    "$gte": min_mastery,
                    "$lte": max_mastery
                }
            }
            
            documents = self.mongodb_operator.find_by_fields(self.collection_name, query)
            
            if limit > 0:
                documents = documents[:limit]
            
            results = []
            for doc in documents:
                mastery_level = doc.get('KT', {}).get(concept_id, 0.0)
                results.append({
                    'uid': doc.get('uid'),
                    'mastery_level': mastery_level,
                    'updated_time': doc.get('updated_time')
                })
            
            logger.info(f"成功获取 {len(results)} 个在掌握水平 [{min_mastery}, {max_mastery}] 内的学习者")
            return results
            
        except Exception as e:
            logger.error(f"根据掌握水平获取学习者失败: {e}")
            return []
    
    def check_learner_exists(self, uid: str) -> bool:
        """
        检查指定UID的学习者是否存在
        
        Args:
            uid: 学习者UID
            
        Returns:
            是否存在
        """
        try:
            count = self.mongodb_operator.count_documents(self.collection_name, {"uid": uid})
            exists = count > 0
            
            logger.debug(f"检查UID {uid} 学习者存在性: {exists}")
            return exists
            
        except Exception as e:
            logger.error(f"检查学习者存在性失败: {e}")
            return False
    
    def get_recent_updated_learners(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近更新的学习者
        
        Args:
            limit: 返回数量限制
            
        Returns:
            最近更新的学习者列表
        """
        try:
            documents = self.mongodb_operator.find_with_sort(
                self.collection_name, 
                "updated_time", 
                "desc", 
                limit
            )
            
            results = []
            for doc in documents:
                kt_count = len(doc.get('KT', {}))
                results.append({
                    'uid': doc.get('uid'),
                    'kt_concept_count': kt_count,
                    'updated_time': doc.get('updated_time')
                })
            
            logger.info(f"成功获取 {len(results)} 个最近更新的学习者")
            return results
            
        except Exception as e:
            logger.error(f"获取最近更新的学习者失败: {e}")
            return []


# 全局仓库实例
learner_repository = LearnerRepository()