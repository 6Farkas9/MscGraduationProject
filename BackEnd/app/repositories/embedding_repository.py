# embedding_repository.py
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)

class EmbeddingRepository(BaseRepository):
    """嵌入向量数据仓库 - 专门处理Embeddings集合的操作"""
    
    def __init__(self):
        super().__init__()
        self.collection_name = "Embeddings"
    
    def get_embedding_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        根据指定的UID获取单个文档中的嵌入表达和更新时间
        
        Args:
            uid: 实体UID
            
        Returns:
            Dict包含embedding和updated_time，如果未找到返回None
        """
        try:
            # 构建查询条件
            query = {"uid": uid}
            
            # 执行查询
            documents = self.mongodb_operator.find_by_fields(self.collection_name, query)
            
            if not documents:
                logger.warning(f"未找到UID为 {uid} 的嵌入向量文档")
                return None
            
            if len(documents) > 1:
                logger.warning(f"找到多个UID为 {uid} 的文档，返回第一个")
            
            document = documents[0]
            
            # 提取所需字段并格式化
            result = {
                'uid': document.get('uid'),
                'entity_type': document.get('entity_type'),
                'embedding': document.get('embedding', []),
                'updated_time': document.get('updated_time')
            }
            
            logger.info(f"成功获取UID {uid} 的嵌入向量，维度: {len(result['embedding'])}")
            return result
            
        except Exception as e:
            logger.error(f"获取UID {uid} 的嵌入向量失败: {e}")
            return None
    
    def get_embeddings_by_uids(self, uids: List[str], return_format: str = "list") -> Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        根据指定的多个UID获取嵌入表达和更新时间
        
        Args:
            uids: 实体UID列表
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
                logger.warning(f"未找到任何UID在 {uids} 中的嵌入向量文档")
                return [] if return_format == "list" else {}
            
            logger.info(f"成功获取 {len(documents)} 个嵌入向量文档，查询UID数量: {len(uids)}")
            
            # 处理结果
            processed_docs = []
            uid_to_doc = {}
            
            for doc in documents:
                processed_doc = {
                    'uid': doc.get('uid'),
                    'entity_type': doc.get('entity_type'),
                    'embedding': doc.get('embedding', []),
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
                            'entity_type': None,
                            'embedding': [],
                            'updated_time': None
                        })
                        logger.warning(f"UID {uid} 未找到对应的嵌入向量")
                
                return ordered_results
            else:
                # 字典格式返回
                return uid_to_doc
            
        except Exception as e:
            logger.error(f"批量获取嵌入向量失败: {e}")
            return [] if return_format == "list" else {}
    
    def get_embeddings_by_entity_type(self, entity_type: str, limit: int = 0) -> List[Dict[str, Any]]:
        """
        根据实体类型获取嵌入向量
        
        Args:
            entity_type: 实体类型 (如 'lrn', 'unt', 'tpc', 'crs' 等)
            limit: 限制返回数量，0表示无限制
            
        Returns:
            嵌入向量文档列表
        """
        try:
            query = {"entity_type": entity_type}
            documents = self.mongodb_operator.find_by_fields(self.collection_name, query)
            
            if limit > 0:
                documents = documents[:limit]
            
            results = []
            for doc in documents:
                results.append({
                    'uid': doc.get('uid'),
                    'entity_type': doc.get('entity_type'),
                    'embedding': doc.get('embedding', []),
                    'updated_time': doc.get('updated_time')
                })
            
            logger.info(f"成功获取 {len(results)} 个类型为 {entity_type} 的嵌入向量")
            return results
            
        except Exception as e:
            logger.error(f"根据实体类型 {entity_type} 获取嵌入向量失败: {e}")
            return []
    
    def get_embedding_stats(self, uid: str = None) -> Dict[str, Any]:
        """
        获取嵌入向量的统计信息
        
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
                    "$group": {
                        "_id": "$entity_type",
                        "count": {"$sum": 1},
                        "avg_embedding_length": {"$avg": {"$size": "$embedding"}},
                        "min_updated_time": {"$min": "$updated_time"},
                        "max_updated_time": {"$max": "$updated_time"}
                    }
                },
                {
                    "$project": {
                        "entity_type": "$_id",
                        "count": 1,
                        "avg_embedding_length": 1,
                        "min_updated_time": 1,
                        "max_updated_time": 1,
                        "_id": 0
                    }
                }
            ])
            
            stats = self.mongodb_operator.aggregate(self.collection_name, pipeline)
            
            # 计算总体统计
            total_count = sum(stat['count'] for stat in stats)
            total_embeddings = self.mongodb_operator.count_documents(
                self.collection_name, 
                {"uid": uid} if uid else {}
            )
            
            result = {
                "total_documents": total_embeddings,
                "statistics_by_type": stats
            }
            
            if uid:
                result["query_uid"] = uid
            
            logger.info(f"成功获取嵌入向量统计信息，总文档数: {total_embeddings}")
            return result
            
        except Exception as e:
            logger.error(f"获取嵌入向量统计信息失败: {e}")
            return {"total_documents": 0, "statistics_by_type": []}
    
    def check_embedding_exists(self, uid: str) -> bool:
        """
        检查指定UID的嵌入向量是否存在
        
        Args:
            uid: 实体UID
            
        Returns:
            是否存在
        """
        try:
            count = self.mongodb_operator.count_documents(self.collection_name, {"uid": uid})
            exists = count > 0
            
            logger.debug(f"检查UID {uid} 嵌入向量存在性: {exists}")
            return exists
            
        except Exception as e:
            logger.error(f"检查嵌入向量存在性失败: {e}")
            return False
    
    def get_recent_updated_embeddings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近更新的嵌入向量
        
        Args:
            limit: 返回数量限制
            
        Returns:
            最近更新的嵌入向量列表
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
                results.append({
                    'uid': doc.get('uid'),
                    'entity_type': doc.get('entity_type'),
                    'embedding': doc.get('embedding', []),
                    'updated_time': doc.get('updated_time')
                })
            
            logger.info(f"成功获取 {len(results)} 个最近更新的嵌入向量")
            return results
            
        except Exception as e:
            logger.error(f"获取最近更新的嵌入向量失败: {e}")
            return []

# 全局仓库实例
embedding_repository = EmbeddingRepository()