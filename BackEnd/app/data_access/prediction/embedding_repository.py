# BackEnd/app/data_access/prediction/embedding_repository.py
import logging
from typing import List, Dict, Any, Optional, Union

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository

logger = logging.getLogger(__name__)


class EmbeddingRepository(MongoDBBaseRepository):
    """
    嵌入向量数据仓库 - 负责 Embeddings 集合的所有读数据操作

    设计要点：
    - 只依赖 MongoDBBaseRepository（内部封装了 MongoDBOperator）
    - 不做连接管理，不做底层 Mongo 细节，只组织查询、聚合和结果格式化
    - 所有返回值结构和原版本保持一致：字段名 / 逻辑不变
    """

    def __init__(self, collection_name: str = "Embeddings") -> None:
        super().__init__()
        self.collection_name = collection_name

    # ------------------------------------------------------------------
    # 基础存在性检查
    # ------------------------------------------------------------------

    def check_embedding_exists(self, uid: str) -> bool:
        """
        检查指定 UID 的嵌入向量是否存在

        Args:
            uid: 实体 UID

        Returns:
            是否存在
        """
        try:
            count = self._mongo.count_documents(self.collection_name, {"uid": uid})
            exists = count > 0
            logger.debug("检查UID %s 嵌入向量存在性: %s", uid, exists)
            return exists
        except Exception as exc:
            logger.error("检查嵌入向量存在性失败: %s", exc)
            return False

    # ------------------------------------------------------------------
    # 单个 UID 查询
    # ------------------------------------------------------------------

    def get_embedding_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        根据指定的 UID 获取单个文档中的嵌入表达和更新时间

        Returns:
            {
                "uid": ...,
                "entity_type": ...,
                "embedding": [...],
                "updated_time": ...
            }
            如果未找到则返回 None
        """
        try:
            query = {"uid": uid}
            documents = self._mongo.find_by_fields(self.collection_name, query)

            if not documents:
                logger.warning("未找到UID为 %s 的嵌入向量文档", uid)
                return None

            if len(documents) > 1:
                logger.warning("找到多个UID为 %s 的文档，返回第一个", uid)

            doc = documents[0]
            result = {
                "uid": doc.get("uid"),
                "entity_type": doc.get("entity_type"),
                "embedding": doc.get("embedding", []),
                "updated_time": doc.get("updated_time"),
            }

            logger.info(
                "成功获取UID %s 的嵌入向量，维度: %d",
                uid,
                len(result["embedding"]),
            )
            return result
        except Exception as exc:
            logger.error("获取UID %s 的嵌入向量失败: %s", uid, exc)
            return None

    # ------------------------------------------------------------------
    # 多 UID 查询（保持返回结构 & 顺序）
    # ------------------------------------------------------------------

    def get_embeddings_by_uids(
        self,
        uids: List[str],
        return_format: str = "list",
    ) -> Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        根据多个 UID 获取嵌入表达和更新时间

        Args:
            uids: 实体 UID 列表
            return_format:
                - "list": 返回与输入 UID 顺序一致的列表，
                          对于未找到的 UID 用空结构占位
                - "dict": 返回 {uid: {...}} 的字典，只包含查到的 UID

        Returns:
            List[dict] 或 Dict[str, dict]
        """
        if not uids:
            logger.warning("提供的UID列表为空")
            return [] if return_format == "list" else {}

        try:
            query = {"uid": {"$in": uids}}
            documents = self._mongo.find_by_fields(self.collection_name, query)

            if not documents:
                logger.warning("未找到任何UID在 %s 中的嵌入向量文档", uids)
                return [] if return_format == "list" else {}

            logger.info(
                "成功获取 %d 个嵌入向量文档，查询UID数量: %d",
                len(documents),
                len(uids),
            )

            # 构建 uid -> 文档 映射
            uid_to_doc: Dict[str, Dict[str, Any]] = {}
            for doc in documents:
                processed = {
                    "uid": doc.get("uid"),
                    "entity_type": doc.get("entity_type"),
                    "embedding": doc.get("embedding", []),
                    "updated_time": doc.get("updated_time"),
                }
                uid_value = processed["uid"]
                if uid_value is not None:
                    uid_to_doc[uid_value] = processed

            # 使用之前提取的通用工具方法：提取结果中的 uid（用于调试或统计）
            found_uids = self.extract_uids_from_results(
                list(uid_to_doc.values()),
                uid_field="uid",
            )
            logger.debug("批量嵌入查询，实际命中的UID: %s", found_uids)

            if return_format == "dict":
                return uid_to_doc

            # 列表模式：按输入顺序返回，并对缺失 UID 填充空结果
            ordered_results: List[Dict[str, Any]] = []
            for uid in uids:
                if uid in uid_to_doc:
                    ordered_results.append(uid_to_doc[uid])
                else:
                    ordered_results.append(
                        {
                            "uid": uid,
                            "entity_type": None,
                            "embedding": [],
                            "updated_time": None,
                        }
                    )
                    logger.warning("UID %s 未找到对应的嵌入向量", uid)

            return ordered_results
        except Exception as exc:
            logger.error("批量获取嵌入向量失败: %s", exc)
            return [] if return_format == "list" else {}

    # ------------------------------------------------------------------
    # 按实体类型查询
    # ------------------------------------------------------------------

    def get_embeddings_by_entity_type(
        self,
        entity_type: str,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        根据实体类型获取嵌入向量

        Args:
            entity_type: 实体类型 (如 "lrn" / "unt" / "tpc" / "crs" 等)
            limit: 限制返回数量，0 表示不限制
        """
        try:
            query = {"entity_type": entity_type}
            documents = self._mongo.find_by_fields(self.collection_name, query)

            if limit > 0:
                documents = documents[:limit]

            results: List[Dict[str, Any]] = []
            for doc in documents:
                results.append(
                    {
                        "uid": doc.get("uid"),
                        "entity_type": doc.get("entity_type"),
                        "embedding": doc.get("embedding", []),
                        "updated_time": doc.get("updated_time"),
                    }
                )

            logger.info(
                "成功获取 %d 个类型为 %s 的嵌入向量",
                len(results),
                entity_type,
            )
            return results
        except Exception as exc:
            logger.error("根据实体类型 %s 获取嵌入向量失败: %s", entity_type, exc)
            return []

    # ------------------------------------------------------------------
    # 嵌入向量统计（按 entity_type 聚合）
    # ------------------------------------------------------------------

    def get_embedding_stats(self, uid: Optional[str] = None) -> Dict[str, Any]:
        """
        获取嵌入向量的统计信息

        Args:
            uid: 可选的特定 UID；如果为 None 则统计整个集合

        Returns:
            {
                "total_documents": <int>,
                "statistics_by_type": [
                    {
                        "entity_type": ...,
                        "count": ...,
                        "avg_embedding_length": ...,
                        "min_updated_time": ...,
                        "max_updated_time": ...,
                    },
                    ...
                ],
                # 如果指定了 uid:
                "query_uid": "<uid>"
            }
        """
        try:
            pipeline: List[Dict[str, Any]] = []

            # 按 UID 过滤（可选）
            if uid:
                pipeline.append({"$match": {"uid": uid}})

            pipeline.extend(
                [
                    {
                        "$group": {
                            "_id": "$entity_type",
                            "count": {"$sum": 1},
                            "avg_embedding_length": {
                                "$avg": {"$size": "$embedding"}
                            },
                            "min_updated_time": {"$min": "$updated_time"},
                            "max_updated_time": {"$max": "$updated_time"},
                        }
                    },
                    {
                        "$project": {
                            "entity_type": "$_id",
                            "count": 1,
                            "avg_embedding_length": 1,
                            "min_updated_time": 1,
                            "max_updated_time": 1,
                            "_id": 0,
                        }
                    },
                ]
            )

            stats = self._mongo.aggregate(self.collection_name, pipeline)

            total_embeddings = self._mongo.count_documents(
                self.collection_name, {"uid": uid} if uid else {}
            )

            result: Dict[str, Any] = {
                "total_documents": total_embeddings,
                "statistics_by_type": stats,
            }
            if uid:
                result["query_uid"] = uid

            logger.info(
                "成功获取嵌入向量统计信息，总文档数: %d",
                total_embeddings,
            )
            return result
        except Exception as exc:
            logger.error("获取嵌入向量统计信息失败: %s", exc)
            return {"total_documents": 0, "statistics_by_type": []}

    # ------------------------------------------------------------------
    # 最近更新的嵌入向量
    # ------------------------------------------------------------------

    def get_recent_updated_embeddings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近更新的嵌入向量（按 updated_time 降序）

        Args:
            limit: 返回数量限制
        """
        try:
            documents = self._mongo.find_with_sort(
                self.collection_name,
                sort_field="updated_time",
                sort_order="desc",
                limit=limit,
            )

            results: List[Dict[str, Any]] = []
            for doc in documents:
                results.append(
                    {
                        "uid": doc.get("uid"),
                        "entity_type": doc.get("entity_type"),
                        "embedding": doc.get("embedding", []),
                        "updated_time": doc.get("updated_time"),
                    }
                )

            logger.info(
                "成功获取 %d 个最近更新的嵌入向量",
                len(results),
            )
            return results
        except Exception as exc:
            logger.error("获取最近更新的嵌入向量失败: %s", exc)
            return []
