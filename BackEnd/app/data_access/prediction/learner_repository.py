# BackEnd/app/data_access/prediction/learner_repository.py
import logging
from typing import List, Dict, Any, Optional, Union

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository

logger = logging.getLogger(__name__)


class LearnerRepository(MongoDBBaseRepository):
    """
    学习者数据仓库 - 负责 Learners 集合中 KT 结果等数据的读取

    设计要点：
    - 只做数据查询和结果组织，不混入 MySQL
    - 核心接口与原文件保持名称和返回结构，方便无痛替换
    - 一些公共模式（多 UID 查询、有序返回）与 EmbeddingRepository 保持一致风格
    """

    def __init__(self, collection_name: str = "Learners") -> None:
        super().__init__()
        self.collection_name = collection_name

    # ------------------------------------------------------------------
    # 单 UID KT 结果
    # ------------------------------------------------------------------

    def get_kt_result_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        根据单个学习者的 UID 获取 KT 结果以及更新时间

        Returns:
            {
                "uid": ...,
                "KT": {...},
                "updated_time": ...
            }
            若未找到则返回 None
        """
        try:
            query = {"uid": uid}
            documents = self._mongo.find_by_fields(self.collection_name, query)

            if not documents:
                logger.warning("未找到UID为 %s 的学习者文档", uid)
                return None

            if len(documents) > 1:
                logger.warning("找到多个UID为 %s 的文档，返回第一个", uid)

            doc = documents[0]
            result = {
                "uid": doc.get("uid"),
                "KT": doc.get("KT", {}),
                "updated_time": doc.get("updated_time"),
            }

            kt_result_count = len(result["KT"])
            logger.info(
                "成功获取UID %s 的KT结果，包含 %d 个知识点掌握概率",
                uid,
                kt_result_count,
            )
            return result
        except Exception as exc:
            logger.error("获取UID %s 的KT结果失败: %s", uid, exc)
            return None

    # ------------------------------------------------------------------
    # 多 UID KT 结果（顺序 & 占位）
    # ------------------------------------------------------------------

    def get_kt_results_by_uids(
        self,
        uids: List[str],
        return_format: str = "list",
    ) -> Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        根据多个学习者 UID 获取 KT 结果和更新时间

        Args:
            uids: 学习者 UID 列表
            return_format: "list" 或 "dict"

        Returns:
            - "list"：按输入 UID 的顺序排列，未命中的 UID 添加占位结果
            - "dict"：{uid: {...}} 只包含命中的 UID
        """
        if not uids:
            logger.warning("提供的UID列表为空")
            return [] if return_format == "list" else {}

        try:
            query = {"uid": {"$in": uids}}
            documents = self._mongo.find_by_fields(self.collection_name, query)

            if not documents:
                logger.warning("未找到任何UID在 %s 中的学习者文档", uids)
                return [] if return_format == "list" else {}

            logger.info(
                "成功获取 %d 个学习者KT结果，查询UID数量: %d",
                len(documents),
                len(uids),
            )

            uid_to_doc: Dict[str, Dict[str, Any]] = {}
            for doc in documents:
                processed = {
                    "uid": doc.get("uid"),
                    "KT": doc.get("KT", {}),
                    "updated_time": doc.get("updated_time"),
                }
                uid_val = processed["uid"]
                if uid_val is not None:
                    uid_to_doc[uid_val] = processed

            # 使用公共工具方法：提取结果中的 uid，便于调试统计
            found_uids = self.extract_uids_from_results(
                list(uid_to_doc.values()),
                uid_field="uid",
            )
            logger.debug("批量KT查询，实际命中的UID: %s", found_uids)

            if return_format == "dict":
                return uid_to_doc

            ordered_results: List[Dict[str, Any]] = []
            for uid in uids:
                if uid in uid_to_doc:
                    ordered_results.append(uid_to_doc[uid])
                else:
                    ordered_results.append(
                        {"uid": uid, "KT": {}, "updated_time": None}
                    )
                    logger.warning("UID %s 未找到对应的KT结果", uid)

            return ordered_results
        except Exception as exc:
            logger.error("批量获取KT结果失败: %s", exc)
            return [] if return_format == "list" else {}

    # ------------------------------------------------------------------
    # KT 聚合统计（使用 MongoDB 聚合管道）
    # ------------------------------------------------------------------

    def get_kt_statistics(self, uid: Optional[str] = None) -> Dict[str, Any]:
        """
        获取 KT 结果的统计信息

        Args:
            uid: 可选的特定 UID，如果为 None 则统计整个集合

        Returns:
            {
                "total_learners": <int>,
                "statistics": [
                    {
                        "uid": ...,
                        "total_concepts": ...,
                        "avg_mastery": ...,
                        "min_mastery": ...,
                        "max_mastery": ...,
                        "high_mastery_count": ...,
                        "medium_mastery_count": ...,
                        "low_mastery_count": ...,
                        "high_mastery_ratio": ...,
                    }, ...
                ],
                # 如果传入 uid:
                "query_uid": "<uid>"
            }
        """
        try:
            pipeline: List[Dict[str, Any]] = []

            if uid:
                pipeline.append({"$match": {"uid": uid}})

            pipeline.extend(
                [
                    {
                        "$project": {
                            "uid": 1,
                            "kt_count": {"$size": {"$objectToArray": "$KT"}},
                            "kt_values": {"$objectToArray": "$KT"},
                        }
                    },
                    {"$unwind": "$kt_values"},
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
                                        {
                                            "$and": [
                                                {
                                                    "$gte": [
                                                        "$kt_values.v",
                                                        0.5,
                                                    ]
                                                },
                                                {
                                                    "$lt": [
                                                        "$kt_values.v",
                                                        0.8,
                                                    ]
                                                },
                                            ]
                                        },
                                        1,
                                        0,
                                    ]
                                }
                            },
                            "low_mastery_count": {
                                "$sum": {
                                    "$cond": [
                                        {"$lt": ["$kt_values.v", 0.5]},
                                        1,
                                        0,
                                    ]
                                }
                            },
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
                                "$divide": [
                                    "$high_mastery_count",
                                    "$total_concepts",
                                ]
                            },
                            "_id": 0,
                        }
                    },
                ]
            )

            stats = self._mongo.aggregate(self.collection_name, pipeline)

            total_learners = self._mongo.count_documents(
                self.collection_name, {"uid": uid} if uid else {}
            )

            result: Dict[str, Any] = {
                "total_learners": total_learners,
                "statistics": list(stats),
            }
            if uid:
                result["query_uid"] = uid

            logger.info("成功获取KT统计信息，总学习者数: %d", total_learners)
            return result
        except Exception as exc:
            logger.error("获取KT统计信息失败: %s", exc)
            return {"total_learners": 0, "statistics": []}

    # ------------------------------------------------------------------
    # 单个学习者的概念掌握度
    # ------------------------------------------------------------------

    def get_concept_mastery(
        self,
        uid: str,
        concept_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        获取特定学习者对指定知识点的掌握情况

        Args:
            uid: 学习者 UID
            concept_ids: 知识点 ID 列表；None 表示返回所有知识点

        Returns:
            {concept_id: mastery 或 None}
        """
        try:
            kt_data = self.get_kt_result_by_uid(uid)
            if not kt_data:
                return {}

            kt_results = kt_data["KT"] or {}

            if concept_ids:
                result: Dict[str, Any] = {}
                for cid in concept_ids:
                    result[cid] = kt_results.get(cid, None)
                return result

            return kt_results
        except Exception as exc:
            logger.error("获取学习者 %s 知识点掌握情况失败: %s", uid, exc)
            return {}

    # ------------------------------------------------------------------
    # 按掌握度区间筛选学习者
    # ------------------------------------------------------------------

    def get_learners_by_mastery_level(
        self,
        concept_id: str,
        min_mastery: float = 0.0,
        max_mastery: float = 1.0,
        limit: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        根据知识点掌握水平获取学习者
        """
        try:
            query = {
                f"KT.{concept_id}": {
                    "$gte": min_mastery,
                    "$lte": max_mastery,
                }
            }
            documents = self._mongo.find_by_fields(self.collection_name, query)

            if limit > 0:
                documents = documents[:limit]

            results: List[Dict[str, Any]] = []
            for doc in documents:
                mastery_level = doc.get("KT", {}).get(concept_id, 0.0)
                results.append(
                    {
                        "uid": doc.get("uid"),
                        "mastery_level": mastery_level,
                        "updated_time": doc.get("updated_time"),
                    }
                )

            logger.info(
                "成功获取 %d 个在掌握水平 [%f, %f] 内的学习者",
                len(results),
                min_mastery,
                max_mastery,
            )
            return results
        except Exception as exc:
            logger.error("根据掌握水平获取学习者失败: %s", exc)
            return []

    # ------------------------------------------------------------------
    # 存在性检查 & 最近更新
    # ------------------------------------------------------------------

    def check_learner_exists(self, uid: str) -> bool:
        """
        检查指定 UID 的学习者是否存在
        """
        try:
            count = self._mongo.count_documents(self.collection_name, {"uid": uid})
            exists = count > 0
            logger.debug("检查UID %s 学习者存在性: %s", uid, exists)
            return exists
        except Exception as exc:
            logger.error("检查学习者存在性失败: %s", exc)
            return False

    def get_recent_updated_learners(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近更新的学习者（按 updated_time 降序）
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
                kt_count = len(doc.get("KT", {}))
                results.append(
                    {
                        "uid": doc.get("uid"),
                        "kt_concept_count": kt_count,
                        "updated_time": doc.get("updated_time"),
                    }
                )

            logger.info("成功获取 %d 个最近更新的学习者", len(results))
            return results
        except Exception as exc:
            logger.error("获取最近更新的学习者失败: %s", exc)
            return []
