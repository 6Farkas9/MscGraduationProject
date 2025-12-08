# app/data_access/partner/learner_profile_repository.py
# -*- coding: utf-8 -*-
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository
from app.core.settings import partner_settings

logger = logging.getLogger(__name__)


class LearnerProfileRepository(MongoDBBaseRepository):
    """
    通用 LearnerProfile 数据仓库（供学习伙伴 & 学习榜样 Engine 共同使用）

    职责说明
    --------
    1. 从 MongoDB 读取 LearnerProfile 文档；
    2. 提供特征视图（knowledge_vector / profile_categorical）；
    3. 可构建候选池（支持排除 UID、按更新时间过滤、限制规模）；
    4. 不负责任何匹配策略，仅做数据准备。
    """

    COLLECTION_NAME: str = partner_settings.learner_profile_collection

    # ------------------- 基础读取 -------------------

    def get_learner_profile_by_uid(self, learner_uid: str) -> Optional[Dict[str, Any]]:
        """
        根据单个 UID 获取最新的 LearnerProfile 文档。
        """
        try:
            docs = self.get_documents(
                self.COLLECTION_NAME,
                {"uid": learner_uid},
            )
            if not docs:
                logger.warning("LearnerProfile not found for uid=%s", learner_uid)
                return None
            docs = sorted(docs, key=lambda d: d.get("updated_time") or datetime.min, reverse=True)
            return docs[0]
        except Exception as exc:
            logger.error("get_learner_profile_by_uid failed: uid=%s, err=%s", learner_uid, exc)
            return None

    def get_learner_profiles_by_uids(self, learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取多个 UID 对应的 LearnerProfile 文档，返回 uid -> 文档 的映射。
        """
        if not learner_uids:
            return {}

        try:
            docs = self.get_documents(
                self.COLLECTION_NAME,
                {"uid": {"$in": learner_uids}},
            )
            result: Dict[str, Dict[str, Any]] = {}
            for doc in docs:
                uid = doc.get("uid")
                if not uid:
                    continue
                prev = result.get(uid)
                if (not prev) or (doc.get("updated_time") or datetime.min) > (
                    prev.get("updated_time") or datetime.min
                ):
                    result[uid] = doc
            return result
        except Exception as exc:
            logger.error("get_learner_profiles_by_uids failed: uids=%s, err=%s", learner_uids, exc)
            return {}

    # ------------------- 候选池 -------------------

    def get_candidate_pool(
        self,
        exclude_uids: Optional[List[str]] = None,
        min_updated_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取候选学习者池的 LearnerProfile 文档列表。

        参数
        ----
        exclude_uids:
            需要排除的 UID 列表（通常包含目标学习者自身等）。
        min_updated_time:
            仅保留最近更新过画像/KT 的学习者。
        limit:
            候选池最大规模；若为 None 或非正数，则退回到 PartnerSettings.partner_candidate_pool_limit。
        """
        query: Dict[str, Any] = {}
        if exclude_uids:
            query["uid"] = {"$nin": exclude_uids}
        if min_updated_time:
            query["updated_time"] = {"$gte": min_updated_time}

        if limit is None or limit <= 0:
            limit = partner_settings.partner_candidate_pool_limit

        try:
            docs = self.get_documents(self.COLLECTION_NAME, query)
            docs = sorted(
                docs,
                key=lambda d: d.get("updated_time") or datetime.min,
                reverse=True,
            )
            if limit is not None and limit > 0:
                docs = docs[:limit]
            return docs
        except Exception as exc:
            logger.error("get_candidate_pool failed: query=%s, err=%s", query, exc)
            return []

    # ------------------- 特征提取 -------------------

    @staticmethod
    def extract_knowledge_vector(learner_doc: Dict[str, Any]) -> Dict[str, float]:
        """
        从 LearnerProfile 文档中抽取知识点预测精度向量 KT。
        """
        kt = learner_doc.get("KT") or {}
        vector: Dict[str, float] = {}
        for kp, val in kt.items():
            try:
                vector[str(kp)] = float(val)
            except Exception:
                # 防御性：跳过异常值
                continue
        return vector

    @staticmethod
    def extract_profile_categorical_features(
        learner_doc: Dict[str, Any]
    ) -> Dict[Tuple[str, str], str]:
        """
        从 LearnerProfile 文档中抽取画像的「(维度, 子键) -> 类别值」视图。
        """
        profiles = learner_doc.get("profiles") or {}
        features: Dict[Tuple[str, str], str] = {}

        for dim_key, dim_content in profiles.items():
            if not isinstance(dim_content, dict):
                continue
            for sub_key, sub_val in dim_content.items():
                features[(str(dim_key), str(sub_key))] = str(sub_val)

        return features

    def build_feature_views(
        self, learner_docs: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        为一批学习者构建特征视图（仅数据准备，不涉及任何匹配策略）。

        返回
        ----
        Dict[str, Dict[str, Any]]:
            uid -> {
                "uid": str,
                "knowledge_vector": Dict[str, float],
                "profile_categorical": Dict[(dim, sub_key), str],
                "raw": Dict[str, Any],  # 原始文档
            }
        """
        result: Dict[str, Dict[str, Any]] = {}
        for doc in learner_docs:
            uid = doc.get("uid")
            if not uid:
                continue

            kv = self.extract_knowledge_vector(doc)
            pf = self.extract_profile_categorical_features(doc)

            result[uid] = {
                "uid": uid,
                "knowledge_vector": kv,
                "profile_categorical": pf,
                "raw": doc,
            }

        return result
