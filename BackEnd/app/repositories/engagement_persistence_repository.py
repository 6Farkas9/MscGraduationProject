# app/repositories/engagement_persistence_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class EngagementPersistenceRepository(BaseRepository):
    """
    行为投入度 & 坚持性（Behavioral Engagement & Persistence）数据仓库

    负责从 MongoDB.MLS.Interaction 中读取与本维度分析相关的事件：
    - initialized
    - completed
    - answered
    - performed-procedure-step
    - explored-extension
    - remained-idle
    - exchanged-value

    注意：
    - 与离线分析脚本 analyze_engagement_persistence.py 保持 verb / collection 命名一致；
    - 仅负责“数据准备”，不做任何复杂计算，把原始 xAPI 事件交给 Engine 处理。
    """

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "initialized": VERB_BASE + "initialized",
        "completed": VERB_BASE + "completed",
        "answered": VERB_BASE + "answered",
        "performed_procedure_step": VERB_BASE + "performed-procedure-step",
        "explored_extension": VERB_BASE + "explored-extension",
        "remained_idle": VERB_BASE + "remained-idle",
        "exchanged_value": VERB_BASE + "exchanged-value",
    }

    # 与分析脚本保持一致：MLS.Interaction
    XAPI_COLLECTION = "Interaction"

    def __init__(self) -> None:
        super().__init__()

    def get_engagement_persistence_events(
        self, learner_uids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        获取指定学习者在 xAPI.Interaction 中的行为投入度与坚持性相关事件。

        返回原始 xAPI 事件文档列表，包含至少：
        - _lrn_uid
        - _course_uid
        - _unt_uid
        - _type
        - verb.id
        - result
        - context.extensions
        - object.id
        - timestamp
        """
        if not learner_uids:
            return []

        verb_ids = [
            self.VERBS["initialized"],
            self.VERBS["completed"],
            self.VERBS["answered"],
            self.VERBS["performed_procedure_step"],
            self.VERBS["explored_extension"],
            self.VERBS["remained_idle"],
            self.VERBS["exchanged_value"],
        ]

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": {"$in": verb_ids},
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                "[EngagementPersistenceRepository] 读取行为投入度/坚持性相关事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取行为投入度/坚持性相关事件失败: {e}")
            return []


# 全局仓库实例（与 attention_allocation_repository 同风格）
engagement_persistence_repository = EngagementPersistenceRepository()
