# app/repositories/interaction_style_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class InteractionStyleRepository(BaseRepository):
    """
    交互与操作熟练度 / 风格（Interaction & Operation Fluency / Style）数据仓库

    负责从 MongoDB.MLS.Interaction 中读取与交互风格相关的事件：
    - manipulated-object
    - performed-procedure-step
    - completed

    仅关注 VR/AR/强交互型单元（_type ∈ {"vr", "ar", "interact"}）。

    仓库层只负责“数据准备”，不做任何数值计算，将原始事件交给 Engine 处理。
    """

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "manipulated_object": VERB_BASE + "manipulated-object",
        "performed_procedure_step": VERB_BASE + "performed-procedure-step",
        "completed": VERB_BASE + "completed",
    }

    # 仅关注 VR/AR/交互型单元
    UNIT_TYPES_FOR_STYLE = {"vr", "ar", "interact"}

    # 与离线分析脚本保持一致：MLS.Interaction
    XAPI_COLLECTION = "Interaction"

    def __init__(self) -> None:
        super().__init__()

    def get_interaction_style_events(
        self, learner_uids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        一次性获取若干学习者在 xAPI.Interaction 中与“交互与操作熟练度 / 风格”
        相关的所有事件。

        返回原始 xAPI 事件文档列表，通常包含：
        - _lrn_uid
        - _course_uid
        - _type
        - verb.id
        - result
        """
        if not learner_uids:
            return []

        verb_ids = list(self.VERBS.values())

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": {"$in": verb_ids},
            "_type": {"$in": list(self.UNIT_TYPES_FOR_STYLE)},
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                "[InteractionStyleRepository] 读取交互风格相关事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取交互风格相关事件失败: {e}")
            return []


# 全局仓库实例
interaction_style_repository = InteractionStyleRepository()
