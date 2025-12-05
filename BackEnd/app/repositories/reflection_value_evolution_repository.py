# app/repositories/reflection_value_evolution_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ReflectionValueEvolutionRepository(BaseRepository):
    """
    反思深度与价值观演变（Reflective Depth & Value Evolution）数据仓库

    负责从 MongoDB.MLS.Interaction 中读取与本维度分析相关的事件：
    - reflected-on-activity    （提交反思文本）
    - explored-extension       （拓展活动探索）

    仓库层只做“数据准备”，不做任何文本分析或指数计算。
    """

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "reflected_on_activity": VERB_BASE + "reflected-on-activity",
        "explored_extension": VERB_BASE + "explored-extension",
    }

    XAPI_COLLECTION = "Interaction"

    def __init__(self) -> None:
        super().__init__()

    def get_reflection_events(
        self,
        learner_uids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        一次性获取若干学习者在 xAPI.Interaction 中与“反思深度与价值观演变”
        相关的所有事件。

        返回原始 xAPI 事件文档列表，通常包含：
        - _lrn_uid
        - _course_uid
        - verb.id
        - timestamp
        - result（其中 result.response 为反思文本）
        - context.extensions（可能包含反思格式等信息）
        """
        if not learner_uids:
            return []

        verb_ids = [
            self.VERBS["reflected_on_activity"],
            self.VERBS["explored_extension"],
        ]

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": {"$in": verb_ids},
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                "[ReflectionValueEvolutionRepository] 读取反思/拓展相关事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取反思/拓展相关事件失败: {e}")
            return []


# 全局仓库实例（与其它 Repository 保持一致）
reflection_value_evolution_repository = ReflectionValueEvolutionRepository()
