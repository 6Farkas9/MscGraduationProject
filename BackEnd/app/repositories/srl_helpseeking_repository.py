# app/repositories/srl_helpseeking_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class SrlHelpSeekingRepository(BaseRepository):
    """
    自我调节与求助策略（Self-Regulated Learning & Help-Seeking）数据仓库

    负责从 MongoDB.MLS.Interaction 中读取与本维度分析相关的事件：
    - answered
    - requested-support
    - reviewed-feedback
    - explored-extension
    - reflected-on-activity

    仓库层只做“数据准备”，不做任何指标或指数计算。
    """

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "answered": VERB_BASE + "answered",
        "requested_support": VERB_BASE + "requested-support",
        "reviewed_feedback": VERB_BASE + "reviewed-feedback",
        "explored_extension": VERB_BASE + "explored-extension",
        "reflected_on_activity": VERB_BASE + "reflected-on-activity",
    }

    XAPI_COLLECTION = "Interaction"

    def __init__(self) -> None:
        super().__init__()

    def get_srl_helpseeking_events(
        self,
        learner_uids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        一次性获取若干学习者在 xAPI.Interaction 中与
        “自我调节与求助策略”相关的所有事件。

        返回原始 xAPI 事件文档列表，通常包含：
        - _lrn_uid
        - _course_uid
        - verb.id
        - object
        - result
        - timestamp
        """
        if not learner_uids:
            return []

        verb_ids = list(self.VERBS.values())

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": {"$in": verb_ids},
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                "[SrlHelpSeekingRepository] 读取自我调节/求助策略相关事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取自我调节/求助策略相关事件失败: {e}")
            return []


# 全局仓库实例（与其它 Repository 保持一致）
srl_helpseeking_repository = SrlHelpSeekingRepository()
