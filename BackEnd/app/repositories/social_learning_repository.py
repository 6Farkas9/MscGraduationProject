# app/repositories/social_learning_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class SocialLearningRepository(BaseRepository):
    """
    社会性学习与同伴取向（Social Learning & Peer Orientation）数据仓库

    负责从 MongoDB.MLS.Interaction 中读取与本维度分析相关的事件：
    - observed-peer
    - collaborated-on-activity

    仓库层只做“数据准备”，不做任何时长统计、标准化或指数计算。
    """

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "observed_peer": VERB_BASE + "observed-peer",
        "collaborated_on_activity": VERB_BASE + "collaborated-on-activity",
    }

    # 与离线分析脚本保持一致
    XAPI_COLLECTION = "Interaction"

    def __init__(self) -> None:
        super().__init__()

    def get_social_learning_events(
        self,
        learner_uids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        一次性获取若干学习者在 xAPI.Interaction 中与
        “社会性学习与同伴取向”相关的所有事件。

        返回原始 xAPI 事件文档列表，通常包含：
        - _lrn_uid
        - _course_uid
        - verb.id
        - result.duration
        - context.extensions（用于观摩对象 learner-id）
        """
        if not learner_uids:
            return []

        verb_ids = [
            self.VERBS["observed_peer"],
            self.VERBS["collaborated_on_activity"],
        ]

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": {"$in": verb_ids},
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                "[SocialLearningRepository] 读取社会性学习相关事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取社会性学习相关事件失败: {e}")
            return []


# 全局仓库实例（与其它 Repository 保持一致）
social_learning_repository = SocialLearningRepository()
