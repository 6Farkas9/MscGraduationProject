# app/repositories/collaborative_role_contribution_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class CollaborativeRoleContributionRepository(BaseRepository):
    """
    协作角色与贡献类型维度的数据仓库

    负责从 MongoDB.MLS.Interaction 中读取协作相关 xAPI 事件：
    - collaborated-on-activity
    - co-edited-artifact
    - contributed-resource
    - responded / referred / followed / managed-resource / took-turn
    """

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "collaborated_on_activity": VERB_BASE + "collaborated-on-activity",
        "co_edited_artifact": VERB_BASE + "co-edited-artifact",
        "contributed_resource": VERB_BASE + "contributed-resource",

        # 社会互动动词（来自画像设计）:
        "responded": VERB_BASE + "responded",
        "referred": VERB_BASE + "referred",
        "followed": VERB_BASE + "followed",
        "managed_resource": VERB_BASE + "managed-resource",
        "took_turn": VERB_BASE + "took-turn",
    }

    # 与原分析脚本一致：MLS.Interaction
    XAPI_COLLECTION = "Interaction"

    def __init__(self):
        super().__init__()

    def get_collaboration_events_for_learners(
        self, learner_uids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        获取多个学习者的协作相关 xAPI 事件。

        事件字段参考 analyze_collaborative_role_contribution.py：
        - _lrn_uid
        - _course_uid
        - verb.id
        - result.duration / result.extensions.edit-type
        - context.extensions.sessionId / participants / collaborator-ids
        - timestamp
        """
        if not learner_uids:
            return []

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": {"$in": list(self.VERBS.values())},
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                f"[CollaborativeRoleContributionRepository] 读取协作相关事件："
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取协作事件失败: {e}")
            return []


# 全局仓库实例（与 hgc_repository / attention_allocation_repository 风格一致）
collaborative_role_contribution_repository = CollaborativeRoleContributionRepository()
