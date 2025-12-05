# app/repositories/contribution_reputation_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ContributionReputationRepository(BaseRepository):
    """
    价值贡献 & 声望（Metaverse Value Contribution & Reputation）数据仓库

    负责从 MongoDB.MLS.Interaction 中读取与价值贡献分析相关的事件：
    - exchanged-value
    - contributed-resource
    - co-edited-artifact
    - collaborated-on-activity

    注意：
    - 与离线分析脚本 analyze_contribution_reputation.py 保持 verb / collection 命名一致；
    - 仅负责“数据准备”，不做任何复杂计算，把原始 xAPI 事件交给 Engine 处理。
    """

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        # 价值交换行为：对应 LEARNER-C 中“价值 token 流水”
        "exchanged_value": VERB_BASE + "exchanged-value",
        # 贡献行为：资源贡献 / 协作编辑 / 协同活动
        "contributed_resource": VERB_BASE + "contributed-resource",
        "co_edited_artifact": VERB_BASE + "co-edited-artifact",
        "collaborated_on_activity": VERB_BASE + "collaborated-on-activity",
    }

    # 与分析脚本保持一致：MLS.Interaction
    XAPI_COLLECTION = "Interaction"

    def __init__(self):
        super().__init__()

    # ----------- 基础查询方法 -----------

    def get_value_and_contribution_events(
        self, learner_uids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        获取指定学习者在 xAPI.Interaction 中的价值交换 + 贡献相关事件。

        返回原始 xAPI 事件文档列表，包含至少：
        - _lrn_uid
        - _course_uid
        - verb.id
        - context.extensions（读取 value-change 等扩展字段）
        - result（备用）
        """
        if not learner_uids:
            return []

        verb_list = [
            self.VERBS["exchanged_value"],
            self.VERBS["contributed_resource"],
            self.VERBS["co_edited_artifact"],
            self.VERBS["collaborated_on_activity"],
        ]

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": {"$in": verb_list},
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                "[ContributionReputationRepository] 读取价值贡献相关事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取价值贡献相关事件失败: {e}")
            return []


# 全局仓库实例（与 attention_allocation_repository 同风格）
contribution_reputation_repository = ContributionReputationRepository()
