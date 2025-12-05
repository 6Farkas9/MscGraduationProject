# app/repositories/attention_allocation_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class AttentionAllocationRepository(BaseRepository):
    """
    注意力分配 & 信息加工方式的数据仓库

    负责从 MongoDB.MLS.Interaction 中读取与注意力分析相关的事件：
    - focused-on-resource
    - observed-peer
    - answered / passed / completed
    """

    # 与分析脚本保持一致的 verb 定义
    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "focused_on_resource": VERB_BASE + "focused-on-resource",
        "observed_peer": VERB_BASE + "observed-peer",
        "answered": VERB_BASE + "answered",
        "passed": VERB_BASE + "passed",
        "completed": VERB_BASE + "completed",
    }

    # 注意，这里集合名与分析脚本保持一致：MLS.Interaction
    # 在当前工程中，只需要传 collection_name 给 mongodb_operator，
    # 具体数据库 / 连接由 db_manager & mongodb_operator 负责。
    XAPI_COLLECTION = "Interaction"

    def __init__(self):
        super().__init__()

    # ----------- 基础查询方法 -----------

    def get_focus_events(self, learner_uids: List[str]) -> List[Dict[str, Any]]:
        """
        获取学习者的 focused-on-resource 事件

        返回原始 xAPI 事件文档列表，包含：
        - _lrn_uid
        - _course_uid
        - result.duration
        - context.extensions
        - object.id（如果有）
        - timestamp
        """
        if not learner_uids:
            return []

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": self.VERBS["focused_on_resource"],
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                f"[AttentionAllocationRepository] 读取 focused-on-resource 事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取 focused-on-resource 事件失败: {e}")
            return []

    def get_observed_peer_events(self, learner_uids: List[str]) -> List[Dict[str, Any]]:
        """
        获取学习者的 observed-peer 事件（视作 example 类注意）
        """
        if not learner_uids:
            return []

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": self.VERBS["observed_peer"],
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                f"[AttentionAllocationRepository] 读取 observed-peer 事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取 observed-peer 事件失败: {e}")
            return []

    def get_performance_events(self, learner_uids: List[str]) -> List[Dict[str, Any]]:
        """
        获取学习者的表现相关事件：answered / passed / completed
        """
        if not learner_uids:
            return []

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": {
                "$in": [
                    self.VERBS["answered"],
                    self.VERBS["passed"],
                    self.VERBS["completed"],
                ]
            },
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                f"[AttentionAllocationRepository] 读取表现相关事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取表现相关事件失败: {e}")
            return []

    # ----------- 汇总接口 -----------

    def get_attention_raw_data_for_learners(
        self, learner_uids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        为多个学习者一次性获取注意力分析所需的所有原始事件数据。

        返回结构：
        {
            "focus_events": [...],
            "observed_events": [...],
            "performance_events": [...],
        }
        """
        focus_events = self.get_focus_events(learner_uids)
        observed_events = self.get_observed_peer_events(learner_uids)
        performance_events = self.get_performance_events(learner_uids)

        return {
            "focus_events": focus_events,
            "observed_events": observed_events,
            "performance_events": performance_events,
        }


# 全局仓库实例（与 hgc_repository 同风格）
attention_allocation_repository = AttentionAllocationRepository()
