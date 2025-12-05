# app/repositories/spatial_exploration_orientation_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class SpatialExplorationOrientationRepository(BaseRepository):
    """
    空间与资源探索倾向（Spatial & Resource Exploration Orientation）数据仓库

    负责从 MongoDB.MLS.Interaction 中读取与本维度相关的事件：
    - navigated-to-space / teleported-to-space：空间导航事件
    - explored-extension：可选拓展单元 / 支线任务事件
    - focused-on-resource：资源聚焦事件
    """

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        # 空间导航事件：对应 OmiLAXR 框架中“进入场景/子场景”的基础记录
        "navigated_to_space": VERB_BASE + "navigated-to-space",
        "teleported_to_space": VERB_BASE + "teleported-to-space",
        # 可选拓展单元事件：Gamified LA 中“支线任务 / 可选挑战”参与行为
        "explored_extension": VERB_BASE + "explored-extension",
        # 资源聚焦事件：资源类别覆盖与停留区域
        "focused_on_resource": VERB_BASE + "focused-on-resource",
    }

    # 与分析脚本保持一致：MLS.Interaction
    XAPI_COLLECTION = "Interaction"

    def __init__(self):
        super().__init__()

    # ----------------- 基础查询方法 -----------------

    def get_navigation_events(self, learner_uids: List[str]) -> List[Dict[str, Any]]:
        """
        获取学习者的空间导航事件：
        - navigated-to-space
        - teleported-to-space

        返回原始 xAPI 事件文档列表，包含：
        - _lrn_uid
        - _course_uid
        - context.extensions（含 space-id, navigation-mode）
        - timestamp
        """
        if not learner_uids:
            return []

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": {
                "$in": [
                    self.VERBS["navigated_to_space"],
                    self.VERBS["teleported_to_space"],
                ]
            },
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                f"[SpatialExplorationOrientationRepository] 读取导航事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取导航事件失败: {e}")
            return []

    def get_extension_events(self, learner_uids: List[str]) -> List[Dict[str, Any]]:
        """
        获取学习者的 explored-extension 事件（可选拓展单元 / 支线任务）
        """
        if not learner_uids:
            return []

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": self.VERBS["explored_extension"],
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                f"[SpatialExplorationOrientationRepository] 读取 explored-extension 事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取 explored-extension 事件失败: {e}")
            return []

    def get_focus_events(self, learner_uids: List[str]) -> List[Dict[str, Any]]:
        """
        获取学习者的 focused-on-resource 事件（资源聚焦，用于资源类别覆盖度）
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
                f"[SpatialExplorationOrientationRepository] 读取 focused-on-resource 事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取 focused-on-resource 事件失败: {e}")
            return []

    # ----------------- 汇总接口 -----------------

    def get_spatial_exploration_raw_data_for_learners(
        self, learner_uids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        为多个学习者一次性获取空间与资源探索分析所需的所有原始事件数据。

        返回结构：
        {
            "navigation_events": [...],   # navigated-to-space / teleported-to-space
            "extension_events": [...],    # explored-extension
            "focus_events": [...],        # focused-on-resource
        }
        """
        navigation_events = self.get_navigation_events(learner_uids)
        extension_events = self.get_extension_events(learner_uids)
        focus_events = self.get_focus_events(learner_uids)

        return {
            "navigation_events": navigation_events,
            "extension_events": extension_events,
            "focus_events": focus_events,
        }


# 全局仓库实例（与 attention_allocation_repository 同风格）
spatial_exploration_orientation_repository = SpatialExplorationOrientationRepository()
