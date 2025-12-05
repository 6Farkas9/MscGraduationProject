# app/repositories/feedback_orientation_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class FeedbackOrientationRepository(BaseRepository):
    """
    反馈敏感度 & 数据使用能力（Feedback Orientation & Data Use Literacy）数据仓库

    负责从 MongoDB.MLS.Interaction 中读取与本维度分析相关的事件：
    - reviewed-feedback       （查看学习仪表盘 / 反馈面板）
    - requested-support       （查看解析 / 示例 / 提示等即时反馈）
    - answered                （题目作答）
    - completed               （任务完成）
    - performed-procedure-step（步骤级交互）

    仓库层只做“数据准备”，不做任何行为指标或指数计算。
    """

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "reviewed_feedback": VERB_BASE + "reviewed-feedback",
        "requested_support": VERB_BASE + "requested-support",
        "answered": VERB_BASE + "answered",
        "completed": VERB_BASE + "completed",
        "performed_procedure_step": VERB_BASE + "performed-procedure-step",
    }

    # 与离线分析脚本保持一致
    XAPI_COLLECTION = "Interaction"

    def __init__(self) -> None:
        super().__init__()

    def get_feedback_orientation_events(
        self,
        learner_uids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        一次性获取若干学习者在 xAPI.Interaction 中与“反馈敏感度与数据使用能力”
        相关的所有事件。

        返回原始 xAPI 事件文档列表，通常包含：
        - _lrn_uid
        - _course_uid
        - verb.id
        - timestamp
        - result
        - context.extensions
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
                "[FeedbackOrientationRepository] 读取反馈敏感度相关事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取反馈敏感度相关事件失败: {e}")
            return []


# 全局仓库实例（与其它 Repository 保持一致）
feedback_orientation_repository = FeedbackOrientationRepository()
