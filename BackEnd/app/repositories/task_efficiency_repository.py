# app/repositories/task_efficiency_repository.py
import logging
from typing import List, Dict, Any

from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class TaskEfficiencyRepository(BaseRepository):
    """
    任务效率（Task Efficiency）数据仓库

    负责从 MongoDB.MLS.Interaction 中读取与任务效率相关的事件：
    - completed
    - performed-procedure-step
    """

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "completed": VERB_BASE + "completed",
        "performed_procedure_step": VERB_BASE + "performed-procedure-step",
    }

    # 与分析脚本保持一致：MLS.Interaction
    XAPI_COLLECTION = "Interaction"

    def __init__(self):
        super().__init__()

    # ----------------- 基础查询方法 -----------------

    def get_task_events(self, learner_uids: List[str]) -> List[Dict[str, Any]]:
        """
        获取学习者的任务级事件（completed / performed-procedure-step）

        返回原始 xAPI 事件文档列表，包含至少：
        - _lrn_uid
        - _course_uid
        - verb.id
        - result.duration
        - result.success / result.completion
        """
        if not learner_uids:
            return []

        query = {
            "_lrn_uid": {"$in": learner_uids},
            "verb.id": {
                "$in": [
                    self.VERBS["completed"],
                    self.VERBS["performed_procedure_step"],
                ]
            },
        }

        try:
            events = self.get_mongodb_documents(self.XAPI_COLLECTION, query)
            logger.info(
                f"[TaskEfficiencyRepository] 读取任务级事件: "
                f"learners={len(learner_uids)}, events={len(events)}"
            )
            return events
        except Exception as e:
            logger.error(f"获取任务级事件失败: {e}")
            return []

    # ----------------- 汇总接口 -----------------

    def get_task_efficiency_raw_data_for_learners(
        self, learner_uids: List[str]
    ) -> Dict[str, Any]:
        """
        一次性为多个学习者获取任务效率分析所需的全部原始数据。

        返回结构：
        {
            "task_events": [...],  # completed / performed-procedure-step 事件
        }
        """
        task_events = self.get_task_events(learner_uids)

        return {
            "task_events": task_events,
        }


# 全局仓库实例（与 attention_allocation_repository 同风格）
task_efficiency_repository = TaskEfficiencyRepository()
