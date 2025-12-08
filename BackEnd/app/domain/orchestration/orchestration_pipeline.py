# orchestration_pipeline.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, List, Optional
import logging

from app.domain.orchestration.resource_planning_engine import ResourcePlanningEngine
from app.domain.orchestration.resource_orchestration_engine import ResourceOrchestrationEngine
from app.domain.orchestration.learning_path_engine import LearningPathEngine

logger = logging.getLogger(__name__)


class OrchestrationPipeline:
    """
    OrchestrationPipeline

    统一编排三个 Engine：
      1. ResourcePlanningEngine
      2. ResourceOrchestrationEngine
      3. LearningPathEngine

    对外只暴露一个 analyze 接口，隐藏中间交互细节。

    输入：
    ----
    analyze(
        learner_uids: List[str],
        kt: Dict[str, Dict[str, float]],
        profile: Dict[str, Dict[str, Any]],
    )

    约定：
    - learner_uids: 需要编排的学习者 UID 列表
    - kt: { learner_uid: { concept_uid: prob, ... }, ... }
    - profile: { learner_uid: { 画像维度: {子标签: 文本描述}, ... }, ... }

    输出：
    ----
    {
      "engine_status": {
        "resource_planning": {...},
        "resource_orchestration": {...},
        "learning_path": {...}
      },
      "results": {
        "<learner_uid>": {
          "planning": { ... 来自 ResourcePlanningEngine 的结果 ... },
          "orchestration": { ... 来自 ResourceOrchestrationEngine 的结果 ... },
          "learning_path": "<LearningPathEngine 生成的 Markdown 文本>"
        },
        ...
      }
    }
    """

    def __init__(
        self,
        planning_engine: Optional[ResourcePlanningEngine] = None,
        orchestration_engine: Optional[ResourceOrchestrationEngine] = None,
        learning_path_engine: Optional[LearningPathEngine] = None,
    ) -> None:
        self.planning_engine = planning_engine or ResourcePlanningEngine()
        self.orchestration_engine = orchestration_engine or ResourceOrchestrationEngine()
        self.learning_path_engine = learning_path_engine or LearningPathEngine()

    # ------------------------------------------------------------------
    # Pipeline 总入口
    # ------------------------------------------------------------------
    def analyze(
        self,
        learner_uids: List[str],
        kt: Dict[str, Dict[str, float]],
        profile: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        综合调用三个 Engine，输出最终结果。

        :param learner_uids: 目标学习者 uid 列表
        :param kt: { uid: { concept_uid: prob, ... } }
        :param profile: { uid: 画像 dict }
        """
        # ================== 1. 调用 ResourcePlanningEngine ==================
        planning_input_data: Dict[str, Any] = {}
        for uid in learner_uids:
            planning_input_data[uid] = {
                "KT": kt.get(uid, {}) or {},
                "Profile": profile.get(uid, {}) or {},
            }

        logger.info("[OrchestrationPipeline] Start ResourcePlanningEngine.analyze ...")
        planning_output = self.planning_engine.analyze(
            learner_uids=learner_uids,
            data=planning_input_data,
        )

        planning_status = planning_output.get("engine_status", {})
        planning_results: Dict[str, Any] = planning_output.get("results", {})

        # ================== 2. 调用 ResourceOrchestrationEngine ==================
        orchestration_input_data: Dict[str, Any] = {}
        for uid in learner_uids:
            plan_for_uid = planning_results.get(uid, {}) or {}
            # ResourceOrchestrationEngine 里会优先寻找 user_data["plan"]，若没有则直接把整个 user_data 当成 plan
            orchestration_input_data[uid] = {"plan": plan_for_uid}

        logger.info("[OrchestrationPipeline] Start ResourceOrchestrationEngine.analyze ...")
        orchestration_output = self.orchestration_engine.analyze(
            learner_uids=learner_uids,
            data=orchestration_input_data,
        )

        orchestration_status = orchestration_output.get("engine_status", {})
        orchestration_results: Dict[str, Any] = orchestration_output.get("results", {})

        # ================== 3. 调用 LearningPathEngine ==================
        learning_path_input_data: Dict[str, Any] = {}
        for uid in learner_uids:
            learning_path_input_data[uid] = {
                "resource_planning": planning_results.get(uid, {}) or {},
                "resource_orchestration": orchestration_results.get(uid, {}) or {},
            }

        logger.info("[OrchestrationPipeline] Start LearningPathEngine.analyze ...")
        learning_path_output = self.learning_path_engine.analyze(
            learner_uids=learner_uids,
            data=learning_path_input_data,
        )

        learning_path_status = learning_path_output.get("engine_status", {})
        learning_path_results: Dict[str, Any] = learning_path_output.get("results", {})

        # ================== 4. 汇总最终结果 ==================
        combined_results: Dict[str, Any] = {}
        for uid in learner_uids:
            combined_results[uid] = {
                "planning": planning_results.get(uid),
                "orchestration": orchestration_results.get(uid),
                "learning_path": (
                    (learning_path_results.get(uid) or {}).get("learning_path_text")
                ),
            }

        return {
            "engine_status": {
                "resource_planning": planning_status,
                "resource_orchestration": orchestration_status,
                "learning_path": learning_path_status,
            },
            "results": combined_results,
        }
