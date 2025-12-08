# BackEnd/app/domain/orchestration/orchestration_pipeline.py
# -*- coding: utf-8 -*-

"""
OrchestrationPipeline
=====================

这是“学习推荐大模块”的对上统一接口（向上层暴露 analyze）。

流程（固定三段）：
  1) 第一次调用 LLM（ResourcePlanningEngine）：根据学习者画像 + 知识点状态/先修关系 -> 生成 plan JSON
  2) 调用资源匹配/编排引擎（ResourceOrchestrationEngine）：plan -> matched_resources(top_k)
  3) 第二次调用 LLM（LearningPathEngine）：输入 learner_info/knowledge_state/first_plan/matched_resources -> 输出学习路径 JSON

该文件不负责：
  - 业务数据库的读取（由上层或测试脚本准备）
  - 资源库检索的实现（由 ResourceOrchestrationEngine 内部实现）

输出：
  - first_plan（第一次LLM输出）
  - matched_resources（资源匹配结果）
  - learning_path（第二次LLM输出，包含理由 why_this_step 等）
  - 汇总字段（方便上层直接拿去展示）
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

from app.domain.orchestration.resource_planning_engine import ResourcePlanningEngine
from app.domain.orchestration.learning_path_engine import LearningPathEngine
from app.domain.orchestration.resource_orchestration_engine import ResourceOrchestrationEngine

logger = logging.getLogger(__name__)


class OrchestrationPipeline:
    """
    大模块统一编排器（对上接口）。

    你可以在实例化时指定：
      - llm_provider：使用哪个 provider（aizex/openai/… 取决于你的 settings 配置）
      - device：如果 BaseEngine 要求 device（一般 cpu 即可）
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self._llm_provider = llm_provider
        self._device = device

        # 两次 LLM 的 engine
        self._planner = ResourcePlanningEngine(device=device, provider=llm_provider)
        self._path = LearningPathEngine(device=device, provider=llm_provider)

        # 中间资源匹配/编排 engine
        self._resource = ResourceOrchestrationEngine(device=device)

        # 初始化（失败则在 analyze 时抛/返回错误）
        self._planner.initialize()
        self._path.initialize()
        self._resource.initialize()

    # ---------------------------------------------------------------------
    # 对上层唯一入口
    # ---------------------------------------------------------------------
    def analyze(
        self,
        learner_uid: str,
        learner_profile: Dict[str, Any],
        knowledge_concepts: List[Dict[str, Any]],
        *,
        # 允许用户在功能使用前选择模型（分别控制两次调用）
        llm1_model: Optional[str] = None,
        llm2_model: Optional[str] = None,
        # 控制资源匹配数量
        top_k: int = 20,
        # 允许上层传一些额外约束（例如总时长/步数等），会透传给第二次LLM输入
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        参数
        ----
        learner_uid:
            学习者 uid，例如 "lrn_..."
        learner_profile:
            学习者画像信息（建议放“已经转成文本标签”的结果）
        knowledge_concepts:
            概念列表（必须包含 uid/name/predecessors/successors/status/(可选)predicted_accuracy 等）
            - status: learned / not_learned
            - predicted_accuracy: learned 可给 float；not_learned 可不给/None（你提出 -1 冗余）
        llm1_model / llm2_model:
            两次 LLM 使用的模型名称，可为空（为空时使用 provider 的 default_model）
        top_k:
            资源匹配返回多少条候选资源
        constraints:
            第二次规划可用的额外硬约束

        返回
        ----
        {
          "learner_uid": "...",
          "first_plan": {...},
          "matched_resources": [...],
          "learning_path": {...},
          "summary": {...}
        }
        """
        # 1) 第一次 LLM：生成 plan（必须严格字段）
        plan_input = {
            "learner_uid": learner_uid,
            "learner_profile": learner_profile,
            "knowledge_concepts": knowledge_concepts,
        }

        # 由于 ResourcePlanningEngine.analyze 接口是按 learner_uids 批处理，
        # 这里传 [learner_uid]，并在内部 build_payload 时取数据。
        #
        # 但你希望 pipeline 由上层传入 learner_profile/knowledge_concepts，
        # 所以我们绕过 engine 内部的 _build_planning_payload，直接调用其 _call_llm。
        # 为了不依赖 private 方法，这里采用“临时子类注入数据”的方式更稳：
        first_plan = self._call_planner_with_payload(plan_input, model=llm1_model)

        # 2) 中间资源匹配（你已经实现）
        matched_resources = self._resource.match_resources(first_plan, top_k=top_k)

        # 3) 第二次 LLM：学习路径规划
        path_input = {
            "learner_uid": learner_uid,
            "learner_profile": learner_profile,
            "knowledge_concepts": knowledge_concepts,
            "first_plan": first_plan,
            "matched_resources": matched_resources,
        }
        if constraints:
            path_input["constraints"] = constraints

        learning_path = self._call_path_with_payload(path_input, model=llm2_model)

        # 输出汇总（方便上层直接取“推荐 + 路径 + 理由”）
        summary = self._build_summary(first_plan, matched_resources, learning_path)

        return {
            "learner_uid": learner_uid,
            "first_plan": first_plan,
            "matched_resources": matched_resources,
            "learning_path": learning_path,
            "summary": summary,
        }

    # ---------------------------------------------------------------------
    # 内部：以“payload直传”的方式调用两次 LLM
    # （避免依赖 engine 内部数据库查询逻辑）
    # ---------------------------------------------------------------------
    def _call_planner_with_payload(self, payload: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
        """
        把上层准备好的 payload 直接送给第一次 LLM engine。
        """
        # 轻量复制，避免 engine 内部改写上层对象
        safe_payload = copy.deepcopy(payload)

        # 直接复用 engine 的 LLM 调用与 JSON 解析能力
        # 这里使用它的 protected 方法约定（属于同工程内部可接受的耦合）
        try:
            return self._planner._call_llm(safe_payload, model=model)  # type: ignore[attr-defined]
        except Exception as e:
            logger.exception("第一次LLM规划失败")
            raise RuntimeError(f"First LLM planning failed: {e}")

    def _call_path_with_payload(self, payload: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
        """
        把上层准备好的 payload 直接送给第二次 LLM engine。
        """
        safe_payload = copy.deepcopy(payload)
        try:
            return self._path._call_llm(safe_payload, model=model)  # type: ignore[attr-defined]
        except Exception as e:
            logger.exception("第二次LLM路径规划失败")
            raise RuntimeError(f"Second LLM path failed: {e}")

    # ---------------------------------------------------------------------
    # 内部：组装 summary（不影响你原始输出结构）
    # ---------------------------------------------------------------------
    @staticmethod
    def _build_summary(
        first_plan: Dict[str, Any],
        matched_resources: List[Dict[str, Any]],
        learning_path: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        给上层一个“可直接展示”的轻量汇总：
        - 推荐目标知识点
        - 推荐资源（前 N 条）
        - 路径步骤（每步的资源 uid + why_this_step）
        """
        target = first_plan.get("target_concepts", [])
        top_resources = matched_resources[:10] if matched_resources else []

        steps = []
        for s in learning_path.get("steps", []) or []:
            steps.append(
                {
                    "step_index": s.get("step_index"),
                    "goal": s.get("goal"),
                    "target_concepts": s.get("target_concepts", []),
                    "resource_uids": s.get("resource_uids", []),
                    "why": s.get("why_this_step"),
                    "time_estimate": s.get("time_estimate"),
                }
            )

        return {
            "target_concepts": target,
            "recommended_resources_top10": top_resources,
            "learning_steps": steps,
            "path_overview": learning_path.get("path_overview", {}),
        }
