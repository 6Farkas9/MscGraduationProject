# learning_path_engine.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, List, Optional

import json
import logging
import requests

from app.domain.common.base_engine import BaseEngine
from app.core.settings import llm_settings

logger = logging.getLogger(__name__)


class LearningPathEngine(BaseEngine):
    """
    LearningPathEngine

    业务链路的最后一环：综合
      - resource_planning_engine 的输出（整体资源规划建议）
      - resource_orchestration_engine 的输出（已匹配好的具体资源列表）

    通过大模型为每个学习者生成：
      - 学习路线（按步骤/阶段组织）
      - 路线设计理由（面向学习者可读的解释）

    特点：
    ----
    - 不访问数据库；
    - 输出不追求严格结构化，仅要求可读、逻辑清晰；
    - 结果只用于前端展示，不会再被下游解析。

    输入约定（推荐，但不强制）：
    --------------------------
    analyze(learner_uids, data) 中的 data 建议形如：

    data = {
        "<learner_uid>": {
            # 对应 resource_planning_engine.analyze(...)[\"results\"][uid]
            "resource_planning": {
                "concept_priority": [...],
                "type_priority": [...],
                "overall_strategy": "...",
                "resource_constraints": {
                    "difficulty_level": "...",
                    "structure_level": "...",
                    "guidance_level": "...",
                    "interaction_level": "...",
                    "collaboration_mode": "...",
                    "pedagogical_function": "..."
                }
            },

            # 对应 resource_orchestration_engine.analyze(...)[\"results\"][uid]
            "resource_orchestration": {
                "learner_uid": "...",
                "top_k": 10,
                "candidate_count": 30,
                "used_relaxation_level": 1,
                "resources": [
                    {
                        "uid": "...",
                        "oid": "...",
                        "type": "video",
                        "concepts": ["...", "..."],
                        "score": 0.93,
                        "difficulty_level": "medium",
                        "structure_level": "medium",
                        "guidance_level": "medium",
                        "interaction_level": "none",
                        "collaboration_mode": "group",
                        "pedagogical_function": "concept_introduction",
                        "time_estimate": 3
                    },
                    ...
                ]
            }
        },
        ...
    }

    输出结构：
    --------
    {
      "engine_status": {...},
      "results": {
        "<learner_uid>": {
          "learning_path_text": "<一段适合展示给学习者的 Markdown 文本，包含路线和理由>"
        },
        ...
      }
    }
    """

    def __init__(self, device: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(device, name)
        self.provider = None
        self.model = None

    # ------------------------------------------------------------------
    # 初始化：选择大模型 provider + model（复用 LLMSettings）
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        try:
            self.provider = llm_settings.get_provider()
            self.model = llm_settings.resolve_model()
            return True
        except Exception as e:
            logger.error("[LearningPathEngine] initialize failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # 内部：调用大模型（OpenAI 兼容接口）
    # ------------------------------------------------------------------
    def _call_llm(self, prompt: str) -> str:
        url = f"{self.provider.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.provider.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": llm_settings.temperature,
            "max_tokens": llm_settings.max_tokens,
        }

        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            proxies=llm_settings.proxies,
            timeout=llm_settings.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # 内部：构建给大模型的 prompt
    # ------------------------------------------------------------------
    def _build_prompt_for_learner(
        self,
        learner_uid: str,
        planning: Dict[str, Any],
        orchestration: Dict[str, Any],
        max_resources_for_prompt: int = 10,
    ) -> str:
        """
        将资源规划结果 + 资源编排结果整合成自然语言上下文，交给大模型生成学习路径。
        """

        concept_priority = planning.get("concept_priority", [])
        type_priority = planning.get("type_priority", [])
        overall_strategy = planning.get("overall_strategy", "")
        resource_constraints = planning.get("resource_constraints", {}) or {}

        # 资源编排结果中，挑前 N 个资源描述给大模型
        resources = orchestration.get("resources", []) or []
        resources_for_prompt = resources[:max_resources_for_prompt]

        # 把资源信息组织成相对紧凑的文本描述，避免上下文过长
        resource_lines = []
        for idx, r in enumerate(resources_for_prompt, start=1):
            r_type = r.get("type")
            r_concepts = ", ".join(r.get("concepts", []) or [])
            r_score = r.get("score")
            r_diff = r.get("difficulty_level")
            r_ped = r.get("pedagogical_function")
            r_time = r.get("time_estimate")
            r_uid = r.get("uid")

            line = (
                f"{idx}. 资源UID={r_uid} / 类型={r_type} / 知识点=[{r_concepts}] / "
                f"教学功能={r_ped} / 难度={r_diff} / 预计时间={r_time}分钟 / 匹配得分={r_score:.3f}"
            )
            resource_lines.append(line)

        resources_block = "\n".join(resource_lines) if resource_lines else "（当前没有匹配到资源）"

        concept_str = ", ".join(concept_priority) if concept_priority else "（无特别优先级）"
        type_str = ", ".join(type_priority) if type_priority else "（无特别偏好）"

        constraints_str = json.dumps(resource_constraints, ensure_ascii=False, indent=2)

        prompt = f"""
你是一个负责给学习者设计个性化学习路径的教学设计专家。

现在已经有两个前置模块的结果：

1）资源规划建议（来自 resource_planning_engine）：
   - 知识点优先级（越前面越重要、越薄弱）：{concept_str}
   - 资源基础类型偏好顺序：{type_str}
   - 整体策略摘要：{overall_strategy}
   - 资源约束（难度、结构、引导、交互与协作等偏好）：
{constraints_str}

2）资源匹配结果（来自 resource_orchestration_engine）：
   - 已经基于上面的规划，在资源库中为学习者 {learner_uid} 匹配出一批候选资源。
   - 以下是前 {len(resources_for_prompt)} 个较优候选资源（按匹配得分排序）：

{resources_block}

任务：
-----
请你基于以上信息，为学习者 {learner_uid} 设计一条清晰、可执行的学习路线，并解释原因。

请注意：
- 你需要**合理利用上述候选资源**来编排学习顺序，而不是凭空想象资源。
- 路线可以分为若干“步骤”或“阶段”，每一步说明使用哪些资源、学习什么知识点、预计用时。
- 解释部分要用学习者能理解的自然语言，说明：
  - 为什么这样排序；
  - 如何照顾到他的薄弱知识点；
  - 难度和负荷是如何渐进的；
  - 如果有协作或高交互资源，为什么适合他。

输出格式（用于直接展示给学习者）：
--------------------------------
请使用 Markdown 进行组织，大致结构如下（可以在此基础上自由发挥）：

### 学习路线规划

1. 第一步：...
   - 使用资源：列出资源 UID 或简要描述
   - 主要知识点：...
   - 预计时间：... 分钟

2. 第二步：...

（视情况增加步骤）

### 为什么为你设计这样的路线

- 理由 1：...
- 理由 2：...
- （可结合你的知识薄弱点、偏好的资源类型、交互/协作特征等进行说明）

要求：
- 使用简体中文；
- 不要输出任何与上述资源完全无关、系统中不存在的资源；
- 可以引用资源的 UID 和主要知识点进行说明，让学习者知道自己会“用什么、学什么、为什么这样安排”。

现在请为学习者 {learner_uid} 输出完整的学习路线规划和理由。
"""
        return prompt

    # ------------------------------------------------------------------
    # 对外接口：analyze
    # ------------------------------------------------------------------
    def analyze(
        self,
        learner_uids: List[str],
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        综合 resource_planning_engine 和 resource_orchestration_engine 的结果，
        为每个学习者生成学习路径规划与理由。

        参数
        ----
        learner_uids: 需要做路径规划的学习者 uid 列表
        data: {
            "<uid>": {
                "resource_planning": {...},       # 来自 resource_planning_engine.results[uid]
                "resource_orchestration": {...}, # 来自 resource_orchestration_engine.results[uid]
            },
            ...
        }

        返回
        ----
        {
          "engine_status": {...},
          "results": {
            "<uid>": {
              "learning_path_text": "<Markdown 文本>"
            },
            ...
          }
        }
        """
        self.ensure_initialized()

        if not data:
            return {"engine_status": self.get_engine_status(), "results": {}}

        results: Dict[str, Any] = {}

        for uid in learner_uids:
            user_data = data.get(uid, {}) or {}

            # 兼容不同命名：尝试 "resource_planning" / "planning"
            planning = (
                user_data.get("resource_planning")
                or user_data.get("planning")
                or {}
            )

            # 兼容 "resource_orchestration" / "orchestration"
            orchestration = (
                user_data.get("resource_orchestration")
                or user_data.get("orchestration")
                or {}
            )

            if not planning or not orchestration:
                logger.warning(
                    "[LearningPathEngine] missing planning or orchestration for learner %s, skip.",
                    uid,
                )
                continue

            prompt = self._build_prompt_for_learner(
                learner_uid=uid,
                planning=planning,
                orchestration=orchestration,
            )

            try:
                raw_text = self._call_llm(prompt)
            except Exception as e:
                logger.error(
                    "[LearningPathEngine] LLM call failed for learner %s: %s", uid, e
                )
                raw_text = f"对不起，在生成学习路线时出现了问题：{e}"

            results[uid] = {
                "learning_path_text": raw_text
            }

        return {
            "engine_status": self.get_engine_status(),
            "results": results,
        }
