# resource_planning_engine.py
# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Optional
import requests
import json

from app.domain.common.base_engine import BaseEngine
from app.core.settings import llm_settings
from app.data_access.orchestration.resource_planning_repository import ResourcePlanningRepository


class ResourcePlanningEngine(BaseEngine):
    """
    学习资源编排 / 推荐的推理引擎。

    输入：
        learner_uids: List[str]
        data: {
            <uid>: {
                "KT": { concept_uid: prob, ... },
                "Profile": { ... }
            }
        }

    输出（适合后续资源匹配使用）：
        {
          "engine_status": {...},
          "results": {
              "<learner_uid>": {
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
              ...
          }
        }
    """

    DIMENSION_KEY = "resource_planning"

    def __init__(self, device: Optional[str] = None, name: Optional[str] = None):
        super().__init__(device, name)
        self.repo = ResourcePlanningRepository()
        self.provider = None
        self.model = None

    # ------------------------------------------------------------
    # 初始化：选择 provider & model（支持快速切换）
    # ------------------------------------------------------------
    def initialize(self) -> bool:
        try:
            # 使用默认 provider（例如 Aizex），也可以通过环境变量切换
            self.provider = llm_settings.get_provider()
            self.model = llm_settings.resolve_model()
            return True
        except Exception as e:
            # 这里可替换为 logging
            print("[ResourcePlanningEngine] initialize failed:", e)
            return False

    # ------------------------------------------------------------
    # 调用大模型（OpenAI 兼容接口）
    # ------------------------------------------------------------
    def _call_llm(self, prompt: str) -> str:
        url = f"{self.provider.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.provider.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
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

    # ------------------------------------------------------------
    # 构建 prompt（包含：KT→name，Profile，全量知识点约束）
    # ------------------------------------------------------------
    def _build_prompt(
        self,
        learner_uid: str,
        kt_pairs: List[tuple],
        profile_dict: Dict[str, Any],
        allowed_concepts: List[str],
    ) -> str:

        resource_schema = """
资源属性结构：
- type: video | vr | ar | interact | cooperate
- concepts: string[] （必须来自系统提供的知识点列表）
- visual_richness: low | medium | high
- audio_richness: low | medium | high
- structure_level: low | medium | high
- guidance_level: low | medium | high
- example_included: true | false
- difficulty_level: easy | medium | hard
- cognitive_load: low | medium | high
- interaction_level: none | low | medium | high
- exploration_freedom: low | medium | high
- task_steps: integer
- error_feedback: none | implicit | explicit
- collaboration_mode: pair | group | open
- social_intensity: none | low | medium | high
- role_requirement: ["leader","coordinator","executor","observer"] 的子集
- communication_format: text | voice | mixed
- environment_complexity: simple | moderate | complex
- spatial_navigation_demand: low | medium | high
- immersion_level: low | medium | high
- pedagogical_function: concept_introduction | practice | assessment | feedback | exploration | collaboration | reflection
- attention_demand: low | medium | high
- time_estimate: integer (minutes)
"""

        kt_str = "\n".join([f"- {name}: {prob}" for name, prob in kt_pairs])
        allowed_str = ", ".join(allowed_concepts)

        prompt = f"""
你是一个学习资源规划大模型，需要根据学习者的知识点掌握情况（KT）、画像（Profile），
生成一个**单条整体资源规划建议**（而不是多条），并且需要提供清晰的优先级结构。

严格限制：
1. 你只能使用系统提供的知识点名：[{allowed_str}]
2. 不允许输出任何系统中不存在的知识点。
3. 输出格式必须是严格 JSON（不要添加反引号，不要说明文字），字段语义如下：

{{
  "concept_priority": ["知识点名称1", "知识点名称2", ...],   // 根据掌握薄弱程度排序
  "type_priority": ["video", "interact", ...],               // 推荐资源基础类型的优先级
  "overall_strategy": "...",                                 // 一段文字总结整体策略（简洁）
  "resource_constraints": {{
      "difficulty_level": "easy | medium | hard",
      "structure_level": "low | medium | high",
      "guidance_level": "low | medium | high",
      "interaction_level": "none | low | medium | high",
      "collaboration_mode": "pair | group | open | none",
      "pedagogical_function": "concept_introduction | practice | assessment | feedback | exploration | collaboration | reflection"
  }}
}}

学习者 uid：{learner_uid}

知识点掌握（KT，格式：知识点名称: 预测正确率）如下：
{kt_str}

学习者画像（Profile，保持原始结构）如下：
{json.dumps(profile_dict, ensure_ascii=False, indent=2)}

请基于以上信息，为该学习者生成一个整体资源规划建议，
并按照上述 JSON 结构输出。不要添加任何额外说明文字或注释。
{resource_schema}
"""
        return prompt

    # ------------------------------------------------------------
    # 主入口：analyze
    # ------------------------------------------------------------
    def analyze(
        self,
        learner_uids: List[str],
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        参数
        ----
        learner_uids: 需要做资源规划的学习者 uid 列表
        data: {
            <uid>: {
                "KT": { concept_uid: prob, ... },
                "Profile": { ... }
            }
        }

        返回
        ----
        {
          "engine_status": {...},
          "results": {
              <uid>: { ... 单条整体建议 JSON ... }
          }
        }
        """
        self.ensure_initialized()

        if not data:
            return {"engine_status": self.get_engine_status(), "results": {}}

        # 1. 获取系统中全部知识点名称，作为“大模型允许使用的概念全集”
        all_concept_names = self.repo.get_all_concept_names()

        results: Dict[str, Any] = {}

        for uid in learner_uids:
            user_data = data.get(uid, {}) or {}

            # --- 处理 KT：uid → name（不让大模型看到 uid）
            kt_raw: Dict[str, float] = user_data.get("KT", {}) or {}
            kt_uids = list(kt_raw.keys())
            uid_to_name = self.repo.get_concepts_by_uids(kt_uids)

            kt_pairs: List[tuple] = []
            for c_uid, prob in kt_raw.items():
                name = uid_to_name.get(c_uid)
                # 只能用系统中确实存在的 name
                if name:
                    kt_pairs.append((name, prob))

            # --- Profile 原样传给大模型
            profile = user_data.get("Profile", {}) or {}

            # --- 构建 prompt
            prompt = self._build_prompt(
                learner_uid=uid,
                kt_pairs=kt_pairs,
                profile_dict=profile,
                allowed_concepts=all_concept_names,
            )

            # --- 调用大模型，解析 JSON
            try:
                raw_output = self._call_llm(prompt)
                parsed = json.loads(raw_output)
            except Exception as e:
                # 若解析失败，保留原始输出，方便调试
                parsed = {
                    "error": str(e),
                    "raw_output": raw_output if "raw_output" in locals() else None,
                }

            # 输出以学习者 uid 作为 key，方便后续匹配
            results[uid] = parsed

        return {
            "engine_status": self.get_engine_status(),
            "results": results,
        }
