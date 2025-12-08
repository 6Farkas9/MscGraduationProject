# BackEnd/app/domain/orchestration/resource_planning_engine.py
# -*- coding: utf-8 -*-

"""
第一次调用大模型：资源规划（ResourcePlanningEngine）

职责：
- 输入：学习者画像（11维标签）+ 知识点状态/先修关系/（可选）预测得分
- 输出：资源匹配引擎（ResourceOrchestrationEngine）可直接消费的 plan JSON：
    {
      "target_concepts": [...],
      "resource_preferences": {
        "unit_type_preferences": [...],
        "feature_constraints": [...]
      },
      "strategy_notes": [...]
    }

注意：
- 你的系统中第一次 LLM 输出会被“资源匹配 engine”解析，因此输出必须严格遵循字段体系/枚举值。
- 用户提出：既然有 learned/not_learned 标记，not_learned 的 predicted_accuracy=-1 冗余。
  -> 这里的提示词改为：not_learned 的 predicted_accuracy 可缺省或为 null；若提供也不得用于判断薄弱。
- 运行时支持 model 参数：调用 analyze(..., model="xxx") 即可切换模型。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests

from app.core.settings import llm_settings
from app.domain.common.base_engine import BaseEngine

logger = logging.getLogger(__name__)


class ResourcePlanningEngine(BaseEngine):
    """
    第一次调用大模型的 Engine：生成资源匹配 plan。
    """

    # ---------------------------------------------------------------------
    # 提示词（严格参考：第一次大模型提示词.txt，并补齐枚举说明/输出约束）
    # ---------------------------------------------------------------------
    _SYSTEM_PROMPT: str = (
        "你是一名“元宇宙智能学习教练（AI Metaverse Learning Coach）”。\n\n"
        "你的任务：\n"
        "接收某个学习者的：\n"
        "1）知识点级别信息（约 1400 个计算机领域知识点），以及\n"
        "2）基于行为数据提取的 11 维学习者画像标签，\n"
        "输出一份**用于资源匹配的结构化 JSON 配置**。\n\n"
        "这些配置将被另一个检索/匹配系统用于到数据库中筛选和排序教学资源分段，因此你的输出必须：\n"
        "- 严格使用给定的字段体系与枚举值；\n"
        "- 避免出现资源池中无法落地的自由描述；\n"
        "- 只输出 JSON（不得输出任何解释、Markdown、代码块）。\n\n"
        "--------------------------------\n"
        "【一、知识点信息与先修关系】\n"
        "输入中会包含该领域全部知识点的列表，每个元素包含（字段名示例）：\n"
        "- concept_uid: string\n"
        "- concept_name: string\n"
        "- status: \"learned\" | \"not_learned\"\n"
        "- predicted_accuracy: number | null（可选）\n"
        "- predecessors: string[]（先修知识点 uid 列表）\n"
        "- successors: string[]（后继知识点 uid 列表）\n\n"
        "重要规则：\n"
        "1) 你必须主要依据 status 判断是否学过；not_learned 的 predicted_accuracy 不可靠/可缺省，禁止把它当成薄弱证据。\n"
        "2) learned 的 predicted_accuracy 可用于识别薄弱/需补救（数值越低越薄弱）。\n"
        "3) 不能枚举全部知识点，只能挑选少量关键目标输出。\n"
        "4) 不要凭空创造不存在的 concept_uid，只能从输入列表中选取。\n\n"
        "--------------------------------\n"
        "【二、资源池字段体系（只能使用这些字段；并严格使用枚举值）】\n\n"
        "资源基础类型：\n"
        "- type: \"video\" | \"vr\" | \"ar\" | \"interact\" | \"cooperate\"\n"
        "- concepts: string[]（资源覆盖的 concept_uid 列表）\n\n"
        "媒体丰富度：\n"
        "- visual_richness: \"low\" | \"medium\" | \"high\"\n"
        "- audio_richness: \"low\" | \"medium\" | \"high\"\n\n"
        "结构与引导：\n"
        "- structure_level: \"low\" | \"medium\" | \"high\"\n"
        "- guidance_level: \"low\" | \"medium\" | \"high\"\n"
        "- example_included: true | false\n\n"
        "难度与负荷：\n"
        "- difficulty_level: \"easy\" | \"medium\" | \"hard\"\n"
        "- cognitive_load: \"low\" | \"medium\" | \"high\"\n\n"
        "交互特征：\n"
        "- interaction_level: \"none\" | \"low\" | \"medium\" | \"high\"\n"
        "- exploration_freedom: \"low\" | \"medium\" | \"high\"\n"
        "- task_steps: integer（0,1,2,...）\n"
        "- error_feedback: \"none\" | \"implicit\" | \"explicit\"\n\n"
        "协作特征：\n"
        "- collaboration_mode: \"pair\" | \"group\" | \"open\"\n"
        "- social_intensity: \"none\" | \"low\" | \"medium\" | \"high\"\n"
        "- role_requirement: [\"leader\",\"coordinator\",\"executor\",\"observer\"] 的子集\n"
        "- communication_format: \"text\" | \"voice\" | \"mixed\"\n\n"
        "元宇宙环境特征：\n"
        "- environment_complexity: \"simple\" | \"moderate\" | \"complex\"\n"
        "- spatial_navigation_demand: \"low\" | \"medium\" | \"high\"\n"
        "- immersion_level: \"low\" | \"medium\" | \"high\"\n\n"
        "教学功能：\n"
        "- pedagogical_function: \"concept_introduction\" | \"practice\" | \"assessment\" | \"feedback\" | \"exploration\" | \"collaboration\" | \"reflection\"\n\n"
        "注意力与时长：\n"
        "- attention_demand: \"low\" | \"medium\" | \"high\"\n"
        "- time_estimate: integer（分钟）\n\n"
        "--------------------------------\n"
        "【三、输出目标（必须严格按此 JSON 结构输出）】\n"
        "输出一个 JSON 对象，包含：\n"
        "1) target_concepts: 目标知识点列表（少量、关键）\n"
        "2) resource_preferences: 资源偏好与硬/软约束\n"
        "3) strategy_notes: 策略备注（字符串数组，简洁）\n\n"
        "--------------------------------\n"
        "【四、输出 JSON 结构定义（必须完全一致，不得增删字段名）】\n"
        "{\n"
        "  \"target_concepts\": [\n"
        "    {\n"
        "      \"concept_uid\": \"...\",\n"
        "      \"goal_type\": \"remedial\" | \"consolidation\" | \"new_learning\",\n"
        "      \"priority\": \"high\" | \"medium\" | \"low\",\n"
        "      \"reason\": \"...\"\n"
        "    }\n"
        "  ],\n"
        "  \"resource_preferences\": {\n"
        "    \"unit_type_preferences\": [\n"
        "      {\n"
        "        \"type\": \"video\" | \"vr\" | \"ar\" | \"interact\" | \"cooperate\",\n"
        "        \"preference_level\": \"high\" | \"medium\" | \"low\",\n"
        "        \"reason\": \"...\"\n"
        "      }\n"
        "    ],\n"
        "    \"feature_constraints\": [\n"
        "      {\n"
        "        \"name\": \"<必须是资源池字段名之一>\",\n"
        "        \"desired_values\": <必须与该字段类型匹配（枚举/boolean/int/数组）>,\n"
        "        \"weight\": <float，默认1.0，越大越重要>,\n"
        "        \"reason\": \"...\"\n"
        "      }\n"
        "    ]\n"
        "  },\n"
        "  \"strategy_notes\": [\"...\"]\n"
        "}\n\n"
        "--------------------------------\n"
        "【五、规模与选择策略（强约束）】\n"
        "1) target_concepts 总数要少：每个 goal_type ≤ 10。\n"
        "2) learned 且 predicted_accuracy 较低的：优先放入 remedial，可提高 priority。\n"
        "3) learned 且 predicted_accuracy 中等、但作为很多 successor 的前驱：可放入 consolidation。\n"
        "4) not_learned 且多数 predecessors 已学且表现较好、并且拥有较多 successors 的：可放入 new_learning。\n"
        "5) reason 简洁，明确基于哪些画像维度、先修关系、learned/得分等判断。\n\n"
        "--------------------------------\n"
        "【六、输出风格】\n"
        "只输出 JSON，不要任何额外文字，不要 Markdown，不要代码块。\n"
    )

    _USER_TEMPLATE: str = (
        "下面是学习者画像与知识状态数据（JSON）。请按 system 约束输出单一 JSON 对象：\n"
        "{input_json}"
    )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def __init__(self, device: str = "cpu", provider: Optional[str] = None) -> None:
        """
        provider:
          - None -> 使用 settings 里的默认 provider
          - "aizex"/"openai"/... -> 使用指定 provider（见 LLMSettings）
        """
        super().__init__(device=device)
        self._provider = provider

    def initialize(self) -> bool:
        """
        本 engine 不加载本地模型；仅校验 provider/key 配置。
        """
        try:
            cfg = llm_settings.get_provider(self._provider)
            if not cfg.api_key:
                logger.error("LLM api_key 未配置：provider=%s", self._provider or "default")
                self.is_initialized = False
                return False

            # 用 model 字段记录当前 provider 信息（便于日志/排错）
            self.model = {"provider": self._provider or "default", "base_url": cfg.base_url}
            self.is_initialized = True
            return True
        except Exception as e:
            logger.exception("ResourcePlanningEngine 初始化失败: %s", e)
            self.is_initialized = False
            return False

    # ------------------------------------------------------------------
    # 核心入口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str], model: Optional[str] = None) -> Dict[str, Any]:
        """
        生成第一次 LLM 输出 plan。
        - model: 运行时可选，允许用户在使用前切换模型（如 "gpt-4o-mini" 等）
        """
        self.ensure_initialized()

        results: Dict[str, Any] = {}
        for uid in learner_uids:
            try:
                payload = self._build_planning_payload(uid)
                plan = self._call_llm(payload, model=model)
                results[uid] = plan
            except Exception as e:
                logger.exception("资源规划失败 learner_uid=%s", uid)
                results[uid] = {"error": str(e)}
        return results

    # ------------------------------------------------------------------
    # 数据组装（由你接入真实数据源）
    # ------------------------------------------------------------------
    def _build_planning_payload(self, learner_uid: str) -> Dict[str, Any]:
        """
        这里必须按你系统真实输入结构组装 JSON（与你的 LLM 调用约定一致）。
        通常包含：
          - learner_info: 11维画像标签
          - knowledge_concepts: 全量概念列表（uid/name/status/(可选)predicted_accuracy/predecessors/successors）
        """
        raise NotImplementedError(
            "请实现 _build_planning_payload：根据 learner_uid 拉取画像与知识点信息并组装成 JSON。"
        )

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------
    def _call_llm(self, input_payload: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        user_prompt = self._USER_TEMPLATE.format(
            input_json=json.dumps(input_payload, ensure_ascii=False)
        )

        content = self._chat_completion(
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
        )
        return self._must_parse_json(content)

    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> str:
        """
        OpenAI 兼容 / Aizex 路由的 chat/completions 调用。
        model 支持运行时传入，从而允许用户在功能使用前选择模型。
        """
        cfg = llm_settings.get_provider(self._provider)
        resolved_model = llm_settings.resolve_model(self._provider, model)

        url = f"{cfg.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        body = {
            "model": resolved_model,
            "messages": messages,
            "temperature": llm_settings.temperature,
            "max_tokens": llm_settings.max_tokens,
        }

        resp = requests.post(
            url,
            headers=headers,
            json=body,
            proxies=llm_settings.proxies,
            timeout=llm_settings.timeout_sec,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"LLM 请求失败: status={resp.status_code}, body={resp.text}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"LLM 返回结构异常: {json.dumps(data, ensure_ascii=False)}")

    # ------------------------------------------------------------------
    # JSON 严格解析（防止模型输出 ```json 代码块/夹杂文本）
    # ------------------------------------------------------------------
    _JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

    def _must_parse_json(self, text: str) -> Dict[str, Any]:
        cleaned = (text or "").strip()
        cleaned = self._JSON_FENCE_RE.sub("", cleaned).strip()

        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"模型输出不是合法 JSON：{e}. raw={text[:800]}")

        if not isinstance(obj, dict):
            raise ValueError(f"模型输出 JSON 不是对象(dict)，实际类型={type(obj)}")
        return obj
