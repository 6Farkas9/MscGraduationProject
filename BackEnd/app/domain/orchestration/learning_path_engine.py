# BackEnd/app/domain/orchestration/learning_path_engine.py
# -*- coding: utf-8 -*-

"""
第二次调用大模型：学习路径规划（LearningPathEngine）

职责：
- 输入：learner_info, knowledge_state, first_plan, matched_resources
- 输出：学习路径 JSON（必须只引用 matched_resources 里存在的 uid，不得虚构资源）

注意：
- 两次调用之间你已有资源匹配 engine，因此第二次输入必须包含 matched_resources（由你的资源匹配 engine 产出）。
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


class LearningPathEngine(BaseEngine):
    # ---------------------------------------------------------------------
    # 提示词（严格参考：第二次大模型提示词.txt，并补充输出 JSON schema 以保证可执行）
    # ---------------------------------------------------------------------
    _SYSTEM_PROMPT: str = (
        "你现在扮演一个“元宇宙智能学习路径规划器”。\n"
        "你的任务是：\n"
        "根据输入的学习者画像、知识状态、第一次规划结果，以及已经匹配好的候选学习资源分段，\n"
        "设计一条结构合理的学习路径。\n"
        "这条路径必须只使用给定候选资源列表中的资源，不得虚构新的资源。\n"
        "你需要同时考虑：\n"
        "知识点的前驱/后继关系与学习目标（补救 / 巩固 / 新学习）；\n"
        "学习者在 11 个画像维度上的特点（如自我调节、认知风格、社交倾向、注意力特征等）；\n"
        "资源的多维标签（类型、教学功能、难度、引导性、交互性、社交强度、认知负荷、时间长度等）；\n"
        "已经给出的多目标匹配分数（overall、concept_score、feature_score、semantic_score 等）。\n"
        "你需要输出一个结构化的学习路径 JSON，包括：\n"
        "路径整体策略说明与预期目标；\n"
        "一个有序的步骤列表，每个步骤明确用到哪些资源、目标是什么、放在这个顺序的理由是什么；\n"
        "关键检查点与自适应调整建议。\n"
        "输出必须是严格符合指定 JSON 结构的单一 JSON 对象，不要加入任何额外说明文字、注释或 Markdown。\n"
        "如果你需要引用资源，请通过 matched_resources 中的 uid 来引用，并保持不变。\n"
        "你可以把这项任务理解为：在一个已经检索好候选资源的“学习资源图”上，进行画像感知、目标感知的“路径搜索与重排”。\n\n"
        "--------------------------------\n"
        "【输出 JSON 结构（必须严格遵守，不得输出额外字段）】\n"
        "{\n"
        "  \"path_overview\": {\n"
        "    \"strategy\": \"...\",\n"
        "    \"objectives\": [\"...\"],\n"
        "    \"estimated_total_time\": <int_minutes>\n"
        "  },\n"
        "  \"steps\": [\n"
        "    {\n"
        "      \"step_index\": <int_start_from_1>,\n"
        "      \"goal\": \"remedial\" | \"consolidation\" | \"new_learning\",\n"
        "      \"target_concepts\": [\"<concept_uid>\"] ,\n"
        "      \"resource_uids\": [\"<matched_resources.uid>\"] ,\n"
        "      \"why_this_step\": \"...\",\n"
        "      \"expected_outcome\": \"...\",\n"
        "      \"time_estimate\": <int_minutes>\n"
        "    }\n"
        "  ],\n"
        "  \"checkpoints\": [\n"
        "    {\n"
        "      \"after_step\": <int>,\n"
        "      \"check\": \"...\",\n"
        "      \"adjustment_if_failed\": \"...\"\n"
        "    }\n"
        "  ],\n"
        "  \"adaptive_rules\": [\"...\"]\n"
        "}\n\n"
        "强约束：\n"
        "- steps[].resource_uids 只能来自输入 matched_resources[].uid；不得杜撰任何 uid。\n"
        "- steps 的顺序要能解释“先修->后继、从易到难、从高引导到低引导（如合适）”。\n"
        "- 输出只能是 JSON。\n"
    )

    _USER_TEMPLATE: str = (
        "下面是当前学习者的信息、知识状态、第一次规划结果以及为该规划检索到的候选资源。\n"
        "请根据这些信息，按照约定的输出格式设计一条学习路径，并给出每一步选择资源与组织顺序的理由。\n"
        "请只输出 JSON，不要输出其他任何文字。\n"
        "{input_json}"
    )

    # ------------------------------------------------------------------
    def __init__(self, device: str = "cpu", provider: Optional[str] = None) -> None:
        super().__init__(device=device)
        self._provider = provider

    def initialize(self) -> bool:
        try:
            cfg = llm_settings.get_provider(self._provider)
            if not cfg.api_key:
                logger.error("LLM api_key 未配置：provider=%s", self._provider or "default")
                self.is_initialized = False
                return False

            self.model = {"provider": self._provider or "default", "base_url": cfg.base_url}
            self.is_initialized = True
            return True
        except Exception as e:
            logger.exception("LearningPathEngine 初始化失败: %s", e)
            self.is_initialized = False
            return False

    def analyze(self, learner_uids: List[str], model: Optional[str] = None) -> Dict[str, Any]:
        """
        生成第二次 LLM 输出：学习路径 JSON
        - model: 运行时可选，允许用户在使用前切换模型
        """
        self.ensure_initialized()

        results: Dict[str, Any] = {}
        for uid in learner_uids:
            try:
                payload = self._build_path_payload(uid)
                path = self._call_llm(payload, model=model)
                self._validate_resource_uids(payload, path)  # 防止模型杜撰 uid
                results[uid] = path
            except Exception as e:
                logger.exception("学习路径规划失败 learner_uid=%s", uid)
                results[uid] = {"error": str(e)}
        return results

    # ------------------------------------------------------------------
    def _build_path_payload(self, learner_uid: str) -> Dict[str, Any]:
        """
        这里必须对齐你系统“第二次调用”的真实输入结构，且要包含 matched_resources。
        典型结构（示例）：
          {
            "learner_info": {...},
            "knowledge_state": {...},
            "first_plan": {...},            # 第一次 LLM 输出（或其关键信息）
            "matched_resources": [          # 资源匹配 engine 输出（uid + 标签 + scores）
              {"uid": "...", "type": "...", "concepts": [...], "scores": {...}, ...}
            ]
          }
        """
        raise NotImplementedError(
            "请实现 _build_path_payload：根据 learner_uid 拉取 first_plan + matched_resources 等并组装 JSON。"
        )

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
    # 保护性校验：严禁模型使用不存在的 resource uid
    # ------------------------------------------------------------------
    def _validate_resource_uids(self, input_payload: Dict[str, Any], output_path: Dict[str, Any]) -> None:
        """
        强校验：steps[].resource_uids 必须全部属于 matched_resources[].uid
        """
        matched = input_payload.get("matched_resources", []) or []
        allowed_uids = {str(x.get("uid")) for x in matched if x.get("uid")}

        steps = output_path.get("steps", []) or []
        for step in steps:
            for rid in (step.get("resource_uids", []) or []):
                if rid not in allowed_uids:
                    raise ValueError(f"模型输出包含未提供的资源 uid：{rid}")

    # ------------------------------------------------------------------
    # JSON 解析
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
