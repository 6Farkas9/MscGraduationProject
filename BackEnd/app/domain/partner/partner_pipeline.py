# app/domain/partner/partner_pipeline.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
PartnerRecommendationPipeline

职责
----
- 作为“学习伙伴 + 学习榜样”两个 Engine 的编排层；
- 对外暴露统一的 analyze(learner_uids, data) 接口；
- data 结构与两个底层 Engine 保持一致：

    data = {
      "<uid>": {
        "learner_profile": {...},       # 或 "learner_profiles"
        "knowledge_concepts": {...},    # 或 "knowledge_concept"
      },
      ...
    }

输出结构示例
------------
{
  "engine_status": {
    "pipeline": {...},
    "partner_engine": {...},
    "role_model_engine": {...}
  },
  "results": {
    "uid_1": {
      "partner": [...],
      "role_model": [...]
    },
    ...
  }
}
"""

import logging
from typing import Any, Dict, List, Optional

from app.domain.common.base_engine import BaseEngine
from app.domain.partner.learning_partner_engine import LearningPartnerMatchingEngine
from app.domain.partner.learning_role_model_engine import LearningRoleModelMatchingEngine
from app.core.settings import profiling_settings

logger = logging.getLogger(__name__)


class PartnerRecommendationPipeline(BaseEngine):
    """
    伙伴 + 榜样联合推荐 Pipeline。
    """

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = profiling_settings.default_device

        super().__init__(device=device, name="PartnerRecommendationPipeline")

        self.partner_engine = LearningPartnerMatchingEngine(device=device)
        self.role_model_engine = LearningRoleModelMatchingEngine(device=device)

    # ------------------------------------------------------------------
    # BaseEngine 接口实现
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        初始化内部两个 Engine，并聚合结果。
        """
        try:
            ok_partner = self.partner_engine.ensure_initialized()
            ok_role = self.role_model_engine.ensure_initialized()
            self.is_initialized = bool(ok_partner and ok_role)

            logger.info(
                "%s 初始化完成: partner_ok=%s, role_model_ok=%s",
                self.engine_name,
                ok_partner,
                ok_role,
            )
            return self.is_initialized
        except Exception as exc:
            logger.error("PartnerRecommendationPipeline.initialize failed: %s", exc)
            self.is_initialized = False
            return False

    def analyze(
        self,
        learner_uids: List[str],
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        综合执行“学习伙伴 + 学习榜样”的推荐分析。
        """
        if not self.ensure_initialized():
            return {
                "engine_status": {
                    "pipeline": self.get_engine_status(),
                    "partner_engine": self.partner_engine.get_engine_status(),
                    "role_model_engine": self.role_model_engine.get_engine_status(),
                },
                "results": {},
            }

        if not learner_uids:
            return {
                "engine_status": {
                    "pipeline": self.get_engine_status(),
                    "partner_engine": self.partner_engine.get_engine_status(),
                    "role_model_engine": self.role_model_engine.get_engine_status(),
                },
                "results": {},
            }

        if not data:
            logger.warning("PartnerRecommendationPipeline.analyze called with empty data.")
            return {
                "engine_status": {
                    "pipeline": self.get_engine_status(),
                    "partner_engine": self.partner_engine.get_engine_status(),
                    "role_model_engine": self.role_model_engine.get_engine_status(),
                },
                "results": {},
            }

        # 1. 调用子引擎
        partner_result = self.partner_engine.analyze(
            learner_uids=learner_uids,
            data=data,
        )
        role_model_result = self.role_model_engine.analyze(
            learner_uids=learner_uids,
            data=data,
        )

        partner_res_map: Dict[str, Any] = partner_result.get("results", {}) or {}
        role_res_map: Dict[str, Any] = role_model_result.get("results", {}) or {}

        # 2. 聚合结果
        merged_results: Dict[str, Any] = {}
        for uid in learner_uids:
            partners = []
            if uid in partner_res_map:
                partners = partner_res_map.get(uid, {}).get("partners", []) or []

            role_models = []
            if uid in role_res_map:
                role_models = role_res_map.get(uid, {}).get("role_models", []) or []

            merged_results[uid] = {
                "partner": partners,
                "role_model": role_models,
            }

        engine_status = {
            "pipeline": self.get_engine_status(),
            "partner_engine": partner_result.get("engine_status", {}),
            "role_model_engine": role_model_result.get("engine_status", {}),
        }

        return {
            "engine_status": engine_status,
            "results": merged_results,
        }

    # ------------------------------------------------------------------
    # 便捷方法（可选）
    # ------------------------------------------------------------------

    def analyze_for_single_learner(
        self,
        learner_uid: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        针对单个学习者的便捷封装。
        """
        res = self.analyze(
            learner_uids=[learner_uid],
            data=data,
        )

        results_map = res.get("results", {}) or {}
        learner_res = results_map.get(learner_uid, {"partner": [], "role_model": []})

        return {
            "engine_status": res.get("engine_status", {}),
            "learner_uid": learner_uid,
            "partner": learner_res.get("partner", []),
            "role_model": learner_res.get("role_model", []),
        }
