# app/domain/partner/partner_pipeline.py
# -*- coding: utf-8 -*-
"""
partner_pipeline

职责
----
- 作为“学习伙伴 + 学习榜样”两个 Engine 的编排层（pipeline）；
- 对外只暴露一个三参版 analyze 接口：
    analyze(learner_uids, learner_profiles, knowledge_concepts)
- 内部隐藏两个 Engine 的调用细节；
- 输出统一整合为以“目标学习者 uid”为键的推荐结果结构。

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
      "partner": [
        {"uid": "p1", "score": 0.92, "explanation": "...", ...},
        ...
      ],
      "role_model": [
        {"uid": "r1", "score": 0.88, "explanation": "...", ...},
        ...
      ]
    },
    "uid_2": {
      ...
    }
  }
}
"""

import logging
from typing import Any, Dict, List, Optional

from app.domain.common.analyze_base_engine import AnalyzeBaseEngine
from app.domain.partner.learning_partner_engine import LearningPartnerMatchingEngine
from app.domain.partner.learning_role_model_engine import LearningRoleModelMatchingEngine
from app.core.settings import profiling_settings

logger = logging.getLogger(__name__)


class PartnerRecommendationPipeline(AnalyzeBaseEngine):
    """
    PartnerRecommendationPipeline

    - 继承 AnalyzeBaseEngine，统一三个输入维度（uids / profiles / knowledge）；
    - 内部组合：
        * LearningPartnerMatchingEngine（学习伙伴匹配）
        * LearningRoleModelMatchingEngine（学习榜样匹配）
    - 对外提供统一的 analyze 接口，隐藏两套引擎的细节。
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """
        参数
        ----
        device:
            运行设备标志，默认使用 profiling_settings.default_device。
        """
        if device is None:
            device = profiling_settings.default_device

        super().__init__(device=device, name="PartnerRecommendationPipeline")

        # 内部组合的两个 Engine，使用同一 device 以便未来可能的向量化/加速
        self.partner_engine = LearningPartnerMatchingEngine(device=device)
        self.role_model_engine = LearningRoleModelMatchingEngine(device=device)

    # ------------------------------------------------------------------
    # AnalyzeBaseEngine 接口实现
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        初始化两个内部 Engine，并聚合初始化结果。
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
        learner_profiles: Dict[str, Any],
        knowledge_concepts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        综合执行“学习伙伴 + 学习榜样”的推荐分析。

        参数
        ----
        learner_uids:
            需要进行推荐的目标学习者 uid 列表。
        learner_profiles:
            uid -> 画像字典（11 维画像等），例如：
                {
                  "uid_1": {
                    "attention_allocation": {...},
                    "social_learning": {...},
                    ...
                  },
                  ...
                }
        knowledge_concepts:
            uid -> 知识点预测精度字典，例如：
                {
                  "uid_1": {"kp_001": 0.92, "kp_002": 0.81, ...},
                  ...
                }

        返回
        ----
        Dict[str, Any]:
            {
              "engine_status": {
                "pipeline": {...},
                "partner_engine": {...},
                "role_model_engine": {...}
              },
              "results": {
                uid: {
                  "partner": [...],
                  "role_model": [...]
                },
                ...
              }
            }
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

        # 可选：做一层轻量的输入完整性检查（日志里会报警告）
        self.validate_inputs(learner_uids, learner_profiles, knowledge_concepts)

        # 1. 调用学习伙伴引擎
        partner_result = self.partner_engine.analyze(
            learner_uids=learner_uids,
            learner_profiles=learner_profiles,
            knowledge_concepts=knowledge_concepts,
        )

        # 2. 调用学习榜样引擎
        role_model_result = self.role_model_engine.analyze(
            learner_uids=learner_uids,
            learner_profiles=learner_profiles,
            knowledge_concepts=knowledge_concepts,
        )

        partner_res_map: Dict[str, Any] = partner_result.get("results", {}) or {}
        role_res_map: Dict[str, Any] = role_model_result.get("results", {}) or {}

        # 3. 聚合两个引擎的结果
        merged_results: Dict[str, Any] = {}
        for uid in learner_uids:
            # 学习伙伴推荐列表
            partners = []
            if uid in partner_res_map:
                partners = partner_res_map.get(uid, {}).get("partners", []) or []

            # 学习榜样推荐列表
            role_models = []
            if uid in role_res_map:
                role_models = role_res_map.get(uid, {}).get("role_models", []) or []

            merged_results[uid] = {
                "partner": partners,
                "role_model": role_models,
            }

        # 4. 汇总状态信息，方便监控/调试
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
    # 可选的便捷方法
    # ------------------------------------------------------------------

    def analyze_for_single_learner(
        self,
        learner_uid: str,
        learner_profiles: Dict[str, Any],
        knowledge_concepts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        针对单个学习者的便捷封装：

        返回结构为：
        {
          "engine_status": {...},
          "learner_uid": xxx,
          "partner": [...],
          "role_model": [...]
        }
        """
        res = self.analyze(
            learner_uids=[learner_uid],
            learner_profiles=learner_profiles,
            knowledge_concepts=knowledge_concepts,
        )

        results_map = res.get("results", {}) or {}
        learner_res = results_map.get(learner_uid, {"partner": [], "role_model": []})

        return {
            "engine_status": res.get("engine_status", {}),
            "learner_uid": learner_uid,
            "partner": learner_res.get("partner", []),
            "role_model": learner_res.get("role_model", []),
        }
