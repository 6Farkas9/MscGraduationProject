# app/domain/partner/learning_partner_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from app.domain.common.base_engine import BaseEngine
from app.core.settings import partner_settings, profiling_settings

logger = logging.getLogger(__name__)


class LearningPartnerMatchingEngine(BaseEngine):
    """
    LearningPartnerMatchingEngine

    基于「画像同质性 + 知识同质性 + 知识互补性」的学习伙伴匹配引擎。

    输入约定（通过 data 参数传入）
    ------------------------------
    data: Dict[str, Any] 结构大致为：

    {
      "<uid>": {
        "learner_profile": { ... },       # 或 "learner_profiles"
        "knowledge_concepts": { ... },    # 或 "knowledge_concept"
        ...
      },
      ...
    }

    其中：
    - learner_profile: 即你一开始描述的 11 维画像结构；
    - knowledge_concepts: 即 KT 预测向量 { concept_uid: accuracy, ... }。

    输出结构
    --------
    {
      "engine_status": {...},
      "results": {
        uid: {
          "partners": [
            {
              "uid": "...",
              "score": float,
              "profile_homophily": float,
              "knowledge_homophily": float,
              "knowledge_complementarity": float,
              "explanation": "..."
            },
            ...
          ]
        },
        ...
      }
    }
    """

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = profiling_settings.default_device

        super().__init__(device=device, name="LearningPartnerMatchingEngine")

        # 从配置中加载多视图融合权重
        score_weights = partner_settings.partner_score_weights
        self.alpha_profile: float = float(score_weights.get("alpha_profile", 0.4))
        self.beta_k_homo: float = float(score_weights.get("beta_k_homophily", 0.3))
        self.gamma_k_comp: float = float(score_weights.get("gamma_k_complementarity", 0.3))

        # 知识互补阈值
        thresholds = partner_settings.partner_knowledge_thresholds
        self.low_threshold: float = float(thresholds.get("low", 0.6))
        self.high_threshold: float = float(thresholds.get("high", 0.85))

        # 默认 top_k（每个学习者的伙伴数）
        self._default_top_k: int = partner_settings.partner_default_top_k

        # 画像子维度权重：配置使用 "dimension.sub_key" 形式，这里转为 (dim, sub_key)
        self.profile_feature_weights: Dict[Tuple[str, str], float] = {}
        for key, w in partner_settings.partner_profile_feature_weights.items():
            try:
                dim, sub_key = key.split(".", 1)
                self.profile_feature_weights[(dim, sub_key)] = float(w)
            except ValueError:
                logger.warning("Invalid partner profile weight key: %s", key)
            except Exception as exc:
                logger.error("Error parsing partner profile weight (%s): %s", key, exc)

    # ------------------------------------------------------------------
    # BaseEngine 接口实现
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        当前版本没有额外模型要加载，只需设置标志位。
        """
        try:
            self.is_initialized = True
            logger.info(
                "%s 初始化完成: alpha=%.3f, beta=%.3f, gamma=%.3f, low=%.2f, high=%.2f",
                self.engine_name,
                self.alpha_profile,
                self.beta_k_homo,
                self.gamma_k_comp,
                self.low_threshold,
                self.high_threshold,
            )
            return True
        except Exception as exc:
            logger.error("LearningPartnerMatchingEngine.initialize failed: %s", exc)
            self.is_initialized = False
            return False

    def analyze(
        self,
        learner_uids: List[str],
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        伙伴匹配主入口。

        learner_uids:
            需要进行伙伴推荐的目标学习者 uid 列表；
        data:
            包含所有候选（包括目标学习者自身）画像与 KT 的大字典。
        """
        if not self.ensure_initialized():
            return {"engine_status": self.get_engine_status(), "results": {}}

        if not learner_uids:
            return {"engine_status": self.get_engine_status(), "results": {}}

        if not data:
            logger.warning("LearningPartnerMatchingEngine.analyze called with empty data.")
            return {"engine_status": self.get_engine_status(), "results": {}}

        # 1. 为 data 中所有 uid 构建特征视图
        features_map = self._build_feature_views_from_data(data)

        # 2. 针对每个目标 uid 计算伙伴列表
        results: Dict[str, Any] = {}
        for uid in learner_uids:
            target_feat = features_map.get(uid)
            if not target_feat:
                results[uid] = {"partners": []}
                continue

            partners = self._match_single_learner(
                target_uid=uid,
                target_feat=target_feat,
                all_features_map=features_map,
                top_k=self._default_top_k,
            )
            results[uid] = {"partners": partners}

        return {
            "engine_status": self.get_engine_status(),
            "results": results,
        }

    # ------------------------------------------------------------------
    # 内部特征构建
    # ------------------------------------------------------------------

    def _build_feature_views_from_data(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        将 data 中的原始输入整理成统一的特征视图。
        """
        features: Dict[str, Dict[str, Any]] = {}

        for uid, entry in data.items():
            if not isinstance(entry, dict):
                continue

            # 兼容 "learner_profile" / "learner_profiles" 两种写法
            profile_obj = (
                entry.get("learner_profile")
                or entry.get("learner_profiles")
                or {}
            )
            # 兼容 "knowledge_concepts" / "knowledge_concept"
            kt_obj = (
                entry.get("knowledge_concepts")
                or entry.get("knowledge_concept")
                or {}
            )

            kv = self._extract_knowledge_vector(kt_obj)
            pf = self._extract_profile_categorical_features(profile_obj)

            features[uid] = {
                "uid": uid,
                "knowledge_vector": kv,
                "profile_categorical": pf,
                "raw": entry,
            }

        return features

    @staticmethod
    def _extract_knowledge_vector(kt_obj: Dict[str, Any]) -> Dict[str, float]:
        """
        将 KT 对象转换为 { concept_uid: float_accuracy }。
        """
        vec: Dict[str, float] = {}
        for k, v in (kt_obj or {}).items():
            try:
                vec[str(k)] = float(v)
            except Exception:
                continue
        return vec

    @staticmethod
    def _extract_profile_categorical_features(
        profile_obj: Dict[str, Any]
    ) -> Dict[Tuple[str, str], str]:
        """
        将画像对象转换为 {(dimension, sub_key): label_str}。
        """
        features: Dict[Tuple[str, str], str] = {}
        profiles = profile_obj or {}

        for dim_key, dim_content in profiles.items():
            if not isinstance(dim_content, dict):
                continue
            for sub_key, sub_val in dim_content.items():
                features[(str(dim_key), str(sub_key))] = str(sub_val)

        return features

    # ------------------------------------------------------------------
    # 核心匹配逻辑
    # ------------------------------------------------------------------

    def _match_single_learner(
        self,
        target_uid: str,
        target_feat: Dict[str, Any],
        all_features_map: Dict[str, Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        针对单个目标学习者，基于多视图相似度从全体特征中选出伙伴。
        """
        t_profile = target_feat["profile_categorical"]
        t_kv = target_feat["knowledge_vector"]

        scored_candidates: List[Tuple[str, Dict[str, Any]]] = []

        for cand_uid, cand_feat in all_features_map.items():
            if cand_uid == target_uid:
                continue

            c_profile = cand_feat["profile_categorical"]
            c_kv = cand_feat["knowledge_vector"]

            s_profile = self._calc_profile_homophily(t_profile, c_profile)
            s_k_homo = self._calc_knowledge_homophily(t_kv, c_kv)
            s_k_comp = self._calc_knowledge_complementarity(t_kv, c_kv)

            score = (
                self.alpha_profile * s_profile
                + self.beta_k_homo * s_k_homo
                + self.gamma_k_comp * s_k_comp
            )

            scored_candidates.append(
                (
                    cand_uid,
                    {
                        "uid": cand_uid,
                        "score": score,
                        "profile_homophily": s_profile,
                        "knowledge_homophily": s_k_homo,
                        "knowledge_complementarity": s_k_comp,
                        "explanation": self._build_explanation(
                            s_profile, s_k_homo, s_k_comp
                        ),
                    },
                )
            )

        scored_candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        return [item[1] for item in scored_candidates[:top_k]]

    # -------- 画像同质性 --------

    def _calc_profile_homophily(
        self,
        t_profile: Dict[Tuple[str, str], str],
        c_profile: Dict[Tuple[str, str], str],
    ) -> float:
        if not t_profile or not c_profile:
            return 0.0

        total_weight = 0.0
        same_weight = 0.0

        for key, w in self.profile_feature_weights.items():
            t_val = t_profile.get(key)
            c_val = c_profile.get(key)
            if t_val is None or c_val is None:
                continue
            total_weight += w
            if t_val == c_val:
                same_weight += w

        if total_weight <= 0:
            return 0.0
        return same_weight / total_weight

    # -------- 知识同质性（余弦） --------

    @staticmethod
    def _calc_knowledge_homophily(
        t_kv: Dict[str, float],
        c_kv: Dict[str, float],
    ) -> float:
        common_keys = set(t_kv.keys()) & set(c_kv.keys())
        if not common_keys:
            return 0.0

        dot = 0.0
        norm_t = 0.0
        norm_c = 0.0
        for k in common_keys:
            tv = t_kv.get(k, 0.0)
            cv = c_kv.get(k, 0.0)
            dot += tv * cv
            norm_t += tv * tv
            norm_c += cv * cv

        if norm_t <= 0 or norm_c <= 0:
            return 0.0

        return float(dot / (math.sqrt(norm_t) * math.sqrt(norm_c)))

    # -------- 知识互补性 --------

    def _calc_knowledge_complementarity(
        self,
        t_kv: Dict[str, float],
        c_kv: Dict[str, float],
    ) -> float:
        if not t_kv and not c_kv:
            return 0.0

        all_keys = set(t_kv.keys()) | set(c_kv.keys())
        if not all_keys:
            return 0.0

        complement_count = 0
        defined_count = 0

        for k in all_keys:
            t_val = t_kv.get(k)
            c_val = c_kv.get(k)
            if t_val is None and c_val is None:
                continue

            defined_count += 1
            if (
                t_val is not None
                and c_val is not None
                and t_val < self.low_threshold
                and c_val > self.high_threshold
            ):
                complement_count += 1

        if defined_count <= 0:
            return 0.0

        return complement_count / defined_count

    # -------- 文本解释 --------

    @staticmethod
    def _build_explanation(
        s_profile: float, s_k_homo: float, s_k_comp: float
    ) -> str:
        parts: List[str] = []

        if s_profile >= 0.7:
            parts.append("在学习风格与行为画像上高度相似")
        elif s_profile >= 0.4:
            parts.append("在学习风格上具有一定相似性")
        else:
            parts.append("画像风格存在差异，可能带来多样化视角")

        if s_k_homo >= 0.7:
            parts.append("整体知识水平接近")
        elif s_k_homo >= 0.4:
            parts.append("知识水平部分接近")
        else:
            parts.append("知识结构差异较大")

        if s_k_comp >= 0.2:
            parts.append("在部分知识点上对你具有明显互补作用")
        else:
            parts.append("知识互补程度有限")

        return "；".join(parts)
