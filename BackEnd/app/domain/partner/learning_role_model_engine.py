# app/domain/partner/learning_role_model_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.domain.common.base_engine import BaseEngine
from app.core.settings import partner_settings, profiling_settings

logger = logging.getLogger(__name__)


class LearningRoleModelMatchingEngine(BaseEngine):
    """
    LearningRoleModelMatchingEngine

    面向“学习榜样（Role Model）”的匹配引擎。

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

    与伙伴引擎共享同一套原始输入，以不同的视图和打分策略进行“向上对标”匹配。
    """

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = profiling_settings.default_device

        super().__init__(device=device, name="LearningRoleModelMatchingEngine")

        # 榜样匹配的多视图权重
        score_weights = partner_settings.role_model_score_weights
        self.alpha_profile: float = float(score_weights.get("alpha_profile", 0.3))
        self.beta_gap: float = float(score_weights.get("beta_global_advancement", 0.4))
        self.gamma_k_comp: float = float(
            score_weights.get("gamma_knowledge_complementarity", 0.3)
        )

        # 向上窗口 [gap_min, gap_max]
        gap_window = partner_settings.role_model_gap_window
        self.gap_min: float = float(gap_window.get("min", 0.05))
        self.gap_max: float = float(gap_window.get("max", 0.25))

        # 互补阈值（复用 partner 的配置）
        thresholds = partner_settings.partner_knowledge_thresholds
        self.low_threshold: float = float(thresholds.get("low", 0.6))
        self.high_threshold: float = float(thresholds.get("high", 0.85))

        # 默认 top_k
        self._default_top_k: int = partner_settings.role_model_default_top_k

        # 画像权重（更强调投入 / 反思等）
        self.profile_feature_weights: Dict[Tuple[str, str], float] = {}
        for key, w in partner_settings.role_model_profile_feature_weights.items():
            try:
                dim, sub_key = key.split(".", 1)
                self.profile_feature_weights[(dim, sub_key)] = float(w)
            except ValueError:
                logger.warning("Invalid role model profile weight key: %s", key)
            except Exception as exc:
                logger.error("Error parsing role model profile weight (%s): %s", key, exc)

    # ------------------------------------------------------------------
    # BaseEngine 接口实现
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            self.is_initialized = True
            logger.info(
                "%s 初始化完成: alpha=%.3f, beta=%.3f, gamma=%.3f, gap=[%.3f, %.3f]",
                self.engine_name,
                self.alpha_profile,
                self.beta_gap,
                self.gamma_k_comp,
                self.gap_min,
                self.gap_max,
            )
            return True
        except Exception as exc:
            logger.error("LearningRoleModelMatchingEngine.initialize failed: %s", exc)
            self.is_initialized = False
            return False

    def analyze(
        self,
        learner_uids: List[str],
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.ensure_initialized():
            return {"engine_status": self.get_engine_status(), "results": {}}

        if not learner_uids:
            return {"engine_status": self.get_engine_status(), "results": {}}

        if not data:
            logger.warning("LearningRoleModelMatchingEngine.analyze called with empty data.")
            return {"engine_status": self.get_engine_status(), "results": {}}

        # 1. 特征视图
        features_map = self._build_feature_views_from_data(data)

        # 2. 每个目标学习者找榜样
        results: Dict[str, Any] = {}
        for uid in learner_uids:
            target_feat = features_map.get(uid)
            if not target_feat:
                results[uid] = {"role_models": []}
                continue

            models = self._match_single_learner(
                target_uid=uid,
                target_feat=target_feat,
                all_features_map=features_map,
                top_k=self._default_top_k,
            )
            results[uid] = {"role_models": models}

        return {
            "engine_status": self.get_engine_status(),
            "results": results,
        }

    # ------------------------------------------------------------------
    # 特征构建（与伙伴引擎保持一致）
    # ------------------------------------------------------------------

    def _build_feature_views_from_data(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        features: Dict[str, Dict[str, Any]] = {}

        for uid, entry in data.items():
            if not isinstance(entry, dict):
                continue

            profile_obj = (
                entry.get("learner_profile")
                or entry.get("learner_profiles")
                or {}
            )
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
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        t_profile = target_feat["profile_categorical"]
        t_kv = target_feat["knowledge_vector"]
        t_E = self._calc_global_expertise(t_kv)

        scored_candidates: List[Tuple[str, Dict[str, Any]]] = []

        for cand_uid, cand_feat in all_features_map.items():
            if cand_uid == target_uid:
                continue

            c_profile = cand_feat["profile_categorical"]
            c_kv = cand_feat["knowledge_vector"]
            c_E = self._calc_global_expertise(c_kv)

            s_profile = self._calc_profile_homophily(t_profile, c_profile)
            s_gap = self._calc_global_advancement(t_E, c_E)
            s_k_comp = self._calc_knowledge_complementarity(t_kv, c_kv)

            score = (
                self.alpha_profile * s_profile
                + self.beta_gap * s_gap
                + self.gamma_k_comp * s_k_comp
            )

            scored_candidates.append(
                (
                    cand_uid,
                    {
                        "uid": cand_uid,
                        "score": score,
                        "profile_homophily": s_profile,
                        "global_advancement": s_gap,
                        "knowledge_complementarity": s_k_comp,
                        "explanation": self._build_explanation(
                            s_profile, s_gap, s_k_comp
                        ),
                    },
                )
            )

        scored_candidates.sort(key=lambda x: x[1]["score"], reverse=True)
        return [item[1] for item in scored_candidates[:top_k]]

    # -------- 全局能力指数 --------

    @staticmethod
    def _calc_global_expertise(kv: Dict[str, float]) -> float:
        if not kv:
            return 0.0
        vals = [v for v in kv.values() if v is not None]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

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

    # -------- 向上窗口（ΔE -> [0,1]） --------

    def _calc_global_advancement(self, E_t: float, E_c: float) -> float:
        delta = E_c - E_t
        if delta <= 0:
            return 0.0

        center = (self.gap_min + self.gap_max) / 2.0
        width = (self.gap_max - self.gap_min) / 2.0
        if width <= 0:
            # 容错：配置不合理时退化为线性缩放
            return max(0.0, min(1.0, delta / max(self.gap_max, 1e-6)))

        s = 1.0 - abs(delta - center) / width
        if s < 0.0:
            return 0.0
        if s > 1.0:
            return 1.0
        return float(s)

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
        s_profile: float, s_gap: float, s_k_comp: float
    ) -> str:
        parts: List[str] = []

        if s_profile >= 0.7:
            parts.append("在非认知画像上与你高度相似，便于产生榜样认同")
        elif s_profile >= 0.4:
            parts.append("在关键学习画像维度上与你有一定相似性")
        else:
            parts.append("学习画像与您差异较大，可提供不同视角的学习范式")

        if s_gap >= 0.7:
            parts.append("整体学习能力显著高于你，但差距处于理想对标区间")
        elif s_gap >= 0.4:
            parts.append("整体能力略高于你，可作为阶段性学习参照")
        else:
            parts.append("整体能力差距不在理想对标区间")

        if s_k_comp >= 0.2:
            parts.append("在你薄弱的若干知识点上具备明显优势，适合作为结构性支架")
        else:
            parts.append("在你薄弱知识点上的互补程度有限")

        return "；".join(parts)
