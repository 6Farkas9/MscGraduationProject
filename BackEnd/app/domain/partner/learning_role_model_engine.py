# app/domain/partner/learning_role_model_engine.py
# -*- coding: utf-8 -*-
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from app.domain.common.analyze_base_engine import AnalyzeBaseEngine
from app.core.settings import partner_settings, profiling_settings

logger = logging.getLogger(__name__)


class LearningRoleModelMatchingEngine(AnalyzeBaseEngine):
    """
    LearningRoleModelMatchingEngine

    - 输入：learner_uids + learner_profiles + knowledge_concepts；
    - 不再访问数据库；
    - 使用同样的全局能力轴 + 局部窗口 + gap window 来控制候选数量，
      应对上万规模学习者时的性能问题。
    """

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = profiling_settings.default_device

        super().__init__(device=device, name="LearningRoleModelMatchingEngine")

        # 多视图权重
        score_weights = partner_settings.role_model_score_weights
        self.alpha_profile: float = float(score_weights.get("alpha_profile", 0.3))
        self.beta_gap: float = float(score_weights.get("beta_global_advancement", 0.4))
        self.gamma_k_comp: float = float(
            score_weights.get("gamma_knowledge_complementarity", 0.3)
        )

        # 向上窗口
        gap_window = partner_settings.role_model_gap_window
        self.gap_min: float = float(gap_window.get("min", 0.05))
        self.gap_max: float = float(gap_window.get("max", 0.25))

        # 知识互补阈值复用 partner 配置
        thresholds = partner_settings.partner_knowledge_thresholds
        self.low_threshold: float = float(thresholds.get("low", 0.6))
        self.high_threshold: float = float(thresholds.get("high", 0.85))

        # top_k & 候选限制
        self._default_top_k: int = partner_settings.role_model_default_top_k
        self._max_candidates_per_target: int = (
            partner_settings.role_model_max_candidates_per_target
        )

        # 画像权重
        self.profile_feature_weights: Dict[Tuple[str, str], float] = {}
        for key, w in partner_settings.role_model_profile_feature_weights.items():
            try:
                dim, sub_key = key.split(".", 1)
                self.profile_feature_weights[(dim, sub_key)] = float(w)
            except Exception:
                logger.warning("Invalid role model profile weight key: %s", key)

    # ------------------------------------------------------------------
    # AnalyzeBaseEngine 接口实现
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            self.is_initialized = True
            logger.info(
                "%s 初始化完成: alpha=%.3f, beta=%.3f, gamma=%.3f, gap=[%.3f, %.3f], max_candidates=%d",
                self.engine_name,
                self.alpha_profile,
                self.beta_gap,
                self.gamma_k_comp,
                self.gap_min,
                self.gap_max,
                self._max_candidates_per_target,
            )
            return True
        except Exception as exc:
            logger.error("LearningRoleModelMatchingEngine.initialize failed: %s", exc)
            self.is_initialized = False
            return False

    def analyze(
        self,
        learner_uids: List[str],
        learner_profiles: Dict[str, Any],
        knowledge_concepts: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.ensure_initialized():
            return {
                "engine_status": self.get_engine_status(),
                "results": {},
            }

        if not learner_uids:
            return {
                "engine_status": self.get_engine_status(),
                "results": {},
            }

        self.validate_inputs(learner_uids, learner_profiles, knowledge_concepts)

        # 1）全量扁平画像 + KT + 全局能力
        all_uids = sorted(
            set(learner_profiles.keys()) | set(knowledge_concepts.keys())
        )
        profile_views: Dict[str, Dict[Tuple[str, str], str]] = {}
        knowledge_views: Dict[str, Dict[str, float]] = {}
        global_expertise: Dict[str, float] = {}

        for uid in all_uids:
            profile_views[uid] = self._flatten_profile(
                learner_profiles.get(uid) or {}
            )
            kv = self._sanitize_knowledge_vector(
                knowledge_concepts.get(uid) or {}
            )
            knowledge_views[uid] = kv
            global_expertise[uid] = self._calc_global_expertise(kv)

        # 2）按全局能力排序
        sorted_by_E: List[str] = sorted(
            all_uids, key=lambda u: global_expertise.get(u, 0.0)
        )
        uid_index: Dict[str, int] = {
            uid: idx for idx, uid in enumerate(sorted_by_E)
        }

        # 3）对每个目标 uid 匹配榜样
        results: Dict[str, Any] = {}
        for uid in learner_uids:
            if uid not in profile_views or uid not in knowledge_views:
                results[uid] = {"role_models": []}
                continue

            candidate_uids = self._select_candidates_for_role_model(
                target_uid=uid,
                sorted_uids=sorted_by_E,
                uid_index=uid_index,
                global_expertise=global_expertise,
            )

            models = self._match_single_learner(
                target_uid=uid,
                candidate_uids=candidate_uids,
                profile_views=profile_views,
                knowledge_views=knowledge_views,
                global_expertise=global_expertise,
                top_k=self._default_top_k,
            )
            results[uid] = {"role_models": models}

        return {
            "engine_status": self.get_engine_status(),
            "results": results,
        }

    # ------------------------------------------------------------------
    # 对外便捷接口
    # ------------------------------------------------------------------

    def find_role_models_for_learner(
        self,
        learner_uid: str,
        learner_profiles: Dict[str, Any],
        knowledge_concepts: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        if top_k is None or top_k <= 0:
            top_k = self._default_top_k

        res = self.analyze(
            learner_uids=[learner_uid],
            learner_profiles=learner_profiles,
            knowledge_concepts=knowledge_concepts,
        )
        learner_res = res.get("results", {}).get(learner_uid, {"role_models": []})
        learner_res["role_models"] = learner_res.get("role_models", [])[:top_k]
        return {
            "engine_status": res.get("engine_status", {}),
            "learner_uid": learner_uid,
            "role_models": learner_res["role_models"],
        }

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_profile(
        profile: Dict[str, Any]
    ) -> Dict[Tuple[str, str], str]:
        features: Dict[Tuple[str, str], str] = {}
        for dim_key, dim_content in profile.items():
            if not isinstance(dim_content, dict):
                continue
            for sub_key, sub_val in dim_content.items():
                features[(str(dim_key), str(sub_key))] = str(sub_val)
        return features

    @staticmethod
    def _sanitize_knowledge_vector(raw: Dict[str, Any]) -> Dict[str, float]:
        kv: Dict[str, float] = {}
        for k, v in raw.items():
            try:
                kv[str(k)] = float(v)
            except Exception:
                continue
        return kv

    @staticmethod
    def _calc_global_expertise(kv: Dict[str, float]) -> float:
        if not kv:
            return 0.0
        vals = [v for v in kv.values() if v is not None]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def _select_candidates_for_role_model(
        self,
        target_uid: str,
        sorted_uids: List[str],
        uid_index: Dict[str, int],
        global_expertise: Dict[str, float],
    ) -> List[str]:
        """
        只向“更强的一侧”采样候选：

        - 从能力轴上 target 的右侧往后看；
        - 只取 ΔE 在 [0, gap_max * 2] 内的若干个（硬截断），并最多 max_candidates_per_target 个；
        - 之后通过 tri-kernel 的 S_gap 进一步细化得分。
        """
        max_cands = self._max_candidates_per_target
        if target_uid not in uid_index:
            return []

        n = len(sorted_uids)
        idx = uid_index[target_uid]
        E_t = global_expertise.get(target_uid, 0.0)

        candidates: List[str] = []
        for j in range(idx + 1, n):
            uid = sorted_uids[j]
            if uid == target_uid:
                continue
            E_c = global_expertise.get(uid, 0.0)
            delta = E_c - E_t
            if delta <= 0:
                continue
            # 简单的硬截断：超过 2 倍 gap_max 就直接停止，避免太远的“高不可攀”式榜样
            if delta > 2.0 * self.gap_max:
                break
            candidates.append(uid)
            if max_cands > 0 and len(candidates) >= max_cands:
                break

        return candidates

    def _match_single_learner(
        self,
        target_uid: str,
        candidate_uids: List[str],
        profile_views: Dict[str, Dict[Tuple[str, str], str]],
        knowledge_views: Dict[str, Dict[str, float]],
        global_expertise: Dict[str, float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        t_profile = profile_views.get(target_uid) or {}
        t_kv = knowledge_views.get(target_uid) or {}
        E_t = global_expertise.get(target_uid, 0.0)

        scored: List[Dict[str, Any]] = []

        for cand_uid in candidate_uids:
            c_profile = profile_views.get(cand_uid) or {}
            c_kv = knowledge_views.get(cand_uid) or {}
            E_c = global_expertise.get(cand_uid, 0.0)

            s_profile = self._calc_profile_homophily(t_profile, c_profile)
            s_gap = self._calc_global_advancement(E_t, E_c)
            s_k_comp = self._calc_knowledge_complementarity(t_kv, c_kv)

            score = (
                self.alpha_profile * s_profile
                + self.beta_gap * s_gap
                + self.gamma_k_comp * s_k_comp
            )

            scored.append(
                {
                    "uid": cand_uid,
                    "score": score,
                    "profile_homophily": s_profile,
                    "global_advancement": s_gap,
                    "knowledge_complementarity": s_k_comp,
                    "explanation": self._build_explanation(
                        s_profile, s_gap, s_k_comp
                    ),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

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

    # -------- ΔE -> S_gap 三角核 --------

    def _calc_global_advancement(self, E_t: float, E_c: float) -> float:
        delta = E_c - E_t
        if delta <= 0:
            return 0.0

        center = (self.gap_min + self.gap_max) / 2.0
        width = (self.gap_max - self.gap_min) / 2.0
        if width <= 0:
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

    # -------- 解释文本 --------

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
