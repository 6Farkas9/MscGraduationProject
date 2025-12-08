# app/domain/partner/learning_partner_engine.py
# -*- coding: utf-8 -*-
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from app.domain.common.analyze_base_engine import AnalyzeBaseEngine
from app.core.settings import partner_settings, profiling_settings

logger = logging.getLogger(__name__)


class LearningPartnerMatchingEngine(AnalyzeBaseEngine):
    """
    LearningPartnerMatchingEngine

    在新的设计下，本 Engine：

    - 输入：learner_uids + learner_profiles + knowledge_concepts（三个外部传入的结构）；
    - 不再直接访问数据库，由上层负责从 Repository 构建好输入；
    - 针对上万规模学习者，通过“全局能力排序 + 局部窗口采样”的方式限制
      每个目标学习者的候选规模，避免 O(N^2) 的爆炸。
    """

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = profiling_settings.default_device

        super().__init__(device=device, name="LearningPartnerMatchingEngine")

        # 多视图融合权重
        score_weights = partner_settings.partner_score_weights
        self.alpha_profile: float = float(score_weights.get("alpha_profile", 0.4))
        self.beta_k_homo: float = float(score_weights.get("beta_k_homophily", 0.3))
        self.gamma_k_comp: float = float(
            score_weights.get("gamma_k_complementarity", 0.3)
        )

        # 知识互补阈值
        thresholds = partner_settings.partner_knowledge_thresholds
        self.low_threshold: float = float(thresholds.get("low", 0.6))
        self.high_threshold: float = float(thresholds.get("high", 0.85))

        # 返回 top_k & 每个目标最多候选规模
        self._default_top_k: int = partner_settings.partner_default_top_k
        self._max_candidates_per_target: int = (
            partner_settings.partner_max_candidates_per_target
        )

        # 画像子维度权重：配置使用 "dimension.sub_key" 形式，这里转为 (dim, sub_key)
        self.profile_feature_weights: Dict[Tuple[str, str], float] = {}
        for key, w in partner_settings.partner_profile_feature_weights.items():
            try:
                dim, sub_key = key.split(".", 1)
                self.profile_feature_weights[(dim, sub_key)] = float(w)
            except Exception:
                logger.warning("Invalid partner profile weight key: %s", key)

    # ------------------------------------------------------------------
    # AnalyzeBaseEngine 接口实现
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            # 当前没有重模型加载，只需标记初始化完成
            self.is_initialized = True
            logger.info(
                "%s 初始化完成: alpha=%.3f, beta=%.3f, gamma=%.3f, low=%.2f, high=%.2f, max_candidates=%d",
                self.engine_name,
                self.alpha_profile,
                self.beta_k_homo,
                self.gamma_k_comp,
                self.low_threshold,
                self.high_threshold,
                self._max_candidates_per_target,
            )
            return True
        except Exception as exc:
            logger.error("LearningPartnerMatchingEngine.initialize failed: %s", exc)
            self.is_initialized = False
            return False

    def analyze(
        self,
        learner_uids: List[str],
        learner_profiles: Dict[str, Any],
        knowledge_concepts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        多视图学习伙伴匹配（三参版接口）。

        learner_profiles[uid] = 11 维画像字典
        knowledge_concepts[uid] = {concept_uid: predicted_accuracy, ...}
        """
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

        # 轻量校验，方便发现数据质量问题
        self.validate_inputs(learner_uids, learner_profiles, knowledge_concepts)

        # 1）构建所有学习者的扁平画像视图 & 知识向量 & 全局能力指数
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

        # 2）按全局能力排序，为后续“局部窗口候选采样”做准备
        sorted_by_E: List[str] = sorted(
            all_uids, key=lambda u: global_expertise.get(u, 0.0)
        )
        uid_index: Dict[str, int] = {
            uid: idx for idx, uid in enumerate(sorted_by_E)
        }

        # 3）针对每个目标 uid 做匹配
        results: Dict[str, Any] = {}
        for uid in learner_uids:
            if uid not in profile_views or uid not in knowledge_views:
                results[uid] = {"partners": []}
                continue

            candidate_uids = self._select_candidates_for_partner(
                target_uid=uid,
                sorted_uids=sorted_by_E,
                uid_index=uid_index,
            )

            partners = self._match_single_learner(
                target_uid=uid,
                candidate_uids=candidate_uids,
                profile_views=profile_views,
                knowledge_views=knowledge_views,
                top_k=self._default_top_k,
            )
            results[uid] = {"partners": partners}

        return {
            "engine_status": self.get_engine_status(),
            "results": results,
        }

    # ------------------------------------------------------------------
    # 对外便捷接口
    # ------------------------------------------------------------------

    def find_partners_for_learner(
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
        learner_res = res.get("results", {}).get(learner_uid, {"partners": []})
        learner_res["partners"] = learner_res.get("partners", [])[:top_k]
        return {
            "engine_status": res.get("engine_status", {}),
            "learner_uid": learner_uid,
            "partners": learner_res["partners"],
        }

    # ------------------------------------------------------------------
    # 内部核心逻辑（性能优化点重点在“候选采样”）
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_profile(
        profile: Dict[str, Any]
    ) -> Dict[Tuple[str, str], str]:
        """
        将 11 维画像的嵌套结构展平成 (dimension, sub_key) -> str(value)。
        """
        features: Dict[Tuple[str, str], str] = {}
        for dim_key, dim_content in profile.items():
            if not isinstance(dim_content, dict):
                continue
            for sub_key, sub_val in dim_content.items():
                features[(str(dim_key), str(sub_key))] = str(sub_val)
        return features

    @staticmethod
    def _sanitize_knowledge_vector(raw: Dict[str, Any]) -> Dict[str, float]:
        """
        确保 KT 向量为 {str: float}，防御性过滤异常值。
        """
        kv: Dict[str, float] = {}
        for k, v in raw.items():
            try:
                kv[str(k)] = float(v)
            except Exception:
                continue
        return kv

    @staticmethod
    def _calc_global_expertise(kv: Dict[str, float]) -> float:
        """
        全局能力指数：知识点预测精度的简单平均，用于建立“一维能力轴”。
        """
        if not kv:
            return 0.0
        vals = [v for v in kv.values() if v is not None]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def _select_candidates_for_partner(
        self,
        target_uid: str,
        sorted_uids: List[str],
        uid_index: Dict[str, int],
    ) -> List[str]:
        """
        使用“全局能力排序 + 局部窗口”策略选择候选学习者：

        - 在能力轴上从 target 的位置向两边扩展；
        - 最多取 partner_max_candidates_per_target 个；
        - 避免对全部学习者 O(N^2) 逐对计算。
        """
        max_cands = self._max_candidates_per_target
        if target_uid not in uid_index:
            return [
                u for u in sorted_uids
                if u != target_uid
            ][:max_cands or None]

        n = len(sorted_uids)
        idx = uid_index[target_uid]

        if max_cands <= 0 or max_cands >= n:
            # 不限候选，退化为全量
            return [u for u in sorted_uids if u != target_uid]

        candidates: List[str] = []
        left = idx - 1
        right = idx + 1
        # 交替从左右两侧取，尽量保证“能力相近”的候选
        while len(candidates) < max_cands and (left >= 0 or right < n):
            if left >= 0:
                u = sorted_uids[left]
                if u != target_uid:
                    candidates.append(u)
                left -= 1
                if len(candidates) >= max_cands:
                    break
            if right < n:
                u = sorted_uids[right]
                if u != target_uid:
                    candidates.append(u)
                right += 1

        return candidates

    def _match_single_learner(
        self,
        target_uid: str,
        candidate_uids: List[str],
        profile_views: Dict[str, Dict[Tuple[str, str], str]],
        knowledge_views: Dict[str, Dict[str, float]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        t_profile = profile_views.get(target_uid) or {}
        t_kv = knowledge_views.get(target_uid) or {}

        scored: List[Dict[str, Any]] = []

        for cand_uid in candidate_uids:
            c_profile = profile_views.get(cand_uid) or {}
            c_kv = knowledge_views.get(cand_uid) or {}

            s_profile = self._calc_profile_homophily(t_profile, c_profile)
            s_k_homo = self._calc_knowledge_homophily(t_kv, c_kv)
            s_k_comp = self._calc_knowledge_complementarity(t_kv, c_kv)

            score = (
                self.alpha_profile * s_profile
                + self.beta_k_homo * s_k_homo
                + self.gamma_k_comp * s_k_comp
            )

            scored.append(
                {
                    "uid": cand_uid,
                    "score": score,
                    "profile_homophily": s_profile,
                    "knowledge_homophily": s_k_homo,
                    "knowledge_complementarity": s_k_comp,
                    "explanation": self._build_explanation(
                        s_profile, s_k_homo, s_k_comp
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

    # -------- 解释文本 --------

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
