# resource_orchestration_engine.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

import json
import logging

from app.domain.common.base_engine import BaseEngine
from app.core.settings import orchestration_settings
from app.data_access.orchestration.resource_orchestration_repository import ResourceOrchestrationRepository

logger = logging.getLogger(__name__)


class ResourceOrchestrationEngine(BaseEngine):
    """
    ResourceOrchestrationEngine

    学习资源编排 / 匹配引擎：
    - 输入：学习者 uid 列表 + 针对每个学习者的大模型“资源规划建议”
    - 输出：为每个学习者匹配到的 Top-K 资源列表（带多目标匹配得分）

    方法论（供论文描述）：
    ----------------------
    - Multi-objective Matching:
        Score(r) = α * S_concept(r)
                   + β * S_type(r)
                   + γ * S_pedagogical(r)
                   + δ * S_interaction(r)

      * S_concept : 基于推荐概念优先级的权重覆盖度
      * S_type : 资源基础类型与推荐类型序列的 rank-aware 匹配度
      * S_pedagogical : 教学功能 / 结构 / 引导等级的一致性
      * S_interaction : 交互/协作模式与画像特征的契合度

    - Progressive Constraint Relaxation:
        从最严格约束开始匹配，若候选资源不足 topK，则逐层放宽：
          Level 0: 概念(top3) + type(top1) + 教学功能 + 难度 + 结构/引导 + 交互 + 协作
          Level 1: 概念(top5) + type(top2) + 教学功能 + 难度 + 交互 + 协作
          Level 2: 概念(all) + type(top3) + 教学功能 + 难度
          Level 3: 概念(all) + type(all) + 教学功能
          Level 4: 概念(all) 仅
          Level 5: 全库兜底

        每层都执行统一的多目标打分函数，保证结果的可比性。
    """

    def __init__(self, device: Optional[str] = None, name: Optional[str] = None) -> None:
        super().__init__(device, name)
        self.repo = ResourceOrchestrationRepository()

        # 从 settings 中读取编排相关配置（top_k / 最大候选数 / 权重）
        self._top_k: int = orchestration_settings.default_top_k
        self._max_candidates: int = orchestration_settings.max_candidates
        self._score_weights: Dict[str, float] = orchestration_settings.score_weights

    # ------------------------------------------------------------------
    # 初始化（本引擎不需要加载大模型，只需标记初始化成功）
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        self.is_initialized = True
        return True

    # ------------------------------------------------------------------
    # 对外接口：analyze
    # ------------------------------------------------------------------
    def analyze(
        self,
        learner_uids: List[str],
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        参数
        ----
        learner_uids:
            需要进行资源匹配的目标学习者 uid 列表。
        data:
            针对每个学习者的大模型资源规划结果，推荐采用结构：

            data = {
                "<learner_uid>": {
                    "plan": {
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
                    }
                },
                ...
            }

        返回
        ----
        {
          "engine_status": {...},
          "results": {
            "<uid>": {
              "top_k": int,
              "candidate_count": int,
              "used_relaxation_level": int,
              "resources": [
                {
                  "uid": "...",
                  "oid": "...",
                  "type": "...",
                  "concepts": [...],
                  "score": float,
                  "difficulty_level": "...",
                  "structure_level": "...",
                  "guidance_level": "...",
                  "interaction_level": "...",
                  "collaboration_mode": "...",
                  "pedagogical_function": "...",
                  "time_estimate": int
                },
                ...
              ]
            }
          }
        }
        """
        self.ensure_initialized()

        if not data:
            return {"engine_status": self.get_engine_status(), "results": {}}

        results: Dict[str, Any] = {}

        for uid in learner_uids:
            user_data = data.get(uid, {}) or {}
            plan = user_data.get("plan") or user_data  # 兼容直接传 plan
            if not plan:
                logger.warning("No plan found for learner %s, skip.", uid)
                continue

            concept_priority: List[str] = plan.get("concept_priority", []) or []
            type_priority: List[str] = plan.get("type_priority", []) or []
            constraints: Dict[str, Any] = plan.get("resource_constraints", {}) or {}

            matched_info = self._match_for_single_learner(
                learner_uid=uid,
                concept_priority=concept_priority,
                type_priority=type_priority,
                constraints=constraints,
            )

            results[uid] = matched_info

        return {
            "engine_status": self.get_engine_status(),
            "results": results,
        }

    # ------------------------------------------------------------------
    # 单个学习者的匹配主流程：带渐进放宽
    # ------------------------------------------------------------------
    def _match_for_single_learner(
        self,
        learner_uid: str,
        concept_priority: List[str],
        type_priority: List[str],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        采用 Progressive Constraint Relaxation，从严格到宽松逐层匹配。
        每层匹配到的候选资源都使用统一的多目标打分函数。
        """

        # 定义放宽策略序列（越往后越宽松）
        # 每个 level 定义：
        #   - concept_top_n: 使用 concept_priority 的前 N 个（None = all）
        #   - type_top_n:    使用 type_priority 的前 N 个（None = all）
        #   - constraint_keys: 要求“严格匹配”的字段名列表
        relaxation_levels: List[Dict[str, Any]] = [
            {
                "level": 0,
                "concept_top_n": 3,
                "type_top_n": 1,
                "constraint_keys": [
                    "difficulty_level",
                    "structure_level",
                    "guidance_level",
                    "interaction_level",
                    "collaboration_mode",
                    "pedagogical_function",
                ],
            },
            {
                "level": 1,
                "concept_top_n": 5,
                "type_top_n": 2,
                "constraint_keys": [
                    "difficulty_level",
                    "interaction_level",
                    "collaboration_mode",
                    "pedagogical_function",
                ],
            },
            {
                "level": 2,
                "concept_top_n": None,  # all
                "type_top_n": 3,
                "constraint_keys": [
                    "difficulty_level",
                    "pedagogical_function",
                ],
            },
            {
                "level": 3,
                "concept_top_n": None,
                "type_top_n": None,  # all
                "constraint_keys": [
                    "pedagogical_function",
                ],
            },
            {
                "level": 4,
                "concept_top_n": None,
                "type_top_n": None,
                "constraint_keys": [],  # 只限定概念
            },
            {
                "level": 5,
                "concept_top_n": None,
                "type_top_n": None,
                "constraint_keys": None,  # 完全兜底
            },
        ]

        all_candidates: List[Dict[str, Any]] = []
        used_level = relaxation_levels[-1]["level"]

        # 依次尝试每个放宽等级
        for level_cfg in relaxation_levels:
            level = level_cfg["level"]
            concept_top_n = level_cfg["concept_top_n"]
            type_top_n = level_cfg["type_top_n"]
            constraint_keys = level_cfg["constraint_keys"]

            # 构造本层使用的概念 / 类型 / 约束
            concept_subset = (
                concept_priority[:concept_top_n]
                if concept_top_n is not None
                else concept_priority
            )
            type_subset = (
                type_priority[:type_top_n]
                if type_top_n is not None
                else type_priority
            )

            extra_filter: Dict[str, Any] = {}
            if constraint_keys is None:
                # 完全兜底：不加任何约束
                extra_filter = {}
            else:
                for k in constraint_keys:
                    v = constraints.get(k)
                    if v is not None:
                        # 注意：此处只是“数据访问侧”的过滤，不做复杂相似度分析
                        # 复杂分析在后续评分函数中完成
                        # 难度这里可以稍作归一（intermediate -> medium）
                        if k == "difficulty_level":
                            v = self._normalize_difficulty_label(v)
                        extra_filter[k] = v

            # 仓库查询候选
            if constraint_keys is None:
                # 完全兜底：从全库取
                candidates = self.repo.get_all_fragments(limit=self._max_candidates)
            else:
                candidates = self.repo.get_fragments_by_concepts_and_types(
                    concept_names=concept_subset or None,
                    types=type_subset or None,
                    extra_filter=extra_filter,
                    limit=self._max_candidates,
                )

            if not candidates:
                # 本层没有任何候选，继续放宽
                continue

            all_candidates = candidates
            used_level = level

            # 如果候选已经足够 top_k，就可以停止放宽
            if len(all_candidates) >= self._top_k:
                break

        # 对候选资源进行多目标打分（即使数量较少也统一打分）
        scored_resources = self._score_candidates(
            candidates=all_candidates,
            concept_priority=concept_priority,
            type_priority=type_priority,
            constraints=constraints,
        )

        top_resources = scored_resources[: self._top_k]

        return {
            "learner_uid": learner_uid,
            "top_k": self._top_k,
            "candidate_count": len(all_candidates),
            "used_relaxation_level": used_level,
            "resources": top_resources,
        }

    # ------------------------------------------------------------------
    # 多目标打分函数（学术吹牛位）
    # ------------------------------------------------------------------
    def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        concept_priority: List[str],
        type_priority: List[str],
        constraints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        对候选资源进行多目标加权打分，并按得分降序返回。
        """

        # 构造概念优先级权重映射：index 越小权重越大
        concept_weight_map: Dict[str, float] = {}
        total_concept_weight = 0.0
        for idx, cname in enumerate(concept_priority):
            w = 1.0 / (idx + 1.0)  # 1, 1/2, 1/3, ...
            concept_weight_map[cname] = w
            total_concept_weight += w
        if total_concept_weight == 0:
            total_concept_weight = 1.0

        # type 的位置权重：类似 soft-ranking
        type_weight_map: Dict[str, float] = {}
        total_type_weight = 0.0
        for idx, t in enumerate(type_priority):
            w = 1.0 / (idx + 1.0)
            type_weight_map[t] = w
            total_type_weight += w
        if total_type_weight == 0:
            total_type_weight = 1.0

        # 期望约束
        expected_diff = self._normalize_difficulty_label(
            constraints.get("difficulty_level")
        )
        expected_struct = constraints.get("structure_level")
        expected_guid = constraints.get("guidance_level")
        expected_interaction = constraints.get("interaction_level")
        expected_collab = constraints.get("collaboration_mode")
        expected_ped = constraints.get("pedagogical_function")

        alpha = self._score_weights.get("alpha", 0.35)  # 概念匹配权重
        beta = self._score_weights.get("beta", 0.15)   # 类型匹配权重
        gamma = self._score_weights.get("gamma", 0.25) # 教学/结构匹配权重
        delta = self._score_weights.get("delta", 0.25) # 交互/协作匹配权重

        scored: List[Dict[str, Any]] = []

        for doc in candidates:
            # ---- 1) 概念匹配：加权覆盖度 ----
            doc_concepts = [c.get("name") for c in doc.get("concepts", []) if c.get("name")]
            concept_score_raw = sum(
                concept_weight_map.get(cname, 0.0) for cname in doc_concepts
            )
            s_concept = concept_score_raw / total_concept_weight

            # ---- 2) 类型匹配：rank-aware ----
            doc_type = doc.get("type")
            type_w = type_weight_map.get(doc_type, 0.0)
            s_type = type_w / total_type_weight

            # ---- 3) 教学功能 / 结构 / 难度一致性 ----
            doc_diff = self._normalize_difficulty_label(doc.get("difficulty_level"))
            doc_struct = doc.get("structure_level")
            doc_guid = doc.get("guidance_level")
            doc_ped = doc.get("pedagogical_function")

            ped_matches = 0
            ped_total = 0

            if expected_diff:
                ped_total += 1
                if doc_diff == expected_diff:
                    ped_matches += 1

            if expected_struct:
                ped_total += 1
                if doc_struct == expected_struct:
                    ped_matches += 1

            if expected_guid:
                ped_total += 1
                if doc_guid == expected_guid:
                    ped_matches += 1

            if expected_ped:
                ped_total += 1
                if doc_ped == expected_ped:
                    ped_matches += 1

            s_pedagogical = (ped_matches / ped_total) if ped_total > 0 else 0.0

            # ---- 4) 交互 / 协作模式一致性 ----
            doc_interaction = doc.get("interaction_level")
            doc_collab = doc.get("collaboration_mode")

            inter_matches = 0
            inter_total = 0

            if expected_interaction:
                inter_total += 1
                if doc_interaction == expected_interaction:
                    inter_matches += 1

            # 若推荐中没指定协作模式，认为不用强制
            if expected_collab and expected_collab != "none":
                inter_total += 1
                if doc_collab == expected_collab:
                    inter_matches += 1

            s_interaction = (inter_matches / inter_total) if inter_total > 0 else 0.0

            # ---- 总分 ----
            score = (
                alpha * s_concept
                + beta * s_type
                + gamma * s_pedagogical
                + delta * s_interaction
            )

            # 收缩到 [0,1] 区间（理论上已经在这个范围，防御性 clamp 一下）
            score = float(max(0.0, min(1.0, score)))

            scored.append(self._build_scored_resource_entry(doc, score))

        # 按得分降序排序
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # 工具：规范化难度标记
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_difficulty_label(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        v = str(value).lower()
        mapping = {
            "easy": "easy",
            "simple": "easy",
            "beginner": "easy",
            "medium": "medium",
            "intermediate": "medium",
            "normal": "medium",
            "hard": "hard",
            "difficult": "hard",
            "advanced": "hard",
        }
        return mapping.get(v, v)

    # ------------------------------------------------------------------
    # 工具：构造输出资源条目（压缩字段，仅保留匹配需要的关键属性）
    # ------------------------------------------------------------------
    @staticmethod
    def _build_scored_resource_entry(doc: Dict[str, Any], score: float) -> Dict[str, Any]:
        concepts = [c.get("name") for c in doc.get("concepts", []) if c.get("name")]
        return {
            "uid": doc.get("uid"),
            "oid": doc.get("oid"),
            "type": doc.get("type"),
            "concepts": concepts,
            "score": score,
            "difficulty_level": doc.get("difficulty_level"),
            "structure_level": doc.get("structure_level"),
            "guidance_level": doc.get("guidance_level"),
            "interaction_level": doc.get("interaction_level"),
            "collaboration_mode": doc.get("collaboration_mode"),
            "pedagogical_function": doc.get("pedagogical_function"),
            "time_estimate": doc.get("time_estimate"),
        }
