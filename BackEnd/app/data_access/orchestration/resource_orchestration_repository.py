# BackEnd/app/data_access/orchestration/resource_orchestration_repository.py
# -*- coding: utf-8 -*-
"""
资源编排仓库（ResourceOrchestrationRepository）

职责：
- 封装所有与 MongoDB 资源分段（Fragments）相关的读写操作
- 实现 HR-PRR 中 Progressive Relaxation 的多阶段候选集获取（数据访问层）
- 不包含任何打分 / 语义模型等“分析逻辑”，这些由 Engine 负责

说明：
- MongoDB 连接与基础操作由 MongoDBBaseRepository / MongoDBOperator 提供
- 集合名 FRAGMENTS_COLLECTION 在本模块中硬编码为 "Fragments"
"""

import logging
from typing import Any, Dict, List, Tuple

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository
from app.core.settings import orchestration_settings

logger = logging.getLogger(__name__)

# 资源分段集合名：根据你的要求，直接硬编码，不放进 settings
FRAGMENTS_COLLECTION: str = "Fragments"


class ResourceOrchestrationRepository(MongoDBBaseRepository):
    """
    针对资源编排（HR-PRR 检索）的仓库层实现。

    - 使用 MongoDBBaseRepository 提供的通用 Mongo 能力
    - 使用硬编码的集合名称 Fragments
    - 使用 settings 中配置的 max_candidates 作为每阶段候选数上限
    - 对外提供 fetch_candidates，供引擎做多阶段检索
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 资源分段所在的集合名（硬编码）
        self._collection: str = FRAGMENTS_COLLECTION
        # 每个阶段最多拉取的候选数量，从 settings 中读取
        self._max_candidates: int = orchestration_settings.max_candidates

    # ------------------------------------------------------------------
    # 内部通用方法：count / limit 查询（基于 aggregate）
    # ------------------------------------------------------------------
    def _count_documents(self, query: Dict[str, Any]) -> int:
        """
        使用聚合管道进行 count。

        之所以不直接调用某个 driver API，是为了统一通过 MongoDBOperator，
        避免在仓库层依赖具体驱动实现。
        """
        if not query:
            pipeline = [{"$count": "count"}]
        else:
            pipeline = [{"$match": query}, {"$count": "count"}]

        docs = self.aggregate(self._collection, pipeline)
        if not docs:
            return 0
        return int(docs[0].get("count", 0))

    def _find_documents_with_limit(self, query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """
        使用聚合管道实现带 limit 的查询。
        """
        pipeline: List[Dict[str, Any]] = []
        if query:
            pipeline.append({"$match": query})
        pipeline.append({"$limit": int(limit)})
        return self.aggregate(self._collection, pipeline)

    # ------------------------------------------------------------------
    # Progressive Relaxation 候选集获取（数据访问视角）
    # ------------------------------------------------------------------
    def fetch_candidates(
        self,
        concept_weights: Dict[str, float],
        type_prefs: Dict[str, float],
        top_k: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Progressive Relaxation 策略的候选集获取（仓库层实现纯数据访问）：

        Stage 1:
          - 概念 + 强偏好类型过滤
          - 若 count >= 3 * top_k 则直接使用
          - 否则继续回退

        Stage 2:
          - 仅按概念过滤
          - 若有结果（count >= top_k 或 > 0）则与 Stage 1 合并去重后使用
          - 否则继续回退

        Stage 3:
          - 不加概念/类型限制，拉全库的 max_candidates 作为兜底

        返回：
            candidates: 候选资源列表
            stage_used: 实际命中的检索阶段（1/2/3）
        """
        # 阈值与原脚本保持一致：3 * top_k 被认为“排序候选足够”
        L1_min = 3 * top_k

        # ---------- Stage 1 ----------
        q1 = self._build_query_stage(1, concept_weights, type_prefs)
        candidates1: List[Dict[str, Any]] = []
        count1 = 0
        if q1:
            count1 = self._count_documents(q1)
            if count1 > 0:
                limit1 = min(count1, self._max_candidates)
                candidates1 = self._find_documents_with_limit(q1, limit1)
        else:
            candidates1, count1 = [], 0

        if count1 >= L1_min:
            # Stage 1 结果已足够排序，直接返回
            return candidates1, 1

        # ---------- Stage 2 ----------
        q2 = self._build_query_stage(2, concept_weights, type_prefs)
        candidates2: List[Dict[str, Any]] = []
        if q2:
            count2 = self._count_documents(q2)
            if count2 > 0:
                limit2 = min(count2, self._max_candidates)
                candidates2 = self._find_documents_with_limit(q2, limit2)
        else:
            candidates2, count2 = [], 0

        if candidates2:
            # 若 Stage 1 也有结果，则合并去重（与原脚本逻辑一致）
            if candidates1:
                existing_ids = {c["_id"] for c in candidates1}
                extra = [c for c in candidates2 if c["_id"] not in existing_ids]
                merged = candidates1 + extra
                return merged, 2
            return candidates2, 2

        # ---------- Stage 3（兜底） ----------
        q3 = self._build_query_stage(3, concept_weights, type_prefs)
        count3 = self._count_documents(q3)
        limit3 = min(count3, self._max_candidates)
        candidates3 = self._find_documents_with_limit(q3, limit3)
        return candidates3, 3

    # ------------------------------------------------------------------
    # 构造各 Stage 的 Mongo 查询条件
    # ------------------------------------------------------------------
    @staticmethod
    def _build_query_stage(
        stage: int,
        concept_weights: Dict[str, float],
        type_prefs: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        根据 stage 构造不同的 MongoDB 查询条件（与原脚本保持一致）：

        Stage 1（strict）:
            - concepts.uid in target_concepts
            - type in {pref_level in [high, medium]}

        Stage 2（relax type）:
            - concepts.uid in target_concepts
            - 不限制 type

        Stage 3（fallback）:
            - 不限制 concepts / type
        """
        query: Dict[str, Any] = {}
        target_concepts = list(concept_weights.keys())

        if stage == 1:
            if target_concepts:
                query["concepts.uid"] = {"$in": target_concepts}
            # “强偏好”类型：偏好分 >= 0.6（high / medium）
            strong_types = [t for t, s in type_prefs.items() if s >= 0.6]
            if strong_types:
                query["type"] = {"$in": strong_types}

        elif stage == 2:
            if target_concepts:
                query["concepts.uid"] = {"$in": target_concepts}
            # 不限制 type

        elif stage == 3:
            # 完全不加 concepts / type 限制，作为兜底
            pass

        return query
