# BackEnd/app/domain/orchestration/resource_orchestration_engine.py
# -*- coding: utf-8 -*-
"""
资源编排引擎（ResourceOrchestrationEngine）

HR-PRR：Hybrid Retrieval with Progressive Relaxation & Re-ranking

职责：
- 解析大模型规划输出 plan（target_concepts + resource_preferences）
- 调用 ResourceOrchestrationRepository 获取候选资源
- 使用多目标打分公式：
    overall = alpha * concept + beta * type + gamma * feature + delta * semantic
  对候选资源重排序
- 对外提供 match_resources(plan, top_k) 作为主入口

说明：
- 语义模型名称 SEM_MODEL_NAME 在本模块中硬编码
- 其他可调参数（top_k、max_candidates、score 权重）从 settings 中读取
"""

import logging
from typing import Any, Dict, List, Tuple, Optional

from sentence_transformers import SentenceTransformer, util

from app.core.settings import orchestration_settings
from app.domain.common.base_engine import BaseEngine
from app.data_access.orchestration.resource_orchestration_repository import (
    ResourceOrchestrationRepository,
)

logger = logging.getLogger(__name__)

# 根据你的要求，语义模型名称直接硬编码在此处，不放入 settings
SEM_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


def _safe_get(d: Dict, key: str, default=None):
    return d[key] if key in d else default


class ResourceOrchestrationEngine(BaseEngine):
    """
    资源编排引擎实现，继承 BaseEngine：

    - initialize：加载 SentenceTransformer 语义模型
    - match_resources：完成整个 HR-PRR 流程
      - 构建概念权重 / 类型偏好 / 特征约束
      - Progressive Relaxation 多阶段候选检索（调用 Repository）
      - 语义相似度 + 多目标打分 + 排序
    """

    def __init__(
        self,
        repository: ResourceOrchestrationRepository,
        device: str = "cpu",
    ) -> None:
        # BaseEngine 会记录 device / model / 初始化状态
        super().__init__(device=device)

        self._repo = repository
        self._sem_model: Optional[SentenceTransformer] = None

        # 语义模型名硬编码
        self._sem_model_name: str = SEM_MODEL_NAME

        # 从 settings 中读取其余可调参数，避免硬编码
        self._default_top_k: int = orchestration_settings.default_top_k
        weights = orchestration_settings.score_weights
        self._alpha: float = weights["alpha"]
        self._beta: float = weights["beta"]
        self._gamma: float = weights["gamma"]
        self._delta: float = weights["delta"]

    # ------------------------------------------------------------------
    # BaseEngine 接口实现
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """
        子类初始化逻辑：
        - 加载 SentenceTransformer 语义模型
        - 设置 model / is_initialized 状态
        """
        try:
            logger.info(
                "初始化 ResourceOrchestrationEngine，加载语义模型：%s",
                self._sem_model_name,
            )
            # BaseEngine 中的 device 字段用于标记设备信息
            self._sem_model = SentenceTransformer(self._sem_model_name, device=self.device)
            self.model = self._sem_model
            self.is_initialized = True
            return True
        except Exception as exc:
            logger.exception("初始化 ResourceOrchestrationEngine 失败: %s", exc)
            self.is_initialized = False
            return False

    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        由于本引擎的核心功能是“资源匹配”（基于 plan），
        这里只是满足 BaseEngine 抽象接口的占位实现。

        实际使用时，请调用：
            match_resources(plan: Dict[str, Any], top_k: Optional[int] = None)
        """
        raise NotImplementedError(
            "ResourceOrchestrationEngine 面向资源匹配，请使用 `match_resources(plan, top_k)`。"
        )

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def match_resources(
        self,
        plan: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        主入口：给第一次大模型输出的 plan，返回排序后的 top_k 个资源分段。
        内部使用 HR-PRR 的三阶段检索策略。
        """
        if not self.ensure_initialized():
            logger.error("ResourceOrchestrationEngine 未能成功初始化，停止匹配。")
            return []

        top_k = top_k or self._default_top_k

        # 1. 从 plan 中构建各种权重 / 偏好
        concept_weights = self._build_concept_weight_map(plan)
        type_prefs = self._build_type_preference_map(plan)
        feature_constraints = self._build_feature_constraints(plan)

        # 2. Progressive Relaxation 多阶段候选检索（仓库层负责与 DB 交互）
        candidates, stage_used = self._repo.fetch_candidates(
            concept_weights=concept_weights,
            type_prefs=type_prefs,
            top_k=top_k,
        )

        # 3. 计算总分并排序
        scored = self._compute_overall_scores(
            plan=plan,
            candidates=candidates,
            concept_weights=concept_weights,
            type_prefs=type_prefs,
            feature_constraints=feature_constraints,
        )

        # 标记使用的检索阶段，方便后续调试和分析
        for _, doc in scored:
            doc["_retrieval_stage"] = stage_used

        return [doc for _, doc in scored[:top_k]]

    # ------------------------------------------------------------------
    # 1. 解析 plan：概念权重 / 类型偏好 / 特征约束
    # ------------------------------------------------------------------
    @staticmethod
    def _build_concept_weight_map(plan: Dict[str, Any]) -> Dict[str, float]:
        """
        根据 target_concepts 构建概念权重：
          w(concept) = priority_weight * goal_type_multiplier
        """
        PRIORITY_WEIGHT = {
            "high": 1.0,
            "medium": 0.6,
            "low": 0.3,
        }
        GOAL_TYPE_MULTIPLIER = {
            "remedial": 1.3,
            "consolidation": 1.0,
            "new_learning": 1.1,
        }

        weights: Dict[str, float] = {}
        for item in plan.get("target_concepts", []):
            cid = item.get("concept_uid")
            if not cid:
                continue
            priority = item.get("priority", "medium")
            goal_type = item.get("goal_type", "consolidation")
            pw = PRIORITY_WEIGHT.get(priority, 0.5)
            gm = GOAL_TYPE_MULTIPLIER.get(goal_type, 1.0)
            w = pw * gm
            # 若同一概念多次出现，取权重最大的一次
            weights[cid] = max(weights.get(cid, 0.0), w)
        return weights

    @staticmethod
    def _build_type_preference_map(plan: Dict[str, Any]) -> Dict[str, float]:
        """
        根据 unit_type_preferences 构建类型偏好：
          type -> [0,1] 分数
        """
        PREF_LEVEL_SCORE = {
            "high": 1.0,
            "medium": 0.6,
            "low": 0.2,
        }

        pref_map: Dict[str, float] = {}
        uprefs = _safe_get(
            plan.get("resource_preferences", {}),
            "unit_type_preferences",
            [],
        )
        for item in uprefs:
            t = (item.get("type") or "").lower()
            level = item.get("preference_level", "medium")
            score = PREF_LEVEL_SCORE.get(level, 0.5)
            if t:
                pref_map[t] = score

        # 没出现的类型默认中性偏好（和原脚本保持一致）
        for t in ["video", "vr", "ar", "interact", "cooperate"]:
            pref_map.setdefault(t, 0.5)

        return pref_map

    @staticmethod
    def _build_feature_constraints(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        清洗 feature_constraints：
          - 保证有 name 和 desired_values
        """
        res: List[Dict[str, Any]] = []
        fcs = _safe_get(
            plan.get("resource_preferences", {}),
            "feature_constraints",
            [],
        )
        for fc in fcs:
            name = fc.get("name")
            desired = fc.get("desired_values")
            if not name or desired is None:
                continue
            weight = float(fc.get("weight", 1.0))
            res.append(
                {
                    "name": name,
                    "desired_values": desired,
                    "weight": weight,
                    "reason": fc.get("reason", ""),
                }
            )
        return res

    # ------------------------------------------------------------------
    # 2. 各子打分模块
    # ------------------------------------------------------------------
    @staticmethod
    def _score_concept_match(
        doc: Dict[str, Any],
        concept_weights: Dict[str, float],
    ) -> float:
        """
        概念相关性得分：累加 doc 中概念的权重，并截断到 [0,1]
        """
        if not concept_weights:
            return 0.0
        total = 0.0
        for c in doc.get("concepts", []):
            cid = c.get("uid")
            if cid in concept_weights:
                total += concept_weights[cid]
        return min(1.0, total)

    @staticmethod
    def _score_type_preference(
        doc: Dict[str, Any],
        type_prefs: Dict[str, float],
    ) -> float:
        """
        类型偏好得分：根据 type_prefs 映射
        """
        t = (doc.get("type") or "").lower()
        return type_prefs.get(t, 0.5)

    @staticmethod
    def _evaluate_numeric_constraint(value: Any, cond: Dict[str, Any]) -> bool:
        """
        数值约束判断：
        - cond: {operator: str, value: Any}
        """
        if value is None:
            return False
        op = cond.get("operator")
        v = cond.get("value")
        if v is None:
            return False
        try:
            val = float(value)
            v = float(v)
        except Exception:
            return False
        if op == "<=":
            return val <= v
        if op == "<":
            return val < v
        if op == ">=":
            return val >= v
        if op == ">":
            return val > v
        if op == "==":
            return val == v
        return False

    @classmethod
    def _score_feature_match(
        cls,
        doc: Dict[str, Any],
        feature_constraints: List[Dict[str, Any]],
    ) -> float:
        """
        特征匹配得分：
          - 枚举/布尔字段： doc[name] ∈ desired_values => 1，否则 0
          - list 字段（如 role_requirement）：有交集则 1
          - 数值字段（task_steps/time_estimate）：desired_values 中可以是 {operator, value}，
            满足任一即 1
        """
        if not feature_constraints:
            return 0.0

        total_weight = 0.0
        score_sum = 0.0

        for fc in feature_constraints:
            name = fc["name"]
            desired = fc["desired_values"]
            w = float(fc["weight"])
            val = doc.get(name)

            # 数值或对象约束
            if name in ["task_steps", "time_estimate"] or any(
                isinstance(v, dict) for v in desired
            ):
                matched = False
                for cond in desired:
                    if isinstance(cond, dict) and cls._evaluate_numeric_constraint(val, cond):
                        matched = True
                        break
                s = 1.0 if matched else 0.0
            else:
                # 枚举 / List 字段
                if isinstance(val, list):
                    s = 1.0 if any(x in val for x in desired) else 0.0
                else:
                    s = 1.0 if val in desired else 0.0

            total_weight += w
            score_sum += s * w

        if total_weight == 0:
            return 0.0
        return score_sum / total_weight

    # ------------------------------------------------------------------
    # 3. 语义相似度 & 总分融合
    # ------------------------------------------------------------------
    def _build_doc_text_repr(self, doc: Dict[str, Any]) -> str:
        """
        将文档中的若干字段拼成语义表示用的文本。
        """
        parts: List[str] = []
        t = doc.get("type", "")
        if t:
            parts.append(f"type: {t}")

        cpt_names = [c.get("name", "") for c in doc.get("concepts", [])]
        if cpt_names:
            parts.append("concepts: " + ", ".join(cpt_names))

        for key in [
            "pedagogical_function",
            "difficulty_level",
            "cognitive_load",
            "guidance_level",
            "interaction_level",
            "social_intensity",
            "environment_complexity",
            "immersion_level",
        ]:
            val = doc.get(key)
            if val is not None:
                parts.append(f"{key}: {val}")

        content = doc.get("content")
        if content:
            parts.append("content: " + content)

        return "\n".join(parts)

    @staticmethod
    def _build_query_text_repr(plan: Dict[str, Any]) -> str:
        """
        将 target_concepts + 偏好 拼成“查询文本”，用于语义编码。
        """
        parts: List[str] = []

        tc_list = plan.get("target_concepts", [])
        if tc_list:
            names = []
            for item in tc_list:
                cname = item.get("concept_name", "")
                gtype = item.get("goal_type", "")
                if cname:
                    names.append(f"{cname} ({gtype})")
            if names:
                parts.append("target concepts: " + ", ".join(names))

        uprefs = _safe_get(
            plan.get("resource_preferences", {}),
            "unit_type_preferences",
            [],
        )
        if uprefs:
            desc = []
            for item in uprefs:
                t = item.get("type")
                pl = item.get("preference_level")
                if t and pl:
                    desc.append(f"{t}: {pl} preference")
            if desc:
                parts.append("unit type preferences: " + "; ".join(desc))

        fcs = _safe_get(
            plan.get("resource_preferences", {}),
            "feature_constraints",
            [],
        )
        if fcs:
            for fc in fcs:
                name = fc.get("name")
                desired = fc.get("desired_values")
                if not name or desired is None:
                    continue
                parts.append(f"prefer {name} in {desired}")

        notes = plan.get("strategy_notes", [])
        for n in notes:
            parts.append("strategy: " + str(n))

        return "\n".join(parts)

    def _compute_semantic_scores(
        self,
        plan: Dict[str, Any],
        docs: List[Dict[str, Any]],
    ) -> List[float]:
        """
        使用 Sentence-Transformers 计算语义相似度分数（映射到 0~1）
        """
        if not docs:
            return []

        if self._sem_model is None:
            raise RuntimeError("语义模型尚未初始化，请先调用 initialize()。")

        query_text = self._build_query_text_repr(plan)
        query_emb = self._sem_model.encode(
            [query_text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        doc_texts = [self._build_doc_text_repr(d) for d in docs]
        doc_embs = self._sem_model.encode(
            doc_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        cos_scores = util.cos_sim(query_emb, doc_embs)[0].cpu().numpy()
        sem_scores = (cos_scores + 1.0) / 2.0
        return sem_scores.tolist()

    def _compute_overall_scores(
        self,
        plan: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        concept_weights: Dict[str, float],
        type_prefs: Dict[str, float],
        feature_constraints: List[Dict[str, Any]],
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        计算总体得分并排序：

        overall = alpha * concept + beta * type + gamma * feature + delta * semantic
        """
        if not candidates:
            return []

        semantic_scores = self._compute_semantic_scores(plan, candidates)
        scored_docs: List[Tuple[float, Dict[str, Any]]] = []

        for doc, sem in zip(candidates, semantic_scores):
            c_score = self._score_concept_match(doc, concept_weights)
            t_score = self._score_type_preference(doc, type_prefs)
            f_score = self._score_feature_match(doc, feature_constraints)
            overall = (
                self._alpha * c_score
                + self._beta * t_score
                + self._gamma * f_score
                + self._delta * sem
            )

            doc["_score"] = {
                "overall": overall,
                "concept_score": c_score,
                "type_score": t_score,
                "feature_score": f_score,
                "semantic_score": sem,
            }
            scored_docs.append((overall, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return scored_docs

if __name__ == "__main__":
    """
    简单自测入口（不依赖 MySQL）：

    使用一个结构与原脚本 plan_basic_remedial 类似的 plan，验证：
    - Engine 能成功初始化语义模型
    - 能从 Mongo 中拉取候选（如果你本地已有 Fragments 集合数据）
    - 整个 HR-PRR 流程能跑通并打印若干结果
    """

    logging.basicConfig(level=logging.INFO)

    # 1. 准备仓库和引擎
    repo = ResourceOrchestrationRepository()
    engine = ResourceOrchestrationEngine(repository=repo, device="cpu")

    if not engine.initialize():
        print("Engine 初始化失败，无法继续测试。")
        raise SystemExit(1)

    # 2. 构造一个简化版的“补救 + 视频优先” plan
    #    这里的概念 uid/name 直接写死，不依赖 MySQL
    c1 = {"uid": "cpt_57192380f5aa48c29685c7217e89db73", "name": "向量中断"}
    c2 = {"uid": "cpt_84c8834c61184813939802a567e9fa99", "name": "贝叶斯因果网"}

    test_plan = {
        "target_concepts": [
            {
                "concept_uid": c1["uid"],
                "concept_name": c1["name"],
                "predicted_accuracy": 0.35,
                "status": "learned",
                "goal_type": "remedial",
                "priority": "high",
                "target_accuracy": 0.8,
                "reason": f"{c1['name']} 是后续内容的基础，该学习者掌握较弱",
            },
            {
                "concept_uid": c2["uid"],
                "concept_name": c2["name"],
                "predicted_accuracy": 0.6,
                "status": "learned",
                "goal_type": "remedial",
                "priority": "medium",
                "target_accuracy": 0.85,
                "reason": f"需要巩固 {c2['name']} 以支持后续学习",
            },
        ],
        "resource_preferences": {
            "unit_type_preferences": [
                {
                    "type": "video",
                    "preference_level": "high",
                    "reason": "学习者偏好有讲解的视频资源",
                },
                {
                    "type": "interact",
                    "preference_level": "medium",
                    "reason": "适当的操作练习有助于理解",
                },
                {
                    "type": "cooperate",
                    "preference_level": "low",
                    "reason": "当前阶段不强制协作任务",
                },
            ],
            "feature_constraints": [
                {
                    "name": "guidance_level",
                    "desired_values": ["high"],
                    "weight": 1.2,
                    "reason": "需要高引导性资源帮助补救薄弱知识点",
                },
                {
                    "name": "difficulty_level",
                    "desired_values": ["basic", "intermediate"],
                    "weight": 1.0,
                    "reason": "以基础/中阶难度为主做补救",
                },
                {
                    "name": "example_included",
                    "desired_values": [True],
                    "weight": 1.0,
                    "reason": "包含示例有助于理解抽象概念",
                },
                {
                    "name": "time_estimate",
                    "desired_values": [
                        {"operator": "<=", "value": 90}
                    ],
                    "weight": 0.8,
                    "reason": "单个资源学习时间不宜过长",
                },
            ],
        },
        "strategy_notes": [
            "优先补救关键前置知识点，再进行巩固练习。",
            "通过高引导性的视频与适量交互任务，逐步提升掌握度。",
        ],
    }

    # 3. 运行匹配
    top_k = 10
    print(f"\nRunning HR-PRR resource matching test (top_k={top_k})...\n")
    results = engine.match_resources(test_plan, top_k=top_k)

    print(f"匹配结果数量：{len(results)}")
    for idx, doc in enumerate(results, start=1):
        score_info = doc.get("_score", {})
        print(f"\n=== Result #{idx} ===")
        print(f"_id: {doc.get('_id')}")
        print(f"type: {doc.get('type')}")
        print(f"concepts: {[c.get('name') for c in doc.get('concepts', [])]}")
        print(
            "scores:",
            {
                "overall": score_info.get("overall"),
                "concept": score_info.get("concept_score"),
                "type": score_info.get("type_score"),
                "feature": score_info.get("feature_score"),
                "semantic": score_info.get("semantic_score"),
            },
        )

    if not results:
        print(
            "\n注意：当前没有匹配到任何资源，这通常是因为 MongoDB 中 "
            "Fragments 集合暂无符合条件的数据，或者连接配置有问题。"
        )