# BackEnd/app/domain/profiling/profiling_pipeline.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.settings import profiling_settings
from app.domain.profiling.attention_allocation_engine import (
    AttentionAllocationEngine,
)
from app.domain.profiling.engagement_persistence_engine import (
    EngagementPersistenceEngine,
)
from app.domain.profiling.feedback_orientation_engine import (
    FeedbackOrientationEngine,
)
from app.domain.profiling.collaborative_role_contribution_engine import (
    CollaborativeRoleContributionEngine,
)
from app.domain.profiling.contribution_reputation_engine import (
    ContributionReputationEngine,
)
from app.domain.profiling.interaction_style_engine import (
    InteractionStyleEngine,
)
from app.domain.profiling.reflection_depth_value_evolution_engine import (
    ReflectionDepthValueEvolutionEngine,
)
from app.domain.profiling.social_learning_engine import (
    SocialLearningEngine,
)
from app.domain.profiling.spatial_exploration_orientation_engine import (
    SpatialExplorationOrientationEngine,
)
from app.domain.profiling.srl_helpseeking_engine import (
    SRLHelpseekingEngine,
)
from app.domain.profiling.task_efficiency_engine import (
    TaskEfficiencyEngine,
)
from app.shared.models.profiles_labels import get_label

logger = logging.getLogger(__name__)


ProfilingResult = Dict[str, Dict[str, Any]]  # 方便类型标注


class ProfilingPipeline:
    """
    统一调度 11 个 learner 画像引擎的流水线。

    对外只暴露一个主入口：

        analyze(learner_uids: List[str]) -> ProfilingResult

    返回结构（示意）：

    {
        "<learner_uid>": {
            "overall": {
                "<dimension_key>": {
                    "<label_category>": "<标签文本或 None>",
                    ...
                },
                ...
            },
            "details": {
                "<dimension_key>": {
                    # 该维度 Engine 的原始结构，
                    # 形如：
                    # {
                    #   "insufficient_data": bool,
                    #   "insufficient_reason": Optional[str],
                    #   "<category>": {
                    #       "final_code": Optional[int],
                    #       "overall_metrics": {...},
                    #       "courses": {
                    #           "<course_uid>": {
                    #               "code": int,
                    #               "metrics": {...}
                    #           },
                    #           ...
                    #       }
                    #   },
                    #   ...
                    # }
                },
                ...
            },
        },
        ...
    }

    其中：

    - overall：给前端/业务使用的“学习者总体画像标签”，
      只保留每个维度每个标签类别的「文本标签」；
    - details：保留各引擎的完整结果（尤其是课程级 courses 数据），
      供前端可视化和后续数据分析使用。
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Args:
            device: 统一传递给各个 Engine 的 device 标记（如 "cpu" / "cuda"），
                    若为 None，则使用 settings.profiling_settings.default_device。
        """
        self.device = device or profiling_settings.default_device
        self._engines = self._build_engines()

    # ------------------------------------------------------------------
    # Engine 初始化
    # ------------------------------------------------------------------
    def _build_engines(self) -> Dict[str, Any]:
        """
        实例化所有画像引擎，并根据 ProfilingSettings.enabled_dimensions 过滤。

        Returns
        -------
        dict:
            {dimension_key: engine_instance}
        """
        all_engines: Dict[str, Any] = {
            AttentionAllocationEngine.DIMENSION_KEY: AttentionAllocationEngine(
                device=self.device
            ),
            EngagementPersistenceEngine.DIMENSION_KEY: EngagementPersistenceEngine(
                device=self.device
            ),
            FeedbackOrientationEngine.DIMENSION_KEY: FeedbackOrientationEngine(
                device=self.device
            ),
            CollaborativeRoleContributionEngine.DIMENSION_KEY: CollaborativeRoleContributionEngine(
                device=self.device
            ),
            ContributionReputationEngine.DIMENSION_KEY: ContributionReputationEngine(
                device=self.device
            ),
            InteractionStyleEngine.DIMENSION_KEY: InteractionStyleEngine(
                device=self.device
            ),
            ReflectionDepthValueEvolutionEngine.DIMENSION_KEY: ReflectionDepthValueEvolutionEngine(
                device=self.device
            ),
            SocialLearningEngine.DIMENSION_KEY: SocialLearningEngine(
                device=self.device
            ),
            SpatialExplorationOrientationEngine.DIMENSION_KEY: SpatialExplorationOrientationEngine(
                device=self.device
            ),
            SRLHelpseekingEngine.DIMENSION_KEY: SRLHelpseekingEngine(
                device=self.device
            ),
            TaskEfficiencyEngine.DIMENSION_KEY: TaskEfficiencyEngine(
                device=self.device
            ),
        }

        enabled_dims = set(profiling_settings.enabled_dimensions)
        engines = {
            dim_key: engine
            for dim_key, engine in all_engines.items()
            if dim_key in enabled_dims
        }

        logger.info(
            "ProfilingPipeline: 初始化完成，启用维度数 = %d，device = %s",
            len(engines),
            self.device,
        )
        return engines

    # ------------------------------------------------------------------
    # 对外主接口
    # ------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> ProfilingResult:
        """
        对一批学习者执行所有已启用的画像引擎，并汇总为统一结果结构。

        Args
        ----
        learner_uids:
            学习者 uid 列表，可以包含重复或空字符串，内部会去重 & 清洗。

        Returns
        -------
        ProfilingResult:
            形如：

            {
                learner_uid: {
                    "overall": {dimension_key: {category: label, ...}, ...},
                    "details": {dimension_key: dim_result, ...},
                },
                ...
            }
        """
        # 1. 清洗 & 保持原有顺序去重
        cleaned_uids: List[str] = [
            uid for uid in dict.fromkeys(learner_uids or []) if uid
        ]
        if not cleaned_uids:
            return {}

        if (
            profiling_settings.max_batch_size is not None
            and len(cleaned_uids) > profiling_settings.max_batch_size
        ):
            logger.warning(
                "ProfilingPipeline.analyze: 学习者数量 %d 超过建议单批上限 %d。",
                len(cleaned_uids),
                profiling_settings.max_batch_size,
            )

        logger.info(
            "ProfilingPipeline.analyze: 开始分析，学习者数 = %d，维度数 = %d",
            len(cleaned_uids),
            len(self._engines),
        )

        # 2. 初始化最终结果骨架
        final_result: ProfilingResult = {
            uid: {"overall": {}, "details": {}} for uid in cleaned_uids
        }

        # 3. 遍历每一个维度的 Engine，执行分析并写入 final_result
        for dim_key, engine in self._engines.items():
            logger.info(
                "ProfilingPipeline.analyze: 运行维度 '%s' 的引擎 %s",
                dim_key,
                engine.__class__.__name__,
            )

            try:
                engine_output: Dict[str, Dict[str, Any]] = engine.analyze(
                    cleaned_uids
                )
            except Exception:
                # 单个维度发生异常时，不阻塞其它维度，统一记录为 insufficient_data
                logger.exception(
                    "ProfilingPipeline.analyze: 引擎 '%s' 执行异常，将该维度视为数据不足。",
                    engine.__class__.__name__,
                )
                for uid in cleaned_uids:
                    final_result[uid]["details"][dim_key] = {
                        "insufficient_data": True,
                        "insufficient_reason": "engine_error",
                    }
                continue

            # 将该维度的结果写入每个 learner
            for uid in cleaned_uids:
                per_learner_payload: Dict[str, Any] = engine_output.get(uid) or {}

                # 各 Engine 的返回一般为：
                # { learner_uid: { "<dimension_key>": dim_result } }
                dim_result: Optional[Dict[str, Any]] = per_learner_payload.get(
                    dim_key
                )

                if dim_result is None:
                    # 该 learner 在此维度没有返回，统一视为 insufficient_data
                    dim_result = {
                        "insufficient_data": True,
                        "insufficient_reason": "no_result_for_learner",
                    }

                # 3.1 写入 details：完整保留该维度的结构（包含 overall_metrics / courses）
                final_result[uid]["details"][dim_key] = dim_result

                # 3.2 从 dim_result 中提取「总体标签」写入 overall
                overall_labels = self._build_overall_labels(
                    dimension=dim_key, dim_result=dim_result
                )
                if overall_labels:
                    final_result[uid]["overall"][dim_key] = overall_labels

        logger.info(
            "ProfilingPipeline.analyze: 完成分析，最终学习者数 = %d",
            len(final_result),
        )
        return final_result

    # ------------------------------------------------------------------
    # 工具：从单个维度结果中提取 overall 文本标签
    # ------------------------------------------------------------------
    @staticmethod
    def _build_overall_labels(
        dimension: str, dim_result: Dict[str, Any]
    ) -> Dict[str, Optional[str]]:
        """
        从某个维度的原始结果中，抽取“整体标签”的文本形式。

        规则约定：
        - 若 dim_result["insufficient_data"] 为 True，则返回空 dict；
        - 对 dim_result 下所有 value 为 dict 且包含 "final_code" 字段的 key，
          视为一个标签类别（label_category），
          例如：
              "level" / "style" / "role" / "efficiency" / "contribution_type" 等；
        - 对每个类别调用 profiles_labels.get_label(dimension, label_category, code)
          将数值 code 转成对应的文本标签。
        """
        if not dim_result or dim_result.get("insufficient_data"):
            return {}

        labels: Dict[str, Optional[str]] = {}

        for key, value in dim_result.items():
            if not isinstance(value, dict):
                continue

            # 只关心包含 final_code 的字段
            if "final_code" not in value:
                continue

            code = value.get("final_code")
            label = get_label(dimension, key, code)

            # 即便 label 为 None 也保留 key，方便前端统一处理“未定义”/“暂无”
            labels[key] = label

        return labels


# ----------------------------------------------------------------------
# 模块级便捷函数：直接调用 analyze
# ----------------------------------------------------------------------
def analyze(learner_uids: List[str], device: Optional[str] = None) -> ProfilingResult:
    """
    便捷方法：不显式构造 ProfilingPipeline 实例，直接完成一次分析。

    使用方式：
        from app.domain.profiling.profiling_pipeline import analyze

        result = analyze(["lrn_xxx", "lrn_yyy"])
    """
    pipeline = ProfilingPipeline(device=device)
    return pipeline.analyze(learner_uids)


__all__ = ["ProfilingPipeline", "analyze"]
