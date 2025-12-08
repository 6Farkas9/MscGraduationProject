# app/domain/common/analyze_base_engine.py
# -*- coding: utf-8 -*-
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AnalyzeBaseEngine(ABC):
    """
    AnalyzeBaseEngine

    设计目的
    --------
    作为一类“基于学习者知识能力 & 画像数据”的分析引擎的统一基类，
    与原来的推理类引擎基类（原 BaseEngine，将要改名为 infer_base_engine）
    做职责分离：

    - infer_base_engine:
        * 典型场景：调用深度模型、LLM 或其他“按样本推理”的引擎；
        * 输入多为原始行为序列 / 文本 / 特征矩阵等；
    - AnalyzeBaseEngine（本类）：
        * 典型场景：基于已计算好的 learner_profile + knowledge_concepts
          做“匹配 / 分群 / 推荐 / 诊断”的分析引擎；
        * 输入统一为：
            - learner_uids: List[str]
            - learner_profiles: Dict[str, Any]
            - knowledge_concepts: Dict[str, Any]

    核心约定
    --------
    1. 输入数据约定：

       learner_uids: List[str]
           需要分析的一批学习者 UID。

       learner_profiles: Dict[str, Any]
           以 uid 为键的画像字典（11 个维度等），例如：
               {
                 "uid_1": {
                   "attention_allocation": {...},
                   "social_learning": {...},
                   ...
                 },
                 ...
               }

       knowledge_concepts: Dict[str, Any]
           以 uid 为键的知识点预测精度字典，例如：
               {
                 "uid_1": {
                   "kp_001": 0.92,
                   "kp_002": 0.81,
                   ...
                 },
                 ...
               }

    2. 输出约定：

       - 返回 Dict[str, Any]，由具体引擎自行约定结构；
       - 建议顶层包含：
           {
             "engine_status": {...},
             "results": { ... }
           }
         以便前端/上层统一读取 engine 状态信息。

    3. 生命周期 & 状态：

       - 子类仍需实现 initialize()，用于：
           * 加载必要的映射 / 超参数 / 轻量模型；
           * 做一次简单的自检；
       - analyze() 前统一调用 ensure_initialized()，
         保证初始化只做一次。
    """

    def __init__(self, device: str = "cpu", name: Optional[str] = None) -> None:
        """
        参数
        ----
        device:
            运行设备标志，主要用于保持接口统一（即使多数分析引擎只用 CPU）。
        name:
            可选的引擎名，用于日志中区分不同子类实例。
        """
        self.device: str = device
        self.model: Any = None  # 保持与原 BaseEngine 一致的状态结构
        self.is_initialized: bool = False
        self.engine_name: str = name or self.__class__.__name__

        logger.info(
            "%s 初始化（AnalyzeBaseEngine），设备: %s",
            self.engine_name,
            self.device,
        )

    # ------------------------------------------------------------------
    # 生命周期接口
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> bool:
        """
        子类负责实现的初始化逻辑，例如：
        - 读取配置 / 权重；
        - 构建必要的数据结构（如权重表、映射表）；
        - 进行一次快速自检。

        返回
        ----
        bool:
            True 表示初始化成功，False 表示失败（失败时 analyze 应避免继续执行）。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 统一分析接口
    # ------------------------------------------------------------------

    @abstractmethod
    def analyze(
        self,
        learner_uids: List[str],
        learner_profiles: Dict[str, Any],
        knowledge_concepts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        核心对外接口：基于画像 + 知识能力进行分析 / 推荐 / 匹配。

        参数
        ----
        learner_uids:
            需要进行分析的一批学习者 UID 列表。
        learner_profiles:
            uid -> 画像字典（11 维画像标签等）。
        knowledge_concepts:
            uid -> 知识点预测精度字典。

        返回
        ----
        Dict[str, Any]:
            分析结果，由具体子类定义结构，但推荐至少包含：
            {
              "engine_status": {...},
              "results": {...}   # 例如 uid -> { ... }
            }
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 通用辅助方法
    # ------------------------------------------------------------------

    def ensure_initialized(self) -> bool:
        """
        在调用 analyze 之前确保引擎已经初始化。

        - 如果尚未初始化，则调用 initialize()；
        - 若初始化失败，记录 error 并返回 False。
        """
        if not self.is_initialized:
            ok = self.initialize()
            if not ok:
                logger.error(
                    "%s 初始化失败，后续 analyze 将返回空结果。",
                    self.engine_name,
                )
                return False
        return True

    def get_engine_status(self) -> Dict[str, Any]:
        """
        返回引擎当前状态信息，供上层调试 & 监控使用。

        字段说明：
        - initialized: 是否已完成初始化；
        - device: 当前声明的运行设备；
        - model_loaded: 是否已加载模型（对于多数分析引擎，可以始终为 False）。
        """
        return {
            "initialized": self.is_initialized,
            "device": self.device,
            "model_loaded": self.model is not None,
            "engine_name": self.engine_name,
        }

    # ------------------------------------------------------------------
    # 输入检查（可选辅助）
    # ------------------------------------------------------------------

    def validate_inputs(
        self,
        learner_uids: List[str],
        learner_profiles: Dict[str, Any],
        knowledge_concepts: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        """
        对传入的 uid / 画像 / 知识向量做一次轻量的完整性检查。

        返回
        ----
        Tuple[List[str], List[str]]:
            (缺画像的 uid 列表, 缺知识向量的 uid 列表)

        说明
        ----
        - 此方法不会抛异常，仅用于记录 warning；
        - 子类可选择性调用，用于 debug 或监控数据质量。
        """
        missing_profile: List[str] = []
        missing_kt: List[str] = []

        for uid in learner_uids:
            if uid not in learner_profiles:
                missing_profile.append(uid)
            if uid not in knowledge_concepts:
                missing_kt.append(uid)

        if missing_profile:
            logger.warning(
                "%s: 部分 learner_uid 在 learner_profiles 中缺失，count=%d, sample=%s",
                self.engine_name,
                len(missing_profile),
                missing_profile[:5],
            )

        if missing_kt:
            logger.warning(
                "%s: 部分 learner_uid 在 knowledge_concepts 中缺失，count=%d, sample=%s",
                self.engine_name,
                len(missing_kt),
                missing_kt[:5],
            )

        return missing_profile, missing_kt
