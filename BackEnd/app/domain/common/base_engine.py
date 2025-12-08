# app/domain/common/base_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """
    所有领域 Engine 的基础抽象类。

    统一约定
    --------
    - 所有 Engine 至少接收「一组学习者 uid」作为分析入口；
    - 对于需要额外特征（如画像、知识点预测）的 Engine，可通过 `data` 传入；
    - `analyze` 统一签名：
        analyze(self, learner_uids: List[str], data: Dict[str, Any] | None = None) -> Dict[str, Any]
    """

    def __init__(self, device: Optional[str] = None, name: Optional[str] = None) -> None:
        """
        参数
        ----
        device:
            运行设备，例如 "cpu" / "cuda" 等，可由子类自由解释；
        name:
            Engine 名称，若不指定则使用类名。
        """
        self.device: str = device or "cpu"
        self.engine_name: str = name or self.__class__.__name__
        self.is_initialized: bool = False

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> bool:
        """
        子类实现：完成模型加载 / 参数初始化等操作。
        返回 True 表示初始化成功。
        """
        raise NotImplementedError

    def ensure_initialized(self) -> bool:
        """
        外部调用前的安全入口：若尚未初始化，则调用 initialize。
        """
        if not self.is_initialized:
            try:
                ok = self.initialize()
                self.is_initialized = bool(ok)
            except Exception as exc:  # 防御性处理，避免异常向外抛
                logger.error("%s.initialize() failed: %s", self.engine_name, exc)
                self.is_initialized = False
        return self.is_initialized

    # ------------------------------------------------------------------
    # 核心分析接口（统一签名）
    # ------------------------------------------------------------------

    @abstractmethod
    def analyze(
        self,
        learner_uids: List[str],
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        核心分析接口（统一签名）。

        参数
        ----
        learner_uids:
            需要进行分析/推荐的目标学习者 uid 列表；
        data:
            可选的附加数据容器，由具体 Engine 自行约定结构。
            对于不需要额外输入的 Engine，可以忽略该参数。

        返回
        ----
        Dict[str, Any]:
            分析结果，通常为:
                {
                  "engine_status": {...},
                  "results": {...}
                }
            具体结构由子类定义。
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 状态查询 / 工具方法
    # ------------------------------------------------------------------

    def get_engine_status(self) -> Dict[str, Any]:
        """
        返回 Engine 当前状态，便于 pipeline / 调试使用。
        """
        return {
            "name": self.engine_name,
            "device": self.device,
            "is_initialized": self.is_initialized,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.engine_name}, device={self.device}, initialized={self.is_initialized})>"
