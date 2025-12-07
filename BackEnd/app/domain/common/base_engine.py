# BackEnd/app/domain/common/base_engine.py
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BaseEngine(ABC):
    """
    引擎基类：统一 device / model / 初始化流程 / 状态查询接口
    """

    def __init__(self, device: str):
        self.device = device
        self.model = None
        self.is_initialized: bool = False
        logger.info("%s 初始化，设备: %s", self.__class__.__name__, self.device)

    @abstractmethod
    def initialize(self) -> bool:
        """
        子类负责实现的初始化逻辑：
        - 加载必要的映射
        - 构建模型
        - 加载权重
        - 做一次简单 self-check
        """
        raise NotImplementedError

    @abstractmethod
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        统一对外接口：
        - 仅接受 List[str] 类型的学习者 UID 列表
        - 返回各自的预测结果（格式由具体引擎定义）
        """
        raise NotImplementedError

    # --- 通用辅助方法 ---

    def ensure_initialized(self) -> bool:
        """
        如果尚未初始化则调用 initialize，避免在各个计算入口重复写逻辑。
        """
        if not self.is_initialized:
            return self.initialize()
        return True

    def get_engine_status(self) -> Dict[str, Any]:
        """
        通用状态信息，子类可以在此基础上扩展。
        """
        return {
            "initialized": self.is_initialized,
            "device": self.device,
            "model_loaded": self.model is not None,
        }
