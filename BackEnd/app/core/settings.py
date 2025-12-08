# BackEnd/app/core/settings.py
import os
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


class PathSettings:
    """
    路径配置：
    - project_root: 包含 BackEnd 与 DeepLearning 的父目录
    - backend_dir: BackEnd 目录
    - deep_learning_dir: DeepLearning 目录
    """

    def __init__(self) -> None:
        # 当前文件所在目录：BackEnd/app/core/
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # BackEnd 目录：core 的上两级
        backend_dir = os.path.dirname(os.path.dirname(current_dir))
        project_root = os.path.dirname(backend_dir)

        self._project_root = project_root
        self._backend_dir = backend_dir
        self._deep_learning_dir = os.path.join(project_root, "DeepLearning")

        # 将 project_root 加入 sys.path，方便直接导入 DeepLearning 下模块
        if self._project_root not in sys.path:
            sys.path.insert(0, self._project_root)

    @property
    def project_root(self) -> str:
        return self._project_root

    @property
    def backend_dir(self) -> str:
        return self._backend_dir

    @property
    def deep_learning_dir(self) -> str:
        return self._deep_learning_dir


class DatabaseSettings:
    """
    数据库配置：
    - MySQL 配置
    - MongoDB 配置
    """

    def __init__(self) -> None:
        # 实际使用时建议从环境变量或配置文件中读取
        self._mysql_config: Dict[str, Any] = {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "123456",
            "database": "mls_sample",
            "charset": "utf8mb4",
            "autocommit": True,
        }

        self._mongodb_config: Dict[str, Any] = {
            "host": "localhost",
            "port": 27017,
            "database": "MLS",
            "username": None,
            "password": None,
            "auth_source": "admin",
        }

    @property
    def mysql_config(self) -> Dict[str, Any]:
        # 返回副本防止外部修改
        return dict(self._mysql_config)

    @property
    def mongodb_config(self) -> Dict[str, Any]:
        return dict(self._mongodb_config)

    def build_mongodb_uri(self) -> str:
        cfg = self._mongodb_config
        if cfg["username"] and cfg["password"]:
            return (
                f"mongodb://{cfg['username']}:{cfg['password']}"
                f"@{cfg['host']}:{cfg['port']}/{cfg['database']}?authSource={cfg['auth_source']}"
            )
        return f"mongodb://{cfg['host']}:{cfg['port']}/{cfg['database']}"
    
class OrchestrationSettings:
    """
    资源编排 / 检索（HR-PRR）相关配置：

    说明：
    - 资源分段集合名 FRAGMENTS_COLLECTION 在 Repository 模块中硬编码
    - 语义模型名称 SEM_MODEL_NAME 在 Engine 模块中硬编码
    - 此处只保留便于调优的数值型参数
    """

    def __init__(self) -> None:
        # 原脚本中的 DEFAULT_TOPK
        self._default_top_k: int = 20

        # 原脚本中的 MAX_CANDIDATES
        self._max_candidates: int = 2000  # 每阶段最多拉这么多候选

        # 原脚本中 score_overall 的默认权重 alpha/beta/gamma/delta
        self._score_weights: Dict[str, float] = {
            "alpha": 0.35,
            "beta": 0.15,
            "gamma": 0.25,
            "delta": 0.25,
        }

    @property
    def default_top_k(self) -> int:
        return self._default_top_k

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

    @property
    def score_weights(self) -> Dict[str, float]:
        # 返回副本，避免外部直接修改内部 dict
        return dict(self._score_weights)
    
@dataclass(frozen=True)
class LLMProviderConfig:
    """
    单个 LLM Provider 配置（OpenAI 兼容 / Aizex 路由等）
    """
    base_url: str
    api_key: str
    default_model: str


class LLMSettings:
    """
    支持“多 provider + 运行时选择 model”的通用大模型配置。

    你可以用环境变量配置多个 provider，例如：
      LLM_PROVIDER=aizex
      LLM_AIZEX_BASE_URL=https://aizex.top/v1
      LLM_AIZEX_API_KEY=sk-xxx
      LLM_AIZEX_DEFAULT_MODEL=gpt-4.1-nano

    如需第二个 provider：
      LLM_PROVIDER=openai
      LLM_OPENAI_BASE_URL=https://api.openai.com/v1
      LLM_OPENAI_API_KEY=sk-yyy
      LLM_OPENAI_DEFAULT_MODEL=gpt-4o-mini
    """

    def __init__(self) -> None:
        self._default_provider: str = os.getenv("LLM_PROVIDER", "aizex").lower()

        self._providers: Dict[str, LLMProviderConfig] = {
            "aizex": LLMProviderConfig(
                base_url=os.getenv("LLM_AIZEX_BASE_URL", "https://aizex.top/v1"),
                api_key=os.getenv("LLM_AIZEX_API_KEY", os.getenv("LLM_API_KEY", "sk-TyyUwhwE1LpVBr24swMwKWhWq9oAdosQGYjI5qumbC6DsDoa")),
                default_model=os.getenv("LLM_AIZEX_DEFAULT_MODEL", os.getenv("LLM_MODEL", "gpt-4.1-nano")),
            )
        }

        # 通用推理参数
        self._temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self._max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        self._timeout_sec: int = int(os.getenv("LLM_TIMEOUT_SEC", "60"))

        # 是否显式禁用代理（与你 test_llm_api.py 类似）
        self._disable_proxies: bool = os.getenv("LLM_DISABLE_PROXIES", "1") == "1"

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def timeout_sec(self) -> int:
        return self._timeout_sec

    @property
    def proxies(self) -> Optional[Dict[str, Optional[str]]]:
        if self._disable_proxies:
            return {"http": None, "https": None}
        return None

    def get_provider(self, provider: Optional[str] = None) -> LLMProviderConfig:
        """
        获取 provider 配置：
        - provider=None -> 使用默认 provider
        - provider 指定但不存在 -> 抛异常
        """
        name = (provider or self._default_provider).lower()
        if name not in self._providers:
            raise ValueError(f"未知 LLM provider: {name}, 可选: {list(self._providers.keys())}")
        return self._providers[name]

    def resolve_model(self, provider: Optional[str] = None, model: Optional[str] = None) -> str:
        """
        运行时模型选择：
        - 若传入 model 则优先使用
        - 否则用 provider.default_model
        """
        cfg = self.get_provider(provider)
        return model or cfg.default_model
    
class KTSettings:
    """
    KT 推理相关配置：

    - history_steps: 返回“最后 K 个有效时间步”的能力结果个数
    - 默认值 10，可通过环境变量 KT_HISTORY_STEPS 覆盖
    """

    def __init__(self) -> None:
        self._history_steps: int = int(os.getenv("KT_HISTORY_STEPS", "10"))

    @property
    def history_steps(self) -> int:
        return self._history_steps
    
class ProfilingSettings:
    """
    学习者画像流水线（Profiling Pipeline）相关配置。

    当前配置项：
    - enabled_dimensions: 启用的画像维度列表（按 DIMENSION_KEY 填写）；
    - default_device: 各个画像 Engine 默认运行设备（如 "cpu" / "cuda"）；
    - max_batch_size: 单次 analyze 推荐的最大 learner 数量，超过会打 warning，
                      但不会强制报错。
    """

    def __init__(self) -> None:
        # 默认启用的所有画像维度（需与各 Engine 的 DIMENSION_KEY 保持一致）
        self._enabled_dimensions: List[str] = [
            "attention_allocation",
            "engagement_persistence",
            "feedback_orientation",
            "collaborative_role_contribution",
            "contribution_reputation",
            "interaction_style",
            "reflection_value_evolution",
            "social_learning",
            "exploration_orientation",
            "srl_helpseeking",
            "task_efficiency",
        ]

        # 可通过环境变量覆盖，便于部署时切换到 "cuda" 等
        self._default_device: str = os.getenv("PROFILING_DEVICE", "cpu")

        # 单批建议最大 learner 数，可通过环境变量覆盖（为 0 或负数时视为不限制）
        max_batch_env = os.getenv("PROFILING_MAX_BATCH_SIZE", "200")
        try:
            max_batch = int(max_batch_env)
        except ValueError:
            max_batch = 200
        self._max_batch_size: Optional[int] = max_batch if max_batch > 0 else None

    # ----------------------- property 封装 -----------------------
    @property
    def enabled_dimensions(self) -> List[str]:
        return list(self._enabled_dimensions)

    @property
    def default_device(self) -> str:
        return self._default_device

    @property
    def max_batch_size(self) -> Optional[int]:
        return self._max_batch_size

# 全局配置实例（这里是“配置对象”，不会直接产生数据库连接）
path_settings = PathSettings()
db_settings = DatabaseSettings()
orchestration_settings = OrchestrationSettings()
llm_settings = LLMSettings()
kt_settings = KTSettings()
profiling_settings = ProfilingSettings()
