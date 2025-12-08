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
    
class PartnerSettings:
    """
    学习伙伴 / 学习榜样等“社群编排（partnering & role modeling）”相关配置。

    说明：
    - 这里只放“可调参数”和“集合名”等环境无关常量；
    - 具体算法逻辑由 domain 层的各个 Engine 决定；
    - 针对上万规模学习者，新增 per-target 的候选上限配置，用来控制算法复杂度。
    """

    def __init__(self) -> None:
        # LearnerProfile 集合名，可通过环境变量覆盖
        self._learner_profile_collection: str = os.getenv(
            "PARTNER_LEARNER_PROFILE_COLLECTION",
            "LearnerProfile",
        )

        # ========================= 学习伙伴（learning partner） =========================

        # 学习伙伴默认返回数量
        self._partner_default_top_k: int = int(
            os.getenv("PARTNER_DEFAULT_TOP_K", "5")
        )

        # 候选池规模上限（通常由上层使用 Repository 构建 learner_profiles 时控制）
        self._partner_candidate_pool_limit: int = int(
            os.getenv("PARTNER_CANDIDATE_POOL_LIMIT", "500")
        )

        # 在 Engine 内部，为每个目标学习者最多考虑多少个候选学习者做精细打分
        # 这是应对“上万规模学习者”的关键参数，用于避免 N * 全量 的 O(N^2) 爆炸。
        self._partner_max_candidates_per_target: int = int(
            os.getenv("PARTNER_MAX_CANDIDATES_PER_TARGET", "300")
        )

        # 仅保留最近 N 天更新过画像 / KT 的用户，用于构建候选池；
        # 若设置为 0 或负数，则不做时间过滤（由上层控制）。
        try:
            days = int(os.getenv("PARTNER_MIN_UPDATED_DAYS", "0"))
        except ValueError:
            days = 0
        self._partner_min_updated_days: int = days

        # 多视图融合的权重系数（可作为论文中的超参数）
        # Score(i, j) = α * S_profile + β * S_K_homo + γ * S_K_comp
        self._partner_score_weights: Dict[str, float] = {
            "alpha_profile": float(os.getenv("PARTNER_ALPHA_PROFILE", "0.4")),
            "beta_k_homophily": float(os.getenv("PARTNER_BETA_K_HOMO", "0.3")),
            "gamma_k_complementarity": float(
                os.getenv("PARTNER_GAMMA_K_COMP", "0.3")
            ),
        }

        # “弱-强”互补阈值
        self._partner_knowledge_thresholds: Dict[str, float] = {
            "low": float(os.getenv("PARTNER_LOW_THRESHOLD", "0.6")),
            "high": float(os.getenv("PARTNER_HIGH_THRESHOLD", "0.85")),
        }

        # 画像子维度权重（key 使用 "dimension.sub_key" 形式，Engine 会转换成 (dim, sub_key)）
        base_weight = 1.0
        self._partner_profile_feature_weights: Dict[str, float] = {
            # 社会性学习取向 / 协作角色
            "social_learning.role": 1.2 * base_weight,
            "collaborative_role_contribution.role": 1.2 * base_weight,
            "collaborative_role_contribution.contribution_type": 1.0 * base_weight,
            # 自我调节 / 求助
            "srl_helpseeking.level": 1.1 * base_weight,
            # 参与与坚持
            "engagement_persistence.level": 1.0 * base_weight,
            # 反馈取向
            "feedback_orientation.level": 0.9 * base_weight,
            # 注意与任务效率
            "attention_allocation.efficiency": 0.8 * base_weight,
            "attention_allocation.style": 0.8 * base_weight,
            "task_efficiency.level": 0.8 * base_weight,
            # 探索
            "exploration_orientation.level": 0.7 * base_weight,
            # 交互风格
            "interaction_style.style": 0.7 * base_weight,
            # 反思 / 价值演化
            "reflection_value_evolution.level": 0.6 * base_weight,
        }

        # ========================= 学习榜样（role model） =========================

        # 学习榜样默认返回数量（通常比学习伙伴少）
        self._role_model_default_top_k: int = int(
            os.getenv("ROLE_MODEL_DEFAULT_TOP_K", "3")
        )

        # 学习榜样候选池规模（由上层在构建输入时控制）
        self._role_model_candidate_pool_limit: int = int(
            os.getenv("ROLE_MODEL_CANDIDATE_POOL_LIMIT", "800")
        )

        # 每个目标学习者最多考虑多少个候选榜样做精细打分
        self._role_model_max_candidates_per_target: int = int(
            os.getenv("ROLE_MODEL_MAX_CANDIDATES_PER_TARGET", "300")
        )

        # 只考虑最近 N 天有画像 / KT 更新的学习者作为候选榜样
        try:
            rm_days = int(os.getenv("ROLE_MODEL_MIN_UPDATED_DAYS", "0"))
        except ValueError:
            rm_days = 0
        self._role_model_min_updated_days: int = rm_days

        # 学习榜样匹配的多视图权重
        # Score_rm(i, j) = α * S_profile + β * S_gap + γ * S_K_comp
        self._role_model_score_weights: Dict[str, float] = {
            "alpha_profile": float(os.getenv("ROLE_MODEL_ALPHA_PROFILE", "0.3")),
            "beta_global_advancement": float(
                os.getenv("ROLE_MODEL_BETA_GLOBAL_ADV", "0.4")
            ),
            "gamma_knowledge_complementarity": float(
                os.getenv("ROLE_MODEL_GAMMA_K_COMP", "0.3")
            ),
        }

        # “向上对标”时，理想的全局能力差距窗（单位：预测精度差值）
        self._role_model_gap_window: Dict[str, float] = {
            "min": float(os.getenv("ROLE_MODEL_GAP_MIN", "0.05")),
            "max": float(os.getenv("ROLE_MODEL_GAP_MAX", "0.25")),
        }

        # 学习榜样场景下，对画像子维度的权重
        rm_base_weight = 1.0
        self._role_model_profile_feature_weights: Dict[str, float] = {
            # 高投入与坚持
            "engagement_persistence.level": 1.3 * rm_base_weight,
            "engagement_persistence.pattern": 1.1 * rm_base_weight,
            # 反思与价值演化
            "reflection_value_evolution.level": 1.2 * rm_base_weight,
            # 自我调节与求助
            "srl_helpseeking.level": 1.1 * rm_base_weight,
            # 反馈取向
            "feedback_orientation.level": 1.0 * rm_base_weight,
            # 社会性学习 / 组织角色
            "social_learning.role": 1.1 * rm_base_weight,
            "collaborative_role_contribution.role": 1.1 * rm_base_weight,
            # 注意与任务效率
            "attention_allocation.efficiency": 0.9 * rm_base_weight,
            "task_efficiency.level": 0.9 * rm_base_weight,
        }

    # ----------------------- property 封装 -----------------------

    @property
    def learner_profile_collection(self) -> str:
        return self._learner_profile_collection

    # ----- partner -----

    @property
    def partner_default_top_k(self) -> int:
        return self._partner_default_top_k

    @property
    def partner_candidate_pool_limit(self) -> int:
        return self._partner_candidate_pool_limit

    @property
    def partner_max_candidates_per_target(self) -> int:
        return self._partner_max_candidates_per_target

    @property
    def partner_min_updated_days(self) -> int:
        return self._partner_min_updated_days

    @property
    def partner_score_weights(self) -> Dict[str, float]:
        return dict(self._partner_score_weights)

    @property
    def partner_knowledge_thresholds(self) -> Dict[str, float]:
        return dict(self._partner_knowledge_thresholds)

    @property
    def partner_profile_feature_weights(self) -> Dict[str, float]:
        return dict(self._partner_profile_feature_weights)

    # ----- role model -----

    @property
    def role_model_default_top_k(self) -> int:
        return self._role_model_default_top_k

    @property
    def role_model_candidate_pool_limit(self) -> int:
        return self._role_model_candidate_pool_limit

    @property
    def role_model_max_candidates_per_target(self) -> int:
        return self._role_model_max_candidates_per_target

    @property
    def role_model_min_updated_days(self) -> int:
        return self._role_model_min_updated_days

    @property
    def role_model_score_weights(self) -> Dict[str, float]:
        return dict(self._role_model_score_weights)

    @property
    def role_model_gap_window(self) -> Dict[str, float]:
        return dict(self._role_model_gap_window)

    @property
    def role_model_profile_feature_weights(self) -> Dict[str, float]:
        return dict(self._role_model_profile_feature_weights)


# 全局配置实例（这里是“配置对象”，不会直接产生数据库连接）
path_settings = PathSettings()
db_settings = DatabaseSettings()
orchestration_settings = OrchestrationSettings()
llm_settings = LLMSettings()
kt_settings = KTSettings()
profiling_settings = ProfilingSettings()
partner_settings = PartnerSettings()