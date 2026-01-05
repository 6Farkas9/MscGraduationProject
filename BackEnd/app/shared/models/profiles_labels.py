# app/models/profiles_labels.py
# -*- coding: utf-8 -*-
"""
统一的人才画像标签映射配置文件（profiles_labels）

设计目标
--------
1. 为各个画像维度的引擎提供「数值标签 → 文本描述」的统一配置入口；
2. 引擎内部只返回数值型标签（code），不直接耦合到具体文案；
3. 其它模块在需要展示时，通过本文件把 code 转成可读文本；
4. 方便后续为不同维度 / 引擎新增或调整标签，而无需改动引擎逻辑。

结构约定
--------
PROFILE_LABELS = {
    "<dimension_key>": {
        "<label_category>": {
            <code:int>: "<人类可读的标签文本>",
            ...
        },
        ...
    },
    ...
}
"""

from typing import Dict, Optional

# 全量服务表征语义配置（去画像化版本）
PROFILE_LABELS: Dict[str, Dict[str, Dict[int, str]]] = {

    "attention_allocation": {
        "efficiency": {
            0: "需强化关键任务与核心资源的注意引导服务",
            1: "可维持基础注意引导并适度优化呈现结构",
            2: "可精简引导提示并提高服务推进节奏",
        },
        "style": {
            0: "应以文本说明类服务为主要引导方式",
            1: "应强化图像或模型类可视化服务支持",
            2: "应提供演示与操作引导类服务", 
            3: "可采用多模态混合服务呈现策略",
            4: "应优先提供基础引导以明确服务响应特征",
        },
    },

    "collaborative_role_contribution": {
        "role": {
            0: "当前情境下无需主动触发协作类服务",
            1: "可维持低频协作支持策略",
            2: "可按需提供基础协作支持服务",
            3: "应侧重协作流程执行与任务分工支持",
            4: "应侧重协作协调与组织支持服务",
            5: "应优先保障协作过程与结果质量",
        },

        "contribution_type": {
            0: "当前情境下无需触发协作贡献类服务",
            1: "可维持低强度协作贡献支持策略",
            2: "应强化讨论与反馈类协作服务",
            3: "应强化资源提交与共享类服务",
            4: "应提供成果优化与修改引导服务",
            5: "应提供内容生成与创作支持服务",
        },
    },

    "contribution_reputation": {
        "level": {
            0: "应增强激励与价值反馈类服务支持",
            1: "可维持常规激励与反馈服务策略",
            2: "可引入长期协作与高价值激励服务",
        },

        "style": {
            0: "应采用基础价值反馈与激励策略",
            1: "可综合采用多类型价值反馈服务",
            2: "应侧重协作成果认可与激励服务",
            3: "应侧重资源贡献价值反馈服务",
        },
    },

    "engagement_persistence": {
        "level": {
            0: "应放缓服务节奏并加强阶段性引导",
            1: "可维持常规服务触发与支持策略",
            2: "可采用延展型与递进式服务策略",
        },
    },

    "feedback_orientation": {
        "level": {
            0: "应增强反馈服务的可见性与使用引导",
            1: "可在关键节点选择性触发反馈服务",
            2: "应频繁提供解析与策略性反馈支持",
        },
    },

    "interaction_style": {
        "style": {
            0: "应简化交互路径并强化操作引导服务",
            1: "可提供探索式操作支持与适应性引导",
            2: "可减少干预并提高服务执行效率",
        },
    },

    "reflection_value_evolution": {
        "level": {
            0: "应提供基础反思提示与价值引导服务",
            1: "可维持周期性反思与总结类服务",
            2: "可引入深度反思与价值导向服务",
        },
    },

    "srl_helpseeking": {
        "level": {
            0: "可维持被动式支持与求助服务策略",
            1: "可在关键节点提供辅助与支持服务",
            2: "应主动提供支持与补救类服务",
        },
    },

    "social_learning": {
        "role": {
            0: "可减少社会交互与关系类服务触发",
            1: "应侧重榜样示例与观摩支持服务",
            2: "应侧重同伴互动与协作支持服务",
            3: "可综合采用观摩与协作类服务策略",
        },
    },

    "exploration_orientation": {
        "level": {
            0: "可采用结构化、线性服务组织策略",
            1: "可在结构与探索之间提供有限调节",
            2: "应放宽路径限制并扩展探索支持服务",
        },
    },

    "task_efficiency": {
        "level": {
            0: "应放缓任务节奏并加强支撑性服务",
            1: "可维持当前任务节奏与服务强度",
            2: "可提高任务挑战强度与推进速度",
        },
    }
}




def get_label(
    dimension: str, label_category: str, code: Optional[int]
) -> Optional[str]:
    """
    获取某个画像维度下、某个标签类别的文本描述。

    参数
    ----
    dimension: 画像维度 key，例如 "attention_allocation"
    label_category: 标签类别，例如 "efficiency" / "style" / "role"
    code: 数值标签（引擎返回的 int），None 则直接返回 None

    返回
    ----
    若配置存在，返回对应的文本标签；否则返回 None。
    """
    if code is None:
        return None

    dim_cfg = PROFILE_LABELS.get(dimension)
    if not dim_cfg:
        return None

    cat_cfg = dim_cfg.get(label_category)
    if not cat_cfg:
        return None

    return cat_cfg.get(code)


def get_all_labels_for_category(
    dimension: str, label_category: str
) -> Dict[int, str]:
    """
    获取某个维度下某个标签类别的全部 code → 文本 映射。

    方便前端一次性加载并缓存所有标签。
    """
    dim_cfg = PROFILE_LABELS.get(dimension, {})
    return dim_cfg.get(label_category, {}).copy()
