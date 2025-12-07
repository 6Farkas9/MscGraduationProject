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

# 全量标签配置
PROFILE_LABELS: Dict[str, Dict[str, Dict[int, str]]] = {
    "attention_allocation": {
        # 注意效率（聚类得到 3 档标签）
        # 0: 低效；1: 中等；2: 高效
        "efficiency": {
            0: "低效注意策略（任务表现较低、任务相关注意比例较低且在非任务 UI 区域停留较多）",
            1: "中等注意策略（任务相关注意与表现处于中间水平）",
            2: "高效注意策略（在关键资源上集中注意、较少停留在无关 UI，且表现较好）",
        },
        # 信息加工风格（AOI 分布与首注视）
        # 0: 文本优先；1: 图像/模型优先；2: 示例/演示优先
        # 3: 均衡整合；4: 未定义/数据不足
        "style": {
            0: "文本优先型加工（进入或整体上更偏向文字信息）",
            1: "图像/模型优先型加工（进入或整体上更偏向图像/3D 模型）",
            2: "示例/演示优先型加工（更偏向提示、示例或同伴演示）",
            3: "均衡整合型加工（在文本/图像/示例之间较为均衡地分配注意）",
            4: "加工风格未明（数据不足或注意非常分散）",
        },
    },

    "collaborative_role_contribution": {
        # 协作角色（会话级 → 课程级 → 全局）
        # code 定义（从“弱”到“强”，方便整体标签并列时按 code 最大者作为“更好”）：
        #   0: 无协作数据
        #   1: 观察者
        #   2: 一般协作者
        #   3: 执行者
        #   4: 协调者
        #   5: 核心贡献者
        "role": {
            0: "无协作数据",
            1: "观察者（贡献与参与份额都较低，偶尔出现于协作会话）",
            2: "一般协作者（在组内有一定参与与贡献，但不突出）",
            3: "执行者（参与份额高，主要负责完成协作任务流程）",
            4: "协调者（互动频繁，善于组织、衔接与协调组内成员）",
            5: "核心贡献者（贡献份额高、参与份额也高，是关键产出者）",
        },

        # 贡献类型（会话内基于 create/update/delete/resource/discussion 的主导形式）
        # code 定义（同样从“弱”到“强/更典型”）：
        #   0: 无协作数据
        #   1: 无有效贡献
        #   2: 讨论参与型
        #   3: 资源提供型
        #   4: 修改完善型
        #   5: 内容创作型
        "contribution_type": {
            0: "无协作数据",
            1: "无有效贡献（几乎没有编辑、资源提交或讨论行为）",
            2: "讨论参与型（以讨论/响应为主，直接产出较少）",
            3: "资源提供型（偏向文件/链接/笔记等资源提交）",
            4: "修改完善型（以 update/delete 等编辑完善现有成果为主）",
            5: "内容创作型（以 create 编辑为主，负责主要内容产出）",
        },
    },

    "contribution_reputation": {
        # 价值贡献水平：对应聚类结果 cluster_rank（0: 低, 1: 中, 2: 高）
        "level": {
            0: "低价值贡献型（在课程中几乎没有价值 token 流入，贡献行为也较少）",
            1: "中等价值贡献型（在课程中具有一定价值 token 流入与贡献行为）",
            2: "高价值贡献 & 高声望型（在课程中获得较多价值 token 奖励并频繁贡献）",
        },

        # 价值贡献风格：根据资源 vs 协作贡献的构成（由引擎计算出的 style_code）
        # code 约定：
        #   0: 未定义/数据不足
        #   1: 平衡型（协作与资源贡献相对均衡）
        #   2: 协作驱动型（共同编辑/协作活动占比较高）
        #   3: 资源驱动型（资源上传/分享占比较高）
        "style": {
            0: "贡献风格未明（数据不足或贡献行为极少）",
            1: "贡献风格平衡型（在协作与资源分享之间较为均衡）",
            2: "协作驱动型贡献者（主要通过共同编辑或协作活动进行价值贡献）",
            3: "资源驱动型贡献者（主要通过上传/分享资源进行价值贡献）",
        },
    },

    "engagement_persistence": {
        # 行为投入度与坚持性水平：对应 EP_norm 聚类后的三档等级（0: 低, 1: 中, 2: 高）
        "level": {
            0: "低投入易放弃型学习者（完成率、交互量和重试/额外练习整体偏弱）",
            1: "中等投入型学习者（整体上能够完成任务并保持一定参与度）",
            2: "高投入高坚持型学习者（完成率较高，且愿意重试并进行额外练习）",
        },
    },

    "feedback_orientation": {
        # 行为侧“反馈敏感度与数据使用能力”三档水平标签：
        # 对应引擎中基于 FO_norm 聚类后的 cluster_rank（0: 低, 1: 中, 2: 高）
        "level": {
            0: "低反馈敏感/低数据使用型（几乎不查看反馈或不使用解析；反馈后正确率提升不明显）",
            1: "中等反馈敏感/一般数据使用型（偶尔查看反馈；会在部分场景使用解析/示例）",
            2: "高反馈敏感/高数据使用型（频繁查看反馈面板/进度板；积极用解析并能调整策略）",
        },
    },

    "interaction_style": {
        # 交互与操作熟练度 / 风格三档水平：
        # 对应 engine 中基于 style_index / (x, y) 聚类后的 cluster_rank（0: 低, 1: 中, 2: 高）
        "style": {
            0: "随便乱点型（操作频率高但成功率低，存在较多无效/误操作）",
            1: "多试多练型（操作频率较高，通过反复尝试逐步掌握）",
            2: "少操作但准确型（操作次数较少但步骤和任务成功率较高）",
        },
    },

    "reflection_value_evolution": {
        # 反思深度与价值观演变三档水平，
        # 对应 engine 中的 reflection_level / label_code（0 = 浅层/不稳定，1 = 稳定深度，2 = 成长型）
        "level": {
            0: "浅层或不稳定反思者（反思频率较低或文本多为描述性，价值相关语汇使用有限/变化不稳定）",
            1: "稳定深度反思者（反思深度较高，但价值相关语汇变化有限或方向较稳定）",
            2: "成长型价值反思者（反思频率与深度较高，且围绕元宇宙/学习价值的语汇在后期明显增强）",
        },
    },

    "srl_helpseeking": {
        # 自我调节与求助策略三档水平：
        # 对应引擎中基于 SRL_help_index 聚类后的 cluster_rank / label_code（0=低, 1=中, 2=高）
        "level": {
            0: "自我调节与求助策略水平偏低（在需要时较少求助或求助模式不够合理，反馈/补救/反思使用有限）",
            1: "自我调节与求助策略水平中等（在部分困难情境下能适度求助，并有一定程度的反馈与补救使用）",
            2: "自我调节与求助策略水平较高（能在遇到困难时适度求助，同时主动使用反馈、补救资源与反思工具）",
        },
    },

    "social_learning": {
        # 社会性学习与同伴取向（Social Learning & Peer Orientation）
        # 对应 analyze_social_learning.py 中基于 obs_total_time / collab_total_time
        # + social_index_normalized 计算得到的 social_label / cluster_rank：
        #   cluster_rank = 0: low_social_participation（低社交参与型）
        #   cluster_rank = 1: observer_dominant（观察型，以观摩为主）
        #   cluster_rank = 2: collab_dominant（协作导向型，以协作为主）
        #   cluster_rank = 3: balanced_active_social（积极社会学习型，观摩+协作均衡且总体较高）
        "role": {
            0: "低社交参与型（在该课程中很少通过观摩同伴或协作来参与学习）",
            1: "观察型（以观摩同伴作品/表现为主，协作参与较少）",
            2: "协作导向型（协作时长和协作次数明显多于观摩，多通过共同编辑/协作完成任务）",
            3: "积极社会学习型（既大量观摩同伴，也积极参与协作，观摩与协作较为均衡）",
        },
    },

    "exploration_orientation": {
        # 空间与资源探索倾向（Spatial & Resource Exploration Orientation）
        # 对应 analyze_spatial_exploration_orientation.py 中的 cluster_rank：
        #   0 = 到点即学型（低探索）
        #   1 = 平衡探索型（中等探索）
        #   2 = 高探索型探索者
        # LearnerProfile 中对应 global_profile.exploration_orientation.score
        "level": {
            0: "到点即学型（低探索）",
            1: "平衡探索型（中等探索）",
            2: "高探索型探索者",
        },
    },

    "task_efficiency": {
        # 任务效率与认知负荷代理（Task Efficiency & Cognitive Load Proxy）
        # 对应 analyze_task_efficiency.py 中基于 E_norm 聚类得到的 cluster_rank：
        #   0 = 低效率型
        #   1 = 中等效率型
        #   2 = 高效率型
        # LearnerProfile 中对应 global_profile.task_efficiency.score
        "level": {
            0: "低效率型学习者（在本课程中任务成功率相对较低、耗时相对较长 / 认知效率指数较低）",
            1: "中等效率型学习者（在本课程中任务成功率与耗时均处于中间水平）",
            2: "高效率型学习者（在本课程中任务成功率相对较高、耗时相对较短 / 认知效率指数较高）",
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
