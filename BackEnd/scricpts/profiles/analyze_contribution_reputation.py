# -*- coding: utf-8 -*-
"""
分析维度：元宇宙价值贡献与声望（Metaverse Value Contribution & Reputation）

脚本目标（与画像设计文档对齐）：
--------------------------------------------------
对应《论文.txt》中“3. 画像设计”里关于“元宇宙价值贡献与声望”维度的设定：
- 在具有代币 / 积分 / 声望机制的教育元宇宙中，学习者通过参与活动、贡献资源、
  帮助他人等产生“价值流”，构成其在元宇宙中的声望与地位。:contentReference[oaicite:0]{index=0}

本脚本的“价值贡献指数”从行为侧刻画：
- 学习者在某门课程中的价值 token 流入量（被系统/他人奖励的价值）；
- 以及与这些价值流紧密相关的贡献行为（资源贡献、协作编辑、协同活动）。

一、行为代理与论文依据：
--------------------------------------------------
1）“价值交换记录 → 区分高贡献 / 低参与”的思路（Hsu et al. 2023：LEARNER-C）
   - 论文列表第 11 篇 Hsu et al. (2023) 在教育元宇宙课程中，引入基于区块链的
     价值交换系统，记录学生在元宇宙环境中参与、贡献、获得认可的 token 流水， 
     并据此区分“高贡献 / 高参与”与“低参与”学习者。:contentReference[oaicite:1]{index=1}
   - 原文强调“价值交换记录本身就是一种 LA 数据源”，可以在不依赖额外问卷的
     情况下用 token 流入量来刻画贡献与声望。:contentReference[oaicite:2]{index=2}
   → 本脚本直接把 xAPI 中的 exchanged-value 动词视为这类“价值交换日志”的实现：
      - context.extensions["https://legend-meta.com/xapi/ext/value-change"] 记录每次
        价值变动（正值为获得 token，负值为支出），用“价值流入总量”作为贡献值核心指标。

2）“协作贡献行为 → 个人贡献度与协作角色”的思路（Student-Facing LAD for Collaborative VR）
   - 论文列表中“Student-Facing Learning Analytics Dashboard for Collaborative Virtual Reality
     Content Creation”（简称 VR 协作 LAD）在协作创建 VR 内容的研究中，记录：
       * 个人编辑次数、修改量、提交资源数；
       * 谁在什么时间段与谁共同编辑同一对象，从而间接构建协作网络。:contentReference[oaicite:3]{index=3}
   - 他们据此定义“发起者、执行者、补充者、观察者”等协作角色，本质上就是
     按“贡献轨迹”与协作行为将学习者分型。:contentReference[oaicite:4]{index=4}
   → 本脚本在 xAPI 中选取以下动词作为“协作/资源贡献行为”的代理：
      - contributed-resource：向公共空间上传或分享资源；
      - co-edited-artifact：共同编辑某个作品或场景；
      - collaborated-on-activity：参与协作活动（如小组任务、共同关卡）。
     这些行为的数量与结构共同支撑“贡献度”这一维度。

3）“积分 / 成就等游戏化指标 → 学习价值与动机”的思路（Papamitsiou 2024/2025 Gamified LA）
   - Papamitsiou (2024, 2025) 将学习分析设计成嵌入 VR 游戏中的游戏化元素（积分、
     成就、排行榜），并在后续实证研究中证明这些 Gamified Learning Analytics（GLA）
     会影响“感知学习价值、愉悦度与继续使用意向”。:contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
   - 这两篇工作都把“游戏内积分 / 奖励”视为具有教育意义的 LA 指标，而非简单的
     娱乐分数。
   → 因此，本脚本将 exchanged-value 产生的 token 视为一种“教育化积分”，
     通过归一化后与贡献行为合成一个“价值贡献指数”，而不是只做简单计数。

二、分析方法概述：
--------------------------------------------------
1）数据来源：
   - 细粒度 xAPI 行为：MongoDB 集合 MLS.Interaction
     * 重点动词：
       - exchanged-value
       - contributed-resource
       - co-edited-artifact
       - collaborated-on-activity
   - 学习者画像：MongoDB 集合 MLS.LearnerProfile
     * 关注字段：global_profile.value_contribution.score
       （人设脚本中基于粗粒度统计/规则预先给出的“价值贡献”分数）

2）按课程的价值贡献指数构建：
   - 粒度：以 (学习者, 课程) 为一个分析单元，和任务效率脚本保持一致。
   - 对每个 (学习者, 课程)，从 xAPI 中统计：
       * token_gain：exchanged-value 事件中 value-change > 0 的累计和
         —— 对应 LEARNER-C 中“获得的价值 token 总量”，代表他在元宇宙中被认可的价值。
       * token_cost：value-change < 0 的绝对值累计（支出），本脚本主要用于补充理解。
       * token_net：token_gain - token_cost（净值），用于存档与后续分析。
       * contrib_counts：
         - resource_contrib_count（contributed-resource 次数）
         - coedit_count（co-edited-artifact 次数）
         - collab_count（collaborated-on-activity 次数）
         - contrib_total = 三者之和
   - 参考 VR 协作 LAD 对“个人贡献度”的处理方式，把“贡献次数”视为体现协作角色与
     个人贡献的关键指标。:contentReference[oaicite:7]{index=7}
   - 参考 LEARNER-C 对“价值交换日志”的处理方式，把“价值 token 流入量”视为
     contribution 的价值权重。:contentReference[oaicite:8]{index=8}

3）课程内标准化与合成指数：
   - 对每门课程，分别对所有 (学习者, 课程) 的：
       * token_gain
       * contrib_total
     计算均值与标准差，得到 z-token 与 z-contrib：
       z_token  = (token_gain_i  - mean_token) / std_token
       z_contrib = (contrib_i    - mean_contrib) / std_contrib
   - 若某个维度标准差为 0，则该维度的 z 值视为 0（所有人相同不区分）。
   - 参考 VR 协作 LAD 中“多指标综合刻画个人贡献度”的思路，本脚本定义
     “价值贡献原始指数”：
       C_i = (z_token + z_contrib) / sqrt(2)
     其中 sqrt(2) 仅用于保持数值尺度稳定。
   - 再将所有课程的 C_i 在全局范围内做 min-max 归一化到 [0, 1]，得到：
       C_norm ∈ [0, 1]
     作为最终用于聚类和人设对比的“价值贡献指数”。

4）基于行为模式进行聚类画像：
   - 参考 Lampropoulos & Evangelidis (2025) 和 Heinemann et al. 对“用行为模式聚类画像”
     的论述，本脚本使用一维 k-means（k=3）对 C_norm 进行聚类，自动划分为三种类型：:contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}
       * 低价值贡献型
       * 中等价值贡献型
       * 高价值贡献 & 高声望型
   - 这种做法延续了任务效率脚本中“以行为指标为基础的三档划分”，便于与整体画像框架对齐。

5）标签设计与价值来源区分：
   - 在聚类得到的贡献水平基础上，本脚本进一步根据贡献行为构成区分价值来源：
       * 任务 / 资源贡献为主：
         - contributed-resource 的比例更高，co-edited / collaborated 相对较少。
       * 协作型贡献为主：
         - co-edited-artifact 与 collaborated-on-activity 占比更高。
   - 最终标签示例：
       * “低价值贡献型学习者（在本课程中几乎没有价值 token 流入与贡献行为）”
       * “高价值贡献 & 高声望型学习者（在本课程中获得大量 token 奖励并频繁进行资源共享与协作），
          且偏协作驱动型”
   - 这种区分参考了 VR 协作 LAD 中“按编辑轨迹与协作行为划分角色”的思路，强调贡献来源
     的不同。:contentReference[oaicite:11]{index=11}

6）与人设的对比验证：
   - 对每个学习者，将其在所有课程上的 C_norm 取平均，得到：
       global_value_contribution（行为侧的全局价值贡献指数）
   - 与 LearnerProfile.global_profile.value_contribution.score 做皮尔逊相关：
       * 若相关为正且较高，说明细粒度 xAPI 分析与人设设定在方向上一致；
       * 若相关较低，则提示需检查人设设定与实际行为是否存在不一致。

三、与原论文方法的差异说明：
--------------------------------------------------
- LEARNER-C 原文在技术实现上使用区块链系统记录价值交换，并结合文本共现网络进行
  更复杂的结构化分析。本脚本：
  * 保留了“基于价值交换日志区分高贡献 / 低参与”的核心思想；
  * 但在算法上采用了更轻量、可直接部署于现有 MongoDB + xAPI 数据上的做法：
    使用标准化 token_gain + 贡献次数构成综合指数，再用一维 k-means 聚类。
- VR 协作 LAD 中有更细致的协作网络分析与可视化（如构建协作网络图），本脚本在不额外
  引入复杂图算法的前提下，采用“贡献次数 + 类型比例”作为协作角色的近似代理，
  以保证在目前数据结构下脚本可直接运行。

脚本功能总结：
--------------------------------------------------
1. 从 MongoDB 中读取：
   - 细粒度 xAPI 行为：MLS.Interaction 集合；
   - 学习者画像：MLS.LearnerProfile 集合（特别是 global_profile.value_contribution.score）。

2. 对每个 (学习者, 课程)：
   - 统计：
     * token_gain / token_cost / token_net；
     * 各类贡献行为次数（resource / co-edit / collaborate）。
   - 在课程内对 token_gain 与 contrib_total 做 z 标准化；
   - 合成价值贡献指数：
       C = (z_token + z_contrib) / sqrt(2)
   - 再在全局范围内对 C 做 min-max 归一化得到 C_norm ∈ [0, 1]。

3. 基于所有 (学习者, 课程) 的 C_norm：
   - 使用一维 k-means（k=3）进行聚类；
   - 根据聚类中心从低到高排序，将每条记录标记为：
       “低价值贡献型 / 中等价值贡献型 / 高价值贡献 & 高声望型”，
     并结合贡献行为类型给出“协作驱动型 / 资源驱动型”等补充描述。

4. 与人设对比：
   - 对每个学习者，把其在所有课程上的 C_norm 做平均，得到行为侧 global_value_contribution；
   - 与 LearnerProfile.global_profile.value_contribution.score 做皮尔逊相关，
     用于粗略验证行为分析与人设的一致性。

5. 数据库存储接口（不在 main() 中调用）：
   - 定义 save_value_contribution_to_db(db, value_results) 函数，
     演示如何把结果写入 MLS.ValueContributionAnalysis 集合，但默认不调用。
   - 你可以在需要时手动取消注释进行持久化。
"""

from pymongo import MongoClient
from datetime import datetime
from math import sqrt
from collections import defaultdict
import random

# ===================== 配置区域 =====================

MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "MLS"
XAPI_COLLECTION = "Interaction"          # 细粒度行为集合（xAPI_interaction_profile.py 生成）
PROFILE_COLLECTION = "LearnerProfile"    # 人设集合（infer_persona_for_course 写入）
VALUE_CONTRIB_COLLECTION = "ValueContributionAnalysis"  # 本维度分析结果集合（仅接口，不在 main 中调用）

# 采样学习者数量：随机选取至多 N_SAMPLE 个学习者进行分析
# 若 N_SAMPLE <= 0 或大于实际可选数量，则使用所有学习者
N_SAMPLE = 3000

VERB_BASE = "https://legend-meta.com/xapi/verb/"

VERBS = {
    # 价值交换行为：对应 LEARNER-C 中“价值 token 流水”概念
    "exchanged_value": VERB_BASE + "exchanged-value",

    # 贡献型行为：对应 VR 协作 LAD 中的“个人贡献度（资源提交 / 编辑 / 协作）”
    "contributed_resource": VERB_BASE + "contributed-resource",
    "co_edited_artifact": VERB_BASE + "co-edited-artifact",
    "collaborated_on_activity": VERB_BASE + "collaborated-on-activity",

    # 可选：若将来为“完成任务奖励 token”建立直接关联，可加入 completed 作为“价值来源”线索
    "completed": VERB_BASE + "completed",
}

# xAPI 扩展字段 IRI（与 xAPI_interaction_profile.py 中一致）
EXT_BASE = "https://legend-meta.com/xapi/ext/"
EXT_VALUE_CHANGE = EXT_BASE + "value-change"       # 单次价值变动（正：获得，负：支出）
EXT_VALUE_TOKEN_TYPE = EXT_BASE + "value-token-type"  # 价值 token 类型（积分、声望等），本脚本主要用于调试显示

# 最小事件数阈值：用于避免极少量事件造成不稳定的统计
MIN_EVENTS_PER_PAIR = 1


# ===================== 工具函数 =====================

def compute_mean_std(values):
    """
    计算一组数的均值和标准差（总体标准差）：
    - 列表为空 -> (0, 0)
    - 仅一个元素 -> 标准差视为 0

    用途：
    - 对课程内所有学习者的 token_gain 与 contrib_total 计算均值和标准差，
      用于后续 z 分数与相关系数计算。
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_v = sum(values) / float(n)
    if n == 1:
        return mean_v, 0.0
    var = sum((v - mean_v) ** 2 for v in values) / float(n)
    std = sqrt(var)
    return mean_v, std


def kmeans_1d(values, k=3, max_iter=50):
    """
    一维 k-means 聚类（Lloyd 算法实现），用于基于价值贡献指数 C_norm 自动划分学习者类型。

    论文与画像依据：
    --------------------------------------------------
    1）“用行为模式做聚类画像”的直接依据：
        - Heinemann et al. 在 OmiLAXR 框架以及 RePiX VR 仪表盘中强调，
          同一任务阶段下不同学生的行为模式可用于聚类画像与识别不同学习者类型。:contentReference[oaicite:12]{index=12}
        - Lampropoulos & Evangelidis 的 XR + LA/EDM 系统综述也将“行为参与度、贡献行为等”
          作为重要的聚类特征来源。:contentReference[oaicite:13]{index=13}
    2）本脚本仅在一维（价值贡献指数）上聚类，保持实现简单、易部署，同时与任务效率脚本
       的做法保持一致，便于在后续仪表盘中统一展示“低 / 中 / 高”三档画像。

    参数：
    - values: list[float]，待聚类的一维数值（这里是 C_norm）
    - k: 聚类簇数（默认 3，对应低/中/高三档）
    - max_iter: 最大迭代次数（防止收敛过慢）

    返回：
    - centers: list[float]，长度为 k 的聚类中心（可能按大小无序）
    - assignments: list[int]，与 values 一一对应的簇编号（0 ~ k-1）
    """
    n = len(values)
    if n == 0 or k <= 0:
        return [], []
    if n <= k:
        # 样本数小于等于簇数：每个样本单独成簇，其余中心复用
        centers = values[:]
        while len(centers) < k:
            centers.append(values[-1])
        assignments = list(range(n))
        return centers, assignments

    # 初始化中心：随机从样本中选 k 个不同的值
    centers = random.sample(values, k)
    assignments = [0] * n

    for _ in range(max_iter):
        # 1. 赋值步骤：每个样本归到最近的中心
        changed = False
        for i, v in enumerate(values):
            dists = [abs(v - c) for c in centers]
            min_idx = dists.index(min(dists))
            if assignments[i] != min_idx:
                assignments[i] = min_idx
                changed = True

        # 若本轮没有任何变化，说明已收敛
        if not changed:
            break

        # 2. 更新中心：对每个簇计算平均值
        for cluster_idx in range(k):
            cluster_vals = [v for v, a in zip(values, assignments) if a == cluster_idx]
            if cluster_vals:
                centers[cluster_idx] = sum(cluster_vals) / float(len(cluster_vals))

    return centers, assignments


# ===================== 数据聚合与分析函数 =====================

def aggregate_value_contribution(events):
    """
    根据传入的 xAPI 事件列表，按 (学习者, 课程) 聚合价值贡献相关统计。

    输入：
    - events: list[dict]，从 MongoDB 读取的 Interaction 文档子集（只包含本维度相关动词）

    输出：
    - value_stats: dict[(lrn_uid, crs_uid) -> dict]，每个键包含：
        {
            "token_gain": float,        # 价值流入总量（value-change > 0 累积和）
            "token_cost": float,        # 价值支出总量（|value-change < 0| 累积和）
            "token_net": float,         # 净价值变动（gain - cost）
            "value_events": int,        # exchanged-value 事件数
            "resource_contrib_count": int,  # contributed-resource 次数
            "coedit_count": int,            # co-edited-artifact 次数
            "collab_count": int,            # collaborated-on-activity 次数
        }
    """
    value_stats = defaultdict(lambda: {
        "token_gain": 0.0,
        "token_cost": 0.0,
        "token_net": 0.0,
        "value_events": 0,
        "resource_contrib_count": 0,
        "coedit_count": 0,
        "collab_count": 0,
    })

    used_events = 0

    for doc in events:
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue

        verb = (doc.get("verb") or {}).get("id")
        if not verb:
            continue

        key = (lrn_uid, crs_uid)
        stats = value_stats[key]

        # 价值交换事件：读取 value-change 扩展
        if verb == VERBS["exchanged_value"]:
            ctx = doc.get("context") or {}
            exts = ctx.get("extensions") or {}
            delta = exts.get(EXT_VALUE_CHANGE)

            try:
                delta_val = float(delta)
            except (TypeError, ValueError):
                # 若缺失或格式不正确，则视为 0（无法参与价值贡献分析）
                continue

            if delta_val > 0:
                stats["token_gain"] += delta_val
            elif delta_val < 0:
                stats["token_cost"] += abs(delta_val)

            stats["token_net"] += delta_val
            stats["value_events"] += 1
            used_events += 1

        # 资源贡献与协作行为：仅计数，用于构建贡献频率指标
        elif verb == VERBS["contributed_resource"]:
            stats["resource_contrib_count"] += 1
            used_events += 1
        elif verb == VERBS["co_edited_artifact"]:
            stats["coedit_count"] += 1
            used_events += 1
        elif verb == VERBS["collaborated_on_activity"]:
            stats["collab_count"] += 1
            used_events += 1
        else:
            # 其他动词（如 completed）在当前版本未直接用于价值贡献聚合
            continue

    print(f"参与价值贡献聚合的有效事件数：{used_events}")
    print(f"有价值/贡献数据的 (学习者, 课程) 数量：{len(value_stats)}")

    # 过滤掉完全没有任何事件的条目（理论上不会出现，但做一次安全清理）
    filtered_stats = {
        k: v for k, v in value_stats.items()
        if (v["value_events"] > 0 or
            v["resource_contrib_count"] > 0 or
            v["coedit_count"] > 0 or
            v["collab_count"] > 0)
    }

    return filtered_stats


def compute_value_contribution_index(value_stats):
    """
    在已经聚合的 value_stats 基础上，计算课程内标准化后的价值贡献指数。

    输入：
    - value_stats: dict[(lrn_uid, crs_uid) -> dict]，由 aggregate_value_contribution 返回，
      且会在本函数中就地扩展字段。

    输出：
    - value_results: dict[(lrn_uid, crs_uid) -> dict]，在原有统计基础上增加：
        {
            ...
            "contrib_total": int,
            "z_token_gain": float,
            "z_contrib": float,
            "C": float,         # 原始价值贡献指数
            "C_norm": float,    # 全局 min-max 归一化后的价值贡献指数
        }

    计算流程说明（与论文依据对应）：
    --------------------------------------------------
    1）课程内标准化：
       - 对每门课程单独计算 token_gain 与 contrib_total 的均值和标准差，
         与 VR 协作 LAD 中“在项目内比较不同学生的贡献度”的思路一致。:contentReference[oaicite:14]{index=14}
       - 这样可以避免不同课程之间任务设计、奖励规模不同带来的偏差。

    2）合成贡献指数：
       - 参考 LEARNER-C 将“价值交换日志”视为 contribution 的价值权重，token_gain 体现
         “被认可的价值大小”；贡献次数体现“参与广度与频度”；通过两者的 z 分数加权合成指数。
    """
    # 先统计每门课程的 token_gain 与 contrib_total
    course_to_token = defaultdict(list)
    course_to_contrib = defaultdict(list)

    # 为每个 (lrn_uid, crs_uid) 计算 contrib_total，并填充基础字段
    for (lrn_uid, crs_uid), stats in value_stats.items():
        contrib_total = (
            stats["resource_contrib_count"] +
            stats["coedit_count"] +
            stats["collab_count"]
        )
        stats["contrib_total"] = contrib_total

        token_gain = float(stats["token_gain"])

        # 若既没有 token 也没有贡献行为，则该条目对“价值贡献”维度信息不足，可以跳过
        if token_gain <= 0 and contrib_total <= 0:
            continue

        course_to_token[crs_uid].append(token_gain)
        course_to_contrib[crs_uid].append(float(contrib_total))

    # 课程内计算均值和标准差
    course_token_stats = {}
    course_contrib_stats = {}
    for crs_uid, vals in course_to_token.items():
        mean_v, std_v = compute_mean_std(vals)
        course_token_stats[crs_uid] = (mean_v, std_v)

    for crs_uid, vals in course_to_contrib.items():
        mean_v, std_v = compute_mean_std(vals)
        course_contrib_stats[crs_uid] = (mean_v, std_v)

    # 计算每个 (学习者, 课程) 的 z 值与原始指数 C
    C_values = []
    for (lrn_uid, crs_uid), stats in value_stats.items():
        token_gain = float(stats["token_gain"])
        contrib_total = float(stats.get("contrib_total", 0))

        # 若双方皆为 0，则不计算指数（信息不足）
        if token_gain <= 0 and contrib_total <= 0:
            continue

        # 课程内 token_gain z 值
        mean_token, std_token = course_token_stats.get(crs_uid, (0.0, 0.0))
        if std_token > 1e-6:
            z_token = (token_gain - mean_token) / std_token
        else:
            z_token = 0.0

        # 课程内 contrib_total z 值
        mean_contrib, std_contrib = course_contrib_stats.get(crs_uid, (0.0, 0.0))
        if std_contrib > 1e-6:
            z_contrib = (contrib_total - mean_contrib) / std_contrib
        else:
            z_contrib = 0.0

        stats["z_token_gain"] = z_token
        stats["z_contrib"] = z_contrib

        # 合成指数：token 与贡献行为同向促进价值贡献
        C = (z_token + z_contrib) / sqrt(2.0)
        stats["C"] = C
        C_values.append(C)

    if not C_values:
        print("没有可用于计算价值贡献指数的样本。")
        return {}

    # 全局 min-max 归一化
    min_C = min(C_values)
    max_C = max(C_values)
    if max_C - min_C < 1e-6:
        # 极端情况：所有 C 完全相同，统一设为 0.5
        for stats in value_stats.values():
            if "C" in stats:
                stats["C_norm"] = 0.5
        print("所有价值贡献指数相同，统一设置 C_norm = 0.5。")
        return value_stats

    for stats in value_stats.values():
        if "C" not in stats:
            continue
        C = float(stats["C"])
        C_norm = (C - min_C) / float(max_C - min_C)
        stats["C_norm"] = C_norm

    print("已完成价值贡献指数的课程内标准化与全局归一化。")
    return value_stats


def assign_value_labels(value_results):
    """
    基于归一化后的价值贡献指数 C_norm，对 (学习者, 课程) 进行聚类并赋予语义标签。

    输入：
    - value_results: dict[(lrn_uid, crs_uid) -> dict]，要求每个条目中已包含 "C_norm" 字段。

    输出：
    - 修改 value_results，就地添加：
        * "cluster_index": int   # k-means 原始簇编号
        * "cluster_rank": int    # 按中心高低排序后的等级（0: 低, 1: 中, 2: 高）
        * "value_label": str     # 文字标签，综合贡献水平与贡献来源
    """
    # 收集所有 C_norm 用于聚类
    C_norm_list = []
    keys_list = []

    for key, stats in value_results.items():
        C_norm = stats.get("C_norm")
        if C_norm is None:
            continue
        C_norm_list.append(float(C_norm))
        keys_list.append(key)

    if not C_norm_list:
        print("没有可用于聚类的价值贡献指数，跳过标签生成。")
        return

    # 运行一维 k-means（k=3）
    centers, assignments = kmeans_1d(C_norm_list, k=3, max_iter=50)
    if not centers or not assignments:
        print("k-means 聚类未能得到有效中心，跳过价值贡献标签生成。")
        return

    # 根据中心数值从小到大排序，映射到 rank（0: 低, 1: 中, 2: 高）
    center_with_idx = list(enumerate(centers))
    center_with_idx.sort(key=lambda x: x[1])  # 按中心数值从小到大排序
    cluster_to_rank = {cluster_idx: rank for rank, (cluster_idx, _) in enumerate(center_with_idx)}

    # rank 对应的基础标签
    base_rank_to_label = {
        0: "低价值贡献型学习者（在本课程中几乎没有价值 token 流入，贡献行为也较少）",
        1: "中等价值贡献型学习者（在本课程中具有一定价值 token 流入与贡献行为）",
        2: "高价值贡献 & 高声望型学习者（在本课程中获得较多价值 token 奖励并频繁贡献）",
    }

    # 依次给每个 (学习者, 课程) 赋予簇编号与标签，并根据贡献类型补充“价值来源”描述
    for key, cluster_idx in zip(keys_list, assignments):
        stats = value_results[key]
        rank = cluster_to_rank.get(cluster_idx, 1)  # 默认视为中等贡献
        base_label = base_rank_to_label.get(rank, "中等价值贡献型学习者（默认）")

        # 贡献来源：根据贡献行为类型比例给出“偏协作型 / 偏资源分享型”等描述
        resource_count = stats.get("resource_contrib_count", 0)
        coedit_count = stats.get("coedit_count", 0)
        collab_count = stats.get("collab_count", 0)
        contrib_total = stats.get("contrib_total", 0)

        extra_desc = ""
        if contrib_total > 0:
            collab_like = coedit_count + collab_count
            # 协作型贡献比例
            collab_ratio = collab_like / float(contrib_total)
            resource_ratio = resource_count / float(contrib_total)

            if collab_ratio >= 0.6 and collab_like >= 2:
                extra_desc = "，且在贡献行为中以协作与共同编辑为主（偏协作驱动型贡献者）"
            elif resource_ratio >= 0.6 and resource_count >= 2:
                extra_desc = "，且在贡献行为中以上传 / 分享资源为主（偏资源驱动型贡献者）"
            else:
                extra_desc = "，贡献行为在协作与资源分享之间较为均衡"

        full_label = base_label + extra_desc

        stats["cluster_index"] = int(cluster_idx)
        stats["cluster_rank"] = int(rank)
        stats["value_label"] = full_label

    # 统计标签分布，便于在控制台快速查看整体情况
    label_counts = defaultdict(int)
    for stats in value_results.values():
        label = stats.get("value_label")
        if label:
            label_counts[label] += 1

    print("价值贡献与声望标签分布（按学习者-课程对统计）：")
    for label, cnt in label_counts.items():
        print(f"- {label}: {cnt} 条记录")


# ===================== 写回数据库接口（默认不调用） =====================

def save_value_contribution_to_db(db, value_results):
    """
    将本脚本得到的价值贡献分析结果写入 MongoDB。

    设计说明：
    --------------------------------------------------
    1）集合选择：
       - MLS.ValueContributionAnalysis
       - 一个文档对应一个 (学习者, 课程) 的分析结果。

    2）存储字段：
       - learner_uid, course_uid
       - token_gain, token_cost, token_net
       - contrib_total, resource_contrib_count, coedit_count, collab_count
       - z_token_gain, z_contrib
       - value_index（原始指数 C）
       - value_normalized（归一化指数 C_norm）
       - value_label（离散标签）
       - cluster_rank（0/1/2：低/中/高）

    3）与画像设计文档的映射关系：
       - value_index / value_normalized：
         * 对应“元宇宙价值贡献与声望”维度中，“价值 token 流动 + 贡献行为”这一综合指标。
       - value_label / cluster_rank：
         * 对应“高贡献 / 低参与”等离散类型，用于在画像中直接显示。
       - 注意：本函数不会在 main() 中自动调用。
         若你希望实际写回数据库，请在 main() 中手动解除注释：
             save_value_contribution_to_db(db, value_results)
    """
    col = db[VALUE_CONTRIB_COLLECTION]

    # 为方便重复实验，这里先清空集合（若你不想清空，可以改为 update 或 upsert）
    db.drop_collection(VALUE_CONTRIB_COLLECTION)
    col = db[VALUE_CONTRIB_COLLECTION]

    docs_to_insert = []
    for (lrn_uid, crs_uid), stats in value_results.items():
        if "C" not in stats or "C_norm" not in stats:
            # 未参与指数计算的条目不写入
            continue

        doc = {
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "token_gain": float(stats.get("token_gain", 0.0)),
            "token_cost": float(stats.get("token_cost", 0.0)),
            "token_net": float(stats.get("token_net", 0.0)),
            "value_events": int(stats.get("value_events", 0)),
            "resource_contrib_count": int(stats.get("resource_contrib_count", 0)),
            "coedit_count": int(stats.get("coedit_count", 0)),
            "collab_count": int(stats.get("collab_count", 0)),
            "contrib_total": int(stats.get("contrib_total", 0)),
            "z_token_gain": float(stats.get("z_token_gain", 0.0)),
            "z_contrib": float(stats.get("z_contrib", 0.0)),
            "value_index": float(stats.get("C", 0.0)),
            "value_normalized": float(stats.get("C_norm", 0.0)),
            "value_label": stats.get("value_label"),
            "cluster_rank": int(stats.get("cluster_rank", 1)),
            "created_at": datetime.utcnow(),
        }
        docs_to_insert.append(doc)

    if docs_to_insert:
        col.insert_many(docs_to_insert, ordered=False)
        col.create_index(
            [("learner_uid", 1), ("course_uid", 1)],
            name="idx_learner_course"
        )
        print(f"[接口调用] 已写入 ValueContributionAnalysis 文档数：{len(docs_to_insert)}")
    else:
        print("[接口调用] 没有可写入 ValueContributionAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]

    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中的“价值贡献”人设 ----------
    print("读取 LearnerProfile 中的 value_contribution 人设信息.")
    persona_scores = {}  # lrn_uid -> persona_value_contribution_score

    cursor_profiles = profile_col.find(
        {},
        {"learner_uid": 1, "global_profile": 1}
    )

    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g_profile = doc.get("global_profile") or {}
        vc = g_profile.get("value_contribution") or {}
        score = vc.get("score")
        if score is not None:
            persona_scores[lrn_uid] = float(score)

    print(f"从人设中读取到具有 value_contribution.score 的学习者数：{len(persona_scores)}")

    # ---------- 3. 选择参与分析的学习者 ----------
    """
    采样策略说明：
    --------------------------------------------------
    - 优先使用具有人设分数的学习者，以便后续进行“行为分析 vs 人设”的相关性检验。
    - 若 N_SAMPLE > 0 且小于具备人设的学习者数量，则随机采样 N_SAMPLE 个。
    - 若人设中没有对应维度（例如尚未在 LearnerProfile 中写入 value_contribution），
      则退而求其次，从 Interaction 集合中按动词查询所有出现过相关行为的学习者。
    """
    if persona_scores:
        all_learners = list(persona_scores.keys())
    else:
        # 从 xAPI 中找出至少有价值/贡献行为的学习者
        print("人设中暂未找到 value_contribution.score，将从 xAPI 行为中推断候选学习者。")
        verb_list = [
            VERBS["exchanged_value"],
            VERBS["contributed_resource"],
            VERBS["co_edited_artifact"],
            VERBS["collaborated_on_activity"],
        ]
        pipeline = [
            {"$match": {"verb.id": {"$in": verb_list}}},
            {"$group": {"_id": "$_lrn_uid"}},
        ]
        learner_ids = list(xapi_col.aggregate(pipeline))
        all_learners = [d["_id"] for d in learner_ids if d.get("_id")]

    if not all_learners:
        print("没有找到任何候选学习者，终止分析。")
        return

    if 0 < N_SAMPLE < len(all_learners):
        sampled_learners = random.sample(all_learners, N_SAMPLE)
    else:
        sampled_learners = all_learners

    print(f"本次分析将使用的学习者数量：{len(sampled_learners)}")

    # ---------- 4. 从 MongoDB 读取相关 xAPI 事件 ----------
    print("从 MongoDB 读取价值交换与贡献相关的 xAPI 事件.")

    verb_filter = [
        VERBS["exchanged_value"],
        VERBS["contributed_resource"],
        VERBS["co_edited_artifact"],
        VERBS["collaborated_on_activity"],
        # 若将来需要联动“完成任务获得奖励”，可以在此加入 VERBS["completed"]
    ]

    cursor_events = xapi_col.find(
        {
            "_lrn_uid": {"$in": sampled_learners},
            "verb.id": {"$in": verb_filter},
        },
        {
            "_id": 0,
            "verb.id": 1,
            "context": 1,
            "result": 1,
            "_lrn_uid": 1,
            "_course_uid": 1,
        }
    )

    events = list(cursor_events)
    print(f"已从 MongoDB 读取事件条数：{len(events)}")

    if not events:
        print("没有任何价值交换或贡献相关事件，无法进行价值贡献分析。")
        return

    # ---------- 5. 聚合 (学习者, 课程) 的价值与贡献统计 ----------
    value_stats = aggregate_value_contribution(events)
    if not value_stats:
        print("聚合后没有有效的 (学习者, 课程) 记录，终止分析。")
        return

    # ---------- 6. 计算价值贡献指数（课程内标准化 + 全局归一化） ----------
    value_results = compute_value_contribution_index(value_stats)
    if not value_results:
        print("无法计算价值贡献指数，终止后续聚类与人设对比。")
        return

    # ---------- 7. 基于指数进行聚类并生成离散标签 ----------
    assign_value_labels(value_results)

    # ---------- 8. （可选）写回数据库接口——默认不调用 ----------
    """
    如你在需求中所述：
    - 当前版本脚本只需完成“读取细粒度 xAPI → 计算价值贡献指数 → 输出标签与人设对比”，
      不需要真正把结果写回数据库。
    - 上面定义的 save_value_contribution_to_db(db, value_results) 即为“写回接口”；
      若未来需要，可手动解除下面的注释。

    示例（默认注释掉）：
        save_value_contribution_to_db(db, value_results)
    """
    # 若需要写回数据库，请取消下一行注释：
    # save_value_contribution_to_db(db, value_results)

    # ---------- 9. 按学习者汇总 global_value_contribution 并与人设对比 ----------
    """
    验证思路与论文参考：
    --------------------------------------------------
    1）行为侧指标：
       - 对每个学习者，把其在所有课程上的 C_norm 取平均，得到 global_value_contribution。
       - global_value_contribution 处于 [0,1] 区间，代表行为数据推断出的“整体元宇宙价值贡献与声望水平”。

    2）人设侧指标：
       - LearnerProfile.global_profile.value_contribution.score 是在人设推断脚本中，
         基于粗粒度统计与规则预先设定的价值贡献分数。

    3）对比目的：
       - 通过皮尔逊相关系数，检验“基于细粒度 xAPI 的价值贡献分析”和
         “基于粗粒度统计的人设价值贡献”在总体趋势上是否一致。
       - 若相关为正且显著，说明细粒度分析与已有画像设计在该维度上方向一致，
         有助于增强画像的可信度。

    4）与文献的关系：
       - Hsu et al. 的 LEARNER-C 研究强调，价值交换日志可以作为数据驱动教育改革的依据，
         本脚本将其进一步量化为课程级与全局级的指数。:contentReference[oaicite:15]{index=15}
       - VR 协作 LAD 与 Gamified LA 的研究则证明，贡献行为与游戏化积分可以稳定刻画学习者
         在协作与价值创造方面的差异，这为我们将行为指数与人设维度对齐提供了理论支撑。
    """
    learner_to_C_vals = defaultdict(list)
    for (lrn_uid, crs_uid), stats in value_results.items():
        C_norm = stats.get("C_norm")
        if C_norm is None:
            continue
        learner_to_C_vals[lrn_uid].append(float(C_norm))

    learner_global_value = {}
    for lrn_uid, vals in learner_to_C_vals.items():
        if vals:
            learner_global_value[lrn_uid] = sum(vals) / float(len(vals))

    xs = []  # 人设中的 value_contribution.score
    ys = []  # 行为分析得到的 global_value_contribution

    for lrn_uid in sampled_learners:
        persona_score = persona_scores.get(lrn_uid)
        analyzed_val = learner_global_value.get(lrn_uid)
        if persona_score is not None and analyzed_val is not None:
            xs.append(float(persona_score))
            ys.append(float(analyzed_val))

    if len(xs) >= 2:
        mean_x, std_x = compute_mean_std(xs)
        mean_y, std_y = compute_mean_std(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / float(len(xs))
        if std_x > 1e-6 and std_y > 1e-6:
            corr = cov / (std_x * std_y)
        else:
            corr = 0.0

        avg_global_value = sum(ys) / float(len(ys))
        avg_persona_score = sum(xs) / float(len(xs))

        print("=========================================================")
        print("【元宇宙价值贡献与声望维度：人设 vs 行为分析 全局对比】")
        print(f"- 采样学习者数量（具备人设）：{len(sampled_learners)}")
        print(f"- 实际参与对比的学习者数量：{len(xs)}")
        print(f"- 行为分析 global_value_contribution 平均值：{avg_global_value:.3f}")
        print(f"- 人设 value_contribution.score 平均值：{avg_persona_score:.3f}")
        print(f"- 皮尔逊相关系数：{corr:.3f}")
        print("  （相关系数用于粗略验证：细粒度 xAPI 分析是否与人设“价值贡献与声望”维度方向一致。）")
        print("=========================================================")
    else:
        print("参与“价值贡献维度”人设对比的学习者样本太少，无法计算相关系数。")

    print("元宇宙价值贡献与声望维度分析完成。")


if __name__ == "__main__":
    main()
