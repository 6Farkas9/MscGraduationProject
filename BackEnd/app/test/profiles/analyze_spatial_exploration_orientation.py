# -*- coding: utf-8 -*-
"""
分析维度：空间与资源探索倾向（Spatial & Resource Exploration Orientation）

脚本目标（与画像设计文档对齐）：
--------------------------------------------------
对应《论文.txt》中“3. 画像设计”里维度：
【空间与资源探索倾向（Spatial & Resource Exploration Orientation）】

该维度在画像文档中的界定：
- 反映学习者在元宇宙空间中探索不同区域、资源的程度，是“到点就学”还是喜欢到处逛 / 探索。
- 推荐使用的行为数据（在 xAPI Profile 中已定义相应动词）：
  * navigated-to-space / teleported-to-space：
    - 不同 space-id 数量与分布 → 探索广度；
    - 访问顺序 → 路径类型（线性、跳跃/回访）。
  * explored-extension：
    - 是否进入可选拓展单元 / 支线任务。
  * focused-on-resource：
    - 覆盖的资源类别数（是否只盯主任务资源）。

本脚本采用的行为代理与论文依据：
--------------------------------------------------
1）空间路径与探索广度：基于 Heinemann et al. 的 RePiX VR 仪表盘研究 + XR+LA/EDM 综述
   - 对应文献与观点：
     * Heinemann 等人在 RePiX VR 学习分析仪表盘中：
       - 使用学习者的位姿与移动记录生成空间热力图和移动路径；
       - 区分“行走/移动”和“teleport（瞬移）”两类导航模式；
       - 用“在虚拟环境中访问区域的范围 + 移动模式”来刻画学习者是否有广泛探索。
     * XR+LA/EDM 系统综述中：
       - 将“空间位置、路径、停留区域”视为表征探索行为的核心数据类型；
       - 探讨了用空间路径和区域覆盖度来描述“探索型”学习者。
   - 本脚本的行为代理设计：
     * 使用 navigated-to-space / teleported-to-space 事件中的 space-id：
       - 按时间顺序构建 space-id 序列，统计访问过的不同 space-id 数量 → 空间探索广度；
       - 用 space-id 序列判断是否出现“离开某空间后又回到该空间”的回访模式。
     * 使用 teleport 导航相关信息：
       - 统计 teleport 导航次数与总导航次数的比例 → teleport_ratio，
         对应原文中“区分不同导航方式”这一思路。
   - 与原论文方法的差异与原因：
     * 原文使用连续 3D 坐标和视角数据生成精细热力图和复杂路径可视化；
       目前细粒度 xAPI 数据主要记录的是离散的 space-id 和导航事件，缺少连续坐标。
     * 本脚本将原来的“连续空间轨迹 + 热力图分析”简化为：
       - 使用“unique space-id 数量 + 路径回访标记 + teleport 比例”三个可直接计算的指标，
         保留“探索范围、路径模式、导航方式”这三个核心视角。
     * 改动原因：
       - 与既有数据结构对齐，保证脚本在当前环境中可直接运行；
       - 在无 3D 坐标的前提下，仍然最大程度保留原论文强调的空间探索特征。

2）场景 / 子场景进入行为作为空间分析基础：基于 OmiLAXR 框架
   - 对应文献与观点：
     * Heinemann 等人提出的 OmiLAXR 框架：
       - 用 xAPI 对 VR 场景的“进入/退出子场景、在空间间移动”等基础行为进行统一建模；
       - 建议将“场景/子场景进入事件”作为空间分析和路径重建的基础。
   - 本脚本的行为代理设计：
     * 将 navigated-to-space / teleported-to-space 视为“进入场景/子场景”的事件；
     * 使用 context.extensions 中的 space-id 作为场景/子场景标识；
     * 所有空间相关指标（空间广度、路径回访、teleport 比例）均基于这些事件统计。
   - 与原论文方法的差异与原因：
     * 原文主要提供框架性建议和 Profile 设计原则，并未给出固定算法；
     * 本脚本落实为具体实现：
       - 显式选定 navigated-to-space / teleported-to-space 作为路径构建的主数据源；
       - 不引入其他 OmiLAXR 行为（如对象交互、视线跟踪）以保持脚本聚焦。
     * 改动原因：
       - 保持脚本职责单一，专注“空间与资源探索”这一画像维度；
       - 便于后续为其他维度单独编写分析脚本。

3）可选拓展单元 / 支线任务参与度：基于 Gamified Learning Analytics 研究
   - 对应文献与观点：
     * Gamified LA 研究（如 Papamitsiou 等）中：
       - 常把“可选挑战 / 支线任务 / 可选关卡”的参与情况视作探索性与成就动机的重要线索；
       - 强调需要在日志中区分“主线任务”和“可选内容”，并记录学习者是否主动参与可选内容。
   - 本脚本的行为代理设计：
     * 定义 explored-extension 动词，对应“进入可选拓展单元 / 支线任务 / 非必修内容”；
     * 对每个 (学习者, 课程)：
       - 若存在至少一次 explored-extension 事件，则 has_extension = 1，否则为 0；
       - 将 has_extension 视为“是否有主动探索可选内容”的二值信号，在课程内部做 z 标准化后，
         作为探索指数的核心组成部分之一。
   - 与原论文方法的差异与原因：
     * 原文中常对“次数、完成率、难度”等做更细粒度建模；
       目前数据结构主要体现“是否进入可选单元”，尚未区分不同支线的状态与得分。
     * 本脚本采用二值特征（是否参与过），而非完整次数 / 完成率：
       - 在当前数据下，二值特征已经可以区分“从不探索支线”和“有拓展探索行为”的学习者；
       - 若将来扩展支线设计，可以在本维度中增加“拓展探索强度”子指标。
     * 改动原因：
       - 保证与现有 xAPI Profile 完整对齐，不额外假定数据库结构；
       - 用最小可行特征完成画像粗分层。

4）资源区域覆盖与探索行为：基于 XR+LA/EDM 综述中“停留区域与资源多样性”的观点
   - 对应文献与观点：
     * XR+LA/EDM 综述中指出：
       - 学习者在虚拟空间中停留的区域、使用的资源类型多样性，是表征探索行为和策略的重要线索；
       - 高探索型学习者往往会浏览更多区域、尝试更多类型资源，而低探索型学习者往往只停留在主任务区。
   - 本脚本的行为代理设计：
     * 使用 focused-on-resource 事件及其中的 focus-target-id：
       - 每一个不同的 focus-target-id 表示一种资源类别 / 界面区域（如 main-screen、diagram-area 等）；
       - 统计 unique_resources（不同资源类别数）作为资源探索广度的代理；
       - 在课程内部对 unique_resources 做 z 标准化。
   - 与原论文方法的差异与原因：
     * 综述提供的是理论视角和数据类型分类，并未给出具体公式；
     * 本脚本将其具体化为：
       - 利用“资源类别覆盖数”直接度量资源使用多样性；
       - 以 z 分数形式纳入总体探索指数。
     * 改动原因：
       - 将抽象理论变为可直接基于 xAPI 字段计算的指标；
       - 与其他维度（如任务效率）保持类似的数值化形式，便于后续联合建模。

5）探索指数与聚类分层：基于 LA/EDM 中的行为聚类画像做法
   - 对应文献与观点：
     * 多篇 LA/EDM 聚类研究与 XR+LA/EDM 综述指出：
       - 可以基于行为特征使用 k-means 等方法将学习者划分为“探索型 / 目标导向型”等群体；
       - 聚类结果常作为后续画像标签和个性化推送策略的基础。
   - 本脚本的分析方法：
     * 特征构建（课程内部）：
       - z_space_breadth      ：unique_spaces（访问到的不同 space-id 数）的 z 分数；
       - z_extension_flag     ：has_extension（是否参与可选拓展）的 z 分数；
       - z_resource_breadth   ：unique_resources（覆盖的资源类别数）的 z 分数；
       - z_path_pattern       ：path_jump（是否存在回访型路径）的 z 分数；
       - z_teleport_ratio     ：teleport_ratio（teleport 导航占比）的 z 分数。
     * 指标合成（探索指数 E）：
       - 采用加权线性组合：
         E_i = (w_space * z_space_breadth
                + w_ext   * z_extension_flag
                + w_res   * z_resource_breadth
                + w_path  * z_path_pattern
                + w_tp    * z_teleport_ratio) / sqrt(w_space^2 + ... + w_tp^2)
       - 权重设计原则：
         · w_space、w_ext 权重最高，对应空间探索广度与支线参与度；
         · w_res 次之，对应资源使用多样性；
         · w_path、w_tp 作为路径风格和导航模式的辅助修正项。
     * 归一化与聚类：
       - 在课程内部对 E 做 min-max 归一化 → exploration_normalized ∈ [0, 1]；
       - 在全局层面，对所有 (学习者, 课程) 的 exploration_normalized 使用一维 k-means（k=3）聚类；
       - 将聚类中心从低到高排序，对应到：
         “到点即学型（低探索） / 平衡探索型（中等探索） / 高探索型探索者”三档。
   - 与原论文方法的差异与原因：
     * 原文一般在多维特征空间上直接聚类，本脚本增加了“探索指数”的压缩步骤：
       - 让本维度在 LearnerProfile 中表现为单一连续指标 + 三档类别，更易解释；
       - 与任务效率等维度保持统一表现形式。
     * 选择一维 k-means（k=3）而非更复杂模型：
       - k=3 对应画像中的“低 / 中 / 高”习惯划分；
       - 一维 k-means 实现简单，可解释性强，也方便在数据库中复现。
     * 改动原因：
       - 在遵循“基于行为聚类形成画像群体”的前提下，尽量降低算法复杂度，
         保证脚本在真实教学环境中的鲁棒性和可维护性。

脚本功能总结：
--------------------------------------------------
1. 从 MongoDB 中读取：
   - 细粒度 xAPI 行为：MLS.Interaction 集合；
   - 学习者画像：MLS.LearnerProfile 集合（尤其是 global_profile.exploration_orientation.score）。

2. 对每个 (学习者, 课程)：
   - 利用 navigated-to-space / teleported-to-space / explored-extension /
     focused-on-resource 等 xAPI 语句，统计：
       · unique_spaces     ：访问到的不同 space-id 数量；
       · unique_resources  ：聚焦过的不同资源类别数（基于 focus-target-id）；
       · has_extension     ：是否参与过至少一次可选拓展单元（支线任务）；
       · path_jump         ：空间访问序列中是否存在回访型路径（离开后再回到同一空间）；
       · teleport_ratio    ：teleport 导航占全部导航的比例。
   - 在“课程内部”对上述指标做 z 标准化，
     以加权组合方式计算探索指数 E，并做 min-max 归一化得到 E_norm ∈ [0, 1]。

3. 基于所有 (学习者, 课程) 的 E_norm：
   - 使用一维 k-means（k=3）进行聚类；
   - 按聚类中心从低到高排序，将每条记录标记为：
       “到点即学型（低探索） / 平衡探索型（中等探索） / 高探索型探索者”。

4. 与人设对比：
   - 对每个学习者，将其在所有课程上的 E_norm 做平均，得到行为侧 global_exploration；
   - 与 LearnerProfile.global_profile.exploration_orientation.score 做皮尔逊相关，
     粗略评估：细粒度 xAPI 空间/资源分析是否与人设中“探索倾向”维度方向一致。

5. 数据库存储接口（不在 main() 中调用）：
   - 定义 save_spatial_exploration_to_db(db, exploration_results) 函数，
     演示如何把本维度的分析结果写入 MLS.SpatialExplorationAnalysis 集合；
   - main() 中默认不调用该函数，满足“保留写回接口但不触发写回”的要求。

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
XAPI_COLLECTION = "Interaction"              # 细粒度行为集合（xAPI_interaction_profile.py 生成）
PROFILE_COLLECTION = "LearnerProfile"        # 人设集合（infer_persona_for_course 写入）
SPATIAL_EXPL_COLLECTION = "SpatialExplorationAnalysis"  # 本维度分析结果集合（仅接口，不在 main 中调用）

# 采样学习者数量：随机选取至多 N_SAMPLE 个学习者进行分析
# 若 N_SAMPLE <= 0 或大于实际可选数量，则使用所有学习者
N_SAMPLE = 3000

VERB_BASE = "https://legend-meta.com/xapi/verb/"

VERBS = {
    # 空间导航事件：对应 OmiLAXR 框架中“进入场景/子场景”的基础记录
    "navigated_to_space": VERB_BASE + "navigated-to-space",
    # 传送式导航事件：用于区分 teleport 与其他导航方式
    "teleported_to_space": VERB_BASE + "teleported-to-space",
    # 可选拓展单元事件：对应 Gamified LA 研究中的“支线任务 / 可选挑战”参与行为
    "explored_extension": VERB_BASE + "explored-extension",
    # 资源聚焦事件：用于统计覆盖的资源类别数，对应 XR+LA/EDM 中“停留区域与资源多样性”视角
    "focused_on_resource": VERB_BASE + "focused-on-resource",
}

# 扩展字段常量（与 xAPI_interaction_profile.py 中定义保持一致）
EXT_SPACE_ID = "https://legend-meta.com/xapi/ext/space-id"
EXT_NAV_MODE = "https://legend-meta.com/xapi/ext/navigation-mode"
EXT_FOCUS_TARGET_ID = "https://legend-meta.com/xapi/ext/focus-target-id"
EXT_UNIT_OPTIONAL = "https://legend-meta.com/xapi/ext/unit-optional"  # 目前只用于 Profile 设计，脚本中不强依赖


# ===================== 工具函数 =====================

def compute_mean_std(values):
    """
    计算一组数的均值和总体标准差（Population Std）：

    设计与用途：
    --------------------------------------------------
    - 本函数在本脚本中用于两个层面：
      1）课程内部标准化：
         对每门课程内所有 (学习者, 课程) 的：
           · 空间探索广度（unique_spaces）
           · 资源探索广度（unique_resources）
           · 可选拓展参与标记（has_extension）
           · 路径回访标记（path_jump）
           · teleport 比例（teleport_ratio）
         分别计算均值与标准差，用于后续 z 分数计算。
      2）全局相关分析：
         对所有参与对比的学习者的：
           · 行为侧 global_exploration
           · 人设侧 exploration_orientation.score
         计算均值与标准差，用于构造皮尔逊相关系数。

    - 采用“总体标准差”而非“样本标准差”的原因：
      在 LA/EDM 相关工作中，对整个“课程内人群”或“全体样本”做标准化时，
      通常将这一批样本视作完整群体（而非更大总体的抽样），
      因此除以 N 而非 N-1 更符合“在人群内部拉齐尺度”的直观含义。

    参数：
        values: List[float]，数值列表

    返回：
        (mean_v, std_v):
            mean_v: 均值，列表为空时为 0.0
            std_v : 总体标准差，若元素个数 < 2 则返回 0.0
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_v = sum(values) / float(n)
    if n == 1:
        return mean_v, 0.0
    var = sum((v - mean_v) ** 2 for v in values) / float(n)
    std_v = sqrt(var)
    return mean_v, std_v


def compute_path_jump_flag(space_seq):
    """
    判断空间访问序列中是否存在“回访型路径”，并给出二值标记。

    设计依据（来自 RePiX VR & XR+LA/EDM 综述）：
    --------------------------------------------------
    - Heinemann 等在 RePiX VR 仪表盘中，通过连续位置数据分析学习者在环境中的移动轨迹，
      对“是否围绕有限区域打转”、“是否在多个区域之间往返游走”进行区分。
    - XR+LA/EDM 综述指出：
      “路径形状（线性 / 往返 / 探索式）”是一类用来刻画探索行为的特征。

    在当前数据结构下的简化实现：
    --------------------------------------------------
    - 我们只有离散的 space-id 序列（表示进入不同场景/子场景），没有连续坐标。
    - 因此用如下规则构造“回访型路径”标记：
      1）先做“相邻去重”：例如 [A, A, B, B, A] → [A, B, A]，
         目的是忽略在同一空间内的短时间微移动；
      2）在去重后的序列中，如果某个 space-id 出现次数 ≥ 2，
         且两次出现之间至少经过了一个不同的 space-id，
         则认为存在“离开后又回到同一空间”的回访型路径；
      3）有回访 → 返回 1；否则返回 0。

    参数：
        space_seq: List[str]，按时间排序的 space-id 序列

    返回：
        0 或 1：
        - 0：近似线性路径（未出现明显回访）
        - 1：存在回访/往返式探索路径
    """
    if not space_seq:
        return 0

    # 相邻去重，过滤掉在同一空间内的连续重复记录
    reduced = []
    for sid in space_seq:
        if not reduced or reduced[-1] != sid:
            reduced.append(sid)

    seen = set()
    for sid in reduced:
        if sid in seen:
            # 说明出现了“离开后再回到该空间”的模式
            return 1
        seen.add(sid)
    return 0


def kmeans_1d(values, k=3, max_iter=50):
    """
    一维 k-means 聚类，用于把连续探索指数划分为“低 / 中 / 高”三档。

    设计依据（来自 XR+LA/EDM 综述 + 行为聚类研究）：
    --------------------------------------------------
    - XR+LA/EDM 综述中多次提到：
      可以将学习者在空间与资源行为上的特征输入聚类算法（如 k-means），
      将其划分为“探索型 / 目标导向型”等群体，用于后续画像与推荐。
    - 本脚本已经通过多维 z 分数构造了单一探索指数 exploration_normalized，
      为了保持画像框架中“每个维度有一个连续指数 + 若干类别标签”的形式，
      采用一维 k-means 对该指数做自动分段划分。

    为何选择 k=3？
    --------------------------------------------------
    - 画像设计中常见“低 / 中 / 高”三档；
    - LearnerProfile 中本维度的人设分类也是三档（低探索 / 中等探索 / 高探索）；
    - 因此设定 k=3，使聚类结果可以自然映射到这三类标签。

    算法说明（Lloyd 算法的一维版）：
    --------------------------------------------------
    1）初始化：
       - 若样本数 n < k，则将 k 改为 n，以避免空簇；
       - 若所有数值几乎相同（max - min 很小），直接将所有样本归为同一簇；
       - 否则在 [min, max] 区间内均匀初始化 k 个中心。

    2）迭代过程（最多 max_iter 次）：
       - 分配阶段：
         对每个样本 v，找到距离最近的中心 c_j，将 v 分配到簇 j；
       - 更新阶段：
         对每个簇 j，计算簇内样本的均值，作为新的中心 c'_j；
       - 若所有中心的移动总量很小（例如 < 1e-6），认为收敛，提前停止。

    3）输出：
       - centers     ：每个簇的中心值；
       - assignments ：每个样本所属簇的编号（0..k-1）。

    参数：
        values: List[float]，需要聚类的一维数值（本脚本中是 exploration_normalized）
        k:      簇数，默认 3
        max_iter: 最大迭代次数

    返回：
        centers: List[float]，聚类中心列表
        assignments: List[int]，与 values 等长的簇编号列表
    """
    n = len(values)
    if n == 0:
        return [], []

    if n < k:
        k = n

    v_min, v_max = min(values), max(values)
    if abs(v_max - v_min) < 1e-6:
        # 所有值几乎相同，视为一个簇
        centers = [v_min for _ in range(k)]
        assignments = [0 for _ in range(n)]
        return centers, assignments

    # 在 [v_min, v_max] 区间内均匀初始化 k 个中心
    centers = [
        v_min + (v_max - v_min) * (i + 0.5) / float(k)
        for i in range(k)
    ]

    for _ in range(max_iter):
        # Step 1: 分配阶段——将每个样本分配给最近的中心
        clusters = [[] for _ in range(k)]
        for idx, v in enumerate(values):
            best_c = 0
            best_dist = abs(v - centers[0])
            for ci in range(1, k):
                d = abs(v - centers[ci])
                if d < best_dist:
                    best_dist = d
                    best_c = ci
            clusters[best_c].append(idx)

        # Step 2: 更新阶段——重新计算每个簇的均值作为新中心
        new_centers = []
        for ci in range(k):
            if not clusters[ci]:
                # 若某簇暂时为空，保持原中心不变，避免产生 NaN
                new_centers.append(centers[ci])
            else:
                mean_v = sum(values[idx] for idx in clusters[ci]) / float(len(clusters[ci]))
                new_centers.append(mean_v)

        # 判断收敛：所有中心移动总量很小则停止
        shift = sum(abs(a - b) for a, b in zip(centers, new_centers))
        centers = new_centers
        if shift < 1e-6:
            break

    # 最终再做一次分配，得到 assignments
    assignments = []
    for v in values:
        best_c = 0
        best_dist = abs(v - centers[0])
        for ci in range(1, k):
            d = abs(v - centers[ci])
            if d < best_dist:
                best_dist = d
                best_c = ci
        assignments.append(best_c)

    return centers, assignments


# ===================== 数据库存储接口（不在 main 中调用） =====================

def save_spatial_exploration_to_db(db, exploration_results):
    """
    将“空间与资源探索倾向”分析结果写入 MongoDB 的通用接口函数。

    设计目的：
    --------------------------------------------------
    - 满足需求：“保留写回数据库的接口，但在主流程中不调用”；
    - 若未来需要把分析结果持久化，调用本函数即可将结果写入
      MLS.SpatialExplorationAnalysis 集合。

    写入数据结构说明：
    --------------------------------------------------
    对于每个 (learner_uid, course_uid) 组合，写入字段包括：

    1）基础统计（直接来自 xAPI 行为）：
       - unique_spaces     ：访问到的不同 space-id 数量；
       - unique_resources  ：聚焦过的不同资源类别数；
       - has_extension     ：是否进入可选拓展单元（0/1）；
       - path_jump         ：是否存在回访型路径（0/1）；
       - teleport_ratio    ：teleport 导航占全部导航的比例。

    2）标准化结果：
       - z_space_breadth   ：空间探索广度 z 分数；
       - z_extension_flag  ：可选拓展参与 z 分数；
       - z_resource_breadth：资源探索广度 z 分数；
       - z_path_pattern    ：路径模式 z 分数；
       - z_teleport_ratio  ：teleport 比例 z 分数。

    3）综合指数与标签：
       - exploration_index     ：未归一化的探索指数 E；
       - exploration_normalized：课程内部 min-max 归一化后的指数 E_norm ∈ [0, 1]；
       - exploration_label     ：文本标签
         （“到点即学型（低探索） / 平衡探索型（中等探索） / 高探索型探索者”）；
       - cluster_rank          ：聚类等级（0=低探索，1=中探索，2=高探索）。

    使用方式：
    --------------------------------------------------
    - 在 main() 末尾打印结果后，如需写回，可显式调用：
        save_spatial_exploration_to_db(db, exploration_results)
    - 本脚本中默认不调用该函数，以避免对现有数据库造成影响。
    """
    col = db[SPATIAL_EXPL_COLLECTION]

    # 简单策略：为便于反复试验，先清空集合再批量插入
    db.drop_collection(SPATIAL_EXPL_COLLECTION)
    col = db[SPATIAL_EXPL_COLLECTION]

    docs_to_insert = []
    for (lrn_uid, crs_uid), res in exploration_results.items():
        doc = {
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "unique_spaces": res.get("unique_spaces", 0),
            "unique_resources": res.get("unique_resources", 0),
            "has_extension": res.get("has_extension", 0),
            "path_jump": res.get("path_jump", 0),
            "teleport_ratio": res.get("teleport_ratio", 0.0),
            "z_space_breadth": res.get("z_space_breadth", 0.0),
            "z_extension_flag": res.get("z_extension_flag", 0.0),
            "z_resource_breadth": res.get("z_resource_breadth", 0.0),
            "z_path_pattern": res.get("z_path_pattern", 0.0),
            "z_teleport_ratio": res.get("z_teleport_ratio", 0.0),
            "exploration_index": res.get("E", 0.0),
            "exploration_normalized": res.get("E_norm", 0.0),
            "exploration_label": res.get("exploration_label"),
            "cluster_rank": res.get("cluster_rank"),
            "created_at": datetime.utcnow(),
        }
        docs_to_insert.append(doc)

    if docs_to_insert:
        col.insert_many(docs_to_insert, ordered=False)
        col.create_index(
            [("learner_uid", 1), ("course_uid", 1)],
            name="idx_learner_course"
        )
        print(f"[接口调用] 已写入 SpatialExplorationAnalysis 文档数：{len(docs_to_insert)}")
    else:
        print("[接口调用] 没有可写入 SpatialExplorationAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    """
    主流程：
    --------------------------------------------------
    1. 连接 MongoDB，获取 Interaction 与 LearnerProfile 集合；
    2. 读取具有 exploration_orientation.score 人设的学习者列表；
    3. 对其中一部分（采样或全部）学习者，加载与空间/资源探索相关的 xAPI 事件；
    4. 按 (学习者, 课程) 聚合基础特征（空间广度、资源广度、支线参与、路径回访、teleport 比例）；
    5. 在课程内部做 z 标准化，并计算探索指数 E 与归一化指数 E_norm；
    6. 使用一维 k-means 对 E_norm 聚类，生成“低 / 中 / 高探索”标签；
    7. 将行为侧 global_exploration 与人设侧 exploration_orientation.score 做相关分析；
    8. 打印整体结果与统计信息（不写回数据库）。
    """
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]

    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中的人设（exploration_orientation） ----------
    print("读取 LearnerProfile 中空间探索倾向维度的人设信息...")
    persona_scores = {}  # learner_uid -> persona_exploration_score

    cursor_profiles = profile_col.find(
        {},
        {"learner_uid": 1, "global_profile": 1}
    )

    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g_profile = doc.get("global_profile") or {}
        expl = g_profile.get("exploration_orientation") or {}
        score = expl.get("score")
        if score is not None:
            persona_scores[lrn_uid] = float(score)

    all_learners_with_persona = list(persona_scores.keys())
    print(f"LearnerProfile 中具有 exploration_orientation.score 的学习者数量：{len(all_learners_with_persona)}")

    if not all_learners_with_persona:
        print("未找到任何具有空间探索人设的学习者，无法进行本维度分析。")
        return

    # ---------- 3. 随机采样学习者 ----------
    if N_SAMPLE > 0 and N_SAMPLE < len(all_learners_with_persona):
        sampled_learners = random.sample(all_learners_with_persona, N_SAMPLE)
    else:
        sampled_learners = all_learners_with_persona
    sampled_set = set(sampled_learners)

    print(f"随机选取用于分析的学习者数量：{len(sampled_learners)} (N_SAMPLE = {N_SAMPLE})")

    # ---------- 4. 加载采样学习者的空间/资源相关事件 ----------
    """
    事件筛选策略与论文依据：
    --------------------------------------------------
    1）选取的 xAPI 动词：
       - navigated-to-space / teleported-to-space：
         * 对应 OmiLAXR 中“场景/子场景进入事件”；
         * 与 RePiX VR 仪表盘中用于构建空间热力图和路径的位姿数据同源；
         * 用于统计空间探索广度和路径模式，并区分 teleport 与其他导航方式。
       - explored-extension：
         * 对应 Gamified LA 中“可选挑战/支线任务参与”的行为；
         * 本脚本以“是否至少出现一次”作为“是否主动探索可选内容”的信号。
       - focused-on-resource：
         * 对应 XR+LA/EDM 综述中提出的“停留区域与资源使用多样性”；
         * 本脚本以“覆盖的资源类别数”作为资源探索广度的代理。

    2）MongoDB 查询条件：
       - 限制 _lrn_uid ∈ sampled_learners；
       - verb.id ∈ {上述四种行为动词}；
       - 按 timestamp 升序排序，便于构造空间访问序列。
    """
    query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": {
            "$in": [
                VERBS["navigated_to_space"],
                VERBS["teleported_to_space"],
                VERBS["explored_extension"],
                VERBS["focused_on_resource"],
            ]
        }
    }

    print("统计待加载的相关事件数量（count_documents）...")
    total_events = xapi_col.count_documents(query)
    print(f"准备加载的空间/资源相关事件条数：{total_events}")

    cursor_events = xapi_col.find(
        query,
        {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "context": 1,
            "timestamp": 1,
        }
    ).sort("timestamp", 1)

    print("开始加载事件...")
    events = list(cursor_events)
    print(f"已从 MongoDB 读取事件条数：{len(events)}")

    if not events:
        print("未找到任何空间/资源相关事件，无法进行本维度分析。")
        return

    # ---------- 5. 按 (学习者, 课程) 聚合原始计数 ----------
    """
    特征聚合目标：
    --------------------------------------------------
    对每个 (学习者, 课程) 组合，构建以下基础统计特征：

    - nav_spaces         : set[str]，访问过的 space-id 集合；
    - nav_sequence       : List[str]，按时间排序的 space-id 序列（用于路径模式分析）；
    - nav_walk_count     : 采用“非 teleport 方式”的导航次数（例如行走、平移）；
    - nav_teleport_count : 采用 teleport 方式的导航次数；
    - extension_count    : explored-extension 事件次数（多为 0 或 1）；
    - focus_targets      : set[str]，聚焦过的资源/界面区域 ID；
    - focus_count        : focused-on-resource 事件次数。

    这些特征对应的论文观点：
    --------------------------------------------------
    - nav_spaces / nav_sequence：
      * 对应 OmiLAXR 与 RePiX VR 中的空间路径分析；
      * 用于刻画空间探索广度与路径形态。
    - nav_walk_count / nav_teleport_count：
      * 对应 RePiX VR 中区分不同导航方式的做法；
      * 用 teleport_ratio 反映“偏好瞬移 vs 慢走”。
    - extension_count：
      * 对应 Gamified LA 中“支线任务参与度”的概念；
      * 用二值 has_extension 表示“是否有探索可选内容”。
    - focus_targets：
      * 对应 XR+LA/EDM 综述中的“资源多样性与停留区域”；
      * 用 unique_resources 表示资源探索广度。
    """
    per_lrn_course = {}

    for doc in events:
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue

        if lrn_uid not in sampled_set:
            continue

        key = (lrn_uid, crs_uid)
        if key not in per_lrn_course:
            per_lrn_course[key] = {
                "nav_spaces": set(),
                "nav_sequence": [],
                "nav_walk_count": 0,
                "nav_teleport_count": 0,
                "extension_count": 0,
                "focus_targets": set(),
                "focus_count": 0,
            }

        rec = per_lrn_course[key]
        verb_id = (doc.get("verb") or {}).get("id")
        context = doc.get("context") or {}
        ext = context.get("extensions") or {}

        if verb_id in (VERBS["navigated_to_space"], VERBS["teleported_to_space"]):
            space_id = ext.get(EXT_SPACE_ID)
            if space_id:
                rec["nav_spaces"].add(space_id)
                rec["nav_sequence"].append(space_id)

            nav_mode = ext.get(EXT_NAV_MODE)
            # 若事件本身就是 teleported-to-space，或扩展字段中标记为 teleport，则记为瞬移
            if verb_id == VERBS["teleported_to_space"] or nav_mode == "teleport":
                rec["nav_teleport_count"] += 1
            else:
                rec["nav_walk_count"] += 1

        elif verb_id == VERBS["explored_extension"]:
            rec["extension_count"] += 1

        elif verb_id == VERBS["focused_on_resource"]:
            target_id = ext.get(EXT_FOCUS_TARGET_ID)
            if target_id:
                rec["focus_targets"].add(target_id)
                rec["focus_count"] += 1

    print(f"参与空间/资源行为统计的 (学习者, 课程) 组合数量：{len(per_lrn_course)}")

    if not per_lrn_course:
        print("没有任何包含空间/资源行为的学习者-课程组合，结束分析。")
        return

    # ---------- 6. 计算每个 (学习者, 课程) 的基础特征 ----------
    """
    对每个 (lrn_uid, crs_uid) 计算：
    - unique_spaces    : len(nav_spaces)，空间探索广度；
    - unique_resources : len(focus_targets)，资源探索广度；
    - has_extension    : 0/1，是否至少参与过一次可选拓展单元；
    - path_jump        : 0/1，是否存在回访型路径（调用 compute_path_jump_flag）；
    - teleport_ratio   : nav_teleport_count / (nav_walk_count + nav_teleport_count)。

    同时为每门课程收集这些值，用于后续“课程内部”均值/标准差计算。
    """
    base_metrics = {}  # (lrn, course) -> dict
    per_course_values = defaultdict(lambda: {
        "space_counts": [],
        "resource_counts": [],
        "extension_flags": [],
        "path_flags": [],
        "teleport_ratios": [],
    })

    for (lrn_uid, crs_uid), rec in per_lrn_course.items():
        unique_spaces = len(rec["nav_spaces"])
        unique_resources = len(rec["focus_targets"])
        has_extension = 1 if rec["extension_count"] > 0 else 0
        path_jump = compute_path_jump_flag(rec["nav_sequence"])

        total_nav = rec["nav_walk_count"] + rec["nav_teleport_count"]
        teleport_ratio = (rec["nav_teleport_count"] / float(total_nav)) if total_nav > 0 else 0.0

        base_metrics[(lrn_uid, crs_uid)] = {
            "unique_spaces": unique_spaces,
            "unique_resources": unique_resources,
            "has_extension": has_extension,
            "path_jump": path_jump,
            "teleport_ratio": teleport_ratio,
        }

        stats = per_course_values[crs_uid]
        stats["space_counts"].append(unique_spaces)
        stats["resource_counts"].append(unique_resources)
        stats["extension_flags"].append(has_extension)
        stats["path_flags"].append(path_jump)
        stats["teleport_ratios"].append(teleport_ratio)

    # ---------- 7. 在课程内部做 z 标准化并计算探索指数 ----------
    """
    标准化与指数构造的思路：
    --------------------------------------------------
    1）课程内部标准化：
       - 对每门课程，分别对以下特征计算均值与标准差，然后转为 z 分数：
         · space_counts      → z_space_breadth
         · resource_counts   → z_resource_breadth
         · extension_flags   → z_extension_flag
         · path_flags        → z_path_pattern
         · teleport_ratios   → z_teleport_ratio
       - 这样可以消除不同课程之间“空间数量、资源布局”等结构性差异的影响，
         让 z 值更多反映“在同一课程中的相对位置”。

    2）加权组合为探索指数 E：
       - 设定权重：
         · w_space = 0.35 （空间探索广度权重最高）
         · w_ext   = 0.30 （支线/可选拓展参与权重次高）
         · w_res   = 0.20 （资源探索广度权重中等）
         · w_path  = 0.10 （路径回访模式权重略低）
         · w_tp    = 0.05 （teleport 比例权重最小）
       - 探索指数：
         E_i = (w_space * z_space
                + w_ext * z_ext
                + w_res * z_res
                + w_path * z_path
                + w_tp * z_tp) / sqrt(w_space^2 + ... + w_tp^2)
       - 分母对权重向量做 L2 归一化，使不同权重组合下的 E 值尺度更稳定。

    3）E 的解释：
       - E 越大，说明学习者在空间广度、支线参与、资源多样性、路径回访等方面都更偏“高探索”；
       - E 越小，说明整体行为更接近“到点就学”（只走主线、少访问新空间/新资源）。
    """
    exploration_results = {}  # (lrn, crs) -> 指标结果
    course_E_values = defaultdict(list)  # crs_uid -> [E_i, ...]

    w_space = 0.35
    w_ext = 0.30
    w_res = 0.20
    w_path = 0.10
    w_tp = 0.05
    w_norm = sqrt(w_space ** 2 + w_ext ** 2 + w_res ** 2 + w_path ** 2 + w_tp ** 2)

    for crs_uid, stats in per_course_values.items():
        mean_space, std_space = compute_mean_std(stats["space_counts"])
        mean_res, std_res = compute_mean_std(stats["resource_counts"])
        mean_ext, std_ext = compute_mean_std(stats["extension_flags"])
        mean_path, std_path = compute_mean_std(stats["path_flags"])
        mean_tp, std_tp = compute_mean_std(stats["teleport_ratios"])

        for (lrn_uid, c_uid), m in base_metrics.items():
            if c_uid != crs_uid:
                continue

            unique_spaces = m["unique_spaces"]
            unique_resources = m["unique_resources"]
            has_extension = m["has_extension"]
            path_jump = m["path_jump"]
            teleport_ratio = m["teleport_ratio"]

            z_space = (unique_spaces - mean_space) / std_space if std_space > 1e-6 else 0.0
            z_res = (unique_resources - mean_res) / std_res if std_res > 1e-6 else 0.0
            z_ext = (has_extension - mean_ext) / std_ext if std_ext > 1e-6 else 0.0
            z_path = (path_jump - mean_path) / std_path if std_path > 1e-6 else 0.0
            z_tp = (teleport_ratio - mean_tp) / std_tp if std_tp > 1e-6 else 0.0

            E = (
                w_space * z_space +
                w_ext * z_ext +
                w_res * z_res +
                w_path * z_path +
                w_tp * z_tp
            ) / (w_norm if w_norm > 1e-6 else 1.0)

            exploration_results[(lrn_uid, crs_uid)] = {
                "unique_spaces": unique_spaces,
                "unique_resources": unique_resources,
                "has_extension": has_extension,
                "path_jump": path_jump,
                "teleport_ratio": teleport_ratio,
                "z_space_breadth": z_space,
                "z_extension_flag": z_ext,
                "z_resource_breadth": z_res,
                "z_path_pattern": z_path,
                "z_teleport_ratio": z_tp,
                "E": E,
            }
            course_E_values[crs_uid].append(E)

    # ---------- 8. 在课程内部对探索指数 E 做 min-max 归一化 ----------
    for crs_uid, E_list in course_E_values.items():
        if not E_list:
            continue
        E_min = min(E_list)
        E_max = max(E_list)
        span = E_max - E_min

        for (lrn_uid, c_uid), res in exploration_results.items():
            if c_uid != crs_uid:
                continue
            E = res["E"]
            if span < 1e-6:
                # 若所有人的指数几乎相同，则统一给 0.5，表示“中等探索”
                E_norm = 0.5
            else:
                E_norm = (E - E_min) / span
            res["E_norm"] = E_norm

    print("已完成课程内标准化与探索指数计算。")

    # ---------- 9. 基于 exploration_normalized 做聚类并打标签 ----------
    all_records = list(exploration_results.items())
    E_norm_values = [res["E_norm"] for _, res in all_records if "E_norm" in res]

    if not E_norm_values:
        print("没有可用于聚类的探索指数，结束分析。")
        return

    centers, assignments = kmeans_1d(E_norm_values, k=3, max_iter=50)

    # 将簇按中心从低到高排序，用 rank 0/1/2 对应“低/中/高探索”
    cluster_with_center = list(enumerate(centers))
    cluster_with_center.sort(key=lambda x: x[1])
    cluster_to_rank = {cluster_idx: rank for rank, (cluster_idx, _) in enumerate(cluster_with_center)}

    rank_to_label = {
        0: "到点即学型（低探索）",
        1: "平衡探索型（中等探索）",
        2: "高探索型探索者",
    }

    for ((key, res), cluster_idx) in zip(all_records, assignments):
        rank = cluster_to_rank.get(cluster_idx, 1)
        label = rank_to_label.get(rank, "平衡探索型（中等探索）")
        res["cluster_index"] = int(cluster_idx)
        res["cluster_rank"] = int(rank)
        res["exploration_label"] = label

    # 统计标签分布
    label_counts = defaultdict(int)
    for res in exploration_results.values():
        label_counts[res["exploration_label"]] += 1

    print("=========================================================")
    print("【空间与资源探索倾向维度：标签分布概览】")
    for label, cnt in label_counts.items():
        print(f"- {label}: {cnt} 个 (学习者, 课程) 记录")
    print("=========================================================")

    # ---------- 10. 与 LearnerProfile 中的人设做全局对比 ----------
    """
    对比思路：
    --------------------------------------------------
    - 行为侧：
      对每个学习者，将其在所有课程上的 exploration_normalized（E_norm）取平均，
      得到 global_exploration（行为侧“空间探索倾向”指数）。
    - 人设侧：
      使用 LearnerProfile.global_profile.exploration_orientation.score 作为预设人设分数。
    - 相关分析：
      使用皮尔逊相关系数衡量两者的一致程度，仅作为整体方向一致性的粗略验证；
      不把人设分数反向用于训练或修正行为分析结果。
    """
    learner_behavior_global = defaultdict(list)
    for (lrn_uid, crs_uid), res in exploration_results.items():
        E_norm = res.get("E_norm")
        if E_norm is not None:
            learner_behavior_global[lrn_uid].append(E_norm)

    learner_behavior_avg = {}
    for lrn_uid, vals in learner_behavior_global.items():
        if vals:
            learner_behavior_avg[lrn_uid] = sum(vals) / float(len(vals))

    xs = []  # persona score
    ys = []  # behavior global_exploration

    for lrn_uid in sampled_learners:
        persona_score = persona_scores.get(lrn_uid)
        analyzed_expl = learner_behavior_avg.get(lrn_uid)
        if persona_score is not None and analyzed_expl is not None:
            xs.append(float(persona_score))
            ys.append(float(analyzed_expl))

    if len(xs) >= 2:
        mean_x, std_x = compute_mean_std(xs)
        mean_y, std_y = compute_mean_std(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / float(len(xs))
        if std_x > 1e-6 and std_y > 1e-6:
            corr = cov / (std_x * std_y)
        else:
            corr = 0.0

        avg_global_expl = sum(ys) / float(len(ys))
        avg_persona_score = sum(xs) / float(len(xs))

        print("=========================================================")
        print("【空间与资源探索倾向维度：人设 vs 行为分析 全局对比】")
        print(f"- 采样学习者数量（具有探索人设分数）：{len(sampled_learners)}")
        print(f"- 实际参与相关分析的学习者数量：{len(xs)}")
        print(f"- 行为侧 global_exploration 平均值：{avg_global_expl:.3f}")
        print(f"- 人设侧 exploration_orientation.score 平均值：{avg_persona_score:.3f}")
        print(f"- 皮尔逊相关系数：{corr:.3f}")
        print("  （相关系数用于粗略验证：细粒度 xAPI 空间/资源分析是否与人设维度方向一致。）")
        print("=========================================================")
    else:
        print("参与人设 vs 行为对比的学习者样本太少，无法计算相关系数。")

    print("空间与资源探索倾向维度分析完成。")

    # 若将来需要将结果写入数据库，可在此处手动解除注释：
    # save_spatial_exploration_to_db(db, exploration_results)


if __name__ == "__main__":
    main()
