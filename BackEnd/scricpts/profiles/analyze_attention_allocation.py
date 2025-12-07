# -*- coding: utf-8 -*-
"""
分析维度：注意力分配与信息加工方式
（Attention Allocation & Information Processing Style）

脚本目标（与画像设计文档对齐）：
--------------------------------------------------
对应画像设计中的维度：
【注意力分配与信息加工方式（Attention Allocation & Information Processing Style）】，
主要关心：
- 学习者在 video / VR / AR / 交互单元中，把注意力（注视）分配在哪些资源类型上：
  文本、图像/3D 模型、提示/示例、界面/其他；
- 在一个学习单元内部，先看哪一类信息（文本先、图像先、示例先），
  以及注视顺序的典型模式；
- 将这种注意力分配 / 顺序模式与最终 answered / passed 的表现关联起来，
  区分“相对高效的注意策略”和“低效或分散的注意策略”。

本脚本依赖的数据来源：
--------------------------------------------------
1）细粒度 xAPI 行为（MongoDB.MLS.Interaction）：
   - 由 xAPI_interaction_profile.py 依据粗粒度数据生成，
     包含以下与本维度强相关的动词与扩展字段：:）
       * verb.id = focused-on-resource
         - result.duration: 注视该资源的时长（秒）被编码为 ISO8601 时长字符串 PT{n}S；
         - context.extensions["https://legend-meta.com/xapi/ext/focus-target-id"]:
             被注视资源/区域的 ID（例如：main-screen, subtitle-area, diagram-area,
             vr-object-1, ar-object-2, hint-panel 等）；
         - context.extensions["https://legend-meta.com/xapi/ext/unit-type"]:
             学习单元类别（video/vr/ar/interact/cooperate）。:contentReference[oaicite:3]{index=3}
       * verb.id = observed-peer
         - result.duration: 观看同伴示范的时长（秒），视作“示例/演示”类注意。:contentReference[oaicite:4]{index=4}
       * verb.id = answered / passed / completed
         - result.success / completion: 用于刻画学习表现（正确率/通过率）。

2）学习者画像（MongoDB.MLS.LearnerProfile）：
   - 由生成细粒度行为脚本中的 infer_persona_for_course 写入，
     包含 global_profile.attention_allocation.score（0~1）。:contentReference[oaicite:5]{index=5}

3）xAPI Profile 设计：
   - 动词与扩展键的命名与含义参考了你给出的元宇宙 xAPI Profile，
     如 attention-event / fixationMs / gazeMethod 等概念。:contentReference[oaicite:6]{index=6}

论文依据与分析方法设计：
--------------------------------------------------
一）为什么用“fixation/AOI 分布 + 注视顺序”刻画信息加工方式？

1. Coşkun et al. (2022) 对沉浸式 VR 学习环境中的眼动研究做系统综述时指出：
   - 最常用的眼动指标之一就是“在不同兴趣区（AOI）上的注视时长与分布”，
     用来推断学习者把注意力放在文字、图片、3D 对象还是界面元素上；
   - AOI 之间的注视转换序列可用于分析“先看哪类信息”和“搜索策略”。:contentReference[oaicite:7]{index=7}

2. Baceviciute et al. (2022) 在“冗余原则”VR 实证研究中，用眼动和 EEG 联合分析：
   - 比较先看文本还是先看图像的学习者，其认知负荷和成绩差异；
   - 通过“文本区 vs 图像区的注视时间比例”和“先看哪种表征”来刻画信息加工差异。:contentReference[oaicite:8]{index=8}

3. Heinemann et al. (2024) 提出的 RePiX VR 学习分析仪表盘中，
   将“视线集中在关键对象 / 文本 / 提示上的程度”和“在非任务区域的游离”作为评价
   学习策略与潜在沉浸度的重要维度。:contentReference[oaicite:9]{index=9}

→ 因此，本脚本采用：
   - “按 AOI 类型统计注视时长比例”刻画注意力分配；
   - “首个注视 AOI 类型分布”刻画进入单元时的加工顺序（文本优先/图像优先/示例优先）。

二）为什么可以使用统计指标和简单聚类，而不是复杂深度模型？

1. Tao et al. (2025) 对沉浸式虚拟学习环境 LA 综述指出：
   - 现有研究中，统计方法（相关、回归、ANOVA）仍然是主流，
     多模态/机器学习方法应用相对较少；:contentReference[oaicite:10]{index=10}
   - 眼动/空间/行为数据与成绩的关系，通常使用“注视时长比例 + 表现”做统计比较。
2. Coşkun 等的眼动综述也强调，大部分实证工作基于注视时长、注视次数、转换序列等
   经典指标进行统计分析，而不是复杂的黑箱模型。:contentReference[oaicite:11]{index=11}

→ 因此，本脚本采用：
   - 基于 fixation 时长比例和表现的标准化指标；
   - 使用一维 k-means 聚类划分“高效/中等/低效注意策略”，
     属于 LA 领域常用、可解释性较好的方法，而不是简单计数。

三）本脚本采用的核心指标和分类标签设计：

1. AOI 类型归类（基于 xAPI ext/focus-target-id）：
   - 文本型 AOI（text）：
     * ID 中包含 "subtitle", "caption", "text", "label"
   - 图像 / 3D 模型型 AOI（visual）：
     * 包含 "diagram", "image", "picture", "screen", "model",
       或以 "vr-object-", "ar-object-" 开头
   - 示例 / 提示 / 演示型 AOI（example）：
     * 包含 "hint", "tip", "example", "demo", "solution"
     * verb = observed-peer 的事件也视作 example 类注意
   - 界面 / 其他 AOI（ui_other）：
     * 不属于上述三类的 AOI，如菜单、背景 UI 等。

2. 注意力分配指标（per (learner, course)）：
   - text_ratio      = 文本类注视时长 / 全部注视时长
   - visual_ratio    = 图像/模型类注视时长 / 全部注视时长
   - example_ratio   = 示例/提示类注视时长 / 全部注视时长
   - ui_ratio        = 界面/其他类注视时长 / 全部注视时长
   - relevant_ratio  = text + visual + example 三类之和，
                       作为“任务相关注意”的比例代理

3. 信息加工顺序指标：
   - 对每个 (学习者, 课程, 单元) 找出最早的 focused-on-resource 事件，
     记录其 AOI 类型，统计在课程中：
       * first_text_ratio    = 首注视为 text 的单元比例
       * first_visual_ratio  = 首注视为 visual 的单元比例
       * first_example_ratio = 首注视为 example 的单元比例

4. 注意策略分类标签（Processing Style Label）：
   - 文本优先型加工（text-first）
     * text_ratio 为四类中最大，且 ≥ 0.55
       或 first_text_ratio 为三种首注视比例中的最大
   - 图像/模型优先型加工（visual-first）
     * visual_ratio 最大且 ≥ 0.55
       或 first_visual_ratio 最大
   - 示例/演示优先型加工（example-first）
     * example_ratio 最大且 ≥ 0.45
       或 first_example_ratio 最大
   - 均衡整合型（balanced-integrative）
     * 不满足以上任何单一偏好，且 relevant_ratio ≥ 0.7，
       即文本、图像、示例三者比较均衡，且大部分时间用于任务相关信息。
   - 其他情况归为“未定义/数据不足”（undefined）。

   这一设计借鉴了 Baceviciute 等对“先看文本还是先看图像”的比较，
   和 Coşkun 等对 AOI 注视分布的分组方法，但这里不直接复制实验分组，
   而是将其思想应用在你的多源 AOI 抽象上。

5. 注意策略效率指数（Attention Efficiency Index）：
   - 目标：区分在注意力分配上“有策略且任务相关” vs “散漫/大量在无关 UI 上游走”的学习者，
     并结合 answered / passed 结果，以识别“高效注意策略”。
   - 对每个 (学习者, 课程) 计算：
       * performance = answered / passed 事件的 success/ completion 平均值（0~1）
       * relevant_ratio（如上）
       * ui_ratio（如上）
   - 在课程内对 performance、relevant_ratio、ui_ratio 做 z 标准化：
       z_perf, z_rel, z_ui
   - 定义注意效率指数：
       E_att = (z_perf + z_rel - z_ui) / sqrt(3)
     解释：
       - performance 高、relevant_ratio 高、ui_ratio 低 → E_att 较大（高效策略）；
       - performance 低、relevant_ratio 低、ui_ratio 高 → E_att 较小（低效策略）；
     该设计与 Heinemann 等在 RePiX VR 中“把 gaze/attention 分布与任务表现结合”的思路一致，
     只是用标准化和线性组合形式进行了实现。

   - 为便于与 LearnerProfile 中 [0,1] 分数对齐，
     在课程内对 E_att 做 min-max 归一化，得到 E_att_norm ∈ [0,1]。

6. 注意策略效率标签（Efficiency Label）：
   - 收集所有 (学习者, 课程) 的 E_att_norm，
     使用一维 k-means（k=3）聚类得到三个簇中心；
   - 将中心从低到高排序，依次标记为：
       * 0 → 低效注意策略
       * 1 → 中等注意策略
       * 2 → 高效注意策略
   - 每条记录保存：
       * efficiency_label（中文描述）
       * cluster_rank（0/1/2，便于与其它维度统一处理）

与 LearnerProfile 人设的对比：
--------------------------------------------------
- 对每个学习者，把其在所有课程上的 E_att_norm 平均得到 behavior_attention_score；
- 与 LearnerProfile.global_profile.attention_allocation.score 做皮尔逊相关，
  用于验证行为分析与先前人设的“一致性程度”。:contentReference[oaicite:12]{index=12}
- 脚本会在控制台打印：
  * 学习者数量、具备注意画像人设的数量；
  * 各注意策略类型的样本量；
  * 行为 attention 指数与人设 attention_allocation.score 的相关系数。

数据库写回接口：
--------------------------------------------------
- 提供 save_attention_allocation_to_db(db, results) 函数，
  演示如何把结果写入 MLS.AttentionAllocationAnalysis 集合，
  但 main() 中不会调用，符合“保留写回接口但不调用”的要求。
"""

from pymongo import MongoClient
from datetime import datetime
from math import sqrt
from collections import defaultdict
import re
import random
from tqdm import tqdm

# ===================== 配置区域 =====================

MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "MLS"
XAPI_COLLECTION = "Interaction"           # 细粒度行为集合
PROFILE_COLLECTION = "LearnerProfile"     # 人设集合
ATTENTION_COLLECTION = "AttentionAllocationAnalysis"  # 注意力分配分析结果集合（仅接口，不在 main 中调用）

# 采样学习者数量：随机选取至多 N_SAMPLE 个学习者进行分析
# 若 N_SAMPLE <= 0 或大于实际数量，则使用所有学习者
N_SAMPLE = 3000

VERB_BASE = "https://legend-meta.com/xapi/verb/"

VERBS = {
    "focused_on_resource": VERB_BASE + "focused-on-resource",
    "observed_peer": VERB_BASE + "observed-peer",
    "answered": VERB_BASE + "answered",
    "passed": VERB_BASE + "passed",
    "completed": VERB_BASE + "completed",
}

# xAPI duration 解析：假定使用 "PT{秒数}S" 形式
DURATION_RE = re.compile(r"^PT(\d+)S$")

# 扩展字段 URL（与生成脚本保持一致）
EXT_UNIT_TYPE = "https://legend-meta.com/xapi/ext/unit-type"
EXT_FOCUS_TARGET = "https://legend-meta.com/xapi/ext/focus-target-id"


# ===================== 工具函数 =====================

def parse_iso8601_duration(duration_str):
    """
    解析简单形式的 ISO8601 时长字符串，例如 "PT120S"。
    若为空或格式不符合预期，返回 None。

    设计依据：
    - xAPI_interaction_profile.py 在生成 focused-on-resource、completed 等事件时，
      使用 result.duration = f"PT{int(seconds)}S" 的形式记录时长。
    """
    if not duration_str:
        return None
    m = DURATION_RE.match(duration_str)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def compute_mean_std(values):
    """
    计算均值和总体标准差：
    - 列表为空 -> (0.0, 0.0)
    - 仅一个元素 -> 标准差视为 0.0

    用于：
    - 课程内对 performance / relevant_ratio / ui_ratio 做 z 标准化；
    - 与 LearnerProfile 中分数做相关。
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
    一维 k-means 聚类（Lloyd 算法），用于基于注意效率指数 E_att_norm 划分学习者类型。

    使用 k-means 的依据：
    --------------------------------------------------
    - Heinemann 等和其他 LA 研究常用聚类把行为模式分成若干“学习者类型”；
    - 在本脚本中，特征是一维的 E_att_norm，k-means 在可解释和可实现性之间
      取得了较好平衡，且方便映射到“低/中/高”三档注意策略效率。

    参数：
        values: List[float]，一维数据（例如所有 (lrn_uid, course_uid) 的 E_att_norm）
        k: 聚类簇数，默认 3
        max_iter: 最大迭代次数

    返回：
        centers: List[float]，每个簇的中心
        assignments: List[int]，与 values 等长的簇编号列表
    """
    n = len(values)
    if n == 0 or k <= 0:
        return [], []

    if n < k:
        k = n

    v_min, v_max = min(values), max(values)
    if abs(v_max - v_min) < 1e-6:
        centers = [v_min for _ in range(k)]
        assignments = [0 for _ in range(n)]
        return centers, assignments

    centers = [
        v_min + (v_max - v_min) * (i + 0.5) / float(k)
        for i in range(k)
    ]

    for _ in range(max_iter):
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

        new_centers = centers[:]
        for ci in range(k):
            if clusters[ci]:
                new_centers[ci] = sum(values[i] for i in clusters[ci]) / float(len(clusters[ci]))
            else:
                new_centers[ci] = centers[ci]

        max_shift = max(abs(new_centers[ci] - centers[ci]) for ci in range(k))
        centers = new_centers
        if max_shift < 1e-4:
            break

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


def categorize_aoi(target_id):
    """
    根据 focus-target-id 粗略判断 AOI 类型：
    - 文本（text）
    - 图像/模型（visual）
    - 示例/提示（example）
    - 界面/其他（ui_other）

    设计依据：
    --------------------------------------------------
    - Coşkun 等和 Baceviciute 等在眼动研究中通常将 AOI 分为文字区、图像/示意图区、
      界面控件等类别，再分别统计注视时长。这里使用字符串规则近似代替实际 AOI 标注。
    """
    if not target_id:
        return "ui_other"

    tid = target_id.lower()

    # 文本区域：字幕、文字面板、说明文字等
    if ("subtitle" in tid or "caption" in tid or "text" in tid or
            "label" in tid or "title" in tid):
        return "text"

    # 图像/示意图/3D 模型/主屏
    if ("diagram" in tid or "image" in tid or "picture" in tid or
            "screen" in tid or "model" in tid or
            tid.startswith("vr-object") or tid.startswith("ar-object")):
        return "visual"

    # 提示/示例/解答/演示
    if ("hint" in tid or "tip" in tid or "example" in tid or
            "demo" in tid or "solution" in tid or "explanation" in tid):
        return "example"

    # 其他界面元素（菜单、背景板等）
    return "ui_other"


def pearson_corr(x_list, y_list):
    """
    计算皮尔逊相关系数：
    - 若样本数少于 2 或标准差为 0，则返回 0.0。
    """
    if not x_list or not y_list:
        return 0.0
    n = min(len(x_list), len(y_list))
    if n < 2:
        return 0.0
    xs = x_list[:n]
    ys = y_list[:n]
    mean_x, std_x = compute_mean_std(xs)
    mean_y, std_y = compute_mean_std(ys)
    if std_x < 1e-6 or std_y < 1e-6:
        return 0.0
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n)) / float(n)
    return cov / (std_x * std_y)


# ===================== 数据库存储接口（不在 main 中调用） =====================

def save_attention_allocation_to_db(db, results):
    """
    把注意力分配与信息加工方式分析结果写入 MongoDB 的接口函数。

    注意：
    - 本函数不会在 main() 中自动调用；
      如需持久化，可在 main() 末尾手动调用。
    - 结果集合名为 MLS.AttentionAllocationAnalysis。

    写入字段设计：
    --------------------------------------------------
    每条文档对应一个 (learner_uid, course_uid)，包含：
    - learner_uid / course_uid
    - text_ratio / visual_ratio / example_ratio / ui_ratio
    - first_text_ratio / first_visual_ratio / first_example_ratio
    - relevant_ratio
    - performance
    - attention_efficiency_index (E_att)
    - attention_efficiency_normalized (E_att_norm)
    - attention_style_label（文本优先/图像优先/示例优先/均衡整合/未定义）
    - efficiency_label（高效注意策略/中等注意策略/低效注意策略）
    - cluster_rank（0/1/2）
    """
    col = db[ATTENTION_COLLECTION]
    db.drop_collection(ATTENTION_COLLECTION)
    col = db[ATTENTION_COLLECTION]

    docs = []
    for (lrn_uid, crs_uid), info in results.items():
        doc = {
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "text_ratio": info.get("text_ratio"),
            "visual_ratio": info.get("visual_ratio"),
            "example_ratio": info.get("example_ratio"),
            "ui_ratio": info.get("ui_ratio"),
            "first_text_ratio": info.get("first_text_ratio"),
            "first_visual_ratio": info.get("first_visual_ratio"),
            "first_example_ratio": info.get("first_example_ratio"),
            "relevant_ratio": info.get("relevant_ratio"),
            "performance": info.get("performance"),
            "attention_efficiency_index": info.get("E_att"),
            "attention_efficiency_normalized": info.get("E_att_norm"),
            "attention_style_label": info.get("style_label"),
            "efficiency_label": info.get("efficiency_label"),
            "cluster_rank": info.get("cluster_rank"),
            "created_at": datetime.utcnow(),
        }
        docs.append(doc)

    if docs:
        col.insert_many(docs, ordered=False)
        col.create_index(
            [("learner_uid", 1), ("course_uid", 1)],
            name="idx_learner_course"
        )
        print(f"[接口调用] 已写入 AttentionAllocationAnalysis 文档数：{len(docs)}")
    else:
        print("[接口调用] 没有可写入 AttentionAllocationAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]

    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中的注意力人设 ----------
    print("读取 LearnerProfile 中的 attention_allocation 人设信息...")
    persona_attention = {}  # lrn_uid -> attention_allocation.score

    cursor_profiles = profile_col.find(
        {},
        {"learner_uid": 1, "global_profile": 1}
    )

    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g_profile = doc.get("global_profile") or {}
        att = g_profile.get("attention_allocation") or {}
        score = att.get("score")
        if score is not None:
            persona_attention[lrn_uid] = float(score)

    learners_with_persona = list(persona_attention.keys())
    print(f"具备注意力分配人设的学习者数量：{len(learners_with_persona)}")

    if not learners_with_persona:
        print("没有任何学习者具备 attention_allocation 人设，无法进行对比分析。")
        return

    # ---------- 3. 随机采样学习者 ----------
    if N_SAMPLE > 0 and N_SAMPLE < len(learners_with_persona):
        sampled_learners = random.sample(learners_with_persona, N_SAMPLE)
    else:
        sampled_learners = learners_with_persona

    sampled_set = set(sampled_learners)
    print(f"随机选取用于分析的学习者数量：{len(sampled_learners)} (N_SAMPLE = {N_SAMPLE})")

    # ---------- 4. 读取与注意相关的 xAPI 事件 ----------
    """
    我们需要的事件类型：
    - focused-on-resource：
      * result.duration：视为“在某资源上的注视时长”；
      * context.extensions[focus-target-id]：资源 ID → 映射到 AOI 类型；
      * _course_uid / _lrn_uid：用于按课程窗口聚合。
    - observed-peer：
      * result.duration：视为“观看同伴示范”的时间，归入 example 类。
    - answered / passed / completed：
      * result.success / completion：用于计算 performance（任务表现）。
    """
    focus_query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": VERBS["focused_on_resource"],
    }
    observed_query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": VERBS["observed_peer"],
    }
    perf_query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": {"$in": [VERBS["answered"], VERBS["passed"], VERBS["completed"]]},
    }

    print("统计 focused-on-resource 事件数量...")
    n_focus = xapi_col.count_documents(focus_query)
    print(f"focused-on-resource 事件总数：{n_focus}")

    print("统计 observed-peer 事件数量...")
    n_obs = xapi_col.count_documents(observed_query)
    print(f"observed-peer 事件总数：{n_obs}")

    print("统计 answered/passed/completed 事件数量...")
    n_perf = xapi_col.count_documents(perf_query)
    print(f"表现相关事件总数：{n_perf}")

    print("加载 focused-on-resource 事件...")
    focus_events = list(xapi_col.find(
        focus_query,
        {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "result": 1,
            "context": 1,
            "timestamp": 1,
        }
    ))
    print(f"已加载 focused-on-resource 事件数：{len(focus_events)}")

    print("加载 observed-peer 事件...")
    observed_events = list(xapi_col.find(
        observed_query,
        {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "result": 1,
        }
    ))
    print(f"已加载 observed-peer 事件数：{len(observed_events)}")

    print("加载 answered/passed/completed 事件...")
    perf_events = list(xapi_col.find(
        perf_query,
        {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "result": 1,
        }
    ))
    print(f"已加载表现事件数：{len(perf_events)}")

    if not focus_events:
        print("没有任何 focused-on-resource 事件，无法进行注意力分配分析。")
        return

    # ---------- 5. 基于 focused-on-resource 聚合 AOI 注视时长 ----------
    """
    这里以 (学习者, 课程) 作为聚合粒度，统计：
    - 各 AOI 类型的总注视时长；
    - 每个单元的“首个注视 AOI 类型”，用于计算 first_* 指标。
    """
    # (lrn_uid, course_uid) -> 累积时长
    aoi_durations = defaultdict(lambda: {
        "text": 0.0,
        "visual": 0.0,
        "example": 0.0,
        "ui_other": 0.0,
    })

    # (lrn_uid, course_uid, unit_key) -> (timestamp, aoi_type)
    first_aoi = {}

    print("聚合 focused-on-resource 事件（AOI 时长与首注视类型）...")
    for doc in tqdm(focus_events, desc="处理 focus 事件", unit="event"):
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue

        result = doc.get("result") or {}
        duration_str = result.get("duration")
        duration_sec = parse_iso8601_duration(duration_str)
        if duration_sec is None or duration_sec <= 0:
            continue

        context = doc.get("context") or {}
        ctx_ext = context.get("extensions") or {}

        target_id = ctx_ext.get(EXT_FOCUS_TARGET)
        aoi_type = categorize_aoi(target_id)

        # 解析 timestamp，用于首注视比较；若无，则用当前时间替代
        ts_str = doc.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.utcnow()
        except Exception:
            ts = datetime.utcnow()

        # 构造 unit_key：这里由于 Interaction 集合中未必直接存 unit_uid，
        # 我们用 object.id 或 context 中的信息做一个粗略 key。
        unit_key = None
        obj = doc.get("object") or {}
        obj_id = obj.get("id")
        if obj_id:
            unit_key = obj_id
        else:
            # 若 object.id 缺失，则用 course + unit-type 做一个粗粒度 key
            unit_type = ctx_ext.get(EXT_UNIT_TYPE, "unknown")
            unit_key = f"{crs_uid}:{unit_type}"

        key = (lrn_uid, crs_uid)

        # 累加 AOI 类型注视时长
        aoi_durations[key][aoi_type] += float(duration_sec)

        # 记录首注视 AOI 类型（针对每个 unit_key）
        fu_key = (lrn_uid, crs_uid, unit_key)
        if fu_key not in first_aoi:
            first_aoi[fu_key] = (ts, aoi_type)
        else:
            prev_ts, _ = first_aoi[fu_key]
            if ts < prev_ts:
                first_aoi[fu_key] = (ts, aoi_type)

    # ---------- 6. 把 observed-peer 也纳入 example 类注视 ----------
    """
    根据画像设计，“observed-peer（看别人示范 vs 看系统讲解）”
    也体现了一种“示例/演示优先”的信息加工方式，因此将其时长并入 example 类。
    """
    print("将 observed-peer 事件并入 example 类注视时长...")
    for doc in tqdm(observed_events, desc="处理 observed-peer 事件", unit="event"):
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue
        result = doc.get("result") or {}
        duration_str = result.get("duration")
        duration_sec = parse_iso8601_duration(duration_str)
        if duration_sec is None or duration_sec <= 0:
            continue
        key = (lrn_uid, crs_uid)
        aoi_durations[key]["example"] += float(duration_sec)

    # ---------- 7. 计算课程窗口内的表现 performance ----------
    """
    performance 定义为：
    - 对 answered / passed / completed 事件的 result.success / completion 的平均值。
    - 若仅有 completion 字段，则 completion=True 视为 1，否则 0。
    """
    perf_stats = defaultdict(lambda: {"sum": 0.0, "cnt": 0})
    print("聚合 answered/passed/completed 事件计算 performance...")
    for doc in tqdm(perf_events, desc="处理表现事件", unit="event"):
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue
        result = doc.get("result") or {}
        success = result.get("success")
        completion = result.get("completion")
        if success is None and completion is None:
            continue
        if success is None:
            val = 1.0 if completion else 0.0
        else:
            val = 1.0 if bool(success) else 0.0
        key = (lrn_uid, crs_uid)
        perf_stats[key]["sum"] += val
        perf_stats[key]["cnt"] += 1

    # ---------- 8. 计算各类比例与首注视比例 ----------
    """
    对于每个 (学习者, 课程)：
    - 计算 text/visual/example/ui_other 四类注视时长比例；
    - 计算 first_* 三类首注视比例。
    """
    attention_metrics = {}  # (lrn_uid, crs_uid) -> dict

    # 先准备 first 注视统计
    first_counts = defaultdict(lambda: {"text": 0, "visual": 0, "example": 0})

    for (lrn_uid, crs_uid, unit_key), (ts, aoi_type) in first_aoi.items():
        key = (lrn_uid, crs_uid)
        if aoi_type in ("text", "visual", "example"):
            first_counts[key][aoi_type] += 1

    print("计算各类型注视比例与首注视比例...")
    for key, dur_dict in aoi_durations.items():
        lrn_uid, crs_uid = key
        total_dur = sum(dur_dict.values())
        if total_dur <= 0:
            continue

        text_dur = dur_dict["text"]
        visual_dur = dur_dict["visual"]
        example_dur = dur_dict["example"]
        ui_dur = dur_dict["ui_other"]

        text_ratio = text_dur / total_dur
        visual_ratio = visual_dur / total_dur
        example_ratio = example_dur / total_dur
        ui_ratio = ui_dur / total_dur
        relevant_ratio = (text_dur + visual_dur + example_dur) / total_dur

        fc = first_counts.get(key, {})
        total_first = sum(fc.values())
        if total_first > 0:
            first_text_ratio = fc.get("text", 0) / float(total_first)
            first_visual_ratio = fc.get("visual", 0) / float(total_first)
            first_example_ratio = fc.get("example", 0) / float(total_first)
        else:
            first_text_ratio = first_visual_ratio = first_example_ratio = 0.0

        # performance
        perf = perf_stats.get(key, {"sum": 0.0, "cnt": 0})
        if perf["cnt"] > 0:
            performance = perf["sum"] / float(perf["cnt"])
        else:
            performance = None  # 可能没有相关题目/任务表现

        attention_metrics[key] = {
            "text_ratio": text_ratio,
            "visual_ratio": visual_ratio,
            "example_ratio": example_ratio,
            "ui_ratio": ui_ratio,
            "relevant_ratio": relevant_ratio,
            "first_text_ratio": first_text_ratio,
            "first_visual_ratio": first_visual_ratio,
            "first_example_ratio": first_example_ratio,
            "performance": performance,
        }

    print(f"完成注意力比例计算的 (学习者, 课程) 数量：{len(attention_metrics)}")
    if not attention_metrics:
        print("没有可用的注意力统计数据，结束。")
        return

    # ---------- 9. 基于规则生成“信息加工风格”标签 ----------
    """
    按照前面定义的规则，为每个 (学习者, 课程) 生成 attention_style_label：
    - 文本优先型 / 图像优先型 / 示例优先型 / 均衡整合型 / 未定义
    """
    def classify_style(m):
        tr = m["text_ratio"]
        vr = m["visual_ratio"]
        er = m["example_ratio"]
        rr = m["relevant_ratio"]
        ftr = m["first_text_ratio"]
        fvr = m["first_visual_ratio"]
        fer = m["first_example_ratio"]

        # 找出整体注视占比最高的类型
        max_ratio = max(tr, vr, er)
        # 找出首注视占比最大的类型
        first_max = max(ftr, fvr, fer)

        # 文本优先
        if (tr == max_ratio and tr >= 0.55) or (ftr == first_max and ftr > 0):
            return "文本优先型加工（进入或整体上更偏向文字信息）"

        # 图像/模型优先
        if (vr == max_ratio and vr >= 0.55) or (fvr == first_max and fvr > 0):
            return "图像/模型优先型加工（进入或整体上更偏向图像/3D 模型）"

        # 示例/演示优先
        if (er == max_ratio and er >= 0.45) or (fer == first_max and fer > 0):
            return "示例/演示优先型加工（更偏向提示、示例或同伴演示）"

        # 均衡整合：文本/图像/示例都不特别极端，且大部分注意力在任务相关信息上
        if rr >= 0.7 and max_ratio <= 0.6:
            return "均衡整合型加工（在文本/图像/示例之间较为均衡地分配注意）"

        # 其他情况视为未定义或数据不足
        return "加工风格未明（数据不足或注意非常分散）"

    for key, m in attention_metrics.items():
        attention_metrics[key]["style_label"] = classify_style(m)

    # ---------- 10. 计算注意效率指数 E_att ----------
    """
    对每门课程内部：
    - 收集该课程中所有 (学习者, 课程) 的：
        * performance
        * relevant_ratio
        * ui_ratio
    - 对三个指标分别做 z 标准化，计算：
        E_att = (z_perf + z_rel - z_ui) / sqrt(3)
    - 再对该课程内的 E_att 做 min-max 归一化，得到 E_att_norm ∈ [0,1]。
    """
    course_to_entries = defaultdict(list)  # crs_uid -> list[(lrn_uid, metrics)]
    for (lrn_uid, crs_uid), m in attention_metrics.items():
        course_to_entries[crs_uid].append((lrn_uid, m))

    print("按课程计算注意效率指数 E_att 及归一化 E_att_norm...")
    attention_results = {}  # (lrn_uid, crs_uid) -> dict（扩展 attention_metrics）

    for crs_uid, entries in course_to_entries.items():
        if not entries:
            continue

        perf_vals = []
        rel_vals = []
        ui_vals = []
        for (lrn_uid, m) in entries:
            # performance 可能为 None（无题目/任务），此时用 0.5 作为中性值
            perf = m["performance"]
            if perf is None:
                perf = 0.5
            perf_vals.append(perf)
            rel_vals.append(m["relevant_ratio"])
            ui_vals.append(m["ui_ratio"])

        mean_perf, std_perf = compute_mean_std(perf_vals)
        mean_rel, std_rel = compute_mean_std(rel_vals)
        mean_ui, std_ui = compute_mean_std(ui_vals)

        E_vals = []
        tmp_store = {}

        for idx, (lrn_uid, m) in enumerate(entries):
            perf = perf_vals[idx]
            rel = rel_vals[idx]
            ui = ui_vals[idx]

            z_perf = (perf - mean_perf) / std_perf if std_perf > 1e-6 else 0.0
            z_rel = (rel - mean_rel) / std_rel if std_rel > 1e-6 else 0.0
            z_ui = (ui - mean_ui) / std_ui if std_ui > 1e-6 else 0.0

            E_att = (z_perf + z_rel - z_ui) / sqrt(3.0)
            tmp_store[(lrn_uid, crs_uid)] = {
                "z_perf": z_perf,
                "z_rel": z_rel,
                "z_ui": z_ui,
                "E_att": E_att,
            }
            E_vals.append(E_att)

        if not E_vals:
            continue

        E_min = min(E_vals)
        E_max = max(E_vals)
        span = E_max - E_min if E_max > E_min else 0.0

        for (lrn_uid, m) in entries:
            key = (lrn_uid, crs_uid)
            base = tmp_store[key]
            E_att = base["E_att"]
            if span > 1e-6:
                E_norm = (E_att - E_min) / span
            else:
                E_norm = 0.5

            # 汇总所有指标到 attention_results
            res = dict(attention_metrics[key])  # 复制前面计算的比例和 style
            res["z_perf"] = base["z_perf"]
            res["z_rel"] = base["z_rel"]
            res["z_ui"] = base["z_ui"]
            res["E_att"] = E_att
            res["E_att_norm"] = E_norm
            attention_results[key] = res

    print("课程层面的注意效率计算完成。")
    print(f"结果条目数（学习者-课程对）：{len(attention_results)}")

    if not attention_results:
        print("没有任何注意效率结果，结束分析。")
        return

    # ---------- 11. 基于 E_att_norm 的注意效率聚类 ----------
    all_E_norm = [res["E_att_norm"] for res in attention_results.values()]
    centers, assignments = kmeans_1d(all_E_norm, k=3, max_iter=50)

    if centers:
        sorted_idx = sorted(range(len(centers)), key=lambda i: centers[i])
        cluster_to_rank = {cluster_idx: rank for rank, cluster_idx in enumerate(sorted_idx)}

        rank_to_label = {
            0: "低效注意策略（任务表现较低、任务相关注意比例较低且在非任务 UI 区域停留较多）",
            1: "中等注意策略（任务相关注意与表现处于中间水平）",
            2: "高效注意策略（在关键资源上集中注意、较少停留在无关 UI，且表现较好）",
        }

        # 给每条记录赋予效率标签
        for (key, res), cluster_idx in zip(attention_results.items(), assignments):
            rank = cluster_to_rank.get(cluster_idx, 1)
            res["cluster_rank"] = rank
            res["efficiency_label"] = rank_to_label.get(rank, "注意效率未定义")
    else:
        # 如果 k-means 退化，全部视为中等
        for res in attention_results.values():
            res["cluster_rank"] = 1
            res["efficiency_label"] = "注意效率未定义（聚类退化）"

    # ---------- 12. 与 LearnerProfile 中的注意力人设对比 ----------
    """
    对每个学习者：
    - behavior_attention_score = 在所有课程上的 E_att_norm 平均值；
    - persona_attention_score  = LearnerProfile.global_profile.attention_allocation.score；
    - 计算两者的皮尔逊相关系数。
    """
    behavior_attention = defaultdict(list)
    for (lrn_uid, crs_uid), res in attention_results.items():
        behavior_attention[lrn_uid].append(res["E_att_norm"])

    behavior_scores = {}
    for lrn_uid, vals in behavior_attention.items():
        if vals:
            behavior_scores[lrn_uid] = sum(vals) / float(len(vals))

    common_learners = [uid for uid in behavior_scores.keys() if uid in persona_attention]
    print(f"既有行为注意指数又有人设注意分数的学习者数量：{len(common_learners)}")

    beh_list = [behavior_scores[uid] for uid in common_learners]
    per_list = [persona_attention[uid] for uid in common_learners]
    corr = pearson_corr(beh_list, per_list)
    print(f"行为注意效率指数 与 人设 attention_allocation.score 的皮尔逊相关：{corr:.4f}")

    # ---------- 13. 输出整体分布摘要 ----------
    style_counter = defaultdict(int)
    eff_counter = defaultdict(int)
    for res in attention_results.values():
        style_counter[res["style_label"]] += 1
        eff_counter[res["efficiency_label"]] += 1

    print("\n=== 注意力分配与信息加工方式：风格分布 ===")
    for label, cnt in style_counter.items():
        print(f"{label}: {cnt}")

    print("\n=== 注意力分配与信息加工方式：效率类型分布 ===")
    for label, cnt in eff_counter.items():
        print(f"{label}: {cnt}")

    print("\n（如需将结果写入数据库，可手动调用 save_attention_allocation_to_db(db, attention_results)）")


if __name__ == "__main__":
    main()
