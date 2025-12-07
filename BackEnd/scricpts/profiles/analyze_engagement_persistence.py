# -*- coding: utf-8 -*-
"""
分析维度：行为投入度与坚持性（Behavioral Engagement & Persistence）

脚本目标（与画像设计文档对齐）：
--------------------------------------------------
对应《论文以及画像.txt》中“3. 画像设计”里关于：
【行为投入度与坚持性（Behavioral Engagement & Persistence）】的维度，
核心关注：
- 学习者在元宇宙/沉浸式环境中的行为参与度（交互量、完成率）；
- 面对困难时的坚持性（持续时间、重试行为、额外练习）；
- “挂机/走神”等低投入行为；
- 长期价值交换（token 等）作为持续参与度的辅助指标。

本脚本采用的行为代理与论文依据：
--------------------------------------------------
1）行为参与度：完成率、交互量、在线时长
   - 论文列表第 4 篇 Lampropoulos & Evangelidis 2025（Immersive Learning LA/EDM 综述）
     在 LA 指标表中将：
       * task completion rate（任务完成率）
       * dropout rate（辍学率）
       * number of tries（尝试次数）
       * time-on-task（任务耗时）
     视为典型的“行为参与度（behavioral engagement）”与“坚持性（persistence）”指标。
   - 因此本脚本中：
       * 使用单元完成率（unit completion rate）作为主要参与度指标；
       * 使用单位单元的交互量（interaction volume / unit）作为次级参与度指标；
       * 使用 active time（任务有效时长）作为整体投入的时间代理。

2）行为数据作为沉浸式学习中最常用的数据类型
   - 论文列表第 5 篇 Tao 等 2025 对沉浸式虚拟学习环境 LA 的系统综述指出：
       * 绝大多数研究使用“行为/日志数据”（完成率、交互次数、点击流）作为主要分析基础；
       * 行为数据常被用于度量参与度、动机与坚持性。
   - 因此本脚本完全基于 xAPI 行为日志（Interaction 集合）构造投入度与坚持性的指标，
     不额外引入问卷或生理数据。

3）重试次数、挑战选择与 gamified LA 中的坚持性
   - 论文列表第 8 篇 Papamitsiou et al. 2024 在 Gamified LA for IVR Games 的研究中，
     将：
       * retry attempts（重试次数）
       * 自选更高难度挑战 / 额外挑战（optional challenges）
     明确和 persistence / frustration tolerance（坚持性/挫折耐受度）相关联。
   - 论文列表第 9 篇 Papamitsiou et al. 2025 进一步在 HICSS 工作中，
     将“失败后重新挑战”和“主动延长游戏参与”的行为视为 gamified LA 中的
     持续参与指标。
   - 因此本脚本中：
       * 对 answered / performed-procedure-step 中“先错后对”的题目/步骤进行统计，
         作为“行为层面的重试率”；
       * 使用 explored-extension（额外练习/可选单元）在整体失败基础上的比例，
         作为“失败后愿不愿意自发多练习”的坚持性代理。

4）AFK / 挂机行为与 remained-idle
   - Lampropoulos & Evangelidis 的综述在“time-on-task / inactivity”相关部分提到，
     长时间无操作可视为低参与度或潜在 dropout 的信号。
   - xAPI_interaction_profile.py 中为 video / cooperate 等单元生成 remained-idle 事件，
     result.duration 表示“超出阈值的空闲时长”，context.extensions 中记录阈值。
   - 本脚本使用：
       * idle_ratio = idle_time / (idle_time + active_time)
     作为“挂机/走神比例”的逆向指标（越高代表越低的行为投入度）。

5）价值交换系统与长期参与度（LEARNER-C）
   - 论文列表第 11 篇 Tsai（LEARNER-C: Analysis in Educational Metaverse Environments）
     将教育元宇宙视为“value exchange system”，强调：
       * 学习者通过自然交互和代币/价值交换持续参与；
       * 长期参与可通过价值交换轨迹来建模。
   - xAPI_interaction_profile.py 中为 exchanged-value 事件设置：
       * context.extensions[".../value-token-type"]
       * context.extensions[".../value-change"]
     用于表达贡献获得的 token。
   - 因此本脚本中：
       * 将 exchanged-value 的频次和 value-change 总量作为“长期参与与价值贡献”
         的一个辅助维度。

6）标签设计与聚类方法
   - 任务效率脚本中已采用一维 k-means（k=3）将连续效率指数划分为三档，
     依据来自：
       * Heinemann et al. 2024（OmiLAXR/xAPI for LA in VR）和
       * Lampropoulos & Evangelidis 2025 对“基于行为模式分群画像”的支持。
   - 本脚本沿用同样的一维 k-means 方法，将“行为投入度与坚持性指数 EP_norm”
     离散化为三种标签：
       * 低投入易放弃型学习者
       * 中等投入型学习者
       * 高投入高坚持型学习者
     以匹配画像框架中 engagement / perseverance.level 的“low/medium/high”三档设置。

7）本脚本与原文方法的差异与改动说明
   - 原文中多以单一指标（完成率、重试次数、在线时长）分别分析参与度或坚持性；
   - 本脚本在充分尊重原文指标选择的前提下：
       * 使用课程内 z 标准化将多种行为指标统一到同一量纲；
       * 依据文献强调程度设置权重，线性合成“行为投入度与坚持性指数 EP”：
           EP = (w1*z_completion + w2*z_retry + w3*z_extension
                 + w4*z_value + w5*z_interaction - w6*z_idle) / sqrt(sum w^2)
       * 再执行一维 k-means 进行三档聚类。
   - 这一“加权合成 + 聚类”的做法是系统工程层面的集成，文献为“选取哪些行为
     作为重要指标”提供依据，而指标聚合与聚类是为了与现有画像框架对齐。

脚本功能总结：
--------------------------------------------------
1. 从 MongoDB 中读取：
   - 细粒度 xAPI 行为：MLS.Interaction 集合；
   - 学习者画像：MLS.LearnerProfile 集合（特别是 global_profile.engagement /
     global_profile.perseverance.score）。

2. 对每个 (学习者, 课程)：
   - 统计如下行为指标：
     * completion_rate：单元完成率（unit_completed / unit_started）
     * interaction_per_unit：单位单元的行为事件数量
     * retry_rate：题目/步骤“先错后对”的比例（体现重试行为）
     * extension_rate：explored-extension 次数 / 失败题目数（失败后额外练习）
     * idle_ratio：空闲时长 / (空闲 + 有效时长)
     * value_rate：exchanged-value 频次/价值变化，归一化到课程规模
   - 在课程内部对上述指标做 z 标准化，并按文献权重线性合成 EP 指数；
   - 在全体 (学习者, 课程) 范围内对 EP 做 min-max 归一化得到 EP_norm ∈ [0, 1]。

3. 基于所有 (学习者, 课程) 的 EP_norm：
   - 使用一维 k-means（k=3）进行聚类；
   - 根据聚类中心从低到高排序，将每条记录标记为：
       “低投入易放弃型学习者 / 中等投入型学习者 / 高投入高坚持型学习者”。

4. 与人设对比：
   - 对每个学习者，将其在所有课程上的 EP_norm 取平均，得到 global_engage_persist；
   - 分别与 LearnerProfile.global_profile.engagement.score 和
     global_profile.perseverance.score 做皮尔逊相关，
     粗略验证行为侧分析与人设在该维度上的一致性。

5. 数据库存储接口（不在 main() 中调用）：
   - 定义 save_engagement_persistence_to_db(db, ep_results) 函数，
     演示如何把结果写入 MLS.EngagementPersistenceAnalysis 集合；
   - 按你的要求，main() 中默认不调用该接口，未来如需持久化可手动解除注释。
"""

from pymongo import MongoClient
from datetime import datetime
from collections import defaultdict
from math import sqrt
import random
import re
from tqdm import tqdm

# ===================== 配置区域 =====================

MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "MLS"
XAPI_COLLECTION = "Interaction"             # 细粒度行为集合（xAPI_interaction_profile.py 生成）
PROFILE_COLLECTION = "LearnerProfile"       # 人设集合
EP_COLLECTION = "EngagementPersistenceAnalysis"  # 行为投入度与坚持性分析结果集合（仅接口）

# 采样学习者数量：随机选取至多 N_SAMPLE 个学习者进行分析
# 若 N_SAMPLE <= 0 或大于实际可选数量，则使用所有具备 engagement/perseverance 人设的学习者
N_SAMPLE = 3000

VERB_BASE = "https://legend-meta.com/xapi/verb/"

VERBS = {
    # 单元初始化与完成：用于估算完成率 / dropout（Lampropoulos & Evangelidis）
    "initialized": VERB_BASE + "initialized",
    "completed": VERB_BASE + "completed",
    # 题目作答：用于统计错误与“先错后对”的重试情况（Papamitsiou 的 gamified LA）
    "answered": VERB_BASE + "answered",
    # 程序步骤执行：同样用于识别步骤级“先错后对”的重试行为
    "performed_procedure_step": VERB_BASE + "performed-procedure-step",
    # 额外练习/扩展单元：failure 之后的主动 extension 视为坚持性的体现
    "explored_extension": VERB_BASE + "explored-extension",
    # AFK / 挂机：remained-idle 的时长对应低投入行为
    "remained_idle": VERB_BASE + "remained-idle",
    # 价值交换：长期参与度与贡献的辅助指标（LEARNER-C）
    "exchanged_value": VERB_BASE + "exchanged-value",
}

# ISO8601 时长解析正则，仅支持 "PT{整数秒}S" 格式（与 xAPI_interaction_profile.py 中生成形式一致）
DURATION_RE = re.compile(r"^PT(\d+)S$")

# 上下文扩展字段常量前缀
CTX_EXT_BASE = "https://legend-meta.com/xapi/ext/"
EXT_STEP_ID = CTX_EXT_BASE + "step-id"
EXT_UNIT_OPTIONAL = CTX_EXT_BASE + "unit-optional"
EXT_IDLE_THRESHOLD = CTX_EXT_BASE + "idle-threshold-seconds"
EXT_VALUE_CHANGE = CTX_EXT_BASE + "value-change"


# ===================== 工具函数 =====================

def parse_iso8601_duration(duration_str):
    """
    解析简单形式的 ISO8601 时长字符串，例如："PT120S"
    若为空或格式不符，返回 None。

    设计说明：
    - xAPI_interaction_profile.py 在生成 completed / remained-idle 等事件时，
      使用 result.duration = f"PT{int(seconds)}S" 的形式保存时长。
    - 为保持与生成脚本一致，这里只需支持“PT{秒数}S”的整数秒格式。
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
    计算一组数的均值和标准差（总体标准差）：
    - 列表为空 -> (0, 0)
    - 仅一个元素 -> 标准差视为 0

    用途：
    - 对课程内所有学习者的行为指标（completion_rate / retry_rate / idle_ratio 等）
      计算 z 分数；
    - 在全局对比中计算人设分数与行为指数的相关系数。
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
    一维 k-means 聚类（Lloyd 算法实现），用于基于 EP_norm 自动划分学习者类型。

    文献与画像依据：
    --------------------------------------------------
    1）“用行为模式做聚类画像”的直接依据：
        - Heinemann et al. 2024（OmiLAXR 框架）在行为类型总结中指出：
          不同学生在相同任务阶段的行为模式可用于学习者分群与画像。
    2）Lampropoulos & Evangelidis 2025 的综述也将“基于行为特征的分群”
       列为 LA/EDM 典型方法之一。
    3）你的画像框架中，engagement / perseverance.level 已被设计为
       low / medium / high 三档，因此使用 k=3 的一维聚类来自动确定三档边界。

    返回：
    - centers: list[float] 聚类中心
    - assignments: list[int] 每个值所属的聚类编号（0 ~ k-1）
    """
    n = len(values)
    if n == 0 or k <= 0:
        return [], []

    # 若样本数小于 k，则退化为每个样本一个簇
    if n <= k:
        centers = list(values)
        assignments = list(range(n))
        return centers, assignments

    # 随机初始化 k 个中心（从样本中随机抽取）
    centers = random.sample(list(values), k)

    for _ in range(max_iter):
        # 分配样本到最近的中心
        clusters = [[] for _ in range(k)]
        for v in values:
            dists = [abs(v - c) for c in centers]
            idx = dists.index(min(dists))
            clusters[idx].append(v)

        # 更新中心
        new_centers = []
        for idx in range(k):
            if clusters[idx]:
                new_centers.append(sum(clusters[idx]) / float(len(clusters[idx])))
            else:
                # 若某个簇空了，随机重新初始化一个中心
                new_centers.append(random.choice(values))

        # 收敛判定
        if all(abs(new_centers[i] - centers[i]) < 1e-6 for i in range(k)):
            centers = new_centers
            break
        centers = new_centers

    # 最终分配
    assignments = []
    for v in values:
        dists = [abs(v - c) for c in centers]
        idx = dists.index(min(dists))
        assignments.append(idx)

    return centers, assignments


# ===================== 数据库存储接口（默认不在 main 中调用） =====================

def save_engagement_persistence_to_db(db, ep_results):
    """
    将 (学习者, 课程) 维度的行为投入度与坚持性分析结果写入 MongoDB。

    参数：
    - db: MongoDB 数据库对象（client[DB_NAME]）
    - ep_results: dict[(lrn_uid, crs_uid)] -> {
          "completion_rate": ...,
          "interaction_per_unit": ...,
          "retry_rate": ...,
          "extension_rate": ...,
          "idle_ratio": ...,
          "value_rate": ...,
          "EP": ...,
          "EP_norm": ...,
          "label": ...,
          "cluster_rank": ...,
      }

    字段与含义（与本脚本分析逻辑对应）：
    --------------------------------------------------
    1）completion_rate：
       - 单元完成率（unit_completed / unit_started），对应 Lampropoulos & Evangelidis 中
         提出的 completion rate / dropout rate 行为参与度指标。
    2）interaction_per_unit：
       - 单位单元行为事件数，体现行为交互量，受到 Tao 等综述中“行为数据是最常用类型”
         的启发，作为参与度的量化代理。
    3）retry_rate：
       - 题目与步骤层面的“先错后对”比例，参考 Papamitsiou 的 Gamified LA 研究中
         将重试行为与坚持性相关联的观点。
    4）extension_rate：
       - explored-extension 次数与失败题目数的比值，用于度量“失败后是否主动去做额外练习”，
         同样参考 Papamitsiou 对 optional challenges 的讨论。
    5）idle_ratio：
       - remained-idle 空闲时长占（空闲 + 有效时长）的比例，逆向反映 AFK / 走神程度，
         与 Lampropoulos & Evangelidis 中关于 time-on-task / inactivity 的讨论一致。
    6）value_rate：
       - exchanged-value 的频次/价值变化归一化到课程规模，借鉴 LEARNER-C
         将 educational metaverse 视为 value exchange system 的思路。
    7）EP / EP_norm：
       - EP 为 z 标准化后加权合成的行为投入度与坚持性指数；
       - EP_norm 为在全体 (学习者, 课程) 上做 min-max 归一化后的结果（∈ [0, 1]）。
    8）label / cluster_rank：
       - 通过一维 k-means 聚类结果给出的三档类型：
         * cluster_rank 越大表示行为投入度与坚持性越高；
         * label 为中文可读标签：
             低投入易放弃型学习者 / 中等投入型学习者 / 高投入高坚持型学习者。

    注意：
    --------------------------------------------------
    - 本函数不会在 main() 中自动调用。
    - 如果你希望实际写回数据库，请在 main() 中手动解除注释：
        save_engagement_persistence_to_db(db, ep_results)
    """
    ep_col = db[EP_COLLECTION]

    # 为了方便重复实验，这里先清空集合（如果不希望清空，可改为 upsert 策略）
    db.drop_collection(EP_COLLECTION)
    ep_col = db[EP_COLLECTION]

    docs = []
    for (lrn_uid, crs_uid), res in ep_results.items():
        doc = {
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "completion_rate": res.get("completion_rate"),
            "interaction_per_unit": res.get("interaction_per_unit"),
            "retry_rate": res.get("retry_rate"),
            "extension_rate": res.get("extension_rate"),
            "idle_ratio": res.get("idle_ratio"),
            "value_rate": res.get("value_rate"),
            "EP": res.get("EP"),
            "EP_norm": res.get("EP_norm"),
            "label": res.get("label"),
            "cluster_rank": res.get("cluster_rank"),
            "created_at": datetime.utcnow(),
        }
        docs.append(doc)

    if docs:
        ep_col.insert_many(docs, ordered=False)
        ep_col.create_index(
            [("learner_uid", 1), ("course_uid", 1)],
            name="idx_learner_course"
        )
        print(f"[接口调用] 已写入 EngagementPersistenceAnalysis 文档数：{len(docs)}")
    else:
        print("[接口调用] 没有可写入 EngagementPersistenceAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]

    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中 engagement / perseverance 人设 ----------
    print("读取 LearnerProfile 中的 engagement / perseverance 人设信息。")
    persona_engagement = {}    # lrn_uid -> engagement.score
    persona_perseverance = {}  # lrn_uid -> perseverance.score

    cursor_profiles = profile_col.find(
        {},
        {"learner_uid": 1, "global_profile": 1}
    )

    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g_profile = doc.get("global_profile") or {}
        eng = (g_profile.get("engagement") or {}).get("score")
        per = (g_profile.get("perseverance") or {}).get("score")

        # 按你的要求：人设数值不用于分析计算，仅做对比，因此这里只做记录。
        if eng is not None:
            persona_engagement[lrn_uid] = float(eng)
        if per is not None:
            persona_perseverance[lrn_uid] = float(per)

    all_learners_with_persona = list(
        set(persona_engagement.keys()) | set(persona_perseverance.keys())
    )
    print(f"具备 engagement / perseverance 人设的学习者数量：{len(all_learners_with_persona)}")

    if not all_learners_with_persona:
        print("没有任何学习者具备 engagement / perseverance 人设，无法进行对比分析。")
        return

    # ---------- 3. 随机采样学习者 ----------
    if N_SAMPLE > 0 and N_SAMPLE < len(all_learners_with_persona):
        sampled_learners = random.sample(all_learners_with_persona, N_SAMPLE)
    else:
        sampled_learners = all_learners_with_persona
    sampled_set = set(sampled_learners)

    print(f"随机选取用于分析的学习者数量：{len(sampled_learners)} (N_SAMPLE = {N_SAMPLE})")

    # ---------- 4. 一次性加载采样学习者的相关事件 ----------
    """
    事件筛选策略与论文依据：
    --------------------------------------------------
    1）使用的 verb：
       - initialized / completed：
         * 对应单元级“开始-完成”行为，用于估计 completion rate 与 dropout rate，
           与 Lampropoulos & Evangelidis 的综述中行为参与度指标一致。
       - answered：
         * 对应题目作答行为，可用于识别“先错后对”的重试情况，
           与 Papamitsiou gamified LA 中的 retry attempts 指标呼应。
       - performed-procedure-step：
         * 对应流程/实验/临床步骤级别的操作行为，同样用于识别步骤级“先错后对”。
       - explored-extension：
         * 对应“额外练习/可选单元”，失败后仍愿意做 extension 被视为坚持性体现。
       - remained-idle：
         * 对应“长时间无操作”，Lampropoulos & Evangelidis 中将 inactivity 视为
           低参与度 / 潜在 dropout 的信号。
       - exchanged-value：
         * 对应 LEARNER-C 框架中的价值交换行为，作为长期参与的辅助指标。
    2）查询条件：
       - 仅针对采样学习者（_lrn_uid in sampled_learners），避免全量计算；
       - 只保留上述几个 verb.id 的事件。
    """
    verb_ids_to_use = [
        VERBS["initialized"],
        VERBS["completed"],
        VERBS["answered"],
        VERBS["performed_procedure_step"],
        VERBS["explored_extension"],
        VERBS["remained_idle"],
        VERBS["exchanged_value"],
    ]

    query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": {"$in": verb_ids_to_use},
    }

    print("统计待加载的事件数量（count_documents）...")
    total_events = xapi_col.count_documents(query)
    print(f"与采样学习者相关的行为投入度/坚持性事件总数：{total_events}")

    print("开始一次性加载所有相关事件到内存（list）...")
    events = list(xapi_col.find(
        query,
        {
            "verb.id": 1,
            "result": 1,
            "context": 1,
            "object.id": 1,
            "_lrn_uid": 1,
            "_course_uid": 1,
            "_unt_uid": 1,
            "_type": 1,
            "timestamp": 1,
        }
    ))
    print(f"已从 MongoDB 读取事件条数：{len(events)}")

    if not events:
        print("没有任何可用于行为投入度与坚持性分析的事件。")
        return

    # ---------- 5. 聚合中间统计量 ----------
    """
    聚合逻辑说明：
    --------------------------------------------------
    聚合粒度：以 (学习者, 课程) 为单位，与画像设计中“按课程窗口刻画维度”保持一致。

    对每个 (学习者, 课程)，累计以下统计：
    1）单元参与与完成：
       - units_started: 开始过的单元 uid 集合（通过任意相关事件判断）
       - units_completed: 有 completed 且 completion=True 的单元 uid 集合
    2）交互量与时长：
       - event_count: 参与上述动词的事件总数
       - active_time: 来自 completed / performed-procedure-step 的有效时长总和
       - idle_time: 来自 remained-idle 的空闲时长总和
    3）重试行为（题目层面与步骤层面）：
       - question_events[(lrn, crs, item_id)] = 按时间顺序的 success 序列
       - step_events[(lrn, crs, step_id)] = 按时间顺序的 success 序列
       后续根据“先错后对”计算 retry 相关统计。
    4）失败后额外练习：
       - extension_count: explored-extension 事件次数
    5）价值交换：
       - value_events: exchanged-value 事件次数
       - value_change_sum: value-change 数值之和（若存在）
    """
    # (lrn_uid, crs_uid) -> 聚合统计
    agg = defaultdict(lambda: {
        "units_started": set(),
        "units_completed": set(),
        "event_count": 0,
        "active_time": 0.0,
        "idle_time": 0.0,
        "extension_count": 0,
        "value_events": 0,
        "value_change_sum": 0.0,
        # retry 相关计数稍后通过 question_events / step_events 计算
        "q_fail_count": 0,
        "q_fail_then_success": 0,
        "step_fail_count": 0,
        "step_fail_then_success": 0,
    })

    # 题目与步骤级别的事件序列
    question_events = defaultdict(list)  # (lrn_uid, crs_uid, obj_id) -> [{'t': ts, 'success': bool}]
    step_events = defaultdict(list)      # (lrn_uid, crs_uid, step_id) -> [{'t': ts, 'success': bool}]

    print("开始遍历事件并构建中间统计量...")
    for doc in tqdm(events, desc="处理事件", unit="event"):
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue

        verb_id = doc.get("verb", {}).get("id") or doc.get("verb.id")
        result = doc.get("result") or {}
        context = doc.get("context") or {}
        extensions = context.get("extensions") or {}
        obj_id = doc.get("object", {}).get("id") or doc.get("object.id")
        unt_uid = doc.get("_unt_uid")
        utype = doc.get("_type")  # video / vr / ar / interact / cooperate / question / course-level
        timestamp = doc.get("timestamp") or ""

        key_lc = (lrn_uid, crs_uid)
        stat = agg[key_lc]

        # 标记单元参与（question / course-level 不计入单元集合）
        if unt_uid and utype and utype not in ("question", "course-level"):
            stat["units_started"].add(unt_uid)

        # 事件计数（行为交互量）
        stat["event_count"] += 1

        # ---- completed：用于完成率与 active_time 统计 ----
        if verb_id == VERBS["completed"]:
            # 仅对单元级 completed 统计完成情况（忽略 course-level）
            if unt_uid and utype and utype not in ("question", "course-level"):
                completion_flag = result.get("completion")
                if completion_flag is True:
                    stat["units_completed"].add(unt_uid)

            # 解析时长，累积 active_time
            dur_sec = parse_iso8601_duration(result.get("duration"))
            if dur_sec is not None and dur_sec > 0:
                stat["active_time"] += float(dur_sec)

        # ---- performed-procedure-step：步骤级 active_time 与 retry 序列 ----
        elif verb_id == VERBS["performed_procedure_step"]:
            dur_sec = parse_iso8601_duration(result.get("duration"))
            if dur_sec is not None and dur_sec > 0:
                stat["active_time"] += float(dur_sec)

            # 记录步骤事件序列，用于后续“先错后对”判断
            step_id = extensions.get(EXT_STEP_ID)
            if step_id:
                success_flag = bool(result.get("success"))
                step_events[(lrn_uid, crs_uid, step_id)].append({
                    "t": timestamp,
                    "success": success_flag,
                })

        # ---- answered：题目级 retry 序列 ----
        elif verb_id == VERBS["answered"]:
            # 对题目级 success 进行记录
            success_flag = bool(result.get("success"))
            if obj_id:
                question_events[(lrn_uid, crs_uid, obj_id)].append({
                    "t": timestamp,
                    "success": success_flag,
                })

        # ---- explored-extension：失败后主动额外练习的代理 ----
        elif verb_id == VERBS["explored_extension"]:
            stat["extension_count"] += 1

        # ---- remained-idle：空闲时长（AFK/挂机） ----
        elif verb_id == VERBS["remained_idle"]:
            dur_sec = parse_iso8601_duration(result.get("duration"))
            if dur_sec is not None and dur_sec > 0:
                stat["idle_time"] += float(dur_sec)

        # ---- exchanged-value：价值交换行为（LEARNER-C） ----
        elif verb_id == VERBS["exchanged_value"]:
            stat["value_events"] += 1
            value_change = extensions.get(EXT_VALUE_CHANGE)
            try:
                if value_change is not None:
                    stat["value_change_sum"] += float(value_change)
            except Exception:
                # 若 value_change 无法转为数值则忽略
                pass

        # initialized 在本分析中仅用于辅助理解整体流程，不单独统计字段
        # （是否初始化不会直接影响最终 EP 指数）

    # ---------- 6. 基于题目与步骤序列统计重试行为 ----------
    """
    重试行为统计说明：
    --------------------------------------------------
    对于每个问题或步骤，若事件序列中存在：
      - 至少一次失败（success=False），且
      - 在失败之后至少一次成功（success=True），
    则认为该问题/步骤上存在“先错后对”的重试行为。

    统计：
      - q_fail_count / q_fail_then_success
      - step_fail_count / step_fail_then_success

    对应 Papamitsiou 的 Gamified LA 研究中对 retry attempts 的定义：
    不只是“做了多少次”，而是特别关注“失败后是否仍然选择继续尝试直到成功”。
    """
    # 题目层面
    for (lrn_uid, crs_uid, qid), seq in question_events.items():
        if not seq:
            continue
        seq_sorted = sorted(seq, key=lambda x: x["t"])
        had_wrong = False
        had_wrong_then_success = False
        for ev in seq_sorted:
            if not ev["success"]:
                had_wrong = True
            elif ev["success"] and had_wrong:
                had_wrong_then_success = True
                break
        if had_wrong:
            agg[(lrn_uid, crs_uid)]["q_fail_count"] += 1
        if had_wrong_then_success:
            agg[(lrn_uid, crs_uid)]["q_fail_then_success"] += 1

    # 步骤层面
    for (lrn_uid, crs_uid, step_id), seq in step_events.items():
        if not seq:
            continue
        seq_sorted = sorted(seq, key=lambda x: x["t"])
        had_wrong = False
        had_wrong_then_success = False
        for ev in seq_sorted:
            if not ev["success"]:
                had_wrong = True
            elif ev["success"] and had_wrong:
                had_wrong_then_success = True
                break
        if had_wrong:
            agg[(lrn_uid, crs_uid)]["step_fail_count"] += 1
        if had_wrong_then_success:
            agg[(lrn_uid, crs_uid)]["step_fail_then_success"] += 1

    # ---------- 7. 计算每个 (学习者, 课程) 的行为指标 ----------
    """
    指标定义（均在 0~1 左右或经标准化）：
    --------------------------------------------------
    1）completion_rate：
       = units_completed / units_started（若分母为 0 则记为 0.0）
       对应 Lampropoulos & Evangelidis 综述中的 completion rate/dropout rate。

    2）interaction_per_unit：
       = event_count / max(units_started, 1)
       行为交互量归一化到单元数，受 Tao 等“行为数据为主”的综述启发。

    3）retry_rate：
       = (q_fail_then_success + step_fail_then_success) /
         max(q_fail_count + step_fail_count, 1)
       对应 Papamitsiou gamified LA 中“失败后仍然重试直到成功”的坚持性指标。

    4）extension_rate：
       = extension_count / max(q_fail_count, 1)
       失败题目越多且 extension 越多，说明越愿意在失败后做额外练习。
       若没有失败题目但有 extension，可以视为较强的探索/坚持，这里仍会产生较高值。

    5）idle_ratio（逆向指标）：
       = idle_time / max(idle_time + active_time, 1)
       对应 Lampropoulos & Evangelidis 中 time-on-task / inactivity 的讨论：
       比例越高，说明挂机/走神越严重，行为投入度越低。

    6）value_rate：
       = value_events / max(units_started, 1)
       或可改为 value_change_sum / max(units_started, 1)，本脚本采用事件频次，
       视为 LEARNER-C 中 value exchange system 的简单代理。
    """
    ep_results = {}  # (lrn_uid, crs_uid) -> 指标与 EP 值
    course_metrics = defaultdict(lambda: defaultdict(list))  # crs_uid -> metric_name -> [values]

    for (lrn_uid, crs_uid), stat in agg.items():
        units_started = len(stat["units_started"])
        units_completed = len(stat["units_completed"])
        event_count = stat["event_count"]
        active_time = stat["active_time"]
        idle_time = stat["idle_time"]
        extension_count = stat["extension_count"]
        value_events = stat["value_events"]
        q_fail = stat["q_fail_count"]
        q_retry = stat["q_fail_then_success"]
        step_fail = stat["step_fail_count"]
        step_retry = stat["step_fail_then_success"]

        if units_started == 0 and event_count == 0:
            # 完全没有有效行为数据的课程窗口直接跳过
            continue

        # 完成率
        if units_started > 0:
            completion_rate = units_completed / float(units_started)
        else:
            completion_rate = 0.0

        # 单位单元交互量
        denom_units = float(units_started) if units_started > 0 else 1.0
        interaction_per_unit = event_count / denom_units

        # 重试率
        total_fail = q_fail + step_fail
        total_retry = q_retry + step_retry
        if total_fail > 0:
            retry_rate = total_retry / float(total_fail)
        else:
            # 没有失败，可以理解为“不需要重试”，这里给一个中性值 0.5
            retry_rate = 0.5

        # 失败后 extension 率
        if q_fail > 0:
            extension_rate = extension_count / float(q_fail)
        else:
            # 没有失败但有 extension，则更偏向探索/坚持；这里使用
            # extension_count / (units_started + 1) 的形式给一点加成
            extension_rate = extension_count / float(units_started + 1)

        # idle 比例
        total_time_for_idle = idle_time + active_time
        if total_time_for_idle > 0:
            idle_ratio = idle_time / float(total_time_for_idle)
        else:
            idle_ratio = 0.0  # 没有时长信息，视为没有挂机

        # value 率（简单用事件频次归一化）
        value_rate = value_events / denom_units

        ep_results[(lrn_uid, crs_uid)] = {
            "completion_rate": completion_rate,
            "interaction_per_unit": interaction_per_unit,
            "retry_rate": retry_rate,
            "extension_rate": extension_rate,
            "idle_ratio": idle_ratio,
            "value_rate": value_rate,
            # EP / EP_norm 暂时留空，后面补充
        }

        # 为课程内标准化收集指标
        course_metrics[crs_uid]["completion_rate"].append(completion_rate)
        course_metrics[crs_uid]["interaction_per_unit"].append(interaction_per_unit)
        course_metrics[crs_uid]["retry_rate"].append(retry_rate)
        course_metrics[crs_uid]["extension_rate"].append(extension_rate)
        course_metrics[crs_uid]["idle_ratio"].append(idle_ratio)
        course_metrics[crs_uid]["value_rate"].append(value_rate)

    if not ep_results:
        print("没有任何 (学习者, 课程) 具备可用的行为指标，无法计算 EP 指数。")
        return

    # ---------- 8. 课程内 z 标准化并合成 EP 指数 ----------
    """
    EP 指数计算说明：
    --------------------------------------------------
    1）对每门课程 crs_uid 内部：
       - 分别对 completion_rate / interaction_per_unit / retry_rate /
         extension_rate / idle_ratio / value_rate 进行 z 标准化。
    2）合成行为投入度与坚持性指数 EP：
       - 受文献中各指标被强调程度启发设定权重：
           w_completion = 1.5   # 完成率是参与度的核心
           w_retry      = 1.5   # 重试行为是坚持性的核心
           w_extension  = 1.2   # 失败后额外练习次之
           w_value      = 1.0   # 价值交换为长期参与的辅助
           w_interact   = 1.0   # 交互量为常规参与度指标
           w_idle       = 1.2   # AFK/挂机为负向指标
       - EP = (w_c*z_c + w_r*z_r + w_e*z_e + w_v*z_v + w_i*z_i - w_idle*z_idle)
              / sqrt(w_c^2 + w_r^2 + w_e^2 + w_v^2 + w_i^2 + w_idle^2)
       - 其中 idle_ratio 的 z 分数以负号进入（挂机高 -> EP 降低）。
    3）再对全体 (学习者, 课程) 的 EP 做 min-max 归一化得到 EP_norm ∈ [0,1]。
    """
    # 预先计算每门课程内各指标的均值与标准差
    course_stats = {}  # crs_uid -> metric_name -> (mean, std)
    for crs_uid, m_dict in course_metrics.items():
        course_stats[crs_uid] = {}
        for metric_name, vals in m_dict.items():
            mean_v, std_v = compute_mean_std(vals)
            course_stats[crs_uid][metric_name] = (mean_v, std_v)

    # 权重设定（见上文说明）
    w_completion = 1.5
    w_retry = 1.5
    w_extension = 1.2
    w_value = 1.0
    w_interact = 1.0
    w_idle = 1.2

    denom_w = sqrt(
        w_completion ** 2 +
        w_retry ** 2 +
        w_extension ** 2 +
        w_value ** 2 +
        w_interact ** 2 +
        w_idle ** 2
    )

    all_EP = []

    for (lrn_uid, crs_uid), res in ep_results.items():
        stats_course = course_stats.get(crs_uid, {})
        def z_score(metric_key):
            val = res.get(metric_key)
            mean_v, std_v = stats_course.get(metric_key, (0.0, 0.0))
            if std_v <= 1e-6:
                return 0.0
            return (val - mean_v) / float(std_v)

        z_c = z_score("completion_rate")
        z_r = z_score("retry_rate")
        z_e = z_score("extension_rate")
        z_v = z_score("value_rate")
        z_i = z_score("interaction_per_unit")
        z_idle = z_score("idle_ratio")

        EP = (
            w_completion * z_c +
            w_retry * z_r +
            w_extension * z_e +
            w_value * z_v +
            w_interact * z_i -
            w_idle * z_idle
        )
        EP = EP / denom_w if denom_w > 0 else EP

        ep_results[(lrn_uid, crs_uid)]["EP"] = EP
        all_EP.append(EP)

    # 全局 min-max 归一化 EP -> EP_norm
    if all_EP:
        min_EP = min(all_EP)
        max_EP = max(all_EP)
        for key, res in ep_results.items():
            EP = res.get("EP", 0.0)
            if max_EP > min_EP:
                EP_norm = (EP - min_EP) / float(max_EP - min_EP)
            else:
                EP_norm = 0.5  # 所有值相同，给中性值
            res["EP_norm"] = EP_norm
    else:
        for key, res in ep_results.items():
            res["EP_norm"] = 0.5

    # ---------- 9. 基于 EP_norm 做一维 k-means 聚类 ----------
    values_norm = [res["EP_norm"] for res in ep_results.values()]
    centers, assignments = kmeans_1d(values_norm, k=3, max_iter=50)

    if not centers:
        print("EP_norm 聚类失败（样本过少），跳过标签划分。")
        for res in ep_results.values():
            res["label"] = "未分组"
            res["cluster_rank"] = None
    else        :
        # 将中心从小到大排序，并映射到 0/1/2 排名
        sorted_centers = sorted((c, idx) for idx, c in enumerate(centers))
        center_to_rank = {orig_idx: rank for rank, (c, orig_idx) in enumerate(sorted_centers)}

        # 将 assignment 映射回每个 (lrn_uid, crs_uid)
        keys_lc = list(ep_results.keys())
        label_counter = defaultdict(int)

        for key_lc, cluster_idx in zip(keys_lc, assignments):
            res = ep_results[key_lc]
            rank = center_to_rank.get(cluster_idx, 1)
            res["cluster_rank"] = rank

            # 根据 rank 指定中文标签
            if rank == 0:
                label = "低投入易放弃型学习者"
            elif rank == 1:
                label = "中等投入型学习者"
            else:
                label = "高投入高坚持型学习者"

            res["label"] = label
            label_counter[label] += 1

        print("=========================================================")
        print("【行为投入度与坚持性维度：聚类标签分布】")
        for label, cnt in label_counter.items():
            print(f"- {label}：{cnt} 条 (学习者, 课程) 记录")
        print("=========================================================")

    # ---------- 10. （可选）写回数据库接口——默认不调用 ----------
    """
    如你在需求中所述：
    - 当前版本脚本只需完成“读取细粒度 xAPI → 计算 EP 指数 → 输出标签与人设对比”，
      不需要真正把结果写回数据库。
    - 上面定义的 save_engagement_persistence_to_db(db, ep_results) 即为“写回接口”；
      若未来需要，可手动解除下面的注释。

    示例（默认注释掉）：
        save_engagement_persistence_to_db(db, ep_results)
    """
    # 若需要写回数据库，请取消下一行注释：
    # save_engagement_persistence_to_db(db, ep_results)

    # ---------- 11. 按学习者汇总 global_engage_persist 并与人设对比 ----------
    """
    验证思路与文献参考：
    --------------------------------------------------
    1）行为侧指标：
       - 对每个学习者，把其在所有课程上的 EP_norm 取平均，得到 global_engage_persist。
       - 该值仍处于 [0,1] 区间，代表行为数据推断出的“整体行为投入度与坚持性水平”。

    2）人设侧指标：
       - LearnerProfile.global_profile.engagement.score 与
         LearnerProfile.global_profile.perseverance.score
         是在粗粒度统计基础上，infer_persona_for_course 推断得到的画像分数。

    3）对比目的：
       - 通过皮尔逊相关系数，检验“基于细粒度 xAPI 的行为投入度与坚持性分析”和
         “基于粗粒度统计的人设维度”在总体趋势上是否一致。
       - 若相关为正且显著，说明细粒度 xAPI 分析与已有画像设计在该维度上方向一致，
         有助于增强画像的可信度。

    4）与文献的关系：
       - Lampropoulos & Evangelidis 综述中强调行为参与度/坚持性是 LA 的重要目标；
         本脚本在其基础上构建了更细粒度的 EP 指数，并与既有画像进行一致性检验。
       - Tao 等综述则支持“使用行为/日志数据度量参与度与坚持性”的整体方法论。
    """
    learner_to_ep_vals = defaultdict(list)
    for (lrn_uid, crs_uid), res in ep_results.items():
        learner_to_ep_vals[lrn_uid].append(res["EP_norm"])

    learner_global_ep = {}
    for lrn_uid, vals in learner_to_ep_vals.items():
        if vals:
            learner_global_ep[lrn_uid] = sum(vals) / float(len(vals))

    # 计算与 engagement 人设的相关
    xs_eng = []  # 人设 engagement 分数
    ys_eng = []  # 行为侧 global_engage_persist

    # 计算与 perseverance 人设的相关
    xs_per = []
    ys_per = []

    for lrn_uid in sampled_learners:
        behavior_score = learner_global_ep.get(lrn_uid)
        if behavior_score is None:
            continue

        eng_score = persona_engagement.get(lrn_uid)
        if eng_score is not None:
            xs_eng.append(float(eng_score))
            ys_eng.append(float(behavior_score))

        per_score = persona_perseverance.get(lrn_uid)
        if per_score is not None:
            xs_per.append(float(per_score))
            ys_per.append(float(behavior_score))

    def compute_pearson(xs, ys):
        if len(xs) < 2 or len(ys) < 2:
            return None, 0.0, 0.0
        mean_x, std_x = compute_mean_std(xs)
        mean_y, std_y = compute_mean_std(ys)
        if std_x <= 1e-6 or std_y <= 1e-6:
            return 0.0, mean_x, mean_y
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / float(len(xs))
        corr = cov / (std_x * std_y)
        return corr, mean_x, mean_y

    # 与 engagement 人设的相关
    corr_eng, mean_eng, mean_behavior_eng = compute_pearson(xs_eng, ys_eng)
    # 与 perseverance 人设的相关
    corr_per, mean_per, mean_behavior_per = compute_pearson(xs_per, ys_per)

    print("=========================================================")
    print("【行为投入度与坚持性维度：人设 vs 行为分析 全局对比】")
    print(f"- 采样学习者数量（具备任一人设）：{len(sampled_learners)}")
    print(f"- 参与 EP 行为汇总的学习者数量：{len(learner_global_ep)}")
    if corr_eng is not None:
        print("—— 与 engagement.score 的对比 ——")
        print(f"  · 参与对比样本数：{len(xs_eng)}")
        print(f"  · 行为侧 global_engage_persist 平均值：{mean_behavior_eng:.3f}")
        print(f"  · 人设 engagement.score 平均值：{mean_eng:.3f}")
        print(f"  · 皮尔逊相关系数：{corr_eng:.3f}")
    else:
        print("—— 与 engagement.score 的对比 ——")
        print("  · 样本过少，无法计算相关系数。")

    if corr_per is not None:
        print("—— 与 perseverance.score 的对比 ——")
        print(f"  · 参与对比样本数：{len(xs_per)}")
        print(f"  · 行为侧 global_engage_persist 平均值：{mean_behavior_per:.3f}")
        print(f"  · 人设 perseverance.score 平均值：{mean_per:.3f}")
        print(f"  · 皮尔逊相关系数：{corr_per:.3f}")
    else:
        print("—— 与 perseverance.score 的对比 ——")
        print("  · 样本过少，无法计算相关系数。")
    print("=========================================================")

    # 简单输出若干示例学习者的对比信息，方便人工检查
    print("【示例学习者行为投入度与坚持性分析（前 5 个）】")
    count_print = 0
    for lrn_uid in sampled_learners:
        if count_print >= 5:
            break
        behavior_score = learner_global_ep.get(lrn_uid)
        if behavior_score is None:
            continue
        eng_score = persona_engagement.get(lrn_uid)
        per_score = persona_perseverance.get(lrn_uid)
        print(f"- learner_uid = {lrn_uid}")
        print(f"  · 行为侧 global_engage_persist：{behavior_score:.3f}")
        if eng_score is not None:
            print(f"  · 人设 engagement.score：{eng_score:.3f}")
        else:
            print("  · 人设 engagement.score：无")
        if per_score is not None:
            print(f"  · 人设 perseverance.score：{per_score:.3f}")
        else:
            print("  · 人设 perseverance.score：无")
        count_print += 1

    print("行为投入度与坚持性维度分析完成。")


if __name__ == "__main__":
    main()
