# -*- coding: utf-8 -*-
"""
分析维度：任务效率（Task Efficiency）

脚本目标（与画像设计文档对齐）：
--------------------------------------------------
对应《论文以及画像.txt》中“3. 画像设计”里第 2 个维度：
【2. 任务效率与认知负荷代理（Task Efficiency & Cognitive Load Proxy）】:contentReference[oaicite:0]{index=0}

任务效率维度在该文档中的界定：
- 反映学习者在完成任务时的时间消耗、重复尝试情况，以及可能的认知负荷（用可采集行为做轻量代理）。:contentReference[oaicite:1]{index=1}
- 推荐使用的行为数据：
  * answered / completed / performed-procedure-step 的 result.duration、score、success；:contentReference[oaicite:2]{index=2}
  * 重复 answered / performed-procedure-step 的次数（重试频率）。:contentReference[oaicite:3]{index=3}

本脚本采用的行为代理与论文依据：
--------------------------------------------------
1）“任务完成时间 + 成绩/正确率 → 操作熟练度与任务效率”的思路
   - 论文列表第 1 篇 Heinemann et al. 2024（RePiX VR）在任务表现维度中，使用
     “交互任务是否完成、是否正确 + 单个任务得分 + 总体平均任务得分（得分与完成时间相关）”分析操作熟练度与任务效率。:contentReference[oaicite:4]{index=4}
   - 论文列表第 3 篇 Mangina et al. 2022 在医学 VR 中，用“每个关键步骤的耗时”和“总流程时间”以及“重试次数”来刻画效率与熟练度。:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}
   → 因此，本脚本在 xAPI 中选取：
      - result.success（或 completion）作为“任务表现 / 正确率 P”代理；
      - result.duration 作为“时间/资源消耗 T”代理。

2）“同样成绩下不同负荷/努力 → 学习效率差异”的思路
   - 论文列表第 7 篇 Baceviciute et al. 2021/2022 指出，在 VR 场景中，同样的学习成绩下，认知负荷可以不同，
     因而可以区分“高效型学习者”（成绩高且负荷低）与“低效型学习者”（成绩相似但负荷高）。:contentReference[oaicite:7]{index=7}
   - 这与经典认知效率研究（Paas & van Merriënboer）中的思想一致：
     用“表现（performance）”和“资源/负荷（resource/effort）”的标准化差值来定义认知效率。
   → 本脚本采用 Paas 等人提出的认知效率偏差公式：
      E = (z_P - z_T) / sqrt(2)
      其中：
      - P：任务成功率（performance）
      - T：任务平均时间（effort/资源消耗）

3）“基于行为模式进行聚类画像”的方法论依据
   - 论文列表第 2 篇 Heinemann et al. 2024（xAPI for LA in VR，OmiLAXR 框架）明确指出：
     “同一任务阶段下不同学生的行为模式可用于聚类画像”。:contentReference[oaicite:8]{index=8}
   - 论文列表第 4 篇 Lampropoulos & Evangelidis 2025 的综述把“行为参与度、坚持性、探索程度”等
     行为指标视为重要 LA 指标，为使用行为特征进行分群提供总体支持。:contentReference[oaicite:9]{index=9}
   → 本脚本在得到“任务效率指数 E_norm（已归一化）”之后，
      使用一维 k-means 聚类（k=3）将 (学习者, 课程) 自动划分为三种效率类型，
      对应画像文档中“任务效率与认知负荷代理”维度中自然存在的“效率高/效率低”的离散类型。

4）“高效率 / 中等效率 / 低效率”三种标签设计依据
   - 画像文档在“任务效率与认知负荷代理”中强调“同等成绩下不同负荷”“高耗时 vs 低耗时”
     以及“重试频率”来区分学习者效率。:contentReference[oaicite:10]{index=10}
   - 同时，人设推断脚本（infer_persona_for_course）中已经把 task_efficiency.score 映射为
     “低/中/高”三个 level（categorize_score），说明你的整体画像框架已经采用了三档离散水平。:contentReference[oaicite:11]{index=11}
   → 因此，本脚本使用 k-means 把连续的效率指数分为三档：
      - 低效率型学习者：在同一课程中，相对表现低且耗时高 / 效率指数较低；
      - 中等效率型学习者：表现和耗时均处于中间水平；
      - 高效率型学习者：表现较高且耗时较低 / 效率指数较高。
     这三档标签与画像框架中的“task_efficiency.level”在概念上一致。

脚本功能总结：
--------------------------------------------------
1. 从 MongoDB 中读取：
   - 细粒度 xAPI 行为：MLS.Interaction 集合；
   - 学习者画像：MLS.LearnerProfile 集合（特别是 global_profile.task_efficiency.score）。:contentReference[oaicite:12]{index=12}

2. 对每个 (学习者, 课程)：
   - 使用 verb = completed / performed-procedure-step 的 xAPI 语句，
     统计任务成功率 P（基于 result.success 或 completion）和平均时长 T（基于 result.duration）。
   - 在课程范围内对 P、T 做 z 标准化：
       z_P = (P_i - mean(P)) / std(P)
       z_T = (T_i - mean(T)) / std(T)
   - 计算认知效率指数：
       E_i = (z_P - z_T) / sqrt(2)
   - 在课程内对 E 做 min-max 归一化得到 E_norm ∈ [0, 1]。

3. 基于所有 (学习者, 课程) 的 E_norm：
   - 使用一维 k-means（k=3）进行聚类；
   - 根据聚类中心从低到高排序，将每条记录标记为：
       “低效率型学习者 / 中等效率型学习者 / 高效率型学习者”。

4. 与人设对比：
   - 对每个学习者，把其在所有课程上的 E_norm 做平均，得到行为侧 global_efficiency；
   - 与 LearnerProfile.global_profile.task_efficiency.score 做皮尔逊相关，
     用于粗略验证行为分析与人设的一致性。

5. 数据库存储接口（不在 main() 中调用）：
   - 定义 save_task_efficiency_to_db(db, efficiency_results) 函数，
     演示如何把结果写入 MLS.TaskEfficiencyAnalysis 集合，但默认不调用。
   - 你可以在需要时手动取消注释进行持久化。
"""

from pymongo import MongoClient
from datetime import datetime
from math import sqrt
import re
from collections import defaultdict
import random
from tqdm import tqdm

# ===================== 配置区域 =====================

MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "MLS"
XAPI_COLLECTION = "Interaction"         # 细粒度行为集合（xAPI_interaction_profile.py 生成）:contentReference[oaicite:13]{index=13}
PROFILE_COLLECTION = "LearnerProfile"   # 人设集合（infer_persona_for_course 写入）:contentReference[oaicite:14]{index=14}
TASK_EFF_COLLECTION = "TaskEfficiencyAnalysis"  # 任务效率分析结果集合（仅接口，不在 main 中调用）

# 采样学习者数量：随机选取至多 N_SAMPLE 个学习者进行分析
# 若 N_SAMPLE <= 0 或大于实际可选数量，则使用所有学习者
N_SAMPLE = 3000

VERB_BASE = "https://legend-meta.com/xapi/verb/"

VERBS = {
    # 学习单元完成事件：对应画像文档中“completed + success + duration”的任务表现与耗时。:contentReference[oaicite:15]{index=15}
    "completed": VERB_BASE + "completed",
    # 程序步骤执行事件：对应画像文档中“performed-procedure-step”记录步骤是否完成及时间。:contentReference[oaicite:16]{index=16}
    "performed_procedure_step": VERB_BASE + "performed-procedure-step",
}

# 预编译 duration 的正则，避免每次 re.compile
DURATION_RE = re.compile(r"^PT(\d+)S$")


# ===================== 工具函数 =====================

def parse_iso8601_duration(duration_str):
    """
    解析简单形式的 ISO8601 时长字符串，例如："PT120S"
    若为空或格式不符，返回 None。

    设计说明：
    - xAPI_interaction_profile.py 在生成 video / interact / vr / ar 等单元的 completed 事件时，
      使用 result.duration = f"PT{int(watch_len)}S" 的形式保存时长。:contentReference[oaicite:17]{index=17}
    - 因此，这里只需支持“PT{秒数}S”的整数秒格式即可覆盖当前数据。
    """
    if not duration_str:
        return None
    m = DURATION_RE.match(duration_str)
    if m:
        return int(m.group(1))
    return None


def compute_mean_std(values):
    """
    计算一组数的均值和标准差（总体标准差）：
    - 列表为空 -> (0, 0)
    - 仅一个元素 -> 标准差视为 0

    用途：
    - 对课程内所有学习者的任务成功率 P、平均时长 T、以及后续 global 对比中的人设分数，
      计算 z 分数与相关系数。
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
    一维 k-means 聚类（Lloyd 算法实现），用于基于效率指数 E_norm 自动划分学习者类型。

    论文与画像依据：
    --------------------------------------------------
    1）“用行为模式做聚类画像”的直接依据：
        - 论文列表第 2 篇 Heinemann et al. 2024（OmiLAXR 框架）在行为类型总结中明确指出：
          “同一任务阶段下不同学生的行为模式可用于聚类画像”。:contentReference[oaicite:18]{index=18}
        - 本脚本认为“任务效率指数 E_norm”是由“任务完成时间 + 正确率”提炼出的行为模式摘要，
          因此使用聚类对 E_norm 进行分群与该观点一致。

    2）选择 k-means 的理由：
        - k-means 是 LA/EDM 领域中常用的无监督聚类算法，能够在一维或多维行为特征空间中
          自动发现“相似行为模式”的学习者群体。
        - 在这里，特征是一维的 E_norm，因此 k-means 的计算和解释都比较直观：
          形成若干“效率水平中心”，每条记录被归到离自己最近的效率中心所属的簇。

    3）k=3 的设定原因：
        - 画像文档中的“任务效率与认知负荷代理”配合人设推断代码中 categorize_score 的实现，
          已经把 task_efficiency 划分为“低 / 中 / 高”三档水平。:contentReference[oaicite:19]{index=19}:contentReference[oaicite:20]{index=20}
        - 因此，这里设定 k=3，让聚类结果自然对应到“低效率型 / 中等效率型 / 高效率型”三种标签。

    参数：
        values: List[float]，要聚类的一维数据（这里是 efficiency_normalized）
        k: 聚类簇数，默认 3
        max_iter: 最多迭代次数

    返回：
        centers: List[float]，每个簇的中心
        assignments: List[int]，与 values 等长的簇编号列表
    """
    n = len(values)
    if n == 0 or k <= 0:
        return [], []

    # 如果样本数少于簇数，则最多聚成 n 类，避免空簇过多
    if n < k:
        k = n

    # 所有值都几乎相同的退化情况：直接视为一个簇
    v_min, v_max = min(values), max(values)
    if abs(v_max - v_min) < 1e-6:
        centers = [v_min for _ in range(k)]
        assignments = [0 for _ in range(n)]
        return centers, assignments

    # 初始化：在 [min, max] 范围内均匀取 k 个初始中心
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
        new_centers = centers[:]
        for ci in range(k):
            if clusters[ci]:
                new_centers[ci] = sum(values[i] for i in clusters[ci]) / float(len(clusters[ci]))
            else:
                # 若某簇暂时为空，保持原中心不变，避免产生 NaN
                new_centers[ci] = centers[ci]

        # 判断收敛：所有中心移动总量很小则停止
        max_shift = max(abs(new_centers[ci] - centers[ci]) for ci in range(k))
        centers = new_centers
        if max_shift < 1e-4:
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

def save_task_efficiency_to_db(db, efficiency_results):
    """
    把效率分析结果写入 MongoDB 的接口函数（默认不在 main 中调用）。

    设计目的：
    --------------------------------------------------
    - 应对你的需求：“最终结果不需要写回数据库，但需要预留写回接口”。
    - 如果将来你希望把任务效率分析结果持久化到 MLS.TaskEfficiencyAnalysis 集合，
      可以在 main() 末尾手动调用本函数。

    写入字段设计依据：
    --------------------------------------------------
    1）字段 learner_uid / course_uid：
       - 与 MLS.Interaction 集合中的 _lrn_uid / _course_uid 保持一一对应。:contentReference[oaicite:21]{index=21}

    2）P_mean / T_mean / z_P / z_T / efficiency_index / efficiency_normalized：
       - P_mean 和 T_mean 分别对应：
         * 行为文档中“任务完成情况、是否正确”的统计（success → 正确率）；:contentReference[oaicite:22]{index=22}
         * “完成关键步骤耗时 / 总流程时间”的统计（duration → 时间）。:contentReference[oaicite:23]{index=23}
       - z_P / z_T / efficiency_index 的计算，采用 Paas 等人的认知效率公式，
         结合 Baceviciute 等关于“同等成绩下不同负荷”的讨论来解释 E 的含义。:contentReference[oaicite:24]{index=24}

    3）efficiency_label / cluster_rank：
       - 通过 k-means 聚类结果给出的分群类型，概念上对应画像文档中
         “任务效率与认知负荷代理”的“高效率 / 低效率”差异，离散化为低/中/高三档。:contentReference[oaicite:25]{index=25}

    注意：
    --------------------------------------------------
    - 本函数不会在 main() 中自动调用。
    - 如果你希望实际写回数据库，请在 main() 中手动解除注释：
        save_task_efficiency_to_db(db, efficiency_results)
    """
    task_eff_col = db[TASK_EFF_COLLECTION]

    # 为方便重复实验，这里先清空集合（若你不想清空，可以改为 update 或 upsert）
    db.drop_collection(TASK_EFF_COLLECTION)
    task_eff_col = db[TASK_EFF_COLLECTION]

    docs_to_insert = []
    for (lrn_uid, crs_uid), res in efficiency_results.items():
        doc = {
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "P_mean": res["P_mean"],
            "T_mean": res["T_mean"],
            "z_P": res["z_P"],
            "z_T": res["z_T"],
            "efficiency_index": res["E"],           # 认知效率指数 E
            "efficiency_normalized": res["E_norm"], # 归一化效率 E_norm
            "efficiency_label": res.get("efficiency_label"),
            "cluster_rank": res.get("cluster_rank"),
            "created_at": datetime.utcnow(),
        }
        docs_to_insert.append(doc)

    if docs_to_insert:
        task_eff_col.insert_many(docs_to_insert, ordered=False)
        task_eff_col.create_index(
            [("learner_uid", 1), ("course_uid", 1)],
            name="idx_learner_course"
        )
        print(f"[接口调用] 已写入 TaskEfficiencyAnalysis 文档数：{len(docs_to_insert)}")
    else:
        print("[接口调用] 没有可写入 TaskEfficiencyAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]

    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中的人设 ----------
    print("读取 LearnerProfile 中的人设信息...")
    persona_scores = {}  # lrn_uid -> persona_task_efficiency_score

    cursor_profiles = profile_col.find(
        {},
        {"learner_uid": 1, "global_profile": 1}
    )

    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g_profile = doc.get("global_profile") or {}
        te = g_profile.get("task_efficiency") or {}
        score = te.get("score")
        if score is not None:
            # infer_persona_for_course 中已经把 task_efficiency.score 定义在 [0,1] 区间，
            # 这里直接读取作为人设中的“先验效率”。:contentReference[oaicite:26]{index=26}
            persona_scores[lrn_uid] = float(score)

    all_learners_with_persona = list(persona_scores.keys())
    print(f"具备任务效率人设的学习者数量：{len(all_learners_with_persona)}")

    if not all_learners_with_persona:
        print("没有任何学习者具备任务效率人设，无法进行对比分析。")
        return

    # ---------- 3. 随机采样学习者 ----------
    if N_SAMPLE > 0 and N_SAMPLE < len(all_learners_with_persona):
        sampled_learners = random.sample(all_learners_with_persona, N_SAMPLE)
    else:
        sampled_learners = all_learners_with_persona
    sampled_set = set(sampled_learners)

    print(f"随机选取用于分析的学习者数量：{len(sampled_learners)} (N_SAMPLE = {N_SAMPLE})")

    # ---------- 4. 一次性加载采样学习者的任务级事件 ----------
    """
    事件筛选策略与论文依据：
    --------------------------------------------------
    1）使用的 verb：
       - completed：
         * 对应视频/VR/AR/交互单元的“完成事件”，包含 completion、duration 等字段。:contentReference[oaicite:27]{index=27}
         * 在 Heinemann RePiX VR 与 Mangina 医学 VR 文献中，都将“任务是否完成 + 完成时间”作为效率分析的核心指标。:contentReference[oaicite:28]{index=28}:contentReference[oaicite:29]{index=29}
       - performed-procedure-step：
         * 对应流程/实验/临床步骤级别的操作行为，画像文档建议用它记录“每个步骤是否完成、顺序是否正确、耗时”。:contentReference[oaicite:30]{index=30}:contentReference[oaicite:31]{index=31}

    2）查询条件：
       - 仅针对采样学习者（_lrn_uid in sampled_learners），避免处理全部数据；
       - 只保留上述两个 verb.id 的事件。
    """
    query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": {"$in": [VERBS["completed"], VERBS["performed_procedure_step"]]}
    }

    print("统计待加载的事件数量（count_documents）...")
    total_events = xapi_col.count_documents(query)
    print(f"与采样学习者相关的任务级事件总数：{total_events}")

    print("开始一次性加载所有相关事件到内存（list）...")
    events = list(xapi_col.find(
        query,
        {"verb.id": 1, "result": 1, "_lrn_uid": 1, "_course_uid": 1}
    ))
    print(f"已从 MongoDB 读取事件条数：{len(events)}")

    if not events:
        print("没有任何任务级事件，无法进行任务效率分析。")
        return

    # ---------- 5. 遍历本地事件列表，聚合任务统计 ----------
    """
    聚合逻辑说明：
    --------------------------------------------------
    - 粒度：以 (学习者, 课程) 为聚合单位，与画像文档中“按课程窗口刻画任务效率”保持一致。:contentReference[oaicite:32]{index=32}
    - 对每个 (学习者, 课程)，累积：
        sum_P：任务成功次数（视 success=True 或 completion=True 为 1）；
        sum_T：任务耗时总和（duration 秒数）；
        count：参与统计的任务数量。
    - 之后会计算：
        P_mean = sum_P / count
        T_mean = sum_T / count
    """
    # (learner_uid, course_uid) -> {"sum_P":..., "sum_T":..., "count":...}
    task_stats = defaultdict(lambda: {"sum_P": 0.0, "sum_T": 0.0, "count": 0})
    used_events = 0

    print("开始遍历任务级事件并进行聚合计算...")
    for doc in tqdm(events, desc="聚合事件", unit="event"):
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue

        result = doc.get("result") or {}
        duration_str = result.get("duration")
        duration_sec = parse_iso8601_duration(duration_str)
        if duration_sec is None or duration_sec <= 0:
            # 无法用于效率分析的事件（缺少时长或时长为 0）
            continue

        # success 字段：对应画像文档中“answered / completed 的 success”用来计算正确率与完成率。:contentReference[oaicite:33]{index=33}
        success = result.get("success")
        completion = result.get("completion")

        if success is None and completion is None:
            # 若既没有 success 也没有 completion，则无法判断表现，跳过
            continue
        elif success is None:
            # 只有 completion 的情况：completion=True 视为成功完成一次任务
            P_task = 1.0 if completion else 0.0
        else:
            # 有 success 的情况：True 为成功，False/None 为失败
            P_task = 1.0 if bool(success) else 0.0

        key = (lrn_uid, crs_uid)
        stat = task_stats[key]
        stat["sum_P"] += P_task
        stat["sum_T"] += float(duration_sec)
        stat["count"] += 1
        used_events += 1

    print(f"参与任务效率统计的有效事件数：{used_events}")
    print(f"有任务数据的 (学习者, 课程) 组合数：{len(task_stats)}")

    if not task_stats:
        print("聚合后没有任何可用任务统计数据，结束分析。")
        return

    # ---------- 6. 计算每个 (学习者, 课程) 的 P_mean / T_mean ----------
    learner_course_metrics = {}  # (lrn_uid, crs_uid) -> {"P_mean":..., "T_mean":...}

    for key, stat in task_stats.items():
        c = stat["count"]
        if c <= 0:
            continue
        P_mean = stat["sum_P"] / float(c)
        T_mean = stat["sum_T"] / float(c)
        learner_course_metrics[key] = {
            "P_mean": P_mean,
            "T_mean": T_mean,
        }

    print(f"成功得到 P_mean / T_mean 的 (学习者, 课程) 数量：{len(learner_course_metrics)}")

    if not learner_course_metrics:
        print("没有可用的 (学习者, 课程) 任务统计数据，结束。")
        return

    # ---------- 7. 按课程进行认知效率 E 计算与归一化 ----------
    """
    课程内标准化与认知效率公式的论文依据：
    --------------------------------------------------
    1）课程内比较而非跨课程比较：
       - 不同课程的内容难度、任务数量与任务类型差异较大，
         直接跨课程比较“绝对正确率 P”或“绝对时间 T”会失真。
       - 因此，本脚本在“每门课程内部”对 P 和 T 进行标准化，并计算认知效率 E，
         仅在同一课程内比较学生之间的效率差异。

    2）认知效率公式：
       - 采用 Paas & van Merriënboer 的偏差公式：
         E = (z_P - z_T) / sqrt(2)
         其中：
         * z_P：表现（正确率）的 z 分数；
         * z_T：时间/资源的 z 分数（越大表示耗时越长）。
       - 结合 Baceviciute 等关于“在同样成绩下，负荷不同”的结论，
         可以把 E 看成“成绩-负荷差”的标准化指标：:contentReference[oaicite:34]{index=34}
         * P 高且 T 低 → E 大 → 高效型学习者；
         * P 低且 T 高 → E 小甚至负 → 低效型学习者。

    3）min-max 归一化：
       - 为了方便后续与人设中的 [0,1] 分数对齐，
         将每门课程内的 E 映射到 [0,1] 区间，得到 E_norm。
    """
    # course_uid -> list[(lrn_uid, P_mean, T_mean)]
    course_to_entries = defaultdict(list)
    for (lrn_uid, crs_uid), mt in learner_course_metrics.items():
        course_to_entries[crs_uid].append((lrn_uid, mt["P_mean"], mt["T_mean"]))

    efficiency_results = {}  # (lrn_uid, crs_uid) -> {P_mean, T_mean, z_P, z_T, E, E_norm}

    print("按课程计算认知效率 E 及归一化 E_norm...")
    for crs_uid, entries in course_to_entries.items():
        if not entries:
            continue

        P_vals = [e[1] for e in entries]
        T_vals = [e[2] for e in entries]
        mean_P, std_P = compute_mean_std(P_vals)
        mean_T, std_T = compute_mean_std(T_vals)

        # 先计算每个学习者的 E 值，为后续 min-max 归一化准备
        E_vals = []
        for (lrn_uid, P_mean, T_mean) in entries:
            # z_P：该学习者在本课程中的“相对正确率”
            z_P = (P_mean - mean_P) / std_P if std_P > 1e-6 else 0.0

            # z_T：该学习者在本课程中的“相对耗时”（时长越长 z_T 越大）
            z_T = (T_mean - mean_T) / std_T if std_T > 1e-6 else 0.0

            # 认知效率公式：E = (z_P - z_T) / sqrt(2)
            # - P 高、T 低 → z_P 大、z_T 小 → E 大，符合“高绩效 + 低负荷”的高效型；
            # - P 低、T 高 → E 小甚至为负，符合“低绩效 + 高负荷”的低效型。
            E = (z_P - z_T) / sqrt(2.0)
            efficiency_results[(lrn_uid, crs_uid)] = {
                "P_mean": P_mean,
                "T_mean": T_mean,
                "z_P": z_P,
                "z_T": z_T,
                "E": E,
            }
            E_vals.append(E)

        # 在当前课程内部对 E 做 [0,1] 的 min-max 归一化
        if E_vals:
            E_min = min(E_vals)
            E_max = max(E_vals)
            span = E_max - E_min if E_max > E_min else 0.0
            for (lrn_uid, P_mean, T_mean) in entries:
                key = (lrn_uid, crs_uid)
                E = efficiency_results[key]["E"]
                if span > 1e-6:
                    E_norm = (E - E_min) / span
                else:
                    # 当所有人的 E 完全相同（只有一个学习者或表现高度一致）时，
                    # 无法区分效率高低，这里统一给 0.5 作为中间值。
                    E_norm = 0.5
                efficiency_results[key]["E_norm"] = E_norm

    print("课程层面的任务效率计算完成。")
    print(f"效率结果条目数（学习者-课程对）：{len(efficiency_results)}")

    if not efficiency_results:
        print("没有任何效率结果，结束分析。")
        return

    # ---------- 8. 基于 E_norm 的学习者类型聚类（k-means 1D） ----------
    all_E_norm = [res["E_norm"] for res in efficiency_results.values()]
    centers, assignments = kmeans_1d(all_E_norm, k=3, max_iter=50)

    if centers:
        # 将中心从低到高排序，给每个簇一个 rank（0=低效率, 1=中等, 2=高效率）
        sorted_idx = sorted(range(len(centers)), key=lambda i: centers[i])
        cluster_to_rank = {cluster_idx: rank for rank, cluster_idx in enumerate(sorted_idx)}

        # 定义按 rank 编号的标签文案
        rank_to_label = {
            0: "低效率型学习者（在本课程中任务成功率相对较低、耗时相对较长 / 认知效率指数较低）",
            1: "中等效率型学习者（在本课程中任务成功率与耗时均处于中间水平）",
            2: "高效率型学习者（在本课程中任务成功率相对较高、耗时相对较短 / 认知效率指数较高）",
        }

        # 依次给每个 (学习者, 课程) 赋予簇编号与标签
        for ((key, res), cluster_idx) in zip(efficiency_results.items(), assignments):
            rank = cluster_to_rank.get(cluster_idx, 1)  # 默认视为中等效率
            label = rank_to_label.get(rank, "中等效率型学习者（默认）")
            res["cluster_index"] = int(cluster_idx)
            res["cluster_rank"] = int(rank)
            res["efficiency_label"] = label

        # 统计标签分布，便于在控制台快速查看整体情况
        label_counts = defaultdict(int)
        for res in efficiency_results.values():
            label = res.get("efficiency_label")
            if label:
                label_counts[label] += 1

        print("任务效率标签分布（按学习者-课程对统计）：")
        for label, cnt in label_counts.items():
            print(f"- {label}: {cnt} 条记录")

    else:
        print("k-means 聚类未能得到有效中心，跳过学习者类型标签生成。")

    # ---------- 9. （可选）写回数据库接口——默认不调用 ----------
    """
    如你在需求中所述：
    - 当前版本脚本只需完成“读取细粒度 xAPI → 计算任务效率 → 输出结果与人设对比”，
      不需要真正把结果写回数据库。
    - 上面定义的 save_task_efficiency_to_db(db, efficiency_results) 即为“写回接口”；
      若未来需要，可手动解除下面的注释。

    示例（默认注释掉）：
        save_task_efficiency_to_db(db, efficiency_results)
    """
    # 若需要写回数据库，请取消下一行注释：
    # save_task_efficiency_to_db(db, efficiency_results)

    # ---------- 10. 按学习者汇总 global_efficiency 并与人设对比 ----------
    """
    验证思路与论文参考：
    --------------------------------------------------
    1）行为侧指标：
       - 对每个学习者，把其在所有课程上的 E_norm 取平均，得到 global_efficiency。
       - global_efficiency 仍然处于 [0,1] 区间，代表行为数据推断出的“整体任务效率水平”。

    2）人设侧指标：
       - LearnerProfile.global_profile.task_efficiency.score 是在粗粒度统计基础上，
         通过 infer_persona_for_course 推断得到的任务效率分数。:contentReference[oaicite:35]{index=35}

    3）对比目的：
       - 通过皮尔逊相关系数，检验“基于细粒度 xAPI 的效率分析”和
         “基于粗粒度统计的人设任务效率”在总体趋势上是否一致。
       - 若相关为正且显著，说明细粒度 xAPI 分析与已有画像设计在任务效率维度上方向一致，
         有助于增强画像的可信度。

    4）与文献的关系：
       - 论文列表第 1 篇与第 3 篇都强调“任务完成时间 + 正确率/重试”可以作为效率指标，:contentReference[oaicite:36]{index=36}:contentReference[oaicite:37]{index=37}
         本脚本对这些指标做进一步加工，形成课程级和全局级的效率指数。
       - 该对比步骤更多是系统工程上的一致性检验，本身不引入新的理论假设。
    """
    learner_to_eff_vals = defaultdict(list)
    for (lrn_uid, crs_uid), res in efficiency_results.items():
        learner_to_eff_vals[lrn_uid].append(res["E_norm"])

    learner_global_eff = {}
    for lrn_uid, vals in learner_to_eff_vals.items():
        if vals:
            learner_global_eff[lrn_uid] = sum(vals) / float(len(vals))

    xs = []  # 人设中的任务效率分数
    ys = []  # 行为分析得到的 global_efficiency

    for lrn_uid in sampled_learners:
        persona_score = persona_scores.get(lrn_uid)
        analyzed_eff = learner_global_eff.get(lrn_uid)
        if persona_score is not None and analyzed_eff is not None:
            xs.append(float(persona_score))
            ys.append(float(analyzed_eff))

    if len(xs) >= 2:
        mean_x, std_x = compute_mean_std(xs)
        mean_y, std_y = compute_mean_std(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / float(len(xs))
        if std_x > 1e-6 and std_y > 1e-6:
            corr = cov / (std_x * std_y)
        else:
            corr = 0.0

        avg_global_eff = sum(ys) / float(len(ys))
        avg_persona_score = sum(xs) / float(len(xs))

        print("=========================================================")
        print("【任务效率维度：人设 vs 行为分析 全局对比】")
        print(f"- 采样学习者数量（具备人设）：{len(sampled_learners)}")
        print(f"- 实际参与对比的学习者数量：{len(xs)}")
        print(f"- 行为分析 global_efficiency 平均值：{avg_global_eff:.3f}")
        print(f"- 人设 task_efficiency.score 平均值：{avg_persona_score:.3f}")
        print(f"- 皮尔逊相关系数：{corr:.3f}")
        print("  （相关系数用于粗略验证：细粒度 xAPI 分析是否与人设任务效率维度方向一致。）")
        print("=========================================================")
    else:
        print("参与对比的学习者样本太少，无法计算相关系数。")

    print("任务效率维度分析完成。")


if __name__ == "__main__":
    main()
