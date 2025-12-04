# -*- coding: utf-8 -*-
"""
交互与操作熟练度 / 风格（Interaction & Operation Fluency / Style）分析脚本
======================================================================

一、脚本目标与画像维度
----------------------------------------------------------------------
本脚本针对画像维度：
    - interaction_style（交互与操作熟练度 / 风格）
    - 描述：学习者在 VR/AR/强交互单元中，对虚拟对象/工具的操作习惯与熟练度。
      希望区分：
        1）多试多练型：操作频率较高，错误不多，通过多次尝试逐渐掌握；
        2）少操作但准确型：操作次数较少但正确率高，步骤执行稳定；
        3）随便乱点型：操作频率高但成功率低，存在大量无效/误操作。

行为数据来源：
    - 由 xAPI_interaction_profile.py 生成并写入 MongoDB 的细粒度 xAPI 行为：
        * verb = manipulated-object：记录 grab / rotate / activate 等关键操作；
        * verb = performed-procedure-step：记录关键步骤的完成情况和时长；
        * verb = completed：记录交互单元完成与否及总时长（success / duration）。

二、论文依据与设计映射
----------------------------------------------------------------------
本脚本在方法设计上直接参考了以下三篇文献（均通过 Web 获取原文）：

1）Heinemann et al. (2024). 
   "A Learning Analytics Dashboard to Investigate the Influence of Interaction in a VR Learning Application."
   - 论文观点：
     * 仪表盘中使用 “对象交互计数 / 导航行为数量 / 任务得分 / 任务耗时” 等聚合指标，
       分析交互模式与学习表现之间的关系。
     * 特别强调：以“对象交互次数 + 任务得分 + 任务时长”作为核心可视化维度，
       帮助教师区分“高频交互 + 高得分”和“高频交互 + 低得分”等不同学习风格。
   - 本脚本对应实现：
     * 采用 “manipulated-object 事件计数 / 单位时间操作频率（每分钟操作次数）”
       作为交互强度指标；
     * 采用 “任务完成 success 比例 + 步骤 success 比例” 作为任务表现/熟练度指标；
     * 在 (交互强度, 错误水平) 的二维空间中进行聚类，将学习者划分为不同交互风格。
     * 即：把 Heinemann 等人在仪表盘中“展示”的指标，转化为可计算的聚类特征。

2）Heinemann et al. (2024).
   "Towards using the xAPI specification for Learning Analytics in Virtual Reality."
   - 论文观点：
     * 给出在 VR 场景中使用 xAPI 的动词和上下文设计示例，如记录用户的导航、
       对象交互（interact with object）、注视（gaze/focus）行为等；
     * 强调为 VR 交互设计专门的 xAPI verb 与 context.extensions，用于后续 Learning Analytics。
   - 本脚本对应实现：
     * 直接使用 mls_xapi_profile.json / xAPI_interaction_profile.py 中已经实现的 VR/交互动词：
       - "manipulated_object" → 对象操作（例如 grab/rotate/activate）；
       - "performed_procedure_step" → 执行关键步骤；
       - "completed" → 完成交互或 VR 单元。
     * 在查询与聚合时，严格按照这些 verb.id 过滤相关事件，
       避免用“任意事件计数”这种与 xAPI 设计不一致的做法。

3）Mangina et al. (2022).
   "Experience API (xAPI) for Virtual Reality (VR) Education in Medicine."
   - 论文观点：
     * 医学 VR 训练中，重点跟踪 “关键操作步骤（procedure steps）” 的完成情况和误操作，
       并通过步骤完成率 / 误操作次数来评估学习者的操作熟练度与安全意识；
     * 强调顺序正确完成所有关键步骤，是衡量“熟练且可靠”操作风格的核心指标。
   - 本脚本对应实现：
     * 把 performed-procedure-step 视为关键操作步骤：
       - 使用 result.success 统计步骤成功率；
       - 使用步骤耗时 result.duration 作为流程熟练度的一个补充指标；
     * 将 “步骤成功率 + 任务完成成功率” 组合为表现指标 performance_score；
     * 将高频对象操作（manipulated-object）但低步骤成功率的模式视为“乱试型 / 误操作多”；
       这与 Mangina 等文中“误操作频繁、关键步骤失败”的学习者特征一致。

三、分析方法与与原文差异说明
----------------------------------------------------------------------
1. 特征构建（受 Heinemann 2024 与 Mangina 2022 启发）
   ----------------------------------------------------
   对每个 (学习者, 课程)：
   - 交互强度：
     * 从 verb = manipulated-object 的 xAPI 中统计对象操作总数 M；
     * 从 verb = completed 且 _type = "interact" 的事件中提取总时长 T_total（单位秒）；
     * 定义操作频率（每分钟操作次数）：
         freq = M / max(T_total / 60, 1)
   - 操作熟练度 / 表现：
     * 步骤成功率：
         step_success_rate = (# performed-procedure-step & result.success=True) / (步骤总数)
     * 单元完成成功率：
         unit_success_rate = (# completed & result.success=True) / (# completed)
     * 综合表现分数（0~1）：
         performance_score = 0.5 * step_success_rate + 0.5 * unit_success_rate

   这些指标与：
     - Heinemann et al. (2024) 中的 “对象交互次数 + 任务得分 + 耗时” 一一对应；
     - Mangina et al. (2022) 中的 “关键步骤正确完成情况 + 误操作频率” 一一对应。

2. 风格空间与聚类（在原文基础上的方法扩展）
   ----------------------------------------------------
   原文中主要使用描述性统计与可视化来区分交互模式，并未给出具体聚类算法。
   本脚本在保持原指标含义不变的前提下，做了如下扩展：
   - 构建二维行为特征：
       x = log(1 + freq)     # 交互强度：对操作频率做对数压缩，缓和极端值
       y = 1 - performance_score  # 错误/不熟练程度：成功率越低，y 越大
   - 在所有 (学习者, 课程) 的 (x, y) 上执行二维 k-means 聚类（k=3）。
   - 根据聚类中心 (x_c, y_c) 的位置，自动映射到三种风格标签：
       * 多试多练型：
           - 交互强度高（x 较大），错误水平较低（y 中等偏低）；
       * 少操作但准确型：
           - 交互强度低（x 较小），错误水平低（y 最小）；
       * 随便乱点型：
           - 错误水平高（y 最大），且交互强度不低（x 中等及以上）。

   与原文差异说明：
   - 原文（Heinemann 2024 / Mangina 2022）主要展示了如何通过计数和时间指标观察差异，
     并用可视化与简单对比分组；
   - 本脚本在相同指标的基础上加入了 k-means 聚类，用于自动划分类别，
     没有用“简单平均 + 手工阈值”替代原研究中的关键指标；
   - 选择 k=3 是为了对应画像中预先设计的三类风格，而非任意设定。

3. 画像标签与 LearnerProfile 对比
   ----------------------------------------------------
   - 对每个 (学习者, 课程) 记录：
       * freq（操作频率）、performance_score（表现）、cluster_index（簇编号）、
         style_label（三类风格文本描述）、style_index（[0,1] 的数值化风格指数）。
   - 对每个学习者，把其在所有课程上的 style_index 取平均，得到：
       global_interaction_style_from_behavior ∈ [0,1]。
   - 与 LearnerProfile.global_profile.interaction_style.score 做皮尔逊相关，
     检验细粒度交互风格分析与粗粒度人设在该维度上的一致性。

4. 数据库存储接口
   ----------------------------------------------------
   - 定义 save_interaction_style_to_db(db, results) 函数：
       * 目标集合：MLS.InteractionStyleAnalysis（仅作为示例，不在 main 中自动调用）；
       * 字段：learner_uid, course_uid, freq, performance_score, x, y, style_label, style_index 等。
   - main() 中默认不调用，仅在末尾保留注释调用示例，与你之前的 analyze_task_efficiency.py 一致。

"""

from pymongo import MongoClient
from datetime import datetime
from math import sqrt, log
import re
from collections import defaultdict
import random
from tqdm import tqdm

# ===================== 配置区域 =====================

MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "MLS"
XAPI_COLLECTION = "Interaction"         # 细粒度行为集合（xAPI_interaction_profile.py 生成）
PROFILE_COLLECTION = "LearnerProfile"   # 人设集合（infer_persona_for_course 写入）
INTERACTION_STYLE_COLLECTION = "InteractionStyleAnalysis"  # 交互风格分析结果集合（仅接口）

# 采样学习者数量：随机选取至多 N_SAMPLE 个学习者进行分析
# 若 N_SAMPLE <= 0 或大于实际可选数量，则使用所有学习者
N_SAMPLE = 3000

VERB_BASE = "https://legend-meta.com/xapi/verb/"

VERBS = {
    # 对象交互：对应 OmiLAXR 与 Heinemann (2024, xAPI for VR) 中关于 grab/activate 等对象操作事件的记录
    "manipulated_object": VERB_BASE + "manipulated-object",
    # 程序步骤执行事件：对应 Mangina (2022) 中的关键操作步骤完成记录
    "performed_procedure_step": VERB_BASE + "performed-procedure-step",
    # 学习单元完成事件：用于获取交互/VR 单元的 success 与总时长
    "completed": VERB_BASE + "completed",
}

# 仅关注 VR/AR/交互型单元
UNIT_TYPES_FOR_STYLE = {"vr", "ar", "interact"}

# 预编译 duration 的正则，避免每次 re.compile
DURATION_RE = re.compile(r"^PT(\d+)S$")


# ===================== 工具函数 =====================

def parse_iso8601_duration(duration_str):
    """
    解析简单形式的 ISO8601 时长字符串，例如："PT120S"
    若为空或格式不符，返回 None。

    设计说明：
    --------------------------------------------------
    - xAPI_interaction_profile.py 生成的 result.duration 字段使用 "PT{秒数}S" 形式；
    - 为了便于统计任务/步骤耗时，与 analyze_task_efficiency.py 中保持一致，
      这里提供统一的解析函数，返回秒数（int）。
    """
    if not duration_str or not isinstance(duration_str, str):
        return None
    m = DURATION_RE.match(duration_str)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def compute_mean_std(values):
    """
    计算一组数值的均值与标准差。
    若有效样本数 < 2，则标准差返回 0.0。

    本函数在后续用于：
    - 对交互频率 / 表现分数等做标准化或检查数值分布。
    """
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    mean_v = sum(vals) / float(n)
    if n < 2:
        return mean_v, 0.0
    var = sum((v - mean_v) ** 2 for v in vals) / float(n - 1)
    return mean_v, sqrt(var)


def kmeans_2d(points, k=3, max_iter=50):
    """
    在二维空间对点集执行 k-means 聚类。

    参数：
    - points: [(x, y), ...]
    - k: 簇数，默认 3（对应三种交互风格）
    - max_iter: 最大迭代次数

    设计依据与说明：
    --------------------------------------------------
    - Heinemann et al. (2024, Dashboard) 中主要通过可视化观察
      “交互次数 × 成绩/时间”的模式，本函数在相同指标基础上
      使用 k-means 对模式进行自动离散化。
    - 由于文献中未给出具体聚类算法，本实现视为工程扩展，
      但不改变原指标的含义，只是自动化地形成三类。

    算法步骤：
    --------------------------------------------------
    1）初始化：在 (x, y) 的最小-最大边界上，沿对角线均匀放置 k 个初始中心；
    2）重复：
       - 分配阶段：把每个点分配给最近的中心（欧氏距离）；
       - 更新阶段：对每个簇取均值作为新中心；
    3）若中心移动量整体小于阈值（1e-4）则视为收敛。
    """
    if not points:
        return [], []

    n = len(points)
    if n < k:
        k = n

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # 所有点几乎重合的退化情况：直接视为一个簇
    if abs(x_max - x_min) < 1e-6 and abs(y_max - y_min) < 1e-6:
        centers = [(x_min, y_min) for _ in range(k)]
        assignments = [0 for _ in range(n)]
        return centers, assignments

    # 初始化中心：在对角线上均匀放置 k 个中心
    centers = []
    for i in range(k):
        alpha = (i + 0.5) / float(k)
        cx = x_min + (x_max - x_min) * alpha
        cy = y_min + (y_max - y_min) * alpha
        centers.append((cx, cy))

    for _ in range(max_iter):
        # Step 1: 分配阶段
        clusters = [[] for _ in range(k)]
        for idx, (x, y) in enumerate(points):
            best_c = 0
            cx, cy = centers[0]
            best_dist = (x - cx) ** 2 + (y - cy) ** 2
            for ci in range(1, k):
                cx, cy = centers[ci]
                d = (x - cx) ** 2 + (y - cy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_c = ci
            clusters[best_c].append(idx)

        # Step 2: 更新阶段
        new_centers = list(centers)
        for ci in range(k):
            if clusters[ci]:
                sum_x = sum(points[idx][0] for idx in clusters[ci])
                sum_y = sum(points[idx][1] for idx in clusters[ci])
                cnt = float(len(clusters[ci]))
                new_centers[ci] = (sum_x / cnt, sum_y / cnt)
            else:
                # 若某簇为空，保持原中心不变，避免 NaN
                new_centers[ci] = centers[ci]

        # 判断收敛
        max_shift = 0.0
        for (ox, oy), (nx, ny) in zip(centers, new_centers):
            shift = (ox - nx) ** 2 + (oy - ny) ** 2
            if shift > max_shift:
                max_shift = shift
        centers = new_centers
        if max_shift < 1e-4:
            break

    # 最终分配
    assignments = []
    for (x, y) in points:
        best_c = 0
        cx, cy = centers[0]
        best_dist = (x - cx) ** 2 + (y - cy) ** 2
        for ci in range(1, k):
            cx, cy = centers[ci]
            d = (x - cx) ** 2 + (y - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_c = ci
        assignments.append(best_c)

    return centers, assignments


def pearson_correlation(xs, ys):
    """
    计算两组数值的皮尔逊相关系数。

    用途：
    --------------------------------------------------
    - 对比行为侧 global_interaction_style_from_behavior
      与 LearnerProfile.global_profile.interaction_style.score 之间的一致性。
    """
    if not xs or not ys:
        return None
    if len(xs) != len(ys):
        return None

    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]

    n = len(xs)
    if n < 2:
        return None

    mean_x, std_x = compute_mean_std(xs)
    mean_y, std_y = compute_mean_std(ys)
    if std_x <= 1e-8 or std_y <= 1e-8:
        return None

    num = 0.0
    for x, y in zip(xs, ys):
        num += (x - mean_x) * (y - mean_y)
    return num / ((n - 1) * std_x * std_y)


# ===================== 数据库存储接口（不在 main 中调用） =====================

def save_interaction_style_to_db(db, style_results):
    """
    把交互风格分析结果写入 MongoDB 的接口函数（默认不在 main 中调用）。

    设计目的：
    --------------------------------------------------
    - 应对需求：“最终结果不需要写回数据库，但需要预留写回接口”；
    - 如果将来希望把交互风格分析结果持久化到 MLS.InteractionStyleAnalysis 集合，
      可以在 main() 末尾手动调用本函数。

    写入字段设计依据：
    --------------------------------------------------
    1）learner_uid / course_uid：
       - 对应 MLS.Interaction 集合中的 _lrn_uid / _course_uid，用于与其他分析结果关联。

    2）freq / performance_score / x / y：
       - freq：每分钟对象操作次数，对应 Heinemann (2024) 中“对象交互计数 × 时间”的组合指标；
       - performance_score：步骤成功率 + 单元成功率的综合表现分数，
         对应 Mangina (2022) 中“关键步骤完成情况”和“任务成功率”；
       - x / y：为聚类使用的二维特征：
           x = log(1 + freq)
           y = 1 - performance_score

    3）style_index / style_label / cluster_index：
       - style_index：将三类风格映射到 [0,1] 区间的数值（用于与人设对比）；
       - style_label：中文文本标签（多试多练型 / 少操作但准确型 / 随便乱点型）；
       - cluster_index：k-means 的原始簇编号，便于后续复现聚类结果。

    注意：
    --------------------------------------------------
    - 本函数不会在 main() 中自动调用；
    - 若希望实际写回数据库，请在 main() 中手动解除注释：
        save_interaction_style_to_db(db, style_results)
    """
    col = db[INTERACTION_STYLE_COLLECTION]

    # 示例实现：先清空集合，再批量写入
    db.drop_collection(INTERACTION_STYLE_COLLECTION)
    col = db[INTERACTION_STYLE_COLLECTION]

    docs_to_insert = []
    for (lrn_uid, crs_uid), res in style_results.items():
        doc = {
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "freq_per_minute": res.get("freq_per_minute"),
            "performance_score": res.get("performance_score"),
            "step_success_rate": res.get("step_success_rate"),
            "unit_success_rate": res.get("unit_success_rate"),
            "x_feature": res.get("x"),
            "y_feature": res.get("y"),
            "style_label": res.get("style_label"),
            "style_index": res.get("style_index"),
            "cluster_index": res.get("cluster_index"),
            "created_at": datetime.utcnow(),
        }
        docs_to_insert.append(doc)

    if docs_to_insert:
        col.insert_many(docs_to_insert, ordered=False)
        col.create_index(
            [("learner_uid", 1), ("course_uid", 1)],
            name="idx_learner_course"
        )
        print(f"[接口调用] 已写入 InteractionStyleAnalysis 文档数：{len(docs_to_insert)}")
    else:
        print("[接口调用] 没有可写入 InteractionStyleAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]

    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中的人设（interaction_style） ----------
    print("读取 LearnerProfile 中的人设（interaction_style）信息...")
    persona_scores = {}  # lrn_uid -> persona_interaction_style_score

    cursor_profiles = profile_col.find(
        {},
        {"learner_uid": 1, "global_profile": 1}
    )

    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g_profile = doc.get("global_profile") or {}
        is_profile = g_profile.get("interaction_style") or {}
        score = is_profile.get("score")
        if score is not None:
            # infer_persona_for_course 中将 score 映射到 [0,1] 区间
            persona_scores[lrn_uid] = float(score)

    all_learners_with_persona = list(persona_scores.keys())
    print(f"具备交互风格人设的学习者数量：{len(all_learners_with_persona)}")

    if not all_learners_with_persona:
        print("没有任何学习者具备交互风格人设，无法进行对比分析。")
        return

    # ---------- 3. 随机采样学习者 ----------
    if N_SAMPLE > 0 and N_SAMPLE < len(all_learners_with_persona):
        sampled_learners = random.sample(all_learners_with_persona, N_SAMPLE)
    else:
        sampled_learners = all_learners_with_persona
    sampled_set = set(sampled_learners)

    print(f"随机选取用于分析的学习者数量：{len(sampled_learners)} (N_SAMPLE = {N_SAMPLE})")

    # ---------- 4. 一次性加载采样学习者的交互相关事件 ----------
    """
    事件筛选策略与论文依据：
    --------------------------------------------------
    1）使用的 verb：
       - manipulated_object：
         * 对应 Heinemann (2024, xAPI for VR) 和 OmiLAXR 框架中强调的对象交互事件
           （grab/rotate/activate 等），用于统计交互强度（操作频率）。
       - performed_procedure_step：
         * 对应 Mangina (2022) 中的关键操作步骤完成情况，用于统计步骤成功率。
       - completed：
         * 对应 Heinemann (2024, Dashboard) 中“任务完成时间 + 成绩”的组合指标，
           这里用于获取交互/VR 单元的 success（任务是否成功）与总时长（duration）。

    2）限制单元类型：
       - 仅关注 _type ∈ {"vr", "ar", "interact"} 的细粒度事件，
         排除视频/题目等非强交互单元，符合该画像维度的定义。
    """
    verb_filter = [
        VERBS["manipulated_object"],
        VERBS["performed_procedure_step"],
        VERBS["completed"],
    ]

    query = {
        "_lrn_uid": {"$in": list(sampled_set)},
        "verb.id": {"$in": verb_filter},
        "_type": {"$in": list(UNIT_TYPES_FOR_STYLE)},
    }

    projection = {
        "_id": 0,
        "_lrn_uid": 1,
        "_course_uid": 1,
        "_type": 1,
        "verb.id": 1,
        "result": 1,
    }

    print("从 xAPI Interaction 集合中加载交互相关事件...")
    cursor_events = xapi_col.find(query, projection)

    # ---------- 5. 汇总 (学习者, 课程) 级别的统计 ----------
    """
    对每个 (lrn_uid, crs_uid) 统计：
    - manip_count：manipulated_object 事件总数（对象操作次数）；
    - step_total / step_success：procedure-step 总数与成功数；
    - unit_total / unit_success：completed 总数与成功数；
    - total_interact_duration：完成事件中的总时长（秒）。

    这些统计将被映射到：
    - 交互强度：freq_per_minute = manip_count / max(total_interact_duration/60, 1)
    - 表现/熟练度：performance_score = 0.5 * step_success_rate + 0.5 * unit_success_rate
    """
    stats_per_lc = {}
    for ev in tqdm(cursor_events, desc="统计交互事件"):
        lrn_uid = ev.get("_lrn_uid")
        crs_uid = ev.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue
        key = (lrn_uid, crs_uid)

        st = stats_per_lc.get(key)
        if st is None:
            st = {
                "manip_count": 0,
                "step_total": 0,
                "step_success": 0,
                "unit_total": 0,
                "unit_success": 0,
                "total_interact_duration": 0.0,
            }
            stats_per_lc[key] = st

        verb_id = ev.get("verb", {}).get("id")
        result = ev.get("result") or {}

        if verb_id == VERBS["manipulated_object"]:
            # 对象操作计数：用于衡量点击/抓取等操作的频率
            st["manip_count"] += 1

        elif verb_id == VERBS["performed_procedure_step"]:
            # 关键步骤完成情况：用于衡量熟练度与安全性（Mangina 2022）
            st["step_total"] += 1
            success = bool(result.get("success"))
            if success:
                st["step_success"] += 1

        elif verb_id == VERBS["completed"]:
            # 单元完成情况与总时长：用于衡量整体任务成功与时间成本
            st["unit_total"] += 1
            success = bool(result.get("success"))
            if success:
                st["unit_success"] += 1
            dur_str = result.get("duration")
            dur_sec = parse_iso8601_duration(dur_str)
            if dur_sec is not None:
                st["total_interact_duration"] += float(dur_sec)

    print(f"汇总得到的 (学习者, 课程) 交互统计条目数：{len(stats_per_lc)}")

    if not stats_per_lc:
        print("没有任何交互统计数据，结束分析。")
        return

    # ---------- 6. 计算交互强度与表现分数 ----------
    """
    对每个 (lrn_uid, crs_uid) 计算：
    - freq_per_minute：每分钟对象操作次数；
    - step_success_rate：关键步骤成功率；
    - unit_success_rate：单元完成成功率；
    - performance_score：综合表现（0~1）。

    然后构造聚类特征：
    - x = log(1 + freq_per_minute)  # 交互强度（对极值做对数压缩）
    - y = 1 - performance_score     # 错误 / 不熟练程度（越大代表越“乱”）
    """
    style_results = {}  # (lrn_uid, crs_uid) -> dict
    feature_points = []  # [(x, y)]
    feature_keys = []    # [(lrn_uid, crs_uid)]

    for key, st in stats_per_lc.items():
        manip = st["manip_count"]
        total_dur = st["total_interact_duration"]
        step_total = st["step_total"]
        step_success = st["step_success"]
        unit_total = st["unit_total"]
        unit_success = st["unit_success"]

        # 交互强度：每分钟操作次数，若时长为 0 则按 1 分钟计算下限
        minutes = max(total_dur / 60.0, 1.0)
        freq_per_minute = manip / minutes if minutes > 0 else 0.0

        # 步骤成功率
        if step_total > 0:
            step_success_rate = step_success / float(step_total)
        else:
            # 没有步骤信息时，用 0.5 作为中性值
            step_success_rate = 0.5

        # 单元成功率
        if unit_total > 0:
            unit_success_rate = unit_success / float(unit_total)
        else:
            unit_success_rate = 0.5

        # 综合表现分数
        performance_score = 0.5 * step_success_rate + 0.5 * unit_success_rate

        # 构造二维特征
        x = log(1.0 + freq_per_minute)  # 交互强度（对数压缩）
        y = 1.0 - performance_score     # 错误 / 不熟练程度（越大越“乱”）

        result = {
            "freq_per_minute": freq_per_minute,
            "step_success_rate": step_success_rate,
            "unit_success_rate": unit_success_rate,
            "performance_score": performance_score,
            "x": x,
            "y": y,
        }
        style_results[key] = result
        feature_points.append((x, y))
        feature_keys.append(key)

    print("已为所有 (学习者, 课程) 计算交互强度与表现分数，并构造聚类特征。")

    # ---------- 7. 基于 (x, y) 的交互风格聚类 ----------
    centers, assignments = kmeans_2d(feature_points, k=3, max_iter=50)

    if not centers:
        print("k-means 聚类未能得到有效中心，跳过交互风格标签生成。")
    else:
        # 根据中心位置映射到三种风格
        """
        聚类中心解释与标签映射逻辑：
        --------------------------------------------------
        - 对三个中心 c_i = (x_i, y_i)，我们希望自动找到：
          1）“少操作但准确型”：x 最小且 y 最小；
          2）“随便乱点型”：y 最大且 x 不低；
          3）“多试多练型”：在剩余中心中，x 较大而 y 中等偏低。

        实现步骤：
        --------------------------------------------------
        1）首先找到 y 最大的中心 → 随便乱点型；
        2）在剩余两个中心中：
           - 若某个中心的 x 和 y 都比较小 → 少操作但准确型；
           - 另一个中心 → 多试多练型。
        3）若数据分布过于特殊（例如所有中心十分接近），则回退为简单规则：
           - y 最大：随便乱点型；
           - x 最小：少操作但准确型；
           - 剩余：多试多练型。
        """
        k = len(centers)
        # 找到错误水平最高的中心：随机乱点型
        y_values = [c[1] for c in centers]
        idx_random = max(range(k), key=lambda i: y_values[i])

        remain_idx = [i for i in range(k) if i != idx_random]

        # 默认初始化
        idx_precise = remain_idx[0] if remain_idx else idx_random
        idx_practice = remain_idx[1] if len(remain_idx) > 1 else idx_random

        if len(remain_idx) == 2:
            i1, i2 = remain_idx
            x1, y1 = centers[i1]
            x2, y2 = centers[i2]
            # 选择 x 和 y 都较小的作为“少操作但准确型”
            if (x1 + y1) <= (x2 + y2):
                idx_precise = i1
                idx_practice = i2
            else:
                idx_precise = i2
                idx_practice = i1

        cluster_to_label = {}
        cluster_to_style_index = {}

        # 少操作但准确型：偏“稳重、准确”，风格指数设为 0.9
        cluster_to_label[idx_precise] = "少操作但准确型（操作次数较少但步骤和任务成功率较高）"
        cluster_to_style_index[idx_precise] = 0.9

        # 多试多练型：偏“多次尝试逐步掌握”，风格指数设为 0.7
        cluster_to_label[idx_practice] = "多试多练型（操作频率较高，通过反复尝试完成任务）"
        cluster_to_style_index[idx_practice] = 0.7

        # 随便乱点型：偏“高频误操作”，风格指数设为 0.3
        cluster_to_label[idx_random] = "随便乱点型（操作频率较高但成功率较低，存在较多无效/误操作）"
        cluster_to_style_index[idx_random] = 0.3

        # 将聚类结果写回 style_results
        label_counts = defaultdict(int)
        for (key, cluster_idx) in zip(feature_keys, assignments):
            res = style_results[key]
            label = cluster_to_label.get(cluster_idx, "多试多练型（默认）")
            s_index = cluster_to_style_index.get(cluster_idx, 0.7)
            res["cluster_index"] = int(cluster_idx)
            res["style_label"] = label
            res["style_index"] = float(s_index)
            label_counts[label] += 1

        print("交互风格标签分布（按学习者-课程对统计）：")
        for label, cnt in label_counts.items():
            print(f"- {label}: {cnt} 条记录")

    # ---------- 8. （可选）写回数据库接口——默认不调用 ----------
    """
    如前所述：
    - 当前版本脚本只需完成“读取细粒度 xAPI → 计算交互风格 → 输出结果与人设对比”，
      不需要真正把结果写回数据库；
    - 上面定义的 save_interaction_style_to_db(db, style_results) 即为“写回接口”，
      若未来需要，可手动解除下面的注释。
    """
    # 若需要写回数据库，请取消下一行注释：
    # save_interaction_style_to_db(db, style_results)

    # ---------- 9. 按学习者汇总 global_interaction_style 并与人设对比 ----------
    """
    验证思路：
    --------------------------------------------------
    1）行为侧指标：
       - 对每个学习者，把其在所有课程上的 style_index 取平均，
         得到 global_interaction_style_from_behavior ∈ [0,1]；
       - 该指标越高，越偏向“少操作但准确型”或“稳健型”，
         越低则越偏向“随便乱点型”。

    2）人设侧指标：
       - LearnerProfile.global_profile.interaction_style.score ∈ [0,1]，
         是基于粗粒度统计与预设规则推断出的交互风格分数。

    3）对比目的：
       - 通过皮尔逊相关系数，检验“基于细粒度 xAPI 的交互风格分析”与
         “基于粗粒度统计的人设交互风格”在总体趋势上是否一致。

    4）与文献关系：
       - Heinemann et al. (2024, Dashboard) 和 Mangina et al. (2022) 均强调
         可以将交互行为聚合为仪表盘指标，用于观察学习者之间的模式差异；
       - 本步骤则进一步把这种模式差异总结为每个学习者的 global 指标，
         与已有画像进行一致性检验。
    """
    learner_to_style_vals = defaultdict(list)
    for (lrn_uid, crs_uid), res in style_results.items():
        s_index = res.get("style_index")
        if s_index is not None:
            learner_to_style_vals[lrn_uid].append(s_index)

    learner_global_style = {}
    for lrn_uid, vals in learner_to_style_vals.items():
        if not vals:
            continue
        avg = sum(vals) / float(len(vals))
        learner_global_style[lrn_uid] = avg

    # 对齐行为侧和人设侧的学习者集合
    common_learners = [
        lrn for lrn in learner_global_style.keys()
        if lrn in persona_scores
    ]
    print(f"行为侧和人设侧均具备交互风格数据的学习者数量：{len(common_learners)}")

    if not common_learners:
        print("没有可用于计算相关性的学习者，结束对比分析。")
        return

    behavior_vals = [learner_global_style[lrn] for lrn in common_learners]
    persona_vals = [persona_scores[lrn] for lrn in common_learners]

    r = pearson_correlation(behavior_vals, persona_vals)
    if r is None:
        print("行为侧与人设侧交互风格的方差过小或样本过少，无法计算皮尔逊相关。")
    else:
        print("行为侧 global_interaction_style 与 人设侧 interaction_style.score 的皮尔逊相关：")
        print(f"r = {r:.4f}")

    # 为方便调试，可以输出部分样例
    print("示例输出（前 5 个学习者的行为侧 vs 人设侧交互风格指数）：")
    for lrn_uid in common_learners[:5]:
        b = learner_global_style.get(lrn_uid)
        p = persona_scores.get(lrn_uid)
        print(f"- Learner {lrn_uid}: behavior={b:.3f}, persona={p:.3f}")


if __name__ == "__main__":
    main()
