# -*- coding: utf-8 -*-
"""
分析维度：自我调节与求助策略
（Self-Regulated Learning & Help-Seeking Strategies）

脚本目标（与画像设计文档对齐）：
--------------------------------------------------
本脚本对应你画像设计中“自我调节与求助策略”维度，重点刻画：
- 学习者在遇到困难时，是否会：
  * 先独立尝试一段时间再适度求助；
  * 一开始就频繁查看提示/例题（过度依赖帮助）；
  * 即便反复错误也很少使用帮助（求助回避）；
- 是否会主动使用系统提供的“反馈、补救资源、反思”等工具。

本脚本使用的 xAPI 行为 / 动词：
- answered            ：做题作答（用于识别是否遇到困难、是否最终解出）
- requested-support   ：请求帮助（hint/example/video/teacher 等）
- reviewed-feedback   ：查看反馈（题目级 or 课程级仪表盘）
- explored-extension  ：使用扩展单元/补救资源
- reflected-on-activity：提交活动后的反思文本

一、论文依据与方法设计说明
--------------------------------------------------
1）“使用帮助资源与补救单元作为自我监控与补救策略的行为指标”
   - Mangina et al. (2022), *Experience API (xAPI) for Virtual Reality (VR) Education in Medicine*：
     在医学 VR 场景下，作者使用 xAPI 记录学习者在任务过程中的
     “请求帮助、查阅参考资料、查看反馈面板”等行为，用于分析学习者
     的自我监控（self-monitoring）与补救策略（remedial strategies）。:contentReference[oaicite:0]{index=0}
   - 由此，本脚本将：
     - requested-support 视为“在遇到困难时主动求助”的关键行为；
     - reviewed-feedback / explored-extension / reflected-on-activity
       视为“监控-补救-反思”阶段的自我调节行为。

2）“以自我调节学习 SRL 框架组织行为指标”
   - Tao, Cukurova & Song (2025), *Learning analytics in immersive virtual learning environments:
     a systematic literature review*：
     综述中强调，沉浸式学习分析常以 SRL 框架（计划-监控-调节）为理论基础，
     如“使用反馈、自我反思、时间管理、使用资源等”行为可映射到 SRL 的不同阶段。:contentReference[oaicite:1]{index=1}
   - 本脚本据此将：
     - “做题时的求助模式”看作在线监控与调节（monitoring & control）；
     - “查看反馈 / 补救扩展单元 / 反思文本”看作学习后期的监控与反思（reflection）。

3）“适应性求助（adaptive help-seeking） vs 过度求助与求助回避”
   - Roll et al. (JLS manuscript, 2016 附近版本), *On the Benefits of Seeking (and Avoiding) Help
     in Online Problem-Solving Environments*：
     利用智能导师系统日志，作者区分了多种求助模式，核心结论包括：:contentReference[oaicite:2]{index=2}
       * 在具有挑战性的步骤上适度求助往往与更好的学习相关；
       * 过度使用帮助（help abuse）反而与较差学习相关；
       * 完全避免求助但反复失败的模式并不总是有利，需要结合是否最终成功来判断。
   - 本脚本借鉴其思想而非完全复制具体模型：
     - 重点区分三类策略：
       a) 适应性求助：在出现错误后适度求助，不在一开始就频繁看提示；
       b) 过度依赖：即便并未多次错误就频繁请求帮助；
       c) 求助回避：反复错误却基本不请求帮助。
     - 我们通过“错误后的求助比例”“首次作答前求助比例”等特征来刻画这些模式。

4）“基于行为特征聚类形成画像类型”
   - Lampropoulos & Evangelidis (2025), *Learning Analytics and Educational Data Mining
     in AR/VR/Metaverse*：
     综述表明，在 XR + LA/EDM 研究中常用 k-means 等聚类算法，将行为特征聚合成
     “策略型/被动型”等学习者类型，为画像提供离散标签。
   - 本脚本在得到“自我调节与求助策略指数（SRL_help_index）”后，
     使用一维 k-means（k=3）自动划分为：
       - 低自我调节与求助策略水平（偏回避/被动型）
       - 中等自我调节与求助策略水平（混合型）
       - 高自我调节与求助策略水平（主动监控 + 有策略求助型）

二、分析方法概述
--------------------------------------------------
1. 从 MongoDB 读取：
   - 细粒度行为：MLS.Interaction 集合（xAPI_interaction_profile.py 生成）
   - 学习者画像：MLS.LearnerProfile 集合（global_profile 中的 feedback_orientation、
     reflection_depth 等，用于“验证”而非参与建模）

2. 事件筛选与特征构造：
   - 仅保留 verb.id 属于：
     answered / requested-support / reviewed-feedback /
     explored-extension / reflected-on-activity 的事件。
   - 以 (学习者, 课程, 题目) 为粒度，在时间序列上统计：
     * 每题错误次数、是否最终做对；
     * 错误后是否出现 requested-support（错误后求助）；
     * 是否在首次作答之前就请求帮助（过早求助）；
     * 哪些题目在没有任何帮助下最终成功（独立解决）。
   - 以 (学习者, 课程) 为粒度，聚合得到：
     * help_need_rate        = 有错误题目中“错误后有求助”的比例；
     * early_help_ratio      = 所有求助中“首答前求助”的比例；
     * no_help_success_ratio = 成功题目中“完全没用帮助”的比例；
     * feedback_density      = reviewed-feedback 次数 / 题目数；
     * extension_flag        = 是否使用过 explored-extension；
     * reflection_flag       = 是否出现过 reflected-on-activity。

3. 自我调节与求助策略指数（SRL_help_index）的计算：
   - 根据 Roll et al. 关于“适度求助”的结论，把 help_need_rate 的理想区间设置在
     0.4～0.8 左右，并用类高斯函数给出适应性得分（过低或过高都降分）。:contentReference[oaicite:4]{index=4}
   - 对 early_help_ratio 使用 “1 - early_help_ratio” 作为“不过早依赖帮助”的得分。
   - 将 no_help_success_ratio 视为“独立解决能力”的得分（在有错误但不求助的情况
     下，会被 help_need_rate 这一维度惩罚，从而区分“健康的独立”与“风险性回避”）。
   - 对 feedback_density、extension_flag、reflection_flag 直接归一化到 [0,1] 作为
     “监控-补救-反思”得分。
   - 最终 SRL_help_index 是以上多个子分数的加权平均，结果归一化到 [0,1]。

4. 聚类与标签设计：
   - 对所有 (学习者, 课程) 的 SRL_help_index 运行一维 k-means (k=3)，
     按聚类中心从低到高排序，得到 cluster_rank ∈ {0,1,2}，
     并映射到：
       0 -> "low"    / 低自我调节与求助策略
       1 -> "medium" / 中等
       2 -> "high"   / 高自我调节与求助策略

5. 与人设对比（验证用途）：
   - 用 LearnerProfile.global_profile 中：
     feedback_orientation.score 与 reflection_depth.score 的平均值
     作为“人设侧自我调节/求助倾向”的先验指标（仅用于对比，不参与建模）。
   - 将每个学习者在所有课程上的 SRL_help_index 取平均，得到行为侧 global_srl_help。
   - 计算两者的皮尔逊相关系数，用于验证细粒度行为分析是否与预设人设方向一致。

6. 数据库存储接口（不在 main() 中调用）：
   - 定义 save_srl_help_analysis_to_db(db, results) 函数，
     演示如何把结果写入 MLS.SRLHelpAnalysis 集合，但默认不调用，
     与 analyze_task_efficiency.py 中的接口风格保持一致。

"""

from pymongo import MongoClient
from datetime import datetime
from math import sqrt, exp
from collections import defaultdict
import random
from tqdm import tqdm

# ===================== 配置区域 =====================

MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "MLS"
XAPI_COLLECTION = "Interaction"       # 细粒度行为集合（xAPI_interaction_profile.py 生成）
PROFILE_COLLECTION = "LearnerProfile" # 人设集合（infer_persona_for_course 写入）

SRL_HELP_COLLECTION = "SRLHelpAnalysis"  # 自我调节与求助策略分析结果集合（仅接口，不在 main 中调用）

# 采样学习者数量：随机选取至多 N_SAMPLE 个学习者进行分析
# 若 N_SAMPLE <= 0 或大于实际可选数量，则使用所有学习者
N_SAMPLE = 3000

VERB_BASE = "https://legend-meta.com/xapi/verb/"
ACTIVITY_TYPE_BASE = "https://legend-meta.com/xapi/activity-type/"

VERBS = {
    "answered": VERB_BASE + "answered",
    "requested_support": VERB_BASE + "requested-support",
    "reviewed_feedback": VERB_BASE + "reviewed-feedback",
    "explored_extension": VERB_BASE + "explored-extension",
    "reflected_on_activity": VERB_BASE + "reflected-on-activity",
}

QUESTION_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "item"
COURSE_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "course"

# ===================== 工具函数 =====================

def compute_mean_std(values):
    """
    计算一组数的均值和标准差（总体标准差）：
    - 列表为空 -> (0, 0)
    - 仅一个元素 -> 标准差视为 0

    用途：
    - 对 (学习者, 课程) 的 SRL_help_index 做整体分析；
    - 与人设分数进行相关性计算。
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
    一维 k-means 聚类（Lloyd 算法实现），用于基于 SRL_help_index 自动划分学习者类型。

    方法论依据：
    --------------------------------------------------
    - Lampropoulos & Evangelidis (2025) 的 XR+LA/EDM 综述中总结了大量使用
      k-means 等聚类方法，以行为特征划分学习者策略类型的研究。
    - 本函数沿用 analyze_task_efficiency.py 中的一维 k-means 思路，
      但聚类对象从“任务效率指数”换成“自我调节与求助策略指数 SRL_help_index”。

    参数：
    - values: List[float]，需聚类的数值列表；
    - k: 聚类数，默认 3（低/中/高）；
    - max_iter: 最大迭代次数。

    返回：
    - centers: 聚类中心列表（长度 k）；
    - labels: 与 values 同长度的列表，每个元素为 0..k-1 的聚类编号。
    """
    if not values:
        return [], []

    # 去重后随机初始化 k 个中心
    unique_vals = sorted(set(values))
    if len(unique_vals) <= k:
        centers = unique_vals[:]
        # 若唯一值少于 k，重复最后一个值填满
        while len(centers) < k:
            centers.append(unique_vals[-1])
    else:
        centers = random.sample(unique_vals, k)

    for _ in range(max_iter):
        # 分配簇
        clusters = [[] for _ in range(k)]
        for v in values:
            # 找到最近的中心
            dists = [abs(v - c) for c in centers]
            idx = dists.index(min(dists))
            clusters[idx].append(v)

        # 更新中心
        new_centers = []
        for i in range(k):
            if clusters[i]:
                new_centers.append(sum(clusters[i]) / float(len(clusters[i])))
            else:
                # 若某簇为空，随机选一个值作为新的中心
                new_centers.append(random.choice(values))

        # 收敛判定
        if all(abs(a - b) < 1e-6 for a, b in zip(centers, new_centers)):
            centers = new_centers
            break
        centers = new_centers

    # 最终再分配一次标签
    labels = []
    for v in values:
        dists = [abs(v - c) for c in centers]
        idx = dists.index(min(dists))
        labels.append(idx)

    return centers, labels


def safe_ratio(num, den):
    """安全计算比例，避免除零，返回浮点数。"""
    if den is None or den == 0:
        return 0.0
    return float(num) / float(den)


def is_question_object(obj):
    """
    判断 xAPI 对象是否为题目级活动：
    - xAPI_interaction_profile.py 中题目对象的 id 形如：
      https://legend-meta.com/item/{question_uid}
    - definition.type 为 QUESTION_ACTIVITY_TYPE。
    """
    if not obj:
        return False
    obj_id = obj.get("id") or ""
    if obj_id.startswith("https://legend-meta.com/item/"):
        return True
    definition = obj.get("definition") or {}
    tp = definition.get("type")
    if tp == QUESTION_ACTIVITY_TYPE:
        return True
    return False


def is_course_object(obj):
    """
    判断 xAPI 对象是否为课程级活动：
    - id 形如 https://legend-meta.com/course/{course_uid}
    - 或 definition.type 为 COURSE_ACTIVITY_TYPE
    """
    if not obj:
        return False
    obj_id = obj.get("id") or ""
    if obj_id.startswith("https://legend-meta.com/course/"):
        return True
    definition = obj.get("definition") or {}
    tp = definition.get("type")
    if tp == COURSE_ACTIVITY_TYPE:
        return True
    return False


def compute_srl_help_index_for_course(course_stat):
    """
    根据单个 (学习者, 课程) 的统计结果，计算自我调节与求助策略指数 SRL_help_index。

    输入结构（course_stat）：
    --------------------------------------------------
    {
        "questions": {
            question_id: {
                "attempts": int,            # 总作答次数
                "wrong": int,               # 错误次数
                "correct": int,             # 正确次数
                "help_total": int,          # 求助总次数
                "help_after_error": int,    # 在出现错误之后的求助次数
                "help_before_attempt": int, # 未作答就求助的次数
            },
            ...
        },
        "feedback_question": int,            # 题目级反馈查看次数
        "feedback_course": int,             # 课程级仪表盘反馈查看次数
        "extension_cnt": int,               # explored-extension 次数
        "reflection_cnt": int,              # reflected-on-activity 次数
    }

    指数计算思路与文献映射：
    --------------------------------------------------
    1）help_need_rate：在“有错误的题目”中，有多少比例在错误后发生了求助。
       - 概念上对应 Roll 等人所谓的“在需要时请求帮助”的适应性求助。:contentReference[oaicite:5]{index=5}
       - 理想情况：0.4～0.8 之间（既不完全回避，也不过度依赖），
         用类高斯函数以 0.6 为中心给出适应性得分 adaptivity_score ∈ [0,1]。

    2）early_help_ratio：所有求助中，有多少是在首次作答之前发生的。
       - 过高意味着倾向于“一开始就看答案”，可能是“过早求助/依赖型”。
       - 使用 early_score = 1 - early_help_ratio 对其进行惩罚。

    3）no_help_success_ratio：最终做对的题目中，有多少是完全没有任何求助的。
       - 反映独立解决能力（independent problem solving），
         但若同时 help_need_rate 非常低且错误题很多，则可能是“带风险的求助回避”。
       - 在综合指数中与 adaptivity_score 共同作用，避免极端回避被误判为高自我调节。

    4）feedback_density：所有题目上的平均反馈查看频率。
       - 体现是否主动使用“reviewed-feedback”监控学习结果，
         对应 Mangina 与 Tao 综述中 SRL 的监控与调节阶段。:contentReference[oaicite:6]{index=6}

    5）extension_flag / reflection_flag：
       - 是否使用补救资源（explored-extension）；
       - 是否在课程后提交反思（reflected-on-activity）。
       - 二者直接作为 0/1 指标，体现“补救策略”和“反思策略”的存在与否。

    最终 SRL_help_index 是以下子分数的加权平均：
        SRL_help_index = w1*adaptivity + w2*early_score + w3*independent_score
                         + w4*feedback_score + w5*extension_score + w6*reflection_score
    权重设计：
        - 适应性求助（w1）和独立解决（w3）权重略高（各 0.25），
          因为它们直接体现 Roll 等人强调的求助模式质量；
        - early_score 作为求助时机的修正因子（0.15）；
        - feedback/extension/reflection 代表 SRL 的监控与反思环节（各 0.15）。
    """
    questions = course_stat.get("questions") or {}
    feedback_question = course_stat.get("feedback_question", 0)
    feedback_course = course_stat.get("feedback_course", 0)
    extension_cnt = course_stat.get("extension_cnt", 0)
    reflection_cnt = course_stat.get("reflection_cnt", 0)

    if not questions:
        # 没有任何题目行为时，很难评估求助策略，返回中性偏低值
        return 0.4, {
            "help_need_rate": 0.0,
            "early_help_ratio": 0.0,
            "no_help_success_ratio": 0.0,
            "feedback_density": 0.0,
            "extension_flag": 1 if extension_cnt > 0 else 0,
            "reflection_flag": 1 if reflection_cnt > 0 else 0,
        }

    num_questions = 0
    num_error_questions = 0
    total_wrong_attempts = 0
    total_help_events = 0
    total_help_after_error = 0
    total_help_before_attempt = 0

    num_success_questions = 0
    num_success_no_help_questions = 0
    num_error_then_success_no_help_questions = 0

    for q_id, qstat in questions.items():
        attempts = qstat.get("attempts", 0)
        wrong = qstat.get("wrong", 0)
        correct = qstat.get("correct", 0)
        help_total = qstat.get("help_total", 0)
        help_after_error = qstat.get("help_after_error", 0)
        help_before_attempt = qstat.get("help_before_attempt", 0)

        if attempts <= 0:
            continue

        num_questions += 1
        total_wrong_attempts += wrong
        total_help_events += help_total
        total_help_after_error += help_after_error
        total_help_before_attempt += help_before_attempt

        if wrong > 0:
            num_error_questions += 1

        if correct > 0:
            num_success_questions += 1
            if help_total == 0:
                num_success_no_help_questions += 1
                if wrong > 0:
                    num_error_then_success_no_help_questions += 1

    # 1) 错误后的求助比例
    if num_error_questions > 0:
        help_need_rate = safe_ratio(total_help_after_error, num_error_questions)
    else:
        # 如果几乎没有错误，则说明题目对其难度较低，此时无法认真评估“需要时是否求助”，
        # 这里将 help_need_rate 设为 0.5 的中性值。
        help_need_rate = 0.5

    # 2) 过早求助比例
    if total_help_events > 0:
        early_help_ratio = safe_ratio(total_help_before_attempt, total_help_events)
    else:
        early_help_ratio = 0.0

    # 3) 无帮助成功比例
    if num_success_questions > 0:
        no_help_success_ratio = safe_ratio(num_success_no_help_questions, num_success_questions)
    else:
        no_help_success_ratio = 0.0

    # 4) 反馈密度（题目数为基础）
    total_feedback = feedback_question + feedback_course
    if num_questions > 0:
        feedback_density = safe_ratio(total_feedback, num_questions)
    else:
        feedback_density = 0.0

    extension_flag = 1 if extension_cnt > 0 else 0
    reflection_flag = 1 if reflection_cnt > 0 else 0

    # ---------- 将上述特征映射到 [0,1] 子分数 ----------

    # 1) 适应性求助得分：help_need_rate 的理想值设在 0.6，一般 0.4~0.8 视为较好。
    #    使用高斯形状：score = exp( - ((x-0.6)^2 / (2*0.3^2)) )
    x = help_need_rate
    adaptivity_score = exp(- ((x - 0.6) ** 2) / (2 * (0.3 ** 2)))
    # clip 到 [0,1]
    adaptivity_score = max(0.0, min(1.0, adaptivity_score))

    # 2) 过早求助得分：early_help_ratio 越小越好，采用 1 - early_help_ratio，但 clip 到 [0,1]
    early_score = 1.0 - early_help_ratio
    early_score = max(0.0, min(1.0, early_score))

    # 3) 独立解决得分：直接使用 no_help_success_ratio，但稍微向中间收缩，
    #    避免极端 1.0（完全不用帮助）在错误很多的情况下被误判。
    #    这里简单采用：independent_score = 0.5 + 0.5 * (no_help_success_ratio - 0.5)
    #    即向 0.5 收缩。
    independent_score = 0.5 + 0.5 * (no_help_success_ratio - 0.5)
    independent_score = max(0.0, min(1.0, independent_score))

    # 4) 反馈得分：以 0.5 次/题为上限做线性归一化，超过则直接记为 1
    feedback_score = min(feedback_density / 0.5, 1.0)

    # 5) 补救与反思得分：0/1 指标
    extension_score = float(extension_flag)
    reflection_score = float(reflection_flag)

    # ---------- 综合加权 ----------

    w1 = 0.25  # 适应性求助
    w2 = 0.15  # 不过早依赖
    w3 = 0.25  # 独立解决
    w4 = 0.15  # 反馈使用
    w5 = 0.10  # 补救资源
    w6 = 0.10  # 反思行为

    total_weight = w1 + w2 + w3 + w4 + w5 + w6
    srl_index = (
        w1 * adaptivity_score
        + w2 * early_score
        + w3 * independent_score
        + w4 * feedback_score
        + w5 * extension_score
        + w6 * reflection_score
    ) / float(total_weight)

    # clip [0,1]
    srl_index = max(0.0, min(1.0, srl_index))

    feature_summary = {
        "help_need_rate": float(help_need_rate),
        "early_help_ratio": float(early_help_ratio),
        "no_help_success_ratio": float(no_help_success_ratio),
        "feedback_density": float(feedback_density),
        "extension_flag": extension_flag,
        "reflection_flag": reflection_flag,
    }

    return srl_index, feature_summary


# ===================== 主分析函数 =====================

def analyze_srl_help_from_xapi(xapi_col, sampled_learners):
    """
    核心分析函数：从 xAPI 行为中推断“自我调节与求助策略”维度。

    步骤：
    1. 从 MLS.Interaction 中筛选：
       - _lrn_uid ∈ sampled_learners
       - verb.id ∈ {answered, requested-support, reviewed-feedback,
                    explored-extension, reflected-on-activity}
    2. 按 (学习者, 课程) 聚合，并在内部按题目粒度统计求助模式。
    3. 对每个 (学习者, 课程) 计算 SRL_help_index 及特征。
    4. 使用一维 k-means 将 SRL_help_index 离散为三类（低/中/高）。

    返回：
    - results: dict[(learner_uid, course_uid)] -> {
          "SRL_help_index": float in [0,1],
          "cluster_rank": int in {0,1,2},
          "cluster_label": str ("low"/"medium"/"high"),
          "features": {...}
      }
    """
    print("统计待加载的自我调节相关事件数量（count_documents）...")
    query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": {
            "$in": [
                VERBS["answered"],
                VERBS["requested_support"],
                VERBS["reviewed_feedback"],
                VERBS["explored_extension"],
                VERBS["reflected_on_activity"],
            ]
        },
    }

    total_events = xapi_col.count_documents(query)
    print(f"与采样学习者相关的自我调节/求助事件总数：{total_events}")

    print("开始一次性加载所有相关事件到内存（list）...")
    events = list(
        xapi_col.find(
            query,
            {
                "verb.id": 1,
                "object": 1,
                "result": 1,
                "_lrn_uid": 1,
                "_course_uid": 1,
                "timestamp": 1,
            },
        )
    )
    print(f"已从 MongoDB 读取事件条数：{len(events)}")

    if not events:
        print("没有任何自我调节/求助相关事件，无法进行分析。")
        return {}

    # ---------- 按 (学习者, 课程) 分组，并在组内按时间排序 ----------

    events_by_lc = defaultdict(list)
    for doc in events:
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue
        events_by_lc[(lrn_uid, crs_uid)].append(doc)

    # 对每个 (learner, course) 的事件按时间排序
    for key, evs in events_by_lc.items():
        evs.sort(key=lambda d: str(d.get("timestamp") or ""))

    # ---------- 在每个 (learner, course) 内统计题目级求助模式与反馈/扩展/反思 ----------

    course_stats = {}  # (lrn_uid, crs_uid) -> stat dict

    print("开始遍历事件并构建 (学习者, 课程) 粗粒度统计...")
    for (lrn_uid, crs_uid), evs in tqdm(
        events_by_lc.items(), desc="按课程统计", unit="course"
    ):
        stat = {
            "questions": defaultdict(
                lambda: {
                    "attempts": 0,
                    "wrong": 0,
                    "correct": 0,
                    "help_total": 0,
                    "help_after_error": 0,
                    "help_before_attempt": 0,
                }
            ),
            "feedback_question": 0,
            "feedback_course": 0,
            "extension_cnt": 0,
            "reflection_cnt": 0,
        }

        for doc in evs:
            verb = (doc.get("verb") or {}).get("id")
            obj = doc.get("object") or {}
            result = doc.get("result") or {}

            is_q = is_question_object(obj)
            is_c = is_course_object(obj)
            obj_id = obj.get("id") or ""

            # 1) 题目作答：统计错误/正确/尝试次数
            if verb == VERBS["answered"] and is_q:
                qstat = stat["questions"][obj_id]
                qstat["attempts"] += 1
                success = result.get("success")
                if success is True:
                    qstat["correct"] += 1
                else:
                    qstat["wrong"] += 1

            # 2) 求助事件：根据当前题目状态区分“错误后求助” vs “首答前求助”
            elif verb == VERBS["requested_support"] and is_q:
                qstat = stat["questions"][obj_id]
                qstat["help_total"] += 1
                # 若还未有任何作答，则视为“首答前求助”
                if qstat["attempts"] == 0:
                    qstat["help_before_attempt"] += 1
                else:
                    # 若已经有错误但尚未正确，则视为“错误后求助”
                    if qstat["wrong"] > 0 and qstat["correct"] == 0:
                        qstat["help_after_error"] += 1

            # 3) 查看反馈：区分题目级与课程级
            elif verb == VERBS["reviewed_feedback"]:
                if is_q:
                    stat["feedback_question"] += 1
                elif is_c:
                    stat["feedback_course"] += 1
                else:
                    # 若无法判断，按课程级反馈计入
                    stat["feedback_course"] += 1

            # 4) 使用扩展/补救单元
            elif verb == VERBS["explored_extension"]:
                stat["extension_cnt"] += 1

            # 5) 提交反思
            elif verb == VERBS["reflected_on_activity"]:
                stat["reflection_cnt"] += 1

        course_stats[(lrn_uid, crs_uid)] = stat

    # ---------- 对每个 (learner, course) 计算 SRL_help_index ----------

    srl_indices = []
    srl_results = {}

    for key, stat in course_stats.items():
        srl_index, feature_summary = compute_srl_help_index_for_course(stat)
        srl_results[key] = {
            "SRL_help_index": srl_index,
            "features": feature_summary,
            "cluster_rank": None,   # 暂时占位，后面用 k-means 填充
            "cluster_label": None,
        }
        srl_indices.append(srl_index)

    if not srl_indices:
        print("在聚合后没有可用的 SRL_help_index，分析中止。")
        return {}

    # ---------- 使用一维 k-means 聚类，将 SRL_help_index 划分为 3 档 ----------

    print("对 SRL_help_index 进行一维 k-means 聚类（k=3）...")
    centers, labels = kmeans_1d(srl_indices, k=3, max_iter=50)

    if not centers:
        print("k-means 聚类失败或数据不足，将跳过聚类标签，仅保留连续指数。")
        return srl_results

    # 将簇中心从低到高排序，并建立簇 -> 排名 的映射
    sorted_centers = sorted([(c, idx) for idx, c in enumerate(centers)], key=lambda x: x[0])
    # rank_map: 原始簇编号 -> 0/1/2（0 为最低 SRL_help_index，2 为最高）
    rank_map = {orig_idx: rank for rank, (c, orig_idx) in enumerate(sorted_centers)}

    # labels 与 srl_indices 一一对应，因此需要再次遍历 srl_results 保持顺序
    # 为方便，这里重新遍历并同步索引
    idx = 0
    for key in course_stats.keys():
        res = srl_results[key]
        srl_val = res["SRL_help_index"]
        # 找到该 srl_val 在 srl_indices 中的标签（按照相同顺序）
        cluster = labels[idx]
        idx += 1
        rank = rank_map.get(cluster, 1)
        if rank == 0:
            label = "low"
        elif rank == 1:
            label = "medium"
        else:
            label = "high"
        res["cluster_rank"] = int(rank)
        res["cluster_label"] = label

    return srl_results


# ===================== 写回数据库的接口（默认不调用） =====================

def save_srl_help_analysis_to_db(db, srl_results):
    """
    将自我调节与求助策略分析结果写入 MongoDB（接口函数，默认不在 main() 中调用）。

    存储设计：
    --------------------------------------------------
    - 集合名：MLS.SRLHelpAnalysis
    - 文档结构：
        {
            "learner_uid": ...,
            "course_uid": ...,
            "SRL_help_index": float in [0,1],
            "help_need_rate": float,
            "early_help_ratio": float,
            "no_help_success_ratio": float,
            "feedback_density": float,
            "extension_flag": int (0/1),
            "reflection_flag": int (0/1),
            "cluster_rank": int (0/1/2),
            "cluster_label": "low"/"medium"/"high",
            "created_at": datetime.utcnow(),
        }

    与 analyze_task_efficiency.py 的对应关系：
    --------------------------------------------------
    - 类似地，这里也：
        1）在写入前 drop 掉 SRLHelpAnalysis 集合（便于重复实验）；
        2）建立 (learner_uid, course_uid) 复合索引；
        3）仅当外部显式调用该函数时才会写入数据库，
           main() 中不会自动调用，以避免影响现有数据。
    """
    srl_col = db[SRL_HELP_COLLECTION]

    # 为方便重复实验，这里选择先清空集合
    db.drop_collection(SRL_HELP_COLLECTION)
    srl_col = db[SRL_HELP_COLLECTION]

    docs_to_insert = []
    for (lrn_uid, crs_uid), res in srl_results.items():
        feat = res.get("features") or {}
        doc = {
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "SRL_help_index": res.get("SRL_help_index"),
            "help_need_rate": feat.get("help_need_rate"),
            "early_help_ratio": feat.get("early_help_ratio"),
            "no_help_success_ratio": feat.get("no_help_success_ratio"),
            "feedback_density": feat.get("feedback_density"),
            "extension_flag": feat.get("extension_flag"),
            "reflection_flag": feat.get("reflection_flag"),
            "cluster_rank": res.get("cluster_rank"),
            "cluster_label": res.get("cluster_label"),
            "created_at": datetime.utcnow(),
        }
        docs_to_insert.append(doc)

    if docs_to_insert:
        srl_col.insert_many(docs_to_insert, ordered=False)
        srl_col.create_index(
            [("learner_uid", 1), ("course_uid", 1)],
            name="idx_learner_course",
        )
        print(f"[接口调用] 已写入 SRLHelpAnalysis 文档数：{len(docs_to_insert)}")
    else:
        print("[接口调用] 没有可写入 SRLHelpAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]

    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中的人设，用于对比验证 ----------
    print("读取 LearnerProfile 中的自我调节/求助相关人设信息...")

    persona_scores = {}  # lrn_uid -> persona_srl_help_score
    cursor_profiles = profile_col.find(
        {},
        {
            "learner_uid": 1,
            "global_profile": 1,
        },
    )

    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g_profile = doc.get("global_profile") or {}

        # 这里选择 feedback_orientation 和 reflection_depth 作为自我调节相关维度，
        # 仅用于“验证性”对比，不参与行为侧建模。
        fb = (g_profile.get("feedback_orientation") or {}).get("score")
        ref = (g_profile.get("reflection_depth") or {}).get("score")

        if fb is None and ref is None:
            continue

        scores = []
        if fb is not None:
            scores.append(float(fb))
        if ref is not None:
            scores.append(float(ref))

        if scores:
            persona_scores[lrn_uid] = sum(scores) / float(len(scores))

    all_learners_with_persona = list(persona_scores.keys())
    print(f"具备自我调节/求助相关人设的学习者数量：{len(all_learners_with_persona)}")

    if not all_learners_with_persona:
        print("没有任何学习者具备相关人设，后续将仅做行为侧分析，不做人设对比。")

    # ---------- 3. 随机采样学习者 ----------
    if all_learners_with_persona:
        candidate_learners = all_learners_with_persona
    else:
        # 若没有人设，则从 xAPI 集合中获取所有学习者 uid 作为候选
        candidate_learners = xapi_col.distinct("_lrn_uid")

    if not candidate_learners:
        print("在 xAPI 或人设中均未找到学习者，分析中止。")
        return

    if N_SAMPLE > 0 and N_SAMPLE < len(candidate_learners):
        sampled_learners = random.sample(candidate_learners, N_SAMPLE)
    else:
        sampled_learners = list(candidate_learners)

    sampled_set = set(sampled_learners)
    print(f"随机选取用于分析的学习者数量：{len(sampled_learners)} (N_SAMPLE = {N_SAMPLE})")

    # ---------- 4. 基于 xAPI 行为计算 SRL_help_index ----------
    srl_results = analyze_srl_help_from_xapi(xapi_col, sampled_learners)
    if not srl_results:
        print("未得到任何 SRL_help_index 结果，分析结束。")
        return

    # ---------- 5. 统计整体分布 & 与人设的相关性 ----------
    # 5.1 统计各类别数量
    cluster_counts = defaultdict(int)
    cluster_indices = defaultdict(list)

    for (lrn_uid, crs_uid), res in srl_results.items():
        rank = res.get("cluster_rank")
        idx = res.get("SRL_help_index")
        if rank is not None:
            cluster_counts[rank] += 1
            cluster_indices[rank].append(idx)

    print("=========================================================")
    print("【自我调节与求助策略维度：行为侧聚类结果概览】")
    for rank in sorted(cluster_counts.keys()):
        label = "低" if rank == 0 else ("中" if rank == 1 else "高")
        cnt = cluster_counts[rank]
        idx_list = cluster_indices[rank]
        if idx_list:
            mean_idx = sum(idx_list) / float(len(idx_list))
        else:
            mean_idx = 0.0
        print(f"- 聚类等级 {rank}（{label} 自我调节/求助策略）：")
        print(f"  样本数：{cnt}，平均 SRL_help_index：{mean_idx:.3f}")
    print("=========================================================")

    # 5.2 计算与人设的相关性
    if persona_scores:
        xs = []  # 人设侧
        ys = []  # 行为侧 global SRL_help_index

        # 先按学习者聚合行为侧 SRL_help_index（对所有课程取平均）
        learner_srl_global = defaultdict(list)
        for (lrn_uid, crs_uid), res in srl_results.items():
            learner_srl_global[lrn_uid].append(res.get("SRL_help_index") or 0.0)

        learner_srl_global_mean = {}
        for lrn_uid, vals in learner_srl_global.items():
            if vals:
                learner_srl_global_mean[lrn_uid] = sum(vals) / float(len(vals))

        for lrn_uid, persona_score in persona_scores.items():
            beh = learner_srl_global_mean.get(lrn_uid)
            if beh is not None:
                xs.append(float(persona_score))
                ys.append(float(beh))

        if len(xs) >= 2:
            mean_x, std_x = compute_mean_std(xs)
            mean_y, std_y = compute_mean_std(ys)
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / float(len(xs))
            if std_x > 1e-6 and std_y > 1e-6:
                corr = cov / (std_x * std_y)
            else:
                corr = 0.0

            avg_behavior = sum(ys) / float(len(ys))
            avg_persona = sum(xs) / float(len(xs))

            print("【自我调节与求助策略维度：人设 vs 行为分析 全局对比】")
            print(f"- 具备相关人设的学习者数量：{len(all_learners_with_persona)}")
            print(f"- 实际参与对比的学习者数量：{len(xs)}")
            print(f"- 行为侧 global SRL_help_index 平均值：{avg_behavior:.3f}")
            print(f"- 人设侧 (feedback_orientation & reflection_depth) 平均值：{avg_persona:.3f}")
            print(f"- 皮尔逊相关系数：{corr:.3f}")
            print("  （该相关系数用于粗略验证：细粒度 xAPI 分析是否与人设中自我调节/求助相关维度在方向上较为一致。）")
            print("=========================================================")
        else:
            print("参与人设对比的学习者样本太少，无法计算相关系数。")
    else:
        print("由于缺乏可用的人设维度，本次分析未进行行为与人设的定量对比。")

    print("自我调节与求助策略维度分析完成。")
    print("如果需要将结果写回 MongoDB，请显式调用 save_srl_help_analysis_to_db(db, srl_results) 函数。")


if __name__ == "__main__":
    main()
