# -*- coding: utf-8 -*-
"""
分析维度：反馈敏感度与数据使用能力
(Feedback Orientation & Data Use Literacy)

脚本目标（与画像设计文档对齐）：
--------------------------------------------------
对应《论文.txt》中“反馈敏感度与自我调节倾向”相关维度：
【反馈敏感度与数据使用能力】：学习者是否愿意查看系统反馈（成绩、进度、协作数据）
并据此调整行为，是“看数据型”还是“感受型”。

本脚本采用的行为代理与论文依据：
--------------------------------------------------
1）“查看学习仪表盘/反馈面板的频率与类型 → 反馈敏感度差异”
   - Heinemann et al. 2024 的 VR 学习仪表盘（Learning Analytics Dashboard）
     设计目标是让学生/教师通过可视化数据理解学习过程，
     并以“是否查看、查看频率、查看何种面板”作为仪表盘使用差异的关键观察点。
   → 因此，本脚本在 xAPI 中选取 verb = reviewed-feedback 的事件，
     统计：
     (a) 反馈查看次数 feedback_view_count
     (b) 反馈查看频率（按“可产生反馈的任务机会数”归一化）feedback_view_rate
     (c) 反馈查看类型分布 feedback_view_type_dist （unit/course/group dashboard）

2）“游戏内嵌入式 GLA 反馈的使用行为 → 数据使用能力/自我监控倾向”
   - Papamitsiou 2024 提出将 GLA 嵌入 IVR 游戏作为可交互的反馈元素；
     Papamitsiou 2025 实证中用“查看个人/团队进度板频率、是否关注实时表现反馈”
     区分不同学习体验与价值感。
   → 因此，本脚本把：
     - reviewed-feedback : 进度板/反馈板查看
     - requested-support 中“查看解析/示例/提示” : 即时反馈使用
     作为 data use literacy 的关键行为。

3）“反馈 → 行为调整（正确率提升）”的序列效应
   - Heinemann 仪表盘与 Papamitsiou GLA 都强调：
     反馈的价值在于触发学习者自我监控/反思并调整后续行为。
   → 因此，本脚本对每个学习者按时间排序的行为序列做窗口化分析：
     (a) 寻找错误事件（answered/completed success=False）
     (b) 观察其后是否在短时间窗内查看反馈（reviewed-feedback）
     (c) 比较查看反馈前后的成功率变化 improvement_after_feedback
     作为“反馈敏感度 + 有效使用反馈”的行为证据。

4）标签与分档设计依据
   - 你的整体画像框架（analyze_task_efficiency.py）已采用三档离散水平并用 k-means 自动分群。
   → 因此，本脚本同样对行为侧反馈指数 FO_norm 进行一维 k-means（k=3）聚类，
     形成：
       “低反馈敏感/低数据使用型”
       “中等反馈敏感/一般数据使用型”
       “高反馈敏感/高数据使用型”
     三种类型。

脚本功能总结：
--------------------------------------------------
1. 从 MongoDB 中读取：
   - 细粒度 xAPI 行为：MLS.Interaction 集合；
   - 学习者画像：MLS.LearnerProfile 集合（用于对比验证，不参与计算）。

2. 对每个 (学习者, 课程)：
   - 统计反馈查看与即时反馈使用行为；
   - 计算机会归一化的反馈查看率、即时反馈使用率；
   - 计算反馈后正确率提升 improvement_after_feedback；
   - 在课程内做 z 标准化并合成为反馈指数 FO；
   - 在课程内 min-max 归一化得到 FO_norm ∈ [0,1]。

3. 基于所有 (学习者, 课程) 的 FO_norm：
   - 使用一维 k-means（k=3）聚类并产出三档标签。

4. 与人设对比：
   - 汇总每个学习者的 global_feedback_orientation = mean(FO_norm over courses)
   - 与 LearnerProfile 中对应维度评分做皮尔逊相关（若字段存在）
     用于粗略验证分析有效性。

5. 数据库存储接口（不在 main() 中调用）：
   - save_feedback_orientation_to_db(db, results)
     预留写回 MLS.FeedbackOrientationAnalysis 集合的接口，但默认不调用。
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
from math import sqrt
from collections import defaultdict
import re
import random
from tqdm import tqdm

# ===================== 配置区域 =====================

MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "MLS"
XAPI_COLLECTION = "Interaction"
PROFILE_COLLECTION = "LearnerProfile"
FO_COLLECTION = "FeedbackOrientationAnalysis"   # 反馈敏感度分析结果集合（仅接口，不在 main 中调用）

# 随机采样学习者数量
N_SAMPLE = 3000

# 与 analyze_task_efficiency.py 保持一致的 verb base
VERB_BASE = "https://legend-meta.com/xapi/verb/"

VERBS = {
    "reviewed_feedback": VERB_BASE + "reviewed-feedback",
    "requested_support": VERB_BASE + "requested-support",
    "answered": VERB_BASE + "answered",
    "completed": VERB_BASE + "completed",
    "performed_procedure_step": VERB_BASE + "performed-procedure-step",
}

# 解析 duration 的简单正则（用于必要时计算机会数）
DURATION_RE = re.compile(r"^PT(\d+)S$")

# 反馈“策略调整”窗口参数
FEEDBACK_WINDOW_MINUTES = 10   # 错误后多久内查看反馈算“使用反馈”
POST_FEEDBACK_K = 3            # 反馈后取多少次任务结果来比较正确率

# ===================== 工具函数 =====================

def parse_iso8601_duration(duration_str):
    """解析简单 ISO8601 时长字符串 PT{秒}S；和任务效率脚本一致。"""
    if not duration_str:
        return None
    m = DURATION_RE.match(duration_str)
    if m:
        return int(m.group(1))
    return None


def compute_mean_std(values):
    """计算均值与总体标准差（同任务效率脚本）。"""
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
    一维 k-means 聚类（与任务效率脚本完全一致的实现与注释风格）。
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


# ===================== 数据库存储接口（不在 main 中调用） =====================

def save_feedback_orientation_to_db(db, results):
    """
    把反馈敏感度分析结果写入 MongoDB 的接口函数（默认不在 main 中调用）。

    写入字段设计依据：
    --------------------------------------------------
    - learner_uid / course_uid：与 MLS.Interaction 的 _lrn_uid / _course_uid 对齐
    - feedback_view_*、support_view_*：
      对应 Heinemann 仪表盘与 Papamitsiou GLA 中的“反馈/进度板使用频率与类型”
    - improvement_after_feedback：
      对应“反馈触发自我监控→后续行为调整”的机制代理
    - FO / FO_norm 与 label：
      行为侧连续指数与三档离散类型
    """
    col = db[FO_COLLECTION]
    db.drop_collection(FO_COLLECTION)
    col = db[FO_COLLECTION]

    docs = []
    for (lrn_uid, crs_uid), r in results.items():
        docs.append({
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "feedback_view_count": r["feedback_view_count"],
            "feedback_view_rate": r["feedback_view_rate"],
            "feedback_view_type_dist": r["feedback_view_type_dist"],
            "support_view_count": r["support_view_count"],
            "support_view_rate": r["support_view_rate"],
            "improvement_after_feedback": r["improvement_after_feedback"],
            "FO": r["FO"],
            "FO_norm": r["FO_norm"],
            "feedback_label": r.get("feedback_label"),
            "cluster_rank": r.get("cluster_rank"),
            "created_at": datetime.utcnow(),
        })

    if docs:
        col.insert_many(docs, ordered=False)
        col.create_index([("learner_uid", 1), ("course_uid", 1)], name="idx_learner_course")
        print(f"[接口调用] 已写入 FeedbackOrientationAnalysis 文档数：{len(docs)}")
    else:
        print("[接口调用] 没有可写入 FeedbackOrientationAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]
    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中的人设（仅用于对比） ----------
    print("读取 LearnerProfile 中的人设信息...")
    persona_scores = {}  # lrn_uid -> persona_feedback_score

    cursor_profiles = profile_col.find(
        {},
        {"learner_uid": 1, "global_profile": 1}
    )

    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g_profile = doc.get("global_profile") or {}

        # 兼容不同命名：优先找 feedback_orientation / data_use_literacy / feedback_sensitivity
        cand_keys = ["feedback_orientation", "data_use_literacy", "feedback_sensitivity"]
        score = None
        for k in cand_keys:
            dim = g_profile.get(k) or {}
            if "score" in dim:
                score = dim.get("score")
                break
        if score is not None:
            persona_scores[lrn_uid] = float(score)

    learners_with_persona = list(persona_scores.keys())
    print(f"具备该维度人设分数的学习者数量：{len(learners_with_persona)}")

    if not learners_with_persona:
        print("没有任何学习者具备该维度人设，仍可输出行为结果，但无法做人设一致性检验。")

    # ---------- 3. 随机采样学习者 ----------
    if learners_with_persona:
        pool = learners_with_persona
    else:
        # 若没有人设，则从全部学习者池采样
        pool = list(xapi_col.distinct("_lrn_uid"))

    if N_SAMPLE > 0 and N_SAMPLE < len(pool):
        sampled_learners = random.sample(pool, N_SAMPLE)
    else:
        sampled_learners = pool

    print(f"随机选取用于分析的学习者数量：{len(sampled_learners)} (N_SAMPLE = {N_SAMPLE})")

    if not sampled_learners:
        print("没有可分析的学习者。")
        return

    # ---------- 4. 一次性加载相关事件 ----------
    """
    事件筛选策略与论文依据：
    --------------------------------------------------
    - reviewed-feedback：核心反馈/仪表盘查看行为（Heinemann 2024；Papamitsiou 2024/2025）
    - requested-support：其中“查看解析/示例/提示”作为即时反馈使用（Papamitsiou GLA 的实时反馈）
    - answered / completed / performed-procedure-step：
      作为“产生反馈机会”的任务事件，也用于计算错误→反馈→提升效应
    """
    query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": {"$in": [
            VERBS["reviewed_feedback"],
            VERBS["requested_support"],
            VERBS["answered"],
            VERBS["completed"],
            VERBS["performed_procedure_step"],
        ]}
    }

    total_events = xapi_col.count_documents(query)
    print(f"与采样学习者相关的事件总数：{total_events}")

    events = list(xapi_col.find(
        query,
        {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "timestamp": 1,
            "result": 1,
            "context": 1
        }
    ))
    print(f"已从 MongoDB 读取事件条数：{len(events)}")

    if not events:
        print("没有任何相关事件，结束分析。")
        return

    # ---------- 5. 预处理：按 (learner, course) 分组并按时间排序 ----------
    lc_to_events = defaultdict(list)
    for e in events:
        lrn_uid = e.get("_lrn_uid")
        crs_uid = e.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue
        ts = e.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                ts = None
        e["_parsed_ts"] = ts
        lc_to_events[(lrn_uid, crs_uid)].append(e)

    for key in lc_to_events:
        lc_to_events[key].sort(key=lambda x: x.get("_parsed_ts") or datetime.min)

    # ---------- 6. 逐 (学习者, 课程) 计算行为指标 ----------
    results = {}

    for (lrn_uid, crs_uid), seq in tqdm(lc_to_events.items(), desc="计算反馈指标", unit="lc"):
        feedback_views = []
        support_views = []
        task_outcomes = []  # [(ts, success_bool)]

        # 6.1 抽取三类事件
        for e in seq:
            vid = e.get("verb", {}).get("id") or e.get("verb.id")
            ts = e.get("_parsed_ts")

            if vid == VERBS["reviewed_feedback"]:
                feedback_views.append(e)

            elif vid == VERBS["requested_support"]:
                # requested-support 如果有 supportType/targetObjectType 等扩展，
                # 且属于“解析/示例/提示”，则视为即时反馈使用；否则也计入 support_use
                support_views.append(e)

            elif vid in (VERBS["answered"], VERBS["completed"], VERBS["performed_procedure_step"]):
                r = e.get("result") or {}
                success = r.get("success")
                completion = r.get("completion")
                if success is None and completion is None:
                    continue
                ok = bool(success) if success is not None else bool(completion)
                task_outcomes.append((ts, ok))

        # 6.2 反馈查看类型分布
        type_counts = defaultdict(int)
        for fv in feedback_views:
            r = fv.get("result") or {}
            c = fv.get("context") or {}
            ext_r = (r.get("extensions") or {})
            ext_c = ((c.get("extensions") or {}) if isinstance(c, dict) else {})
            view_type = ext_r.get("feedback-view-type") or ext_c.get("feedback-view-type")
            if not view_type:
                view_type = "unknown"
            type_counts[str(view_type)] += 1

        total_type = sum(type_counts.values()) or 1
        feedback_view_type_dist = {k: v / total_type for k, v in type_counts.items()}

        # 6.3 机会归一化频率
        feedback_view_count = len(feedback_views)
        support_view_count = len(support_views)
        opportunity_count = len(task_outcomes)  # 视作“产生反馈机会”的任务数

        if opportunity_count <= 0:
            feedback_view_rate = 0.0
            support_view_rate = 0.0
        else:
            feedback_view_rate = feedback_view_count / float(opportunity_count)
            support_view_rate = support_view_count / float(opportunity_count)

        # 6.4 反馈后正确率提升（策略调整代理）
        """
        计算流程：
        --------------------------------------------------
        a) 在任务序列中找到错误点 (success=False)
        b) 若错误后 FEEDBACK_WINDOW_MINUTES 内出现 reviewed-feedback，
           则记为一次“反馈使用”
        c) 取该反馈之后的 POST_FEEDBACK_K 次任务结果，
           与该反馈之前的 POST_FEEDBACK_K 次任务结果做正确率差
        d) 对所有可用窗口取平均，作为 improvement_after_feedback
        """
        improvements = []
        window_delta = timedelta(minutes=FEEDBACK_WINDOW_MINUTES)

        # 把 task_outcomes 与 feedback 时间点统一到 seq 上扫描
        # 先做一个方便的列表：事件时间、类型、success
        timeline = []
        for e in seq:
            ts = e.get("_parsed_ts")
            if not ts:
                continue
            vid = e.get("verb", {}).get("id") or e.get("verb.id")

            if vid in (VERBS["answered"], VERBS["completed"], VERBS["performed_procedure_step"]):
                r = e.get("result") or {}
                success = r.get("success")
                completion = r.get("completion")
                if success is None and completion is None:
                    continue
                ok = bool(success) if success is not None else bool(completion)
                timeline.append(("task", ts, ok))

            elif vid == VERBS["reviewed_feedback"]:
                timeline.append(("feedback", ts, None))

        timeline.sort(key=lambda x: x[1])

        # 扫描窗口
        for i, (typ, ts, ok) in enumerate(timeline):
            if typ != "task" or ok is True:
                continue  # 只关注错误任务

            # 找最近的反馈查看
            fb_idx = None
            for j in range(i + 1, len(timeline)):
                if timeline[j][0] == "feedback" and timeline[j][1] - ts <= window_delta:
                    fb_idx = j
                    break
                if timeline[j][1] - ts > window_delta:
                    break

            if fb_idx is None:
                continue

            # 取反馈前后 K 个 task 结果
            pre = []
            post = []

            # pre
            k = i - 1
            while k >= 0 and len(pre) < POST_FEEDBACK_K:
                if timeline[k][0] == "task":
                    pre.append(timeline[k][2])
                k -= 1

            # post
            k = fb_idx + 1
            while k < len(timeline) and len(post) < POST_FEEDBACK_K:
                if timeline[k][0] == "task":
                    post.append(timeline[k][2])
                k += 1

            if pre and post:
                pre_acc = sum(1 for x in pre if x) / float(len(pre))
                post_acc = sum(1 for x in post if x) / float(len(post))
                improvements.append(post_acc - pre_acc)

        improvement_after_feedback = sum(improvements) / float(len(improvements)) if improvements else 0.0

        results[(lrn_uid, crs_uid)] = {
            "feedback_view_count": feedback_view_count,
            "feedback_view_rate": feedback_view_rate,
            "feedback_view_type_dist": feedback_view_type_dist,
            "support_view_count": support_view_count,
            "support_view_rate": support_view_rate,
            "improvement_after_feedback": improvement_after_feedback,
        }

    if not results:
        print("没有可用的 (学习者, 课程) 结果，结束。")
        return

    # ---------- 7. 按课程做 z 标准化并合成 FO ----------
    """
    合成指数思路（与论文一致）：
    --------------------------------------------------
    - Papamitsiou 2025 以“查看进度板/反馈频率、关注实时反馈”解释差异
    - Heinemann 仪表盘强调“查看与理解反馈”的学习者差异
    → 因此我们用三项行为代理：
       (1) feedback_view_rate
       (2) support_view_rate
       (3) improvement_after_feedback
      在课程内做 z 标准化后取平均得到 FO
    """
    course_to_keys = defaultdict(list)
    for (lrn_uid, crs_uid) in results:
        course_to_keys[crs_uid].append((lrn_uid, crs_uid))

    for crs_uid, keys in course_to_keys.items():
        fv_rates = [results[k]["feedback_view_rate"] for k in keys]
        sp_rates = [results[k]["support_view_rate"] for k in keys]
        imps = [results[k]["improvement_after_feedback"] for k in keys]

        m_fv, s_fv = compute_mean_std(fv_rates)
        m_sp, s_sp = compute_mean_std(sp_rates)
        m_im, s_im = compute_mean_std(imps)

        FO_vals = []
        for k in keys:
            z_fv = (results[k]["feedback_view_rate"] - m_fv) / s_fv if s_fv > 1e-6 else 0.0
            z_sp = (results[k]["support_view_rate"] - m_sp) / s_sp if s_sp > 1e-6 else 0.0
            z_im = (results[k]["improvement_after_feedback"] - m_im) / s_im if s_im > 1e-6 else 0.0

            FO = (z_fv + z_sp + z_im) / 3.0
            results[k]["FO"] = FO
            FO_vals.append(FO)

        # min-max 归一化
        if FO_vals:
            FO_min, FO_max = min(FO_vals), max(FO_vals)
            span = FO_max - FO_min if FO_max > FO_min else 0.0
            for k in keys:
                FO = results[k]["FO"]
                if span > 1e-6:
                    FO_norm = (FO - FO_min) / span
                else:
                    FO_norm = 0.5
                results[k]["FO_norm"] = FO_norm

    print("课程层面的反馈敏感度指数 FO_norm 计算完成。")
    print(f"结果条目数（学习者-课程对）：{len(results)}")

    # ---------- 8. 基于 FO_norm 的学习者类型聚类 ----------
    all_FO_norm = [r["FO_norm"] for r in results.values()]
    centers, assignments = kmeans_1d(all_FO_norm, k=3, max_iter=50)

    if centers:
        sorted_idx = sorted(range(len(centers)), key=lambda i: centers[i])
        cluster_to_rank = {cluster_idx: rank for rank, cluster_idx in enumerate(sorted_idx)}

        rank_to_label = {
            0: "低反馈敏感/低数据使用型（几乎不查看反馈或不使用解析；反馈后正确率提升不明显）",
            1: "中等反馈敏感/一般数据使用型（偶尔查看反馈；会在部分场景使用解析/示例）",
            2: "高反馈敏感/高数据使用型（频繁查看反馈面板/进度板；积极用解析并能调整策略）",
        }

        for ((key, r), cluster_idx) in zip(results.items(), assignments):
            rank = cluster_to_rank.get(cluster_idx, 1)
            r["cluster_index"] = int(cluster_idx)
            r["cluster_rank"] = int(rank)
            r["feedback_label"] = rank_to_label.get(rank, rank_to_label[1])

        label_counts = defaultdict(int)
        for r in results.values():
            label_counts[r["feedback_label"]] += 1

        print("反馈敏感度标签分布（按学习者-课程对统计）：")
        for label, cnt in label_counts.items():
            print(f"- {label}: {cnt} 条记录")
    else:
        print("k-means 聚类未能得到有效中心，跳过类型标签生成。")

    # ---------- 9. （可选）写回数据库接口——默认不调用 ----------
    """
    若需要写回数据库，请取消下一行注释：
        save_feedback_orientation_to_db(db, results)
    """
    # save_feedback_orientation_to_db(db, results)

    # ---------- 10. 按学习者汇总 global_feedback_orientation 并与人设对比 ----------
    learner_to_vals = defaultdict(list)
    for (lrn_uid, _crs_uid), r in results.items():
        learner_to_vals[lrn_uid].append(r["FO_norm"])

    learner_global_FO = {}
    for lrn_uid, vals in learner_to_vals.items():
        learner_global_FO[lrn_uid] = sum(vals) / float(len(vals))

    print("\n学习者全局反馈敏感度（行为侧）示例：")
    sample_items = list(learner_global_FO.items())[:10]
    for lrn_uid, gfo in sample_items:
        print(f"- learner={lrn_uid} global_feedback_orientation={gfo:.3f}")

    # 与人设做相关
    if persona_scores:
        common = [(lrn, learner_global_FO[lrn], persona_scores[lrn])
                  for lrn in learner_global_FO if lrn in persona_scores]

        if len(common) >= 3:
            beh_vals = [x[1] for x in common]
            per_vals = [x[2] for x in common]

            m_b, s_b = compute_mean_std(beh_vals)
            m_p, s_p = compute_mean_std(per_vals)

            cov = sum((b - m_b) * (p - m_p) for b, p in zip(beh_vals, per_vals)) / float(len(common))
            corr = cov / (s_b * s_p) if s_b > 1e-6 and s_p > 1e-6 else 0.0

            print("\n[一致性检验] 行为侧 global_feedback_orientation 与人设 score 的皮尔逊相关：")
            print(f"- n_common={len(common)}  corr={corr:.4f}")
        else:
            print("\n[一致性检验] 行为侧与人设的交集样本不足，无法计算相关。")
    else:
        print("\n未读取到该维度人设分数，跳过一致性检验。")


if __name__ == "__main__":
    main()
