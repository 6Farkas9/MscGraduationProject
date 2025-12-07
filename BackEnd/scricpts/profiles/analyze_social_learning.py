# -*- coding: utf-8 -*-
"""
分析维度：社会性学习与同伴取向（Social Learning & Peer Orientation）

脚本目标（与画像设计文档对齐）：
--------------------------------------------------
对应画像维度：
【社会性学习与同伴取向（Social Learning & Peer Orientation）】

在你的画像设计中，该维度主要关注：
- 学习者是否通过“观察同伴、查看同伴作品”进行学习；
- 是否在协作空间中保持“社交可见”（即既出现为观众，也参与共同创作）。 
- xAPI 中对应的核心动词有：
  * observed-peer：观摩同伴作品 / 行为；
  * collaborated-on-activity：在协作学习单元中进行共同操作 / 编辑。

本脚本采用的行为代理与论文依据：
--------------------------------------------------
1）协作学习中“贡献 + 参与 + 社交动态”的框架
   - Hu, Ng, & Chu (2022) 在 wiki 协作学习环境中提出了协作质量分析框架，
     将日志指标拆分为：
       * contribution（贡献）：编辑次数、增加内容量等；
       * participation（参与）：参与天数、查看次数等；
       * social dynamics（社会动态）：与他人交互、对他人作品操作的比例等。
   - 在 VR 内容共创的协作 LA 研究（如 VR content creation collaboration quality）
     中，该框架被迁移到 VR 场景，以“共同编辑对象的时长与频率”
     作为协作质量的重要指标。
   → 本脚本对应地选择：
      - collaborated-on-activity 的 result.duration 作为“协作贡献”代理；
      - observed-peer 的 result.duration 作为“观摩参与 / 社会性观察”代理。

2）XR/Metaverse + LA/EDM 综述中的“社会维度”画像
   - Lampropoulos & Evangelidis (2025) 在 XR/Metaverse 的 LA/EDM 综述中，
     将“协作水平、群体参与度、社交角色”作为 XR 学习环境中的
     关键画像维度之一，并强调可以通过行为日志
     （协作操作日志 + 社交互动日志）来建模。
   → 因此，本脚本将“观摩同伴 + 协作活动”的行为组合
      视为社会性学习与同伴取向的主要可观测代理。

3）协作 VR Dashboard 中的参与角色（尤其是“观察者型”）
   - 协作 VR 学习分析 Dashboard 相关研究中，常区分：
       * 仅浏览他人作品、不参与编辑的“观察者型（observer-type）”；
       * 既浏览他人作品又参与编辑的“积极协作者 / 社会学习型”。
   → 本脚本在标签设计上显式区分：
       - 观察型：观摩时间占绝大多数，协作很少；
       - 积极社会学习型：观摩与协作都较多且较均衡；
       - 协作导向型：协作远多于观摩；
       - 低社交参与型：观摩与协作都极少。

4）社会性学习指数（Social Learning Index）的构造方式
   - 借鉴 Hu et al. 框架中对多个行为指标进行标准化并合成维度得分的方式，
     本脚本在“课程内部”对每个 (学习者, 课程) 的：
       * 观摩总时长（observed-peer duration total）,
       * 协作总时长（collaborated-on-activity duration total）
     进行 z 标准化：
       z_obs, z_collab
     然后构造：
       S = (z_obs + z_collab) / sqrt(2)
     再在同一课程内对 S 做 min-max 归一化，得到：
       social_index_normalized ∈ [0, 1]
   - 这一做法与任务效率分析脚本中对 E_norm 的处理保持一致，
     方便后续在画像层面进行统一比较与可视化。

5）分类标签设计与阈值选择
   - 标签类型：
       1. "low_social_participation" / "低社交参与型"
       2. "observer_dominant" / "观察型（观摩为主）"
       3. "collab_dominant" / "协作导向型"
       4. "balanced_active_social" / "积极社会学习型（观摩+协作均衡）"
   - 主要依据：
       * 协作 VR LAD 中“观察者型”的定义（只看不写 / 观摩远多于编辑）。
       * 协作学习 LA 文献中常用的“相对贡献比例”作为角色划分标准。
   - 具体规则（在课程内部统计的基础上）：
       - 记 total_time = obs_time + collab_time（两类时长之和）；
       - 若 total_time < MIN_SOCIAL_TIME 或两类事件总数几乎为 0：
           → 低社交参与型；
       - 否则，记 ratio_obs = obs_time / total_time：
           * ratio_obs ≥ 0.8（观摩时间占比极高）：
               → 观察型（observer_dominant）；
           * ratio_obs ≤ 0.3 且协作时长较大：
               → 协作导向型（collab_dominant）；
           * 0.3 < ratio_obs < 0.8 且 social_index_normalized 较高：
               → 积极社会学习型（balanced_active_social）。

脚本功能总结：
--------------------------------------------------
1. 从 MongoDB 中读取：
   - 细粒度 xAPI 行为：MLS.Interaction 集合；
   - 学习者画像：MLS.LearnerProfile 集合（特别是 global_profile.social_learning.score）。

2. 对每个 (学习者, 课程)：
   - 基于 verb = observed-peer / collaborated-on-activity 的 xAPI 语句，
     计算：
       * obs_count：观摩事件次数；
       * obs_total_time：观摩总时长（秒）；
       * obs_unique_peers：观摩对象数量（去重）；
       * collab_count：协作事件次数；
       * collab_total_time：协作总时长（秒）。
   - 在“课程内部”对 obs_total_time 与 collab_total_time 做 z 标准化，
     得到 z_obs, z_collab；
   - 构造社会性学习指数：
       S = (z_obs + z_collab) / sqrt(2)
     并做 min-max 归一化为 social_index_normalized ∈ [0, 1]。

3. 基于 total_time 与 ratio_obs 进行角色分类：
   - 给每个 (学习者, 课程) 一个：
       * social_label（上述四种标签之一）；
       * cluster_rank（0~3 的整数，用于代表从“低”到“强社交学习”的等级）。

4. 与人设对比：
   - 对每个学习者，把其在所有课程上的 social_index_normalized 做平均，
     得到行为侧 global_social_learning_index；
   - 与 LearnerProfile.global_profile.social_learning.score 做皮尔逊相关，
     粗略验证行为分析结果与预设人设的一致性。

5. 数据库存储接口（不在 main() 中调用）：
   - 定义 save_social_learning_to_db(db, social_results) 函数，
     演示如何把结果写入 MLS.SocialLearningAnalysis 集合，但默认不调用。
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
XAPI_COLLECTION = "Interaction"          # 细粒度行为集合（xAPI_interaction_profile.py 生成）
PROFILE_COLLECTION = "LearnerProfile"    # 人设集合（infer_persona_for_course 写入）
SOCIAL_COLLECTION = "SocialLearningAnalysis"  # 社会性学习维度分析结果集合（仅接口，不在 main 中调用）

# 采样学习者数量：随机选取至多 N_SAMPLE 个学习者进行分析
# 若 N_SAMPLE <= 0 或大于实际可选数量，则使用所有学习者
N_SAMPLE = 3000

VERB_BASE = "https://legend-meta.com/xapi/verb/"

VERBS = {
    "observed_peer": VERB_BASE + "observed-peer",
    "collaborated_on_activity": VERB_BASE + "collaborated-on-activity",
}

# 预编译 duration 的正则，避免每次 re.compile
DURATION_RE = re.compile(r"^PT(\d+)S$")

# 判定“低社交参与”的总时长阈值（秒）
# 这里设置为 30 秒：少于 30 秒的观摩+协作时长视为偶然行为，不将其视为稳定倾向。
MIN_SOCIAL_TIME = 30.0


# ===================== 工具函数 =====================

def parse_iso8601_duration(duration_str):
    """
    解析简单形式的 ISO8601 时长字符串，例如："PT120S"
    若为空或格式不符，返回 None。

    设计说明：
    - xAPI_interaction_profile 中生成的 duration 字段统一使用整数秒的 PT{n}S 形式，
      因此这里采用简单正则解析即可。
    - 若未来扩展为更复杂的 ISO8601 表达式，可以在此处集中升级解析逻辑。
    """
    if not duration_str or not isinstance(duration_str, str):
        return None
    m = DURATION_RE.match(duration_str)
    if not m:
        return None
    try:
        seconds = int(m.group(1))
        if seconds < 0:
            return None
        return float(seconds)
    except ValueError:
        return None


def compute_mean_std(values):
    """
    计算一组数的平均值与标准差（总体标准差）。

    - 若列表为空，返回 (0.0, 0.0)
    - 若只有一个样本，标准差视为 0.0

    本函数与任务效率脚本中的实现思路一致，用于在课程内部
    对行为指标（观摩时长 / 协作时长）做标准化。
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


def ratio_safe(numerator, denominator):
    """
    安全计算比例，避免除零。
    - 若分母为 0 或 None，返回 0.0
    """
    if denominator is None or denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


# ===================== 数据库存储接口（默认不调用） =====================

def save_social_learning_to_db(db, social_results):
    """
    将社会性学习维度分析结果写入 MongoDB（仅接口，不在 main 中自动调用）。

    设计说明：
    --------------------------------------------------
    1）集合名：
        - 使用 SOCIAL_COLLECTION = "SocialLearningAnalysis"
          与 TaskEfficiencyAnalysis 区分开。

    2）字段设计（每条文档对应一个 (learner_uid, course_uid)）：
        - learner_uid, course_uid：主键字段；
        - obs_count, obs_total_time, obs_unique_peers：
            观摩事件次数/总时长/观摩对象数；
        - collab_count, collab_total_time：
            协作事件次数/总时长；
        - z_obs, z_collab：
            课程内部标准化后的观摩/协作 z 值；
        - social_index：合成后的社会性学习指数 S；
        - social_index_normalized：在课程内做 min-max 后的归一化结果；
        - social_label：四类社会性角色标签之一；
        - cluster_rank：0~3 的整数，代表从“低社交参与”到“积极社会学习”的等级；
        - created_at：写入时间戳。

    注意：
    --------------------------------------------------
    - 本函数不会在 main() 中自动调用。
    - 如果你希望实际写回数据库，请在 main() 中手动解除注释：
        save_social_learning_to_db(db, social_results)
    """
    col = db[SOCIAL_COLLECTION]

    # 为方便重复实验，这里先清空集合（若你不想清空，可以改为 update 或 upsert）
    db.drop_collection(SOCIAL_COLLECTION)
    col = db[SOCIAL_COLLECTION]

    docs_to_insert = []
    for (lrn_uid, crs_uid), res in social_results.items():
        doc = {
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "obs_count": res.get("obs_count", 0),
            "obs_total_time": res.get("obs_total_time", 0.0),
            "obs_unique_peers": res.get("obs_unique_peers", 0),
            "collab_count": res.get("collab_count", 0),
            "collab_total_time": res.get("collab_total_time", 0.0),
            "z_obs": res.get("z_obs"),
            "z_collab": res.get("z_collab"),
            "social_index": res.get("social_index"),
            "social_index_normalized": res.get("social_index_normalized"),
            "social_label": res.get("social_label"),
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
        print(f"[接口调用] 已写入 SocialLearningAnalysis 文档数：{len(docs_to_insert)}")
    else:
        print("[接口调用] 没有可写入 SocialLearningAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]

    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中的社会性学习人设 ----------
    print("读取 LearnerProfile 中的人设信息（social_learning 维度）.")
    persona_scores = {}  # lrn_uid -> persona_social_learning_score

    cursor_profiles = profile_col.find(
        {},
        {"learner_uid": 1, "global_profile": 1}
    )

    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g_profile = doc.get("global_profile") or {}
        soc = g_profile.get("social_learning") or {}
        score = soc.get("score")
        if score is not None:
            # infer_persona_for_course 中应该已将 social_learning.score
            # 映射到 [0,1] 区间，这里直接读取作为人设中的“先验社会性学习倾向”。
            persona_scores[lrn_uid] = float(score)

    all_learners_with_persona = list(persona_scores.keys())
    print(f"具备社会性学习人设的学习者数量：{len(all_learners_with_persona)}")

    if not all_learners_with_persona:
        print("没有任何学习者具备社会性学习人设，无法进行对比分析。")
        return

    # ---------- 3. 随机采样学习者 ----------
    if N_SAMPLE > 0 and N_SAMPLE < len(all_learners_with_persona):
        sampled_learners = random.sample(all_learners_with_persona, N_SAMPLE)
    else:
        sampled_learners = all_learners_with_persona
    sampled_set = set(sampled_learners)

    print(f"随机选取用于分析的学习者数量：{len(sampled_learners)} (N_SAMPLE = {N_SAMPLE})")

    # ---------- 4. 一次性加载采样学习者的社会性相关事件 ----------
    """
    事件筛选策略与论文依据：
    --------------------------------------------------
    1）使用的 verb：
       - observed-peer：
         * 对应“观摩同伴作品 / 观看同伴表现”的事件；
         * 在协作学习文献中，相当于 participation / social viewing 行为，
           用于刻画“通过观察他人来学习”的社会性学习方式。
       - collaborated-on-activity：
         * 对应协作单元中的共同操作 / 编辑行为；
         * 在 Hu et al. 框架和 VR 内容共创 LA 研究中，被视为
           贡献/参与（contribution & participation）的核心指标。

    2）查询条件：
       - 仅针对采样学习者（_lrn_uid in sampled_learners），避免处理全部数据；
       - 只保留上述两个 verb.id 的事件。
    """
    query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": {"$in": [VERBS["observed_peer"], VERBS["collaborated_on_activity"]]}
    }

    print("统计待加载的社会性事件数量（count_documents）.")
    total_events = xapi_col.count_documents(query)
    print(f"与采样学习者相关的社会性学习事件总数：{total_events}")

    print("开始一次性加载所有相关事件到内存（list）.")
    events = list(xapi_col.find(
        query,
        {
            "verb.id": 1,
            "result": 1,
            "_lrn_uid": 1,
            "_course_uid": 1,
            "context": 1,
        }
    ))
    print(f"已从 MongoDB 读取事件条数：{len(events)}")

    if not events:
        print("没有任何 observed-peer / collaborated-on-activity 事件，无法进行社会性学习分析。")
        return

    # ---------- 5. 遍历事件，按 (学习者, 课程) 聚合社会性统计 ----------
    """
    聚合逻辑说明：
    --------------------------------------------------
    - 粒度：以 (学习者, 课程) 为聚合单位，与整体画像框架的“按课程窗口刻画行为倾向”保持一致。
    - 对每个 (学习者, 课程)，累积：
        obs_count：observed-peer 事件次数；
        obs_total_time：观摩总时长（秒，基于 result.duration）；
        obs_unique_peers：观摩对象数量（基于 context.extensions.observed-learner-id 去重）；
        collab_count：collaborated-on-activity 事件次数；
        collab_total_time：协作总时长（秒）。
    - 这些指标对应 Hu et al. 的协作质量框架中的：
        participation（参与）与 contribution（贡献），
      也与 XR/Metaverse 文献中“通过日志刻画协作与社会角色”的思路一致。
    """
    # (learner_uid, course_uid) -> 聚合统计
    social_stats = defaultdict(lambda: {
        "obs_count": 0,
        "obs_total_time": 0.0,
        "obs_peers": set(),
        "collab_count": 0,
        "collab_total_time": 0.0,
    })
    used_events = 0

    print("开始遍历社会性事件并进行聚合计算.")
    for doc in tqdm(events, desc="聚合社会性事件", unit="event"):
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue

        verb_id = (doc.get("verb") or {}).get("id")
        result = doc.get("result") or {}
        duration_str = result.get("duration")
        duration_sec = parse_iso8601_duration(duration_str)

        # 注意：某些 observed-peer 可能没有 duration，此时 duration_sec 为 None
        # 对于这类事件，仍然可以计数，但不计入时长。
        if duration_sec is not None and duration_sec < 0:
            continue

        key = (lrn_uid, crs_uid)
        stat = social_stats[key]

        if verb_id == VERBS["observed_peer"]:
            stat["obs_count"] += 1
            if duration_sec is not None:
                stat["obs_total_time"] += float(duration_sec)

            context = doc.get("context") or {}
            ext = context.get("extensions") or {}
            peer_id = ext.get("https://legend-meta.com/xapi/ext/observed-learner-id")
            if peer_id:
                stat["obs_peers"].add(peer_id)

            used_events += 1

        elif verb_id == VERBS["collaborated_on_activity"]:
            stat["collab_count"] += 1
            if duration_sec is not None:
                stat["collab_total_time"] += float(duration_sec)
            used_events += 1

    print(f"参与社会性学习统计的有效事件数：{used_events}")
    print(f"有社会性数据的 (学习者, 课程) 组合数：{len(social_stats)}")

    if not social_stats:
        print("聚合后没有任何可用社会性行为统计数据，结束分析。")
        return

    # ---------- 6. 计算每个 (学习者, 课程) 的社会性学习指标（课程内标准化） ----------
    """
    课程内标准化与社会性学习指数公式的依据：
    --------------------------------------------------
    1）课程内比较：
       - 不同课程的协作任务设计、协作单元数量和时长差异较大，
         直接跨课程比较“绝对观摩时长 / 协作时长”容易失真。
       - 因此，本脚本在“每门课程内部”对 obs_total_time 和 collab_total_time
         进行标准化，并计算社会性学习指数 S，
         仅在同一课程内比较学生之间的社会性学习差异。

    2）社会性学习指数公式：
       - 借鉴 Hu et al. 框架中“多指标标准化后合成维度得分”的思路，
         将观摩时长与协作时长均视为社会性学习的正向指标。
       - 定义：
           z_obs = 标准化后的观摩时长
           z_collab = 标准化后的协作时长
           S = (z_obs + z_collab) / sqrt(2)
         * 两个指标越大，S 越大，表示该学习者在社交观察与协作贡献方面
           都相对活跃。
       - 再在课程内对 S 做 min-max 归一化，得到
           social_index_normalized ∈ [0,1]
         以便与画像中 [0,1] 区间的分数对齐。
    """
    # course_uid -> list[(lrn_uid, obs_total, collab_total, obs_count, collab_count, obs_peers_count)]
    course_to_entries = defaultdict(list)
    for (lrn_uid, crs_uid), stat in social_stats.items():
        obs_total = stat["obs_total_time"]
        collab_total = stat["collab_total_time"]
        obs_count = stat["obs_count"]
        collab_count = stat["collab_count"]
        obs_peers_count = len(stat["obs_peers"]) if stat["obs_peers"] else 0
        course_to_entries[crs_uid].append(
            (lrn_uid, obs_total, collab_total, obs_count, collab_count, obs_peers_count)
        )

    social_results = {}  # (lrn_uid, crs_uid) -> {...}

    print("按课程计算社会性学习指数 S 及归一化 social_index_normalized.")
    for crs_uid, entries in course_to_entries.items():
        if not entries:
            continue

        obs_vals = [e[1] for e in entries]
        collab_vals = [e[2] for e in entries]
        mean_obs, std_obs = compute_mean_std(obs_vals)
        mean_collab, std_collab = compute_mean_std(collab_vals)

        S_vals = []
        # 先计算每个学习者的 S 值，为后续 min-max 归一化准备
        for (lrn_uid, obs_total, collab_total, obs_count, collab_count, obs_peers_count) in entries:
            z_obs = (obs_total - mean_obs) / std_obs if std_obs > 1e-6 else 0.0
            z_collab = (collab_total - mean_collab) / std_collab if std_collab > 1e-6 else 0.0

            # 社会性学习指数：两个行为指标都越大越“社会化”，因此采用正向合成。
            S = (z_obs + z_collab) / sqrt(2.0)

            key = (lrn_uid, crs_uid)
            social_results[key] = {
                "obs_count": obs_count,
                "obs_total_time": obs_total,
                "obs_unique_peers": obs_peers_count,
                "collab_count": collab_count,
                "collab_total_time": collab_total,
                "z_obs": z_obs,
                "z_collab": z_collab,
                "social_index": S,
            }
            S_vals.append(S)

        # 在当前课程内部对 S 做 [0,1] 的 min-max 归一化
        if S_vals:
            S_min = min(S_vals)
            S_max = max(S_vals)
            span = S_max - S_min if S_max > S_min else 0.0
            for (lrn_uid, obs_total, collab_total, obs_count, collab_count, obs_peers_count) in entries:
                key = (lrn_uid, crs_uid)
                S = social_results[key]["social_index"]
                if span > 1e-6:
                    S_norm = (S - S_min) / span
                else:
                    # 当所有人的 S 完全相同（只有一个学习者或行为高度一致）时，
                    # 无法区分社会性水平，这里统一给 0.5 作为中间值。
                    S_norm = 0.5
                social_results[key]["social_index_normalized"] = S_norm

    print(f"成功得到 social_index_normalized 的 (学习者, 课程) 数量：{len(social_results)}")

    if not social_results:
        print("没有可用的社会性学习指数，结束。")
        return

    # ---------- 7. 基于观摩/协作比例进行角色分类 ----------
    """
    角色分类设计说明：
    --------------------------------------------------
    - 参考协作 VR LAD 及 CSCL 角色划分研究中的“观察者型 / 积极协作者”等角色，
      本脚本结合 total_time 和 ratio_obs（观摩时间在总社会性时间中的占比）
      将每个 (学习者, 课程) 分类为以下四类角色：

      1）低社交参与型（low_social_participation，cluster_rank = 0）
         - 总社会性时间 total_time < MIN_SOCIAL_TIME，或
           obs_count + collab_count 极少；
         - 表示该课程中几乎没有通过观摩或协作来学习。

      2）观察型（observer_dominant，cluster_rank = 1）
         - total_time ≥ MIN_SOCIAL_TIME；
         - ratio_obs ≥ 0.8（绝大部分时间用于观摩）；
         - 协作时长很少或没有；
         - 对应“只看不写或主要看别人的作品”的参与方式。

      3）协作导向型（collab_dominant，cluster_rank = 2）
         - total_time ≥ MIN_SOCIAL_TIME；
         - ratio_obs ≤ 0.3 且协作时长较为可观；
         - 表示更偏向于作为“共同创作者”参与，而较少通过观摩他人作品学习。

      4）积极社会学习型（balanced_active_social，cluster_rank = 3）
         - total_time ≥ MIN_SOCIAL_TIME；
         - 0.3 < ratio_obs < 0.8；
         - social_index_normalized 较高（例如 ≥ 0.5）；
         - 既通过观摩他人来获取灵感，也积极参与共同编辑，
           对应文献中经常强调的“积极社会学习型”角色。
    """
    label_counts = defaultdict(int)

    for key, res in social_results.items():
        obs_time = res.get("obs_total_time", 0.0)
        collab_time = res.get("collab_total_time", 0.0)
        obs_count = res.get("obs_count", 0)
        collab_count = res.get("collab_count", 0)
        S_norm = res.get("social_index_normalized", 0.0)

        total_time = obs_time + collab_time
        total_events = obs_count + collab_count

        if total_time < MIN_SOCIAL_TIME or total_events <= 0:
            label = "low_social_participation"
            cluster_rank = 0
        else:
            ratio_obs = ratio_safe(obs_time, total_time)

            if ratio_obs >= 0.8 and collab_time <= total_time * 0.2:
                label = "observer_dominant"
                cluster_rank = 1
            elif ratio_obs <= 0.3 and collab_time >= total_time * 0.5:
                label = "collab_dominant"
                cluster_rank = 2
            else:
                # 处于中间区域，进一步看整体社会性水平高低
                if S_norm >= 0.5:
                    label = "balanced_active_social"
                    cluster_rank = 3
                else:
                    # 社会性总体不高但比例居中，视为低社交参与的稍强版本
                    label = "low_social_participation"
                    cluster_rank = 0

        res["social_label"] = label
        res["cluster_rank"] = cluster_rank
        label_counts[label] += 1

    print("=========================================================")
    print("【社会性学习与同伴取向维度：课程内角色标签分布】")
    total_records = len(social_results)
    print(f"- 参与分类的 (学习者, 课程) 总数：{total_records}")
    for label in ["low_social_participation", "observer_dominant",
                  "collab_dominant", "balanced_active_social"]:
        cnt = label_counts.get(label, 0)
        pct = ratio_safe(cnt, total_records) * 100.0
        print(f"  * {label}: {cnt} ({pct:.1f}%)")
    print("=========================================================")

    # ---------- 8. 与 LearnerProfile 中的社会性学习人设做全局对比 ----------
    """
    对比逻辑说明：
    --------------------------------------------------
    - 与任务效率维度的做法保持一致：
        * 对每个学习者，把其在所有课程上的 social_index_normalized 做平均，
          得到行为侧 global_social_learning_index；
        * 与 LearnerProfile.global_profile.social_learning.score
          做皮尔逊相关，粗略验证：
            “基于细粒度 xAPI 行为计算的社会性学习指数”
            是否与预设的人设方向一致。
    - 注意：LearnerProfile 中的分数只用于对比验证，不参与
      本脚本中行为指数的计算，符合你对“人设不可直接参与行为分析计算”的要求。
    """
    learner_global_social = defaultdict(list)
    for (lrn_uid, crs_uid), res in social_results.items():
        S_norm = res.get("social_index_normalized")
        if S_norm is None:
            continue
        learner_global_social[lrn_uid].append(float(S_norm))

    # 计算每个学习者的平均行为社会性指数
    learner_global_social_avg = {}
    for lrn_uid, vals in learner_global_social.items():
        if not vals:
            continue
        learner_global_social_avg[lrn_uid] = sum(vals) / float(len(vals))

    xs = []  # persona_social_learning.score
    ys = []  # 行为侧 global_social_learning_index

    for lrn_uid, persona_score in persona_scores.items():
        analyzed_val = learner_global_social_avg.get(lrn_uid)
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

        avg_global_soc = sum(ys) / float(len(ys))
        avg_persona_score = sum(xs) / float(len(xs))

        print("=========================================================")
        print("【社会性学习与同伴取向维度：人设 vs 行为分析 全局对比】")
        print(f"- 采样学习者数量（具备人设）：{len(sampled_learners)}")
        print(f"- 实际参与对比的学习者数量：{len(xs)}")
        print(f"- 行为分析 global_social_learning_index 平均值：{avg_global_soc:.3f}")
        print(f"- 人设 social_learning.score 平均值：{avg_persona_score:.3f}")
        print(f"- 皮尔逊相关系数：{corr:.3f}")
        print("  （相关系数用于粗略验证：细粒度 xAPI 社会性学习分析是否与人设维度方向一致。）")
        print("=========================================================")
    else:
        print("参与社会性维度人设对比的学习者样本太少，无法计算相关系数。")

    print("社会性学习与同伴取向维度分析完成。")

    # 如需将结果写回数据库，可在此手动解除注释：
    # save_social_learning_to_db(db, social_results)


if __name__ == "__main__":
    main()
