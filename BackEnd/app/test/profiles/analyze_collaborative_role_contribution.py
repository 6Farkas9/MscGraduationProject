# -*- coding: utf-8 -*-
"""
分析维度：协作角色与贡献类型（Collaborative Role & Contribution Type）

脚本目标（与画像设计文档对齐）：
--------------------------------------------------
对应《论文.txt / 画像设计》中的协作社会画像维度之一：
【协作角色与贡献类型（Collaborative Role & Contribution Type）】

该维度在画像框架中的界定：
- 学习者在协作任务/ cooperative 单元中扮演的角色：
  核心贡献者、执行者、协调者、观察者等；
- 学习者贡献的主要形式：
  内容创作（create）、修改（update/delete）、资源提供（file/link/note）、
  或仅参与讨论/响应。

本脚本采用的行为代理与论文依据：
--------------------------------------------------
1）协作 VR 内容创作 LAD 中的“任务板进度 + 编辑次数/类型 + 资源提交 → 协作角色/贡献类型”
   - Wang & Hu, 2025 ICALT《Student-Facing Learning Analytics Dashboard for Collaborative VR Content Creation》
     使用任务清单(Task checklist)与编辑行为可视化来促进组内觉察，并以
     “编辑次数、编辑类型(create/update/delete)、资源提交进度”
     区分学生的协作参与度与贡献差异。
   → 因此，本脚本选择 xAPI 中与上述日志等价的动词/字段：
      - collaborated-on-activity：协作参与频率/时长
      - co-edited-artifact：编辑行为与 edit-type(create/update/delete)
      - contributed-resource：资源提供次数与 resource-type(file/link/note)

2）“Contribution / Participation / Transactivity / Social Dynamics”协作质量框架
   - Wang, Ng & Hu, 2024 ICALT（并在 CSCL 2025 短文中复述）
     以四维框架评估协作：
       * Contribution(= Creation/Update)：新增与修改 artefact 的频次
       * Participation：参与协作活动的频率/持续投入
       * Transactivity / Social dynamics：与同伴互动质量（响应/互指/资源管理/话轮）
   → 本脚本中：
      - Contribution 由 create/update/delete 编辑数 + 资源提交数表示；
      - Participation 由 collaborated-on-activity 次数、协作时长表示；
      - Transactivity/Social dynamics 由 responded/referred/followed/managed-resource/took-turn 次数表示；
      这些都与该框架对日志的操作化一致。

3）社交网络结构是社会画像的重要维度
   - Lampropoulos & Evangelidis, 2025 XR+LA/EDM SLR 指出
     协作学习中应关注学习者间的社交网络结构、中心度等网络指标来刻画社会画像维度。
   → 若 xAPI 事件里存在 collaborator-ids / participants 字段，
     本脚本构建协作网络并计算 degree / betweenness 等中心度，
     用于辅助识别“协调者/枢纽型”角色。

4）“价值贡献 → 功能角色”的元宇宙画像思想
   - LEARNER-C（Hsu et al., 2023）把“价值贡献”与学习活动绑定，
     区分不同学生在元宇宙中的功能角色与价值位置。
   → 本脚本把“贡献强度(Contribution/Participation)”与“贡献形式(creation/modification/resource/discussion)”
     映射为角色标签，从而与 LEARNER-C 的“价值贡献型角色”一致。

与原文不同的地方 & 改动原因：
--------------------------------------------------
- 原 VR 创作研究中通常有显式的任务板日志、场景/对象级创建记录与明确的小组边界；
  你的模拟 xAPI 数据可能不包含完整 task-board 字段或稳定的 group-id。
- 因此本脚本：
  1) 优先使用 context.extensions 中 sessionId / participants / collaborator-ids 来划分协作会话与网络；
  2) 若缺失，则退化为“按课程内 cooperative 单元 + 时间窗口(同日)近似会话”的方式。
  这属于“在不改变论文核心算法思想前提下的可部署平替”，符合你的第 2 条要求。

脚本功能总结：
--------------------------------------------------
1. 从 MongoDB 中读取：
   - 细粒度 xAPI 行为：MLS.Interaction
   - 学习者画像：MLS.LearnerProfile（用于筛选与后验对比，不直接参与计算）

2. 对每个协作会话（优先 sessionId，否则按课程+日期近似）聚合每个学习者的协作特征：
   - Contribution：
       * create_edits / update_edits / delete_edits （co-edited-artifact + edit-type）
       * resources_contributed            （contributed-resource）
   - Participation：
       * collaborated_count              （collaborated-on-activity）
       * collaborated_duration_seconds   （collaborated-on-activity.result.duration）
   - Transactivity/Social dynamics：
       * responded / referred / followed / managed-resource / took-turn 的次数
   - Network（若有 collaborator-ids/participants）：计算度中心度与介数中心度

3. 角色分类（论文驱动的规则离散化）：
   - 先在“同一会话内”把 Contribution 与 Participation 归一化为相对份额；
   - 标签依据 Wang 2024/2025 的“贡献/参与/互动质量”框架解释：
       * 核心贡献者：贡献份额高（create+update+resource 高）且参与份额高
       * 执行者：参与份额高，但贡献份额中低（更多完成协作任务而非产出）
       * 协调者：Transactivity/Social 次数高，且网络介数/度中心度高（若可算）
       * 观察者：参与与贡献份额都低，仅少量协作出现
   - 会话级标签再汇总为课程级与全局标签。

4. 贡献类型分类：
   - 内容创作型：create_edits 占贡献的大头
   - 修改完善型：update/delete 占贡献的大头
   - 资源提供型：resources_contributed 占贡献的大头
   - 讨论参与型：讨论/响应多，但编辑/资源少

5. 结果输出：
   - 打印每个学习者的全局协作角色、贡献类型、关键指标；
   - 与 LearnerProfile.global_profile.collaboration / value_contribution 做相关性对比（验证有效性）。

6. 数据库存储接口（不在 main() 调用）：
   - save_collaboration_role_to_db(db, results)
   - 写入 MLS.CollaborationRoleAnalysis（接口保留但默认不调用）

"""

from pymongo import MongoClient
from datetime import datetime
import re
from collections import defaultdict
from math import sqrt
import random
from tqdm import tqdm

# ===================== 配置区域 =====================

MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "MLS"
XAPI_COLLECTION = "Interaction"          # 细粒度行为集合
PROFILE_COLLECTION = "LearnerProfile"    # 人设集合
COLLAB_ROLE_COLLECTION = "CollaborationRoleAnalysis"  # 协作角色分析结果集合（仅接口，不在 main 中调用）

# 采样学习者数量：随机选取至多 N_SAMPLE 个学习者进行分析
N_SAMPLE = 3000

VERB_BASE = "https://legend-meta.com/xapi/verb/"

VERBS = {
    "collaborated_on_activity": VERB_BASE + "collaborated-on-activity",
    "co_edited_artifact": VERB_BASE + "co-edited-artifact",
    "contributed_resource": VERB_BASE + "contributed-resource",

    # 社会互动动词（来自 profile 设计，用于 Transactivity / Social dynamics）:
    "responded": VERB_BASE + "responded",
    "referred": VERB_BASE + "referred",
    "followed": VERB_BASE + "followed",
    "managed_resource": VERB_BASE + "managed-resource",
    "took_turn": VERB_BASE + "took-turn",
}

# 解析简单 ISO8601 秒级 duration
DURATION_RE = re.compile(r"^PT(\d+)S$")


# ===================== 工具函数 =====================

def parse_iso8601_duration(duration_str):
    """解析 "PT120S" -> 120, 若格式不符返回 None"""
    if not duration_str:
        return None
    m = DURATION_RE.match(duration_str)
    if m:
        return int(m.group(1))
    return None


def compute_mean_std(values):
    """计算均值与总体标准差"""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_v = sum(values) / float(n)
    if n == 1:
        return mean_v, 0.0
    var = sum((v - mean_v) ** 2 for v in values) / float(n)
    return mean_v, sqrt(var)


def safe_div(a, b):
    """安全除法"""
    if b <= 1e-9:
        return 0.0
    return float(a) / float(b)


def get_session_id(doc):
    """
    取得协作会话标识：
    - 优先使用 context.extensions.sessionId（profile 中要求项）
    - 其次使用 context.registration
    - 再次使用 timestamp 的日期作为近似会话（原文任务板日志缺失时的平替）
    """
    ctx = doc.get("context") or {}
    exts = (ctx.get("extensions") or {})
    sid = exts.get("https://legend-meta.com/xapi/ext/sessionId") \
          or exts.get("https://legend-meta.com/xapi/ext/session-id") \
          or exts.get("sessionId")
    if sid:
        return str(sid)

    reg = ctx.get("registration")
    if reg:
        return str(reg)

    ts = doc.get("timestamp")
    if ts:
        # 只取日期部分作为近似会话
        return str(ts)[:10]

    return "unknown-session"


def extract_collaborators(doc):
    """
    提取协作伙伴列表（用于建网络）：
    - 优先 participants / collaborator-ids
    - 若没有，则返回空
    """
    ctx = doc.get("context") or {}
    exts = (ctx.get("extensions") or {})
    ids = exts.get("https://legend-meta.com/xapi/ext/participants") \
          or exts.get("participants") \
          or exts.get("collaborator-ids") \
          or exts.get("collaborator_ids")

    if isinstance(ids, list):
        return [str(x) for x in ids if x]
    return []


def pearson_corr(xs, ys):
    """计算皮尔逊相关（用于与人设粗验证）"""
    n = len(xs)
    if n == 0 or n != len(ys):
        return 0.0
    mean_x, std_x = compute_mean_std(xs)
    mean_y, std_y = compute_mean_std(ys)
    if std_x < 1e-9 or std_y < 1e-9:
        return 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / float(n)
    return cov / (std_x * std_y)


# ===================== 网络指标计算（无外部依赖版） =====================

def build_undirected_graph(edges):
    """edges: list[(u,v)] -> dict[u]=set(vs)"""
    g = defaultdict(set)
    for u, v in edges:
        if u == v:
            continue
        g[u].add(v)
        g[v].add(u)
    return g


def degree_centrality(graph):
    """度中心度 = degree / (n-1)"""
    n = len(graph)
    if n <= 1:
        return {u: 0.0 for u in graph}
    return {u: len(vs) / float(n - 1) for u, vs in graph.items()}


def betweenness_centrality(graph):
    """
    介数中心度（Brandes 简化实现，适用于小图/中等规模）
    用于识别“协调者/枢纽型”角色（Lampropoulos & Evangelidis 2025 的网络画像建议）。
    """
    nodes = list(graph.keys())
    bc = {v: 0.0 for v in nodes}

    for s in nodes:
        stack = []
        pred = {w: [] for w in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        sigma[s] = 1.0
        dist = dict.fromkeys(nodes, -1)
        dist[s] = 0
        queue = [s]

        # BFS
        for v in queue:
            stack.append(v)
            for w in graph[v]:
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # 回溯累积依赖
        delta = dict.fromkeys(nodes, 0.0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta_v = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    delta[v] += delta_v
            if w != s:
                bc[w] += delta[w]

    # 归一化
    n = len(nodes)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2) / 2.0)
        for v in bc:
            bc[v] *= scale
    return bc


# ===================== 数据库存储接口（不在 main 中调用） =====================

def save_collaboration_role_to_db(db, results):
    """
    把协作角色分析结果写入 MongoDB 的接口函数（默认不在 main 中调用）。

    字段设计依据：
    - learner_uid / course_uid 与 Interaction 中保持一致；
    - role_label / contribution_label 对应本脚本的离散画像输出；
    - contribution_share / participation_share / transactivity_share / centrality
      对应 Wang 2024/2025 的协作质量四维与 Lampropoulos 2025 的网络画像建议。

    注意：
    - 本函数不会在 main() 中自动调用。
    """
    col = db[COLLAB_ROLE_COLLECTION]
    db.drop_collection(COLLAB_ROLE_COLLECTION)
    col = db[COLLAB_ROLE_COLLECTION]

    docs = []
    for (lrn_uid, crs_uid), r in results.items():
        docs.append({
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "role_label": r.get("role_label"),
            "contribution_label": r.get("contribution_label"),
            "metrics": r.get("metrics"),
            "created_at": datetime.utcnow()
        })
    if docs:
        col.insert_many(docs, ordered=False)
        print(f"[接口调用] 已写入 CollaborationRoleAnalysis 文档数：{len(docs)}")
    else:
        print("[接口调用] 没有可写入 CollaborationRoleAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]

    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中的人设（用于筛选与对比，不参与计算） ----------
    print("读取 LearnerProfile 中的人设信息...")
    persona_collab_scores = {}   # lrn_uid -> collaboration.score
    persona_value_scores = {}    # lrn_uid -> value_contribution.score

    cursor_profiles = profile_col.find(
        {},
        {"learner_uid": 1, "global_profile": 1}
    )
    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g = doc.get("global_profile") or {}

        cdim = g.get("collaboration") or {}
        vdim = g.get("value_contribution") or {}

        if cdim.get("score") is not None:
            persona_collab_scores[lrn_uid] = float(cdim["score"])
        if vdim.get("score") is not None:
            persona_value_scores[lrn_uid] = float(vdim["score"])

    learners_with_persona = list(set(persona_collab_scores.keys()) | set(persona_value_scores.keys()))
    print(f"具备协作/价值贡献人设的学习者数量：{len(learners_with_persona)}")

    if not learners_with_persona:
        print("没有任何学习者具备协作相关人设，无法进行对比分析。")
        return

    # ---------- 3. 随机采样学习者 ----------
    if N_SAMPLE > 0 and N_SAMPLE < len(learners_with_persona):
        sampled_learners = random.sample(learners_with_persona, N_SAMPLE)
    else:
        sampled_learners = learners_with_persona

    print(f"随机选取用于分析的学习者数量：{len(sampled_learners)} (N_SAMPLE = {N_SAMPLE})")

    # ---------- 4. 加载协作相关事件 ----------
    """
    事件筛选策略与论文依据：
    --------------------------------------------------
    - Wang & Hu 2025 / Wang et al. 2024 的协作 VR 创作分析依赖：
        1) 协作参与日志（participation）
        2) artefact 编辑日志（creation/update/delete）
        3) 资源提交日志（resource contribution）
        4) 同伴互动日志（transactivity/social）
    - 因此我们只拉取以下 verb 的事件：
        collaborated-on-activity / co-edited-artifact / contributed-resource
        responded / referred / followed / managed-resource / took-turn
    """
    query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": {"$in": list(VERBS.values())}
    }

    total_events = xapi_col.count_documents(query)
    print(f"与采样学习者相关的协作事件总数：{total_events}")

    events = list(xapi_col.find(
        query,
        {"verb.id": 1, "result": 1, "context": 1, "_lrn_uid": 1, "_course_uid": 1, "timestamp": 1}
    ))
    print(f"已从 MongoDB 读取事件条数：{len(events)}")

    if not events:
        print("没有任何协作相关事件，无法进行分析。")
        return

    # ---------- 5. 按会话聚合协作特征 ----------
    """
    聚合粒度：
    --------------------------------------------------
    - 原 VR 创作研究以“小组协作会话/任务阶段”为粒度；
    - 本脚本优先用 sessionId/registration，否则用课程+日期近似会话。
    """
    # (session_id, course_uid) -> lists of docs
    session_events = defaultdict(list)

    for doc in events:
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue
        sid = get_session_id(doc)
        session_events[(sid, crs_uid)].append(doc)

    print(f"识别到协作会话数（session, course）：{len(session_events)}")

    # 会话内每人特征
    # (sid, crs_uid, lrn_uid) -> metrics
    per_session_metrics = defaultdict(lambda: {
        "create_edits": 0,
        "update_edits": 0,
        "delete_edits": 0,
        "resources_contributed": 0,
        "collaborated_count": 0,
        "collaborated_duration": 0,
        "responded": 0,
        "referred": 0,
        "followed": 0,
        "managed_resource": 0,
        "took_turn": 0,
    })

    # 构建网络边
    session_edges = defaultdict(list)  # (sid, crs_uid) -> [(u,v),...]

    print("开始遍历会话并聚合协作指标...")
    for (sid, crs_uid), docs in tqdm(session_events.items(), desc="聚合会话", unit="session"):
        for doc in docs:
            lrn_uid = doc["_lrn_uid"]
            verb = doc.get("verb", {}).get("id")

            m = per_session_metrics[(sid, crs_uid, lrn_uid)]
            result = doc.get("result") or {}

            if verb == VERBS["co_edited_artifact"]:
                # edit-type 可能在 result.extensions 或 result.edit-type 中
                exts = (result.get("extensions") or {})
                etype = exts.get("edit-type") or result.get("edit-type") or result.get("edit_type")
                etype = str(etype).lower() if etype else "update"

                if etype == "create":
                    m["create_edits"] += 1
                elif etype == "delete":
                    m["delete_edits"] += 1
                else:
                    m["update_edits"] += 1

            elif verb == VERBS["contributed_resource"]:
                m["resources_contributed"] += 1

            elif verb == VERBS["collaborated_on_activity"]:
                m["collaborated_count"] += 1
                dur = parse_iso8601_duration(result.get("duration"))
                if dur:
                    m["collaborated_duration"] += dur

                # 同时尝试从 context 中提取协作伙伴，建立网络边
                peers = extract_collaborators(doc)
                for p in peers:
                    session_edges[(sid, crs_uid)].append((lrn_uid, p))

            elif verb == VERBS["responded"]:
                m["responded"] += 1
            elif verb == VERBS["referred"]:
                m["referred"] += 1
            elif verb == VERBS["followed"]:
                m["followed"] += 1
            elif verb == VERBS["managed_resource"]:
                m["managed_resource"] += 1
            elif verb == VERBS["took_turn"]:
                m["took_turn"] += 1

    # ---------- 6. 会话内归一化份额 + 角色/贡献类型判定 ----------
    """
    判定方法与论文依据：
    --------------------------------------------------
    - Wang 2024/2025 的框架以会话内相对“贡献/参与/互动”来区分角色类型；
    - Lampropoulos 2025 SLR 建议结合网络中心度识别社会位置；
    - 因此本脚本在每个会话内：
        1) 计算每人 Contribution / Participation / Transactivity 的相对份额
        2) 若网络可建，则加入 centrality
        3) 按论文解释规则离散化为角色标签与贡献类型标签
    """
    per_session_role = {}  # (sid, crs_uid, lrn_uid) -> {"role_label", "contribution_label", "shares", "raw"}

    for (sid, crs_uid), docs in session_events.items():
        # 收集本会话所有人
        learners_in_session = set(d["_lrn_uid"] for d in docs if d.get("_lrn_uid"))

        # 计算本会话总量
        totals = {
            "contribution": 0.0,
            "participation": 0.0,
            "transactivity": 0.0,
        }
        raw_by_learner = {}
        for lrn_uid in learners_in_session:
            m = per_session_metrics[(sid, crs_uid, lrn_uid)]
            contribution = m["create_edits"] + m["update_edits"] + m["delete_edits"] + m["resources_contributed"]
            participation = m["collaborated_count"] + safe_div(m["collaborated_duration"], 60.0)  # 时长折算为分钟份额
            transactivity = m["responded"] + m["referred"] + m["followed"] + m["managed_resource"] + m["took_turn"]

            raw_by_learner[lrn_uid] = (contribution, participation, transactivity, m)

            totals["contribution"] += contribution
            totals["participation"] += participation
            totals["transactivity"] += transactivity

        # 构建网络并算中心度（若有边）
        edges = session_edges.get((sid, crs_uid), [])
        centrality_deg = {}
        centrality_bet = {}
        if edges:
            g = build_undirected_graph(edges)
            centrality_deg = degree_centrality(g)
            centrality_bet = betweenness_centrality(g)

        # 对每个学习者判定
        for lrn_uid, (contribution, participation, transactivity, m) in raw_by_learner.items():
            share_contrib = safe_div(contribution, totals["contribution"])
            share_partic = safe_div(participation, totals["participation"])
            share_trans = safe_div(transactivity, totals["transactivity"])

            deg = centrality_deg.get(lrn_uid, 0.0)
            bet = centrality_bet.get(lrn_uid, 0.0)

            # ------- 角色判定（会话内） -------
            # 核心贡献者：贡献份额高 + 参与份额高
            if share_contrib >= 0.35 and share_partic >= 0.35:
                role = "核心贡献者"
            # 协调者：互动份额高 + 网络中心度高（若无网络则仅靠互动份额）
            elif share_trans >= 0.40 and (deg >= 0.30 or bet >= 0.10 or not edges):
                role = "协调者"
            # 执行者：参与高但贡献中低
            elif share_partic >= 0.40 and share_contrib < 0.30:
                role = "执行者"
            # 观察者：参与与贡献都低
            elif share_partic < 0.20 and share_contrib < 0.20:
                role = "观察者"
            else:
                role = "一般协作者"

            # ------- 贡献类型判定（会话内） -------
            create = m["create_edits"]
            modify = m["update_edits"] + m["delete_edits"]
            resource = m["resources_contributed"]
            discuss = m["responded"] + m["referred"] + m["followed"] + m["managed_resource"] + m["took_turn"]

            total_contrib_actions = create + modify + resource + discuss
            if total_contrib_actions <= 0:
                contrib_type = "无有效贡献"
            else:
                # 贡献大头所属类别
                if create >= max(modify, resource, discuss):
                    contrib_type = "内容创作型（以 create 编辑为主）"
                elif modify >= max(create, resource, discuss):
                    contrib_type = "修改完善型（以 update/delete 编辑为主）"
                elif resource >= max(create, modify, discuss):
                    contrib_type = "资源提供型（以资源提交为主）"
                else:
                    contrib_type = "讨论参与型（讨论/响应为主，产出较少）"

            per_session_role[(sid, crs_uid, lrn_uid)] = {
                "role_label": role,
                "contribution_label": contrib_type,
                "metrics": {
                    "share_contribution": share_contrib,
                    "share_participation": share_partic,
                    "share_transactivity": share_trans,
                    "degree_centrality": deg,
                    "betweenness_centrality": bet,
                    "raw": m
                }
            }

    print(f"完成会话内角色/贡献类型判定，条目数：{len(per_session_role)}")

    # ---------- 7. 会话 -> 课程 -> 全局汇总 ----------
    # (lrn_uid, crs_uid) -> 累积
    course_agg = defaultdict(lambda: {
        "role_counts": defaultdict(int),
        "contrib_counts": defaultdict(int),
        "share_contribution": [],
        "share_participation": [],
        "share_transactivity": [],
        "degree_centrality": [],
        "betweenness_centrality": [],
        "sessions": 0
    })

    for (sid, crs_uid, lrn_uid), res in per_session_role.items():
        ca = course_agg[(lrn_uid, crs_uid)]
        ca["role_counts"][res["role_label"]] += 1
        ca["contrib_counts"][res["contribution_label"]] += 1
        ca["share_contribution"].append(res["metrics"]["share_contribution"])
        ca["share_participation"].append(res["metrics"]["share_participation"])
        ca["share_transactivity"].append(res["metrics"]["share_transactivity"])
        ca["degree_centrality"].append(res["metrics"]["degree_centrality"])
        ca["betweenness_centrality"].append(res["metrics"]["betweenness_centrality"])
        ca["sessions"] += 1

    # 课程级主标签 = 出现最多的会话标签
    collaboration_results = {}  # (lrn_uid, crs_uid) -> result
    for (lrn_uid, crs_uid), ca in course_agg.items():
        role_label = max(ca["role_counts"].items(), key=lambda x: x[1])[0] if ca["role_counts"] else "无协作数据"
        contrib_label = max(ca["contrib_counts"].items(), key=lambda x: x[1])[0] if ca["contrib_counts"] else "无协作数据"

        collaboration_results[(lrn_uid, crs_uid)] = {
            "role_label": role_label,
            "contribution_label": contrib_label,
            "metrics": {
                "avg_share_contribution": safe_div(sum(ca["share_contribution"]), len(ca["share_contribution"])),
                "avg_share_participation": safe_div(sum(ca["share_participation"]), len(ca["share_participation"])),
                "avg_share_transactivity": safe_div(sum(ca["share_transactivity"]), len(ca["share_transactivity"])),
                "avg_degree_centrality": safe_div(sum(ca["degree_centrality"]), len(ca["degree_centrality"])),
                "avg_betweenness_centrality": safe_div(sum(ca["betweenness_centrality"]), len(ca["betweenness_centrality"])),
                "sessions_count": ca["sessions"],
                "role_counts": dict(ca["role_counts"]),
                "contrib_counts": dict(ca["contrib_counts"])
            }
        }

    print(f"课程级协作画像条目数（学习者-课程对）：{len(collaboration_results)}")

    # ---------- 8. 与人设对比（粗验证） ----------
    """
    验证思路：
    --------------------------------------------------
    - 与 analyze_task_efficiency.py 一致：
      用行为侧的连续指标与人设 score 做皮尔逊相关，检查一致性。
    - 行为侧这里使用 avg_share_contribution 作为“协作贡献强度”代理，
      与人设 collaboration.score / value_contribution.score 对比。
    """
    behavior_scores = []
    persona_c_scores = []
    persona_v_scores = []

    for (lrn_uid, crs_uid), r in collaboration_results.items():
        b = r["metrics"]["avg_share_contribution"]
        behavior_scores.append(b)

        if lrn_uid in persona_collab_scores:
            persona_c_scores.append(persona_collab_scores[lrn_uid])
        else:
            persona_c_scores.append(0.0)

        if lrn_uid in persona_value_scores:
            persona_v_scores.append(persona_value_scores[lrn_uid])
        else:
            persona_v_scores.append(0.0)

    corr_c = pearson_corr(behavior_scores, persona_c_scores)
    corr_v = pearson_corr(behavior_scores, persona_v_scores)

    print("\n========== 协作角色与贡献类型分析结果（课程级） ==========")
    # 只展示前若干条示例，避免终端爆炸；你可按需改大
    show_n = 30
    i = 0
    for (lrn_uid, crs_uid), r in collaboration_results.items():
        print(f"\n[学习者] {lrn_uid}  [课程] {crs_uid}")
        print(f"  - 课程级协作角色：{r['role_label']}")
        print(f"  - 课程级贡献类型：{r['contribution_label']}")
        print("  - 关键指标：")
        for k, v in r["metrics"].items():
            if k in ("role_counts", "contrib_counts"):
                print(f"      * {k}: {v}")
            else:
                print(f"      * {k}: {v:.4f}" if isinstance(v, float) else f"      * {k}: {v}")

        i += 1
        if i >= show_n:
            break

    print("\n========== 与人设的粗一致性验证 ==========")
    print(f"行为侧 avg_share_contribution 与人设 collaboration.score 的相关：{corr_c:.4f}")
    print(f"行为侧 avg_share_contribution 与人设 value_contribution.score 的相关：{corr_v:.4f}")

    print("\n（如需写回数据库，可在 main() 末尾手动调用：")
    print("   save_collaboration_role_to_db(db, collaboration_results) ）")


if __name__ == "__main__":
    main()
