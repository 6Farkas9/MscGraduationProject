# BackEnd/app/engine/collaborative_role_contribution_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from math import sqrt
import re

from app.repositories.collaborative_role_contribution_repository import (
    collaborative_role_contribution_repository,
)

logger = logging.getLogger(__name__)

# 与分析脚本保持一致的动词常量（从 repository 复用）
VERBS = collaborative_role_contribution_repository.VERBS

# 解析 "PT120S" 这样的持续时间
DURATION_RE = re.compile(r"^PT(\d+)S$")


def parse_iso8601_duration(duration_str: Optional[str]) -> Optional[int]:
    """解析 'PT120S' -> 120, 若格式不符返回 None"""
    if not duration_str:
        return None
    m = DURATION_RE.match(duration_str)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def safe_div(a: float, b: float) -> float:
    """安全除法"""
    if b <= 1e-9:
        return 0.0
    return float(a) / float(b)


def get_session_id(doc: Dict[str, Any]) -> str:
    """
    取得协作会话标识（与脚本一致）：
    - 优先 context.extensions.sessionId / session-id
    - 其次 context.registration
    - 否则用 timestamp 的日期部分做近似
    """
    ctx = doc.get("context") or {}
    exts = ctx.get("extensions") or {}
    sid = (
        exts.get("https://legend-meta.com/xapi/ext/sessionId")
        or exts.get("https://legend-meta.com/xapi/ext/session-id")
        or exts.get("sessionId")
    )
    if sid:
        return str(sid)

    reg = ctx.get("registration")
    if reg:
        return str(reg)

    ts = doc.get("timestamp")
    if ts:
        return str(ts)[:10]

    return "unknown-session"


def extract_collaborators(doc: Dict[str, Any]) -> List[str]:
    """
    提取协作伙伴（用于网络指标）：
    - context.extensions.participants / collaborator-ids / collaborator_ids
    """
    ctx = doc.get("context") or {}
    exts = ctx.get("extensions") or {}
    ids = (
        exts.get("https://legend-meta.com/xapi/ext/participants")
        or exts.get("participants")
        or exts.get("collaborator-ids")
        or exts.get("collaborator_ids")
    )

    if isinstance(ids, list):
        return [str(x) for x in ids if x]
    return []


def build_undirected_graph(edges: List[Tuple[str, str]]) -> Dict[str, set]:
    """edges: [(u,v)] -> graph[u] = set(vs)"""
    g: Dict[str, set] = defaultdict(set)
    for u, v in edges:
        if not u or not v or u == v:
            continue
        g[u].add(v)
        g[v].add(u)
    return g


def degree_centrality(graph: Dict[str, set]) -> Dict[str, float]:
    """度中心度 = degree / (n-1)"""
    n = len(graph)
    if n <= 1:
        return {u: 0.0 for u in graph}
    return {u: len(vs) / float(n - 1) for u, vs in graph.items()}


def betweenness_centrality(graph: Dict[str, set]) -> Dict[str, float]:
    """
    简易 Brandes 算法版本的介数中心度（适合中小规模协作会话）
    与原脚本保持一致，用于识别“协调者/枢纽型”角色。
    """
    nodes = list(graph.keys())
    bc = {v: 0.0 for v in nodes}

    for s in nodes:
        stack: List[str] = []
        pred: Dict[str, List[str]] = {w: [] for w in nodes}
        sigma: Dict[str, float] = dict.fromkeys(nodes, 0.0)
        sigma[s] = 1.0
        dist: Dict[str, int] = dict.fromkeys(nodes, -1)
        dist[s] = 0
        queue: List[str] = [s]

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


class CollaborativeRoleContributionEngine:
    """
    协作角色 & 贡献类型分析引擎。

    流程：
    1. 从 Repository 拉协作相关事件（Interaction 集合）。
    2. 按 (session_id, course_uid) 聚合为协作会话。
    3. 会话内计算每个学习者：
       - create/update/delete 编辑次数
       - 资源提交数
       - 协作次数 + 协作时长
       - 社会互动次数
       - 网络度中心度 / 介数中心度（若能建图）
    4. 会话内归一化为份额，并判定：
       - role_label：核心贡献者 / 协调者 / 执行者 / 观察者 / 一般协作者
       - contribution_label：内容创作型 / 修改完善型 / 资源提供型 / 讨论参与型 / 无有效贡献
    5. 按 (学习者, 课程) 聚合，得到课程级结果。
    6. 再按学习者聚合：
       - overall_contribution_share：所有课程 avg_share_contribution 的均值
       - overall_role_label：课程级角色标签众数并偏向“好”的那类
       - overall_contribution_label：课程级贡献类型众数并偏向“好”的那类
    """

    # 角色“好坏顺序”（用于多课程并列时往“好”里偏）
    ROLE_PRIORITY = [
        "无协作数据",
        "观察者",
        "一般协作者",
        "执行者",
        "协调者",
        "核心贡献者",
    ]

    CONTRIBUTION_PRIORITY = [
        "无协作数据",
        "无有效贡献",
        "讨论参与型（讨论/响应为主，产出较少）",
        "资源提供型（以资源提交为主）",
        "修改完善型（以 update/delete 编辑为主）",
        "内容创作型（以 create 编辑为主）",
    ]

    def __init__(self) -> None:
        logger.info("CollaborativeRoleContributionEngine 初始化完成")

    # ------------------------------------------------------------------
    # 内部核心：从事件列表构造 (learner, course) 级协作画像
    # ------------------------------------------------------------------

    def _compute_course_level_results(
        self, events: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        完整复刻脚本中的：
        - 按会话聚合协作特征
        - 会话级角色+贡献类型判定
        - 课程级汇总

        返回：
        {
          (lrn_uid, crs_uid): {
              "role_label": ...,
              "contribution_label": ...,
              "metrics": {
                  "avg_share_contribution": ...,
                  "avg_share_participation": ...,
                  "avg_share_transactivity": ...,
                  "avg_degree_centrality": ...,
                  "avg_betweenness_centrality": ...,
                  "sessions_count": int,
                  "role_counts": {...},
                  "contrib_counts": {...},
              },
          },
          ...
        }
        """

        if not events:
            return {}

        # ---------- 5. 按会话聚合协作特征 ----------
        # (session_id, course_uid) -> [docs]
        session_events: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

        for doc in events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue
            sid = get_session_id(doc)
            session_events[(sid, crs_uid)].append(doc)

        logger.info(
            f"[CollaborativeRoleContributionEngine] 识别到协作会话数: {len(session_events)}"
        )

        # 会话内 per-learner 指标
        per_session_metrics: Dict[
            Tuple[str, str, str], Dict[str, float]
        ] = defaultdict(
            lambda: {
                "create_edits": 0.0,
                "update_edits": 0.0,
                "delete_edits": 0.0,
                "resources_contributed": 0.0,
                "collaborated_count": 0.0,
                "collaborated_duration": 0.0,
                "responded": 0.0,
                "referred": 0.0,
                "followed": 0.0,
                "managed_resource": 0.0,
                "took_turn": 0.0,
            }
        )

        # 网络边： (sid, crs_uid) -> [(u,v)]
        session_edges: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

        for (sid, crs_uid), docs in session_events.items():
            for doc in docs:
                lrn_uid = doc.get("_lrn_uid")
                if not lrn_uid:
                    continue
                verb_id = (doc.get("verb") or {}).get("id")
                result = doc.get("result") or {}
                m = per_session_metrics[(sid, crs_uid, lrn_uid)]

                if verb_id == VERBS["co_edited_artifact"]:
                    exts = result.get("extensions") or {}
                    etype = (
                        exts.get("edit-type")
                        or result.get("edit-type")
                        or result.get("edit_type")
                    )
                    etype = str(etype).lower() if etype else "update"

                    if etype == "create":
                        m["create_edits"] += 1
                    elif etype == "delete":
                        m["delete_edits"] += 1
                    else:
                        m["update_edits"] += 1

                elif verb_id == VERBS["contributed_resource"]:
                    m["resources_contributed"] += 1

                elif verb_id == VERBS["collaborated_on_activity"]:
                    m["collaborated_count"] += 1
                    dur = parse_iso8601_duration(result.get("duration"))
                    if dur:
                        m["collaborated_duration"] += float(dur)

                    peers = extract_collaborators(doc)
                    for p in peers:
                        session_edges[(sid, crs_uid)].append((lrn_uid, p))

                elif verb_id == VERBS["responded"]:
                    m["responded"] += 1
                elif verb_id == VERBS["referred"]:
                    m["referred"] += 1
                elif verb_id == VERBS["followed"]:
                    m["followed"] += 1
                elif verb_id == VERBS["managed_resource"]:
                    m["managed_resource"] += 1
                elif verb_id == VERBS["took_turn"]:
                    m["took_turn"] += 1

        # ---------- 6. 会话内归一化 + 角色/贡献类型判定 ----------

        per_session_role: Dict[
            Tuple[str, str, str], Dict[str, Any]
        ] = {}  # (sid, crs_uid, lrn_uid) -> ...

        for (sid, crs_uid), docs in session_events.items():
            learners_in_session = {
                d.get("_lrn_uid") for d in docs if d.get("_lrn_uid")
            }

            # 计算本会话总量
            totals = {
                "contribution": 0.0,
                "participation": 0.0,
                "transactivity": 0.0,
            }
            raw_by_learner: Dict[str, Tuple[float, float, float, Dict[str, float]]] = {}

            for lrn_uid in learners_in_session:
                m = per_session_metrics[(sid, crs_uid, lrn_uid)]
                contribution = (
                    m["create_edits"]
                    + m["update_edits"]
                    + m["delete_edits"]
                    + m["resources_contributed"]
                )
                participation = m["collaborated_count"] + safe_div(
                    m["collaborated_duration"], 60.0
                )  # 秒 → 分钟
                transactivity = (
                    m["responded"]
                    + m["referred"]
                    + m["followed"]
                    + m["managed_resource"]
                    + m["took_turn"]
                )

                raw_by_learner[lrn_uid] = (contribution, participation, transactivity, m)

                totals["contribution"] += contribution
                totals["participation"] += participation
                totals["transactivity"] += transactivity

            # 网络指标（若有边）
            edges = session_edges.get((sid, crs_uid), [])
            centrality_deg: Dict[str, float] = {}
            centrality_bet: Dict[str, float] = {}
            if edges:
                g = build_undirected_graph(edges)
                centrality_deg = degree_centrality(g)
                centrality_bet = betweenness_centrality(g)

            for lrn_uid, (
                contribution,
                participation,
                transactivity,
                m,
            ) in raw_by_learner.items():
                share_contrib = safe_div(contribution, totals["contribution"])
                share_partic = safe_div(participation, totals["participation"])
                share_trans = safe_div(transactivity, totals["transactivity"])

                deg = centrality_deg.get(lrn_uid, 0.0)
                bet = centrality_bet.get(lrn_uid, 0.0)

                # --- 会话级角色判定 ---
                if share_contrib >= 0.35 and share_partic >= 0.35:
                    role = "核心贡献者"
                elif share_trans >= 0.40 and (
                    deg >= 0.30 or bet >= 0.10 or not edges
                ):
                    role = "协调者"
                elif share_partic >= 0.40 and share_contrib < 0.30:
                    role = "执行者"
                elif share_partic < 0.20 and share_contrib < 0.20:
                    role = "观察者"
                else:
                    role = "一般协作者"

                # --- 会话级贡献类型判定 ---
                create = m["create_edits"]
                modify = m["update_edits"] + m["delete_edits"]
                resource = m["resources_contributed"]
                discuss = (
                    m["responded"]
                    + m["referred"]
                    + m["followed"]
                    + m["managed_resource"]
                    + m["took_turn"]
                )

                total_contrib_actions = create + modify + resource + discuss
                if total_contrib_actions <= 0:
                    contrib_type = "无有效贡献"
                else:
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
                        "raw": m,
                    },
                }

        # ---------- 7. 会话 -> 课程 汇总 ----------

        course_agg: Dict[
            Tuple[str, str], Dict[str, Any]
        ] = defaultdict(
            lambda: {
                "role_counts": defaultdict(int),
                "contrib_counts": defaultdict(int),
                "share_contribution": [],
                "share_participation": [],
                "share_transactivity": [],
                "degree_centrality": [],
                "betweenness_centrality": [],
                "sessions": 0,
            }
        )

        for (sid, crs_uid, lrn_uid), res in per_session_role.items():
            ca = course_agg[(lrn_uid, crs_uid)]
            ca["role_counts"][res["role_label"]] += 1
            ca["contrib_counts"][res["contribution_label"]] += 1
            ca["share_contribution"].append(res["metrics"]["share_contribution"])
            ca["share_participation"].append(res["metrics"]["share_participation"])
            ca["share_transactivity"].append(res["metrics"]["share_transactivity"])
            ca["degree_centrality"].append(res["metrics"]["degree_centrality"])
            ca["betweenness_centrality"].append(
                res["metrics"]["betweenness_centrality"]
            )
            ca["sessions"] += 1

        collaboration_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), ca in course_agg.items():
            role_label = (
                max(ca["role_counts"].items(), key=lambda x: x[1])[0]
                if ca["role_counts"]
                else "无协作数据"
            )
            contrib_label = (
                max(ca["contrib_counts"].items(), key=lambda x: x[1])[0]
                if ca["contrib_counts"]
                else "无协作数据"
            )

            collaboration_results[(lrn_uid, crs_uid)] = {
                "role_label": role_label,
                "contribution_label": contrib_label,
                "metrics": {
                    "avg_share_contribution": safe_div(
                        sum(ca["share_contribution"]),
                        len(ca["share_contribution"]),
                    ),
                    "avg_share_participation": safe_div(
                        sum(ca["share_participation"]),
                        len(ca["share_participation"]),
                    ),
                    "avg_share_transactivity": safe_div(
                        sum(ca["share_transactivity"]),
                        len(ca["share_transactivity"]),
                    ),
                    "avg_degree_centrality": safe_div(
                        sum(ca["degree_centrality"]),
                        len(ca["degree_centrality"]),
                    ),
                    "avg_betweenness_centrality": safe_div(
                        sum(ca["betweenness_centrality"]),
                        len(ca["betweenness_centrality"]),
                    ),
                    "sessions_count": ca["sessions"],
                    "role_counts": dict(ca["role_counts"]),
                    "contrib_counts": dict(ca["contrib_counts"]),
                },
            }

        logger.info(
            f"[CollaborativeRoleContributionEngine] 课程级协作画像条目数: "
            f"{len(collaboration_results)}"
        )
        return collaboration_results

    # ------------------------------------------------------------------
    # 学习者级聚合
    # ------------------------------------------------------------------

    def _pick_best_label(
        self, counts: Dict[str, int], priority: List[str]
    ) -> Optional[str]:
        if not counts:
            return None
        max_cnt = max(counts.values())
        candidates = [lab for lab, c in counts.items() if c == max_cnt]
        # 在优先级列表中索引越大越“好”
        candidates = sorted(
            candidates,
            key=lambda lab: priority.index(lab) if lab in priority else -1,
        )
        return candidates[-1] if candidates else None

    def _build_learner_summaries(
        self, course_results: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (learner, course) 级结果聚合为按学习者的结果。
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), r in course_results.items():
            if lrn_uid not in learner_data:
                learner_data[lrn_uid] = {
                    "learner_uid": lrn_uid,
                    "has_data": True,
                    "overall_contribution_share": None,
                    "overall_role_label": None,
                    "overall_contribution_label": None,
                    "per_course_results": [],
                }

            m = r["metrics"]
            learner_data[lrn_uid]["per_course_results"].append(
                {
                    "course_uid": crs_uid,
                    "role_label": r["role_label"],
                    "contribution_label": r["contribution_label"],
                    "avg_share_contribution": m["avg_share_contribution"],
                    "avg_share_participation": m["avg_share_participation"],
                    "avg_share_transactivity": m["avg_share_transactivity"],
                    "avg_degree_centrality": m["avg_degree_centrality"],
                    "avg_betweenness_centrality": m["avg_betweenness_centrality"],
                    "sessions_count": m["sessions_count"],
                    "role_counts": m["role_counts"],
                    "contrib_counts": m["contrib_counts"],
                }
            )

        # 计算 overall 数值 + 标签
        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            # 数值：多课程 avg_share_contribution 的均值
            contrib_shares = [
                it["avg_share_contribution"] for it in pcs
            ]
            info["overall_contribution_share"] = (
                sum(contrib_shares) / float(len(contrib_shares))
                if contrib_shares
                else 0.0
            )

            # 角色标签众数（并列时往更“好”的角色偏）
            role_counts: Dict[str, int] = defaultdict(int)
            contrib_counts: Dict[str, int] = defaultdict(int)
            for it in pcs:
                role_counts[it["role_label"]] += 1
                contrib_counts[it["contribution_label"]] += 1

            info["overall_role_label"] = self._pick_best_label(
                role_counts, self.ROLE_PRIORITY
            )
            info["overall_contribution_label"] = self._pick_best_label(
                contrib_counts, self.CONTRIBUTION_PRIORITY
            )

        return learner_data

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def analyze_multiple_learners(
        self, learner_uids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        多学习者协作角色 & 贡献类型分析。

        返回：
        {
          learner_uid: {
            "learner_uid": "...",
            "has_data": bool,
            "overall_contribution_share": float 或 None,
            "overall_role_label": str 或 None,
            "overall_contribution_label": str 或 None,
            "per_course_results": [
              {
                "course_uid": "...",
                "role_label": "...",
                "contribution_label": "...",
                "avg_share_contribution": ...,
                "avg_share_participation": ...,
                "avg_share_transactivity": ...,
                "avg_degree_centrality": ...,
                "avg_betweenness_centrality": ...,
                "sessions_count": int,
                "role_counts": {...},
                "contrib_counts": {...}
              },
              ...
            ]
          },
          ...
        }
        """
        if not learner_uids:
            return {}

        try:
            events = collaborative_role_contribution_repository.get_collaboration_events_for_learners(
                learner_uids
            )
            if not events:
                # 所有人都没有协作数据
                return {
                    uid: {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_contribution_share": None,
                        "overall_role_label": None,
                        "overall_contribution_label": None,
                        "per_course_results": [],
                    }
                    for uid in learner_uids
                }

            course_results = self._compute_course_level_results(events)
            learner_summaries = self._build_learner_summaries(course_results)

            # 确保每个传入的 UID 都有一条记录
            for uid in learner_uids:
                if uid not in learner_summaries:
                    learner_summaries[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_contribution_share": None,
                        "overall_role_label": None,
                        "overall_contribution_label": None,
                        "per_course_results": [],
                    }

            return learner_summaries

        except Exception as e:
            logger.error(f"多学习者协作角色分析失败: {e}", exc_info=True)
            # 出错时也保证结构一致
            return {
                uid: {
                    "learner_uid": uid,
                    "has_data": False,
                    "overall_contribution_share": None,
                    "overall_role_label": None,
                    "overall_contribution_label": None,
                    "per_course_results": [],
                    "error": str(e),
                }
                for uid in learner_uids
            }

    def analyze_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """
        单学习者便捷接口：
        返回结构同 analyze_multiple_learners()[learner_uid]
        """
        results = self.analyze_multiple_learners([learner_uid])
        return results.get(
            learner_uid,
            {
                "learner_uid": learner_uid,
                "has_data": False,
                "overall_contribution_share": None,
                "overall_role_label": None,
                "overall_contribution_label": None,
                "per_course_results": [],
            },
        )


# 全局实例 + 便捷函数（与 hgc_engine / attention_allocation_engine 一致）
_engine_instance: Optional[CollaborativeRoleContributionEngine] = None


def get_collaborative_role_contribution_engine() -> CollaborativeRoleContributionEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CollaborativeRoleContributionEngine()
    return _engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_collaborative_role_contribution_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(
    learner_uids: List[str],
) -> Dict[str, Dict[str, Any]]:
    engine = get_collaborative_role_contribution_engine()
    return engine.analyze_multiple_learners(learner_uids)


if __name__ == "__main__":
    # 与你现在的 attention_allocation_engine main 一致，使用相同测试 UID
    logging.basicConfig(level=logging.INFO)

    engine = CollaborativeRoleContributionEngine()

    # 真实存在的学习者UID
    test_learner_uids = [
        "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
        "lrn_004a9c3f5bf246faab3d390ce716e658",
    ]

    print("=== 单学习者测试 ===")
    res_single = engine.analyze_single_learner(test_learner_uids[0])
    print(res_single)

    print("=== 多学习者测试 ===")
    res_multi = engine.analyze_multiple_learners(test_learner_uids)
    print(res_multi)
