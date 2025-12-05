# BackEnd/app/engine/social_learning_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from math import sqrt
from collections import defaultdict
import re

from app.repositories.social_learning_repository import (
    social_learning_repository,
    SocialLearningRepository,
)

logger = logging.getLogger(__name__)

# 预编译 duration 的正则，解析 "PT{n}S"
DURATION_RE = re.compile(r"^PT(\d+)S$")

# 判定“低社交参与”的总时长阈值（秒）
MIN_SOCIAL_TIME = 30.0

# 观摩对象 learner-id 扩展字段
EXT_OBSERVED_LEARNER_ID = (
    "https://legend-meta.com/xapi/ext/observed-learner-id"
)


class SocialLearningEngine:
    """
    社会性学习与同伴取向分析引擎

    功能：
    - 给定一个或多个学习者 UID，从 Repository 读取 xAPI 行为；
    - 以 (学习者, 课程) 为单位计算：
        * obs_count              观摩事件次数（observed-peer）
        * obs_total_time         观摩总时长（秒）
        * obs_unique_peers       被观摩同伴的数量（去重）
        * collab_count           协作事件次数（collaborated-on-activity）
        * collab_total_time      协作总时长（秒）
        * z_obs, z_collab        课程内标准化后的观摩/协作 z 值
        * social_index           合成社会性学习指数 S
        * social_index_normalized课程内 min-max 归一化 [0,1]
        * social_label           行为角色标签：
                                   - low_social_participation
                                   - observer_dominant
                                   - collab_dominant
                                   - balanced_active_social
        * cluster_rank           0~3：从低社交参与到积极社会学习
    - 对单个学习者的多门课程结果做聚合：
        * overall_score          多课程 social_index_normalized 的均值
        * overall_cluster_rank   多课程 cluster_rank 的众数（并列取“更好”的一档）
        * overall_label          综合中文描述标签
    """

    def __init__(self) -> None:
        logger.info("SocialLearningEngine 初始化完成")

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_iso8601_duration(duration_str: Any) -> Optional[float]:
        """
        解析简单形式的 ISO8601 时长字符串，例如："PT120S"
        若为空或格式不符，返回 None。
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

    @staticmethod
    def _compute_mean_std(values: List[float]) -> Tuple[float, float]:
        """
        计算一组数的平均值与总体标准差：
        - 若列表为空，返回 (0.0, 0.0)
        - 若只有一个样本，标准差视为 0.0
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

    @staticmethod
    def _ratio_safe(num: float, den: float) -> float:
        """
        安全计算比例，避免除零。
        """
        if not den:
            return 0.0
        return float(num) / float(den)

    # ------------------------------------------------------------------
    # 第一步：从事件构建 (学习者, 课程) 粗粒度统计
    # ------------------------------------------------------------------
    def _build_social_stats(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        对每个 (lrn_uid, crs_uid) 统计：
        - obs_count：observed-peer 事件次数
        - obs_total_time：观摩总时长（秒）
        - obs_peers：被观摩同伴集合
        - collab_count：协作事件次数
        - collab_total_time：协作总时长（秒）
        """
        if not events:
            return {}

        verb_dict = SocialLearningRepository.VERBS

        social_stats: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "obs_count": 0,
                "obs_total_time": 0.0,
                "obs_peers": set(),
                "collab_count": 0,
                "collab_total_time": 0.0,
            }
        )

        used_events = 0

        for doc in events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            verb_id = (doc.get("verb") or {}).get("id") or doc.get("verb.id")
            result = doc.get("result") or {}
            duration_str = result.get("duration")
            duration_sec = self._parse_iso8601_duration(duration_str)

            if duration_sec is not None and duration_sec < 0:
                continue

            key = (lrn_uid, crs_uid)
            stat = social_stats[key]

            if verb_id == verb_dict["observed_peer"]:
                stat["obs_count"] += 1
                if duration_sec is not None:
                    stat["obs_total_time"] += float(duration_sec)

                context = doc.get("context") or {}
                ext = context.get("extensions") or {}
                peer_id = ext.get(EXT_OBSERVED_LEARNER_ID)
                if peer_id:
                    stat["obs_peers"].add(peer_id)

                used_events += 1

            elif verb_id == verb_dict["collaborated_on_activity"]:
                stat["collab_count"] += 1
                if duration_sec is not None:
                    stat["collab_total_time"] += float(duration_sec)
                used_events += 1

        logger.info(
            "[SocialLearningEngine] 参与社会性学习统计的有效事件数: %d, "
            "有社会性数据的 (学习者, 课程) 数量: %d",
            used_events,
            len(social_stats),
        )

        return social_stats

    # ------------------------------------------------------------------
    # 第二步：按课程计算社会性学习指数（标准化 + 归一化）
    # ------------------------------------------------------------------
    def _compute_course_social_indices(
        self,
        social_stats: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在每门课程内部，对 obs_total_time 与 collab_total_time 标准化：
        - z_obs, z_collab
        - social_index = (z_obs + z_collab) / sqrt(2)
        然后做 min-max 归一化得到 social_index_normalized ∈ [0,1]。
        """
        if not social_stats:
            return {}

        # course_uid -> list[(lrn_uid, obs_total, collab_total, obs_count, collab_count, obs_peers_count)]
        course_to_entries: Dict[str, List[Tuple[str, float, float, int, int, int]]] = defaultdict(
            list
        )
        for (lrn_uid, crs_uid), stat in social_stats.items():
            obs_total = float(stat.get("obs_total_time", 0.0))
            collab_total = float(stat.get("collab_total_time", 0.0))
            obs_count = int(stat.get("obs_count", 0))
            collab_count = int(stat.get("collab_count", 0))
            obs_peers_count = len(stat.get("obs_peers") or set())
            course_to_entries[crs_uid].append(
                (lrn_uid, obs_total, collab_total, obs_count, collab_count, obs_peers_count)
            )

        social_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_to_entries.items():
            if not entries:
                continue

            obs_vals = [e[1] for e in entries]
            collab_vals = [e[2] for e in entries]
            mean_obs, std_obs = self._compute_mean_std(obs_vals)
            mean_collab, std_collab = self._compute_mean_std(collab_vals)

            S_vals: List[float] = []

            # 先计算 z 与 S
            for (
                lrn_uid,
                obs_total,
                collab_total,
                obs_count,
                collab_count,
                obs_peers_count,
            ) in entries:
                z_obs = (
                    (obs_total - mean_obs) / std_obs
                    if std_obs > 1e-6
                    else 0.0
                )
                z_collab = (
                    (collab_total - mean_collab) / std_collab
                    if std_collab > 1e-6
                    else 0.0
                )

                S = (z_obs + z_collab) / sqrt(2.0)

                key = (lrn_uid, crs_uid)
                social_results[key] = {
                    "obs_count": obs_count,
                    "obs_total_time": obs_total,
                    "obs_unique_peers": obs_peers_count,
                    "collab_count": collab_count,
                    "collab_total_time": collab_total,
                    "z_obs": float(z_obs),
                    "z_collab": float(z_collab),
                    "social_index": float(S),
                    # social_index_normalized 后面再填
                }
                S_vals.append(S)

            # 再在课程内对 S 做 [0,1] min-max 归一化
            if S_vals:
                S_min = min(S_vals)
                S_max = max(S_vals)
                span = S_max - S_min if S_max > S_min else 0.0

                for (
                    lrn_uid,
                    obs_total,
                    collab_total,
                    obs_count,
                    collab_count,
                    obs_peers_count,
                ) in entries:
                    key = (lrn_uid, crs_uid)
                    S = social_results[key]["social_index"]
                    if span > 1e-6:
                        S_norm = (S - S_min) / span
                    else:
                        S_norm = 0.5
                    social_results[key]["social_index_normalized"] = float(S_norm)

        logger.info(
            "[SocialLearningEngine] 完成 social_index_normalized 计算，条目数: %d",
            len(social_results),
        )

        return social_results

    # ------------------------------------------------------------------
    # 第三步：基于观摩/协作比例进行角色分类
    # ------------------------------------------------------------------
    def _assign_social_labels(
        self,
        social_results: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> None:
        """
        使用 total_time + ratio_obs + social_index_normalized 进行角色划分：

        - low_social_participation（cluster_rank=0）
        - observer_dominant（cluster_rank=1）
        - collab_dominant（cluster_rank=2）
        - balanced_active_social（cluster_rank=3）
        """
        if not social_results:
            return

        label_counts: Dict[str, int] = defaultdict(int)

        for key, res in social_results.items():
            obs_time = float(res.get("obs_total_time", 0.0))
            collab_time = float(res.get("collab_total_time", 0.0))
            obs_count = int(res.get("obs_count", 0))
            collab_count = int(res.get("collab_count", 0))
            S_norm = float(res.get("social_index_normalized", 0.0))

            total_time = obs_time + collab_time
            total_events = obs_count + collab_count

            if total_time < MIN_SOCIAL_TIME or total_events <= 0:
                label = "low_social_participation"
                cluster_rank = 0
            else:
                ratio_obs = self._ratio_safe(obs_time, total_time)

                if ratio_obs >= 0.8 and collab_time <= total_time * 0.2:
                    label = "observer_dominant"
                    cluster_rank = 1
                elif ratio_obs <= 0.3 and collab_time >= total_time * 0.5:
                    label = "collab_dominant"
                    cluster_rank = 2
                else:
                    if S_norm >= 0.5:
                        label = "balanced_active_social"
                        cluster_rank = 3
                    else:
                        label = "low_social_participation"
                        cluster_rank = 0

            res["social_label"] = label
            res["cluster_rank"] = int(cluster_rank)
            label_counts[label] += 1

        # 简单日志输出角色分布
        total_records = len(social_results)
        logger.info(
            "[SocialLearningEngine] 角色分类完成，总记录数: %d", total_records
        )
        for label in [
            "low_social_participation",
            "observer_dominant",
            "collab_dominant",
            "balanced_active_social",
        ]:
            cnt = label_counts.get(label, 0)
            pct = self._ratio_safe(cnt, total_records) * 100.0
            logger.info(
                "  - %s: %d (%.1f%%)", label, cnt, pct
            )

    # ------------------------------------------------------------------
    # 第四步：聚合到学习者级别
    # ------------------------------------------------------------------
    def _build_learner_summaries(
        self,
        social_results: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (学习者, 课程) 级别结果聚合为按学习者的结果。

        返回结构：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,          # 多课程 social_index_normalized 均值
                "overall_label": str 或 None,            # 综合标签
                "overall_cluster_rank": int 或 None,     # 0~3
                "per_course_results": [...],             # 每门课程详情
            },
            ...
        }
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), res in social_results.items():
            S_norm = res.get("social_index_normalized")
            if S_norm is None:
                continue

            if lrn_uid not in learner_data:
                learner_data[lrn_uid] = {
                    "learner_uid": lrn_uid,
                    "has_data": True,
                    "overall_score": None,
                    "overall_label": None,
                    "overall_cluster_rank": None,
                    "per_course_results": [],
                }

            item = {
                "course_uid": crs_uid,
                "obs_count": int(res.get("obs_count", 0)),
                "obs_total_time": float(res.get("obs_total_time", 0.0)),
                "obs_unique_peers": int(res.get("obs_unique_peers", 0)),
                "collab_count": int(res.get("collab_count", 0)),
                "collab_total_time": float(res.get("collab_total_time", 0.0)),
                "z_obs": float(res.get("z_obs", 0.0)),
                "z_collab": float(res.get("z_collab", 0.0)),
                "social_index": float(res.get("social_index", 0.0)),
                "social_index_normalized": float(
                    res.get("social_index_normalized", 0.0)
                ),
                "social_label": res.get("social_label"),
                "cluster_rank": int(res.get("cluster_rank", 0)),
            }
            learner_data[lrn_uid]["per_course_results"].append(item)

        overall_rank_label = {
            0: "整体社交参与水平偏低（观摩与协作行为都较少，多数课程中几乎不通过同伴进行学习）。",
            1: "整体以观察他人为主（更多通过观摩同伴作品或表现来学习，协作参与相对较少）。",
            2: "整体协作导向（在协作单元中参与较多联合编辑与操作，相对较少停留在单纯观摩）。",
            3: "整体为积极社会学习型（在观摩同伴与参与协作两方面都较活跃，兼具观察与贡献）。",
        }

        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            scores = [it["social_index_normalized"] for it in pcs]
            info["overall_score"] = float(sum(scores) / float(len(scores)))

            rank_counts: Dict[int, int] = {}
            for it in pcs:
                rnk = int(it.get("cluster_rank", 0))
                rank_counts[rnk] = rank_counts.get(rnk, 0) + 1

            if rank_counts:
                max_count = max(rank_counts.values())
                candidate_ranks = [
                    r for r, c in rank_counts.items() if c == max_count
                ]
                best_rank = max(candidate_ranks)  # 并列时选择“更好”的一档

                info["overall_cluster_rank"] = int(best_rank)
                info["overall_label"] = overall_rank_label.get(
                    best_rank,
                    "整体社交参与水平中等（默认）。",
                )
            else:
                info["overall_cluster_rank"] = None
                info["overall_label"] = None

        return learner_data

    # ------------------------------------------------------------------
    # 对外公开接口
    # ------------------------------------------------------------------
    def analyze_multiple_learners(
        self,
        learner_uids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        对多个学习者进行“社会性学习与同伴取向”分析。

        返回：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,
                "overall_label": str 或 None,
                "overall_cluster_rank": int 或 None,
                "per_course_results": [...],
                # 如出错，还会包含 "error": str
            },
            ...
        }
        """
        if not learner_uids:
            return {}

        try:
            events = social_learning_repository.get_social_learning_events(
                learner_uids
            )

            if not events:
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            social_stats = self._build_social_stats(events)
            if not social_stats:
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            social_results = self._compute_course_social_indices(social_stats)
            if not social_results:
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            self._assign_social_labels(social_results)

            learner_summaries = self._build_learner_summaries(social_results)

            # 确保所有传入 UID 都有结构化返回
            for uid in learner_uids:
                if uid not in learner_summaries:
                    learner_summaries[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }

            return learner_summaries

        except Exception as e:
            logger.error(
                f"多学习者社会性学习与同伴取向分析失败: {e}", exc_info=True
            )
            result: Dict[str, Dict[str, Any]] = {}
            for uid in learner_uids:
                result[uid] = {
                    "learner_uid": uid,
                    "has_data": False,
                    "overall_score": None,
                    "overall_label": None,
                    "overall_cluster_rank": None,
                    "per_course_results": [],
                    "error": str(e),
                }
            return result

    def analyze_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """
        单学习者便捷接口：返回结构等同于 analyze_multiple_learners()[learner_uid]
        """
        results = self.analyze_multiple_learners([learner_uid])
        return results.get(
            learner_uid,
            {
                "learner_uid": learner_uid,
                "has_data": False,
                "overall_score": None,
                "overall_label": None,
                "overall_cluster_rank": None,
                "per_course_results": [],
            },
        )


# 全局引擎实例 + 便捷函数（与其它 Engine 保持一致）
_social_engine_instance: Optional[SocialLearningEngine] = None


def get_social_learning_engine() -> SocialLearningEngine:
    global _social_engine_instance
    if _social_engine_instance is None:
        _social_engine_instance = SocialLearningEngine()
    return _social_engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_social_learning_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(
    learner_uids: List[str],
) -> Dict[str, Dict[str, Any]]:
    engine = get_social_learning_engine()
    return engine.analyze_multiple_learners(learner_uids)


# 简单本地测试（使用与其它 engine 相同的测试 UID）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = SocialLearningEngine()
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
