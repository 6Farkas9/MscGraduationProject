# BackEnd/app/engine/engagement_persistence_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from math import sqrt
import re
import random
from collections import defaultdict

from app.repositories.engagement_persistence_repository import (
    engagement_persistence_repository,
    EngagementPersistenceRepository,
)

logger = logging.getLogger(__name__)

# 上下文扩展字段常量前缀（与分析脚本保持一致）
CTX_EXT_BASE = "https://legend-meta.com/xapi/ext/"
EXT_STEP_ID = CTX_EXT_BASE + "step-id"
EXT_IDLE_THRESHOLD = CTX_EXT_BASE + "idle-threshold-seconds"
EXT_UNIT_OPTIONAL = CTX_EXT_BASE + "unit-optional"
EXT_VALUE_CHANGE = CTX_EXT_BASE + "value-change"

# ISO8601 时长解析正则，仅支持 "PT{整数秒}S"
DURATION_RE = re.compile(r"^PT(\d+)S$")


class EngagementPersistenceEngine:
    """
    行为投入度与坚持性分析引擎

    功能：
    - 给定一个或多个学习者 UID，从 Repository 读取细粒度 xAPI 行为；
    - 以 (学习者, 课程) 为单位计算：
        * completion_rate
        * interaction_per_unit
        * retry_rate
        * extension_rate
        * idle_ratio
        * value_rate
        * EP / EP_norm（行为投入度与坚持性指数）
        * 聚类标签 label / cluster_rank（0/1/2 = 低/中/高）
    - 对单个学习者的多门课程结果做聚合：
        * overall_score：多课程 EP_norm 的均值；
        * overall_cluster_rank & overall_label：根据 cluster_rank 众数得到综合分类，
          若并列则选择“更好”的那一档（rank 数值更大）。
    """

    def __init__(self) -> None:
        logger.info("EngagementPersistenceEngine 初始化完成")

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_iso8601_duration(duration_str: Any) -> Optional[int]:
        """解析简单形式的 ISO8601 时长，例如 'PT120S' -> 120（秒）"""
        if not duration_str:
            return None
        m = DURATION_RE.match(str(duration_str))
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _compute_mean_std(values: List[float]) -> Tuple[float, float]:
        """
        计算一组数的均值和总体标准差：
        - 列表为空 -> (0.0, 0.0)
        - 仅一个元素 -> std 视为 0.0
        """
        n = len(values)
        if n == 0:
            return 0.0, 0.0
        mean_v = sum(values) / float(n)
        if n == 1:
            return mean_v, 0.0
        var = sum((v - mean_v) ** 2 for v in values) / float(n)
        return mean_v, sqrt(var)

    @staticmethod
    def _kmeans_1d(
        values: List[float], k: int = 3, max_iter: int = 50
    ) -> Tuple[List[float], List[int]]:
        """
        一维 k-means 聚类（Lloyd 算法），用于基于 EP_norm 自动划分学习者类型。

        返回：
        - centers: 聚类中心列表
        - assignments: 与 values 一一对应的簇编号（0 ~ k-1）
        """
        n = len(values)
        if n == 0 or k <= 0:
            return [], []
        if n <= k:
            centers = list(values)
            assignments = list(range(n))
            return centers, assignments

        centers = random.sample(list(values), k)
        for _ in range(max_iter):
            clusters = [[] for _ in range(k)]
            for v in values:
                dists = [abs(v - c) for c in centers]
                idx = dists.index(min(dists))
                clusters[idx].append(v)

            new_centers: List[float] = []
            for idx in range(k):
                if clusters[idx]:
                    new_centers.append(sum(clusters[idx]) / float(len(clusters[idx])))
                else:
                    new_centers.append(random.choice(values))

            if all(abs(new_centers[i] - centers[i]) < 1e-6 for i in range(k)):
                centers = new_centers
                break
            centers = new_centers

        assignments: List[int] = []
        for v in values:
            dists = [abs(v - c) for c in centers]
            idx = dists.index(min(dists))
            assignments.append(idx)

        return centers, assignments

    # ------------------------------------------------------------------
    # 第一步：聚合中间统计量
    # ------------------------------------------------------------------
    def _build_intermediate_stats(
        self, events: List[Dict[str, Any]]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[Tuple[str, str, str], List[Dict[str, Any]]],
        Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    ]:
        """
        将原始 xAPI 事件聚合为中间统计量：

        返回：
        - agg[(lrn_uid, crs_uid)] = {
              "units_started": set(),
              "units_completed": set(),
              "event_count": int,
              "active_time": float,
              "idle_time": float,
              "extension_count": int,
              "value_events": int,
              "value_change_sum": float,
              "q_fail_count": int,
              "q_fail_then_success": int,
              "step_fail_count": int,
              "step_fail_then_success": int,
          }
        - question_events[(lrn_uid, crs_uid, obj_id)] = [{"t": ts, "success": bool}, ...]
        - step_events[(lrn_uid, crs_uid, step_id)] = [{"t": ts, "success": bool}, ...]
        """
        verb_dict = EngagementPersistenceRepository.VERBS

        agg: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {
                "units_started": set(),
                "units_completed": set(),
                "event_count": 0,
                "active_time": 0.0,
                "idle_time": 0.0,
                "extension_count": 0,
                "value_events": 0,
                "value_change_sum": 0.0,
                "q_fail_count": 0,
                "q_fail_then_success": 0,
                "step_fail_count": 0,
                "step_fail_then_success": 0,
            }
        )

        question_events: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        step_events: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

        for doc in events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            verb = (doc.get("verb") or {}).get("id") or doc.get("verb.id")
            result = doc.get("result") or {}
            context = doc.get("context") or {}
            extensions = context.get("extensions") or {}
            obj_id = (doc.get("object") or {}).get("id") or doc.get("object.id")
            unt_uid = doc.get("_unt_uid")
            utype = doc.get("_type")  # video / vr / ar / interact / cooperate / question / course-level
            timestamp = doc.get("timestamp") or ""

            key_lc = (lrn_uid, crs_uid)
            stat = agg[key_lc]

            # 标记单元参与（question / course-level 不计入单元集合）
            if unt_uid and utype and utype not in ("question", "course-level"):
                stat["units_started"].add(unt_uid)

            # 行为事件数量
            stat["event_count"] += 1

            # completed：完成率 + active_time
            if verb == verb_dict["completed"]:
                if unt_uid and utype and utype not in ("question", "course-level"):
                    if result.get("completion") is True:
                        stat["units_completed"].add(unt_uid)

                dur_sec = self._parse_iso8601_duration(result.get("duration"))
                if dur_sec and dur_sec > 0:
                    stat["active_time"] += float(dur_sec)

            # performed-procedure-step：步骤级 active_time + retry 序列
            elif verb == verb_dict["performed_procedure_step"]:
                dur_sec = self._parse_iso8601_duration(result.get("duration"))
                if dur_sec and dur_sec > 0:
                    stat["active_time"] += float(dur_sec)

                step_id = extensions.get(EXT_STEP_ID)
                if step_id:
                    success_flag = bool(result.get("success"))
                    step_events[(lrn_uid, crs_uid, step_id)].append(
                        {"t": timestamp, "success": success_flag}
                    )

            # answered：题目级 retry 序列
            elif verb == verb_dict["answered"]:
                if obj_id:
                    success_flag = bool(result.get("success"))
                    question_events[(lrn_uid, crs_uid, obj_id)].append(
                        {"t": timestamp, "success": success_flag}
                    )

            # explored-extension：失败后主动额外练习
            elif verb == verb_dict["explored_extension"]:
                stat["extension_count"] += 1

            # remained-idle：挂机 / 走神
            elif verb == verb_dict["remained_idle"]:
                dur_sec = self._parse_iso8601_duration(result.get("duration"))
                if dur_sec and dur_sec > 0:
                    stat["idle_time"] += float(dur_sec)

            # exchanged-value：价值交换
            elif verb == verb_dict["exchanged_value"]:
                stat["value_events"] += 1
                value_change = extensions.get(EXT_VALUE_CHANGE)
                if value_change is not None:
                    try:
                        stat["value_change_sum"] += float(value_change)
                    except Exception:
                        pass

            # initialized 在本分析中不参与统计字段

        # 基于题目与步骤序列统计“先错后对”的重试行为
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

        return agg, question_events, step_events

    # ------------------------------------------------------------------
    # 第二步：计算 (学习者, 课程) 级别指标 + EP / EP_norm
    # ------------------------------------------------------------------
    def _compute_ep_index(
        self, agg: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        从中间统计量计算每个 (学习者, 课程) 的行为指标与 EP / EP_norm。
        """
        ep_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        course_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # 先计算原始指标
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
                # 完全没有有效行为的课程窗口直接跳过
                continue

            # 完成率：units_completed / units_started
            completion_rate = (
                units_completed / float(units_started)
                if units_started > 0
                else 0.0
            )

            # 单位单元交互量：event_count / max(units_started, 1)
            denom_units = float(units_started) if units_started > 0 else 1.0
            interaction_per_unit = event_count / denom_units

            # 重试率：先错后对的比率
            total_fail = q_fail + step_fail
            total_retry = q_retry + step_retry
            if total_fail > 0:
                retry_rate = total_retry / float(total_fail)
            else:
                # 没有失败，给中性值 0.5
                retry_rate = 0.5

            # 失败后 extension 率
            if q_fail > 0:
                extension_rate = extension_count / float(q_fail)
            else:
                extension_rate = extension_count / float(units_started + 1)

            # idle 比例：空闲时长 / (空闲 + 有效时长)
            total_time_for_idle = idle_time + active_time
            idle_ratio = (
                idle_time / float(total_time_for_idle)
                if total_time_for_idle > 0
                else 0.0
            )

            # value 率：价值交换事件数 / 单元数量
            value_rate = value_events / denom_units

            res = {
                "completion_rate": float(completion_rate),
                "interaction_per_unit": float(interaction_per_unit),
                "retry_rate": float(retry_rate),
                "extension_rate": float(extension_rate),
                "idle_ratio": float(idle_ratio),
                "value_rate": float(value_rate),
            }
            ep_results[(lrn_uid, crs_uid)] = res

            # 为课程内标准化收集指标
            course_metrics[crs_uid]["completion_rate"].append(completion_rate)
            course_metrics[crs_uid]["interaction_per_unit"].append(
                interaction_per_unit
            )
            course_metrics[crs_uid]["retry_rate"].append(retry_rate)
            course_metrics[crs_uid]["extension_rate"].append(extension_rate)
            course_metrics[crs_uid]["idle_ratio"].append(idle_ratio)
            course_metrics[crs_uid]["value_rate"].append(value_rate)

        if not ep_results:
            logger.warning(
                "[EngagementPersistenceEngine] 没有任何 (学习者, 课程) 具备可用行为指标"
            )
            return {}

        # 课程内 mean/std
        course_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for crs_uid, metrics in course_metrics.items():
            course_stats[crs_uid] = {}
            for metric_name, vals in metrics.items():
                course_stats[crs_uid][metric_name] = self._compute_mean_std(vals)

        # 权重（与分析脚本保持一致的设计思路）
        w_completion = 1.5
        w_retry = 1.5
        w_extension = 1.2
        w_value = 1.0
        w_interact = 1.0
        w_idle = 1.2

        denom_w = sqrt(
            w_completion ** 2
            + w_retry ** 2
            + w_extension ** 2
            + w_value ** 2
            + w_interact ** 2
            + w_idle ** 2
        )

        all_EP: List[float] = []

        # 计算 EP
        for (lrn_uid, crs_uid), res in ep_results.items():
            stats_course = course_stats.get(crs_uid, {})

            def z_score(metric_key: str) -> float:
                val = res.get(metric_key, 0.0)
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
                w_completion * z_c
                + w_retry * z_r
                + w_extension * z_e
                + w_value * z_v
                + w_interact * z_i
                - w_idle * z_idle
            )
            if denom_w > 0:
                EP = EP / denom_w

            res["EP"] = float(EP)
            all_EP.append(EP)

        # 全局 min-max 归一化得到 EP_norm
        if all_EP:
            min_EP = min(all_EP)
            max_EP = max(all_EP)
            if max_EP > min_EP:
                span = max_EP - min_EP
                for res in ep_results.values():
                    EP = res.get("EP", 0.0)
                    res["EP_norm"] = float((EP - min_EP) / float(span))
            else:
                for res in ep_results.values():
                    res["EP_norm"] = 0.5
        else:
            for res in ep_results.values():
                res["EP_norm"] = 0.5

        return ep_results

    # ------------------------------------------------------------------
    # 第三步：聚类 & 标签
    # ------------------------------------------------------------------
    def _assign_labels(
        self, ep_results: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> None:
        """
        基于 EP_norm 对 (学习者, 课程) 进行聚类并赋予语义标签：
        - cluster_index: 原始簇编号
        - cluster_rank: 按中心高低排序后的等级（0: 低, 1: 中, 2: 高）
        - label: 中文标签
        """
        values_norm: List[float] = []
        keys_list: List[Tuple[str, str]] = []

        for key, res in ep_results.items():
            ep_norm = res.get("EP_norm")
            if ep_norm is None:
                continue
            values_norm.append(float(ep_norm))
            keys_list.append(key)

        if not values_norm:
            logger.warning("[EngagementPersistenceEngine] 无 EP_norm 可用于聚类")
            return

        centers, assignments = self._kmeans_1d(values_norm, k=3, max_iter=50)
        if not centers or not assignments:
            logger.warning("[EngagementPersistenceEngine] k-means 聚类失败")
            return

        # 将中心从小到大排序，映射为 rank 0/1/2
        center_with_idx = list(enumerate(centers))
        center_with_idx.sort(key=lambda x: x[1])
        cluster_to_rank = {cluster_idx: rank for rank, (cluster_idx, _) in enumerate(center_with_idx)}

        label_map = {
            0: "低投入易放弃型学习者",
            1: "中等投入型学习者",
            2: "高投入高坚持型学习者",
        }

        label_counter: Dict[str, int] = defaultdict(int)

        for key, cluster_idx in zip(keys_list, assignments):
            res = ep_results[key]
            rank = cluster_to_rank.get(cluster_idx, 1)
            label = label_map.get(rank, "中等投入型学习者")

            res["cluster_index"] = int(cluster_idx)
            res["cluster_rank"] = int(rank)
            res["label"] = label

            label_counter[label] += 1

        for label, cnt in label_counter.items():
            logger.info(f"[EngagementPersistenceEngine] 标签分布: {label} -> {cnt}")

    # ------------------------------------------------------------------
    # 第四步：聚合到学习者级别
    # ------------------------------------------------------------------
    def _build_learner_summaries(
        self, ep_results: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (学习者, 课程) 级别结果聚合为按学习者的结果。

        返回结构：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,          # 多课程 EP_norm 均值
                "overall_label": str 或 None,            # 综合标签
                "overall_cluster_rank": int 或 None,     # 0/1/2
                "per_course_results": [...],             # 每门课程详情
            },
            ...
        }
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), res in ep_results.items():
            if "EP_norm" not in res:
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
                "completion_rate": float(res.get("completion_rate", 0.0)),
                "interaction_per_unit": float(res.get("interaction_per_unit", 0.0)),
                "retry_rate": float(res.get("retry_rate", 0.0)),
                "extension_rate": float(res.get("extension_rate", 0.0)),
                "idle_ratio": float(res.get("idle_ratio", 0.0)),
                "value_rate": float(res.get("value_rate", 0.0)),
                "EP": float(res.get("EP", 0.0)),
                "EP_norm": float(res.get("EP_norm", 0.0)),
                "label": res.get("label"),
                "cluster_rank": int(res.get("cluster_rank", 1)),
            }
            learner_data[lrn_uid]["per_course_results"].append(item)

        overall_rank_label = {
            0: "整体行为投入度与坚持性偏低（在所参与课程中完成率、交互量和重试/延展行为整体偏弱）",
            1: "整体行为投入度与坚持性中等（整体上能够完成任务并保持一定的参与度）",
            2: "整体行为投入度与坚持性较高（整体上完成率较高，且愿意重试与进行额外练习）",
        }

        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            scores = [it["EP_norm"] for it in pcs]
            info["overall_score"] = float(sum(scores) / float(len(scores)))

            rank_counts: Dict[int, int] = {}
            for it in pcs:
                r = int(it["cluster_rank"])
                rank_counts[r] = rank_counts.get(r, 0) + 1

            if rank_counts:
                max_count = max(rank_counts.values())
                candidate_ranks = [r for r, c in rank_counts.items() if c == max_count]
                best_rank = max(candidate_ranks)  # 并列时选择“更好”的一档

                info["overall_cluster_rank"] = best_rank
                info["overall_label"] = overall_rank_label.get(
                    best_rank,
                    "整体行为投入度与坚持性中等（默认）",
                )
            else:
                info["overall_cluster_rank"] = None
                info["overall_label"] = None

        return learner_data

    # ------------------------------------------------------------------
    # 对外公开接口
    # ------------------------------------------------------------------
    def analyze_multiple_learners(
        self, learner_uids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        对多个学习者进行“行为投入度与坚持性”分析。

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
            # 1) 从 Repository 获取原始事件
            events = engagement_persistence_repository.get_engagement_persistence_events(
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

            # 2) 构建中间统计量
            agg, _, _ = self._build_intermediate_stats(events)
            if not agg:
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

            # 3) 计算 EP 指数
            ep_results = self._compute_ep_index(agg)
            if not ep_results:
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

            # 4) 聚类 + 标签
            self._assign_labels(ep_results)

            # 5) 聚合为学习者级别结果
            learner_summaries = self._build_learner_summaries(ep_results)

            # 6) 对于传入但没有任何结果的学习者，也返回结构化空结果
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
            logger.error(f"多学习者行为投入度与坚持性分析失败: {e}", exc_info=True)
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


# 全局引擎实例 + 便捷函数（与 attention_allocation_engine 风格一致）
_engagement_engine_instance: Optional[EngagementPersistenceEngine] = None


def get_engagement_persistence_engine() -> EngagementPersistenceEngine:
    global _engagement_engine_instance
    if _engagement_engine_instance is None:
        _engagement_engine_instance = EngagementPersistenceEngine()
    return _engagement_engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_engagement_persistence_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(
    learner_uids: List[str],
) -> Dict[str, Dict[str, Any]]:
    engine = get_engagement_persistence_engine()
    return engine.analyze_multiple_learners(learner_uids)


# 简单本地测试（使用与 attention_allocation_engine 相同的测试 UID）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = EngagementPersistenceEngine()
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
