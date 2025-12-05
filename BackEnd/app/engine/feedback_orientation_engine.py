# BackEnd/app/engine/feedback_orientation_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from math import sqrt
from datetime import datetime, timedelta
import random
from collections import defaultdict

from app.repositories.feedback_orientation_repository import (
    feedback_orientation_repository,
    FeedbackOrientationRepository,
)

logger = logging.getLogger(__name__)


class FeedbackOrientationEngine:
    """
    反馈敏感度与数据使用能力分析引擎

    功能：
    - 给定一个或多个学习者 UID，从 Repository 读取细粒度 xAPI 行为；
    - 以 (学习者, 课程) 为单位计算：
        * feedback_view_count          - 反馈面板查看次数
        * feedback_view_rate           - 机会归一化的反馈查看率
        * feedback_view_type_dist      - 反馈查看类型分布（unit/course/group/...）
        * support_view_count           - 即时反馈（解析/示例/提示）使用次数
        * support_view_rate            - 机会归一化的即时反馈使用率
        * improvement_after_feedback   - 查看反馈后正确率提升（策略调整代理）
        * FO / FO_norm                 - 反馈敏感度指数及归一化结果
        * feedback_label / cluster_rank- 三档分类标签（低/中/高）
    - 对单个学习者的多门课程结果做聚合：
        * overall_score：多课程 FO_norm 的均值；
        * overall_cluster_rank & overall_label：
          以 cluster_rank 众数为准，若并列则选“更好”的一档（rank 值更大）。
    """

    # 与分析脚本保持一致的窗口参数
    FEEDBACK_WINDOW_MINUTES = 10
    POST_FEEDBACK_K = 3

    def __init__(self) -> None:
        logger.info("FeedbackOrientationEngine 初始化完成")

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_mean_std(values: List[float]) -> Tuple[float, float]:
        """
        计算均值与总体标准差：
        - 空列表 -> (0.0, 0.0)
        - 单元素 -> (value, 0.0)
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
        一维 k-means 聚类（与离线脚本风格一致）。

        返回：
        - centers      聚类中心列表
        - assignments  与 values 一一对应的簇编号（0 ~ k-1）
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

        # 用区间等分初始化中心
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

    @staticmethod
    def _parse_timestamp(ts_val: Any) -> Optional[datetime]:
        """
        将字符串/Datetime 类型时间戳统一解析为 datetime，用于排序。
        """
        if isinstance(ts_val, datetime):
            return ts_val
        if isinstance(ts_val, str):
            try:
                return datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # 第一步：从事件构建 (学习者, 课程) 级别的基础行为指标
    # ------------------------------------------------------------------
    def _build_course_level_metrics(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        将原始 xAPI 事件分组为 (学习者, 课程)，并计算基础行为指标：
        - feedback_view_count / rate / type_dist
        - support_view_count / rate
        - improvement_after_feedback
        """
        verb_dict = FeedbackOrientationRepository.VERBS

        # 1) 先按 (learner, course) 分组 + 时间排序
        lc_events: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for doc in events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            ts = self._parse_timestamp(doc.get("timestamp"))
            doc["_parsed_ts"] = ts
            lc_events[(lrn_uid, crs_uid)].append(doc)

        for key in lc_events:
            lc_events[key].sort(key=lambda d: d.get("_parsed_ts") or datetime.min)

        results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        window_delta = timedelta(minutes=self.FEEDBACK_WINDOW_MINUTES)

        for (lrn_uid, crs_uid), seq in lc_events.items():
            if not seq:
                continue

            # 6.1 抽取三类事件
            feedback_views: List[Dict[str, Any]] = []
            support_views: List[Dict[str, Any]] = []
            task_outcomes: List[Tuple[datetime, bool]] = []

            for e in seq:
                vid = (e.get("verb") or {}).get("id") or e.get("verb.id")
                ts = e.get("_parsed_ts")

                if vid == verb_dict["reviewed_feedback"]:
                    feedback_views.append(e)

                elif vid == verb_dict["requested_support"]:
                    support_views.append(e)

                elif vid in (
                    verb_dict["answered"],
                    verb_dict["completed"],
                    verb_dict["performed_procedure_step"],
                ):
                    r = e.get("result") or {}
                    success = r.get("success")
                    completion = r.get("completion")
                    if success is None and completion is None:
                        continue
                    ok = bool(success) if success is not None else bool(completion)
                    task_outcomes.append((ts, ok))

            # 没有任何机会则直接给 0 指标
            opportunity_count = len(task_outcomes)

            # 6.2 反馈查看类型分布
            type_counts: Dict[str, int] = defaultdict(int)
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
            feedback_view_type_dist = {
                k: v / float(total_type) for k, v in type_counts.items()
            }

            # 6.3 机会归一化频率
            feedback_view_count = len(feedback_views)
            support_view_count = len(support_views)

            if opportunity_count <= 0:
                feedback_view_rate = 0.0
                support_view_rate = 0.0
            else:
                feedback_view_rate = feedback_view_count / float(opportunity_count)
                support_view_rate = support_view_count / float(opportunity_count)

            # 6.4 反馈后正确率提升（策略调整代理）
            timeline: List[Tuple[str, datetime, Optional[bool]]] = []
            for e in seq:
                ts = e.get("_parsed_ts")
                if not ts:
                    continue
                vid = (e.get("verb") or {}).get("id") or e.get("verb.id")

                if vid in (
                    verb_dict["answered"],
                    verb_dict["completed"],
                    verb_dict["performed_procedure_step"],
                ):
                    r = e.get("result") or {}
                    success = r.get("success")
                    completion = r.get("completion")
                    if success is None and completion is None:
                        continue
                    ok = bool(success) if success is not None else bool(completion)
                    timeline.append(("task", ts, ok))

                elif vid == verb_dict["reviewed_feedback"]:
                    timeline.append(("feedback", ts, None))

            timeline.sort(key=lambda x: x[1])

            improvements: List[float] = []

            for i, (typ, ts, ok) in enumerate(timeline):
                if typ != "task" or ok is True:
                    continue  # 只关注错误任务

                # 找错误后的最近一次反馈查看
                fb_idx = None
                for j in range(i + 1, len(timeline)):
                    if timeline[j][0] == "feedback" and timeline[j][1] - ts <= window_delta:
                        fb_idx = j
                        break
                    if timeline[j][1] - ts > window_delta:
                        break

                if fb_idx is None:
                    continue

                # 取反馈前后各 K 个任务结果
                pre: List[bool] = []
                post: List[bool] = []

                k = i - 1
                while k >= 0 and len(pre) < self.POST_FEEDBACK_K:
                    if timeline[k][0] == "task":
                        pre.append(bool(timeline[k][2]))
                    k -= 1

                k = fb_idx + 1
                while k < len(timeline) and len(post) < self.POST_FEEDBACK_K:
                    if timeline[k][0] == "task":
                        post.append(bool(timeline[k][2]))
                    k += 1

                if pre and post:
                    pre_acc = sum(1 for x in pre if x) / float(len(pre))
                    post_acc = sum(1 for x in post if x) / float(len(post))
                    improvements.append(post_acc - pre_acc)

            improvement_after_feedback = (
                sum(improvements) / float(len(improvements))
                if improvements
                else 0.0
            )

            results[(lrn_uid, crs_uid)] = {
                "feedback_view_count": int(feedback_view_count),
                "feedback_view_rate": float(feedback_view_rate),
                "feedback_view_type_dist": feedback_view_type_dist,
                "support_view_count": int(support_view_count),
                "support_view_rate": float(support_view_rate),
                "improvement_after_feedback": float(improvement_after_feedback),
            }

        return results

    # ------------------------------------------------------------------
    # 第二步：课程内标准化并合成 FO / FO_norm
    # ------------------------------------------------------------------
    def _compute_feedback_index(
        self,
        results: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        在每门课程内，对下列指标做 z 标准化并合成反馈指数 FO：
        - feedback_view_rate
        - support_view_rate
        - improvement_after_feedback

        再在课程内做 min-max 归一化得到 FO_norm ∈ [0,1]。
        """
        if not results:
            return {}

        course_to_keys: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for (lrn_uid, crs_uid) in results:
            course_to_keys[crs_uid].append((lrn_uid, crs_uid))

        for crs_uid, keys in course_to_keys.items():
            fv_rates = [results[k]["feedback_view_rate"] for k in keys]
            sp_rates = [results[k]["support_view_rate"] for k in keys]
            imps = [results[k]["improvement_after_feedback"] for k in keys]

            m_fv, s_fv = self._compute_mean_std(fv_rates)
            m_sp, s_sp = self._compute_mean_std(sp_rates)
            m_im, s_im = self._compute_mean_std(imps)

            FO_vals: List[float] = []

            for k in keys:
                z_fv = (
                    (results[k]["feedback_view_rate"] - m_fv) / s_fv
                    if s_fv > 1e-6
                    else 0.0
                )
                z_sp = (
                    (results[k]["support_view_rate"] - m_sp) / s_sp
                    if s_sp > 1e-6
                    else 0.0
                )
                z_im = (
                    (results[k]["improvement_after_feedback"] - m_im) / s_im
                    if s_im > 1e-6
                    else 0.0
                )

                FO = (z_fv + z_sp + z_im) / 3.0
                results[k]["FO"] = float(FO)
                FO_vals.append(FO)

            # 课程内 min-max 归一化
            if FO_vals:
                FO_min, FO_max = min(FO_vals), max(FO_vals)
                span = FO_max - FO_min if FO_max > FO_min else 0.0
                for k in keys:
                    FO = results[k]["FO"]
                    if span > 1e-6:
                        FO_norm = (FO - FO_min) / float(span)
                    else:
                        FO_norm = 0.5
                    results[k]["FO_norm"] = float(FO_norm)

        return results

    # ------------------------------------------------------------------
    # 第三步：聚类 & 标签
    # ------------------------------------------------------------------
    def _assign_labels(
        self,
        results: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> None:
        """
        基于 FO_norm 对 (学习者, 课程) 进行聚类并赋予语义标签：
        - cluster_index
        - cluster_rank (0: 低, 1: 中, 2: 高)
        - feedback_label
        """
        fo_values: List[float] = []
        keys_list: List[Tuple[str, str]] = []

        for key, r in results.items():
            fo_norm = r.get("FO_norm")
            if fo_norm is None:
                continue
            fo_values.append(float(fo_norm))
            keys_list.append(key)

        if not fo_values:
            logger.warning("[FeedbackOrientationEngine] 无 FO_norm 可用于聚类")
            return

        centers, assignments = self._kmeans_1d(fo_values, k=3, max_iter=50)
        if not centers or not assignments:
            logger.warning("[FeedbackOrientationEngine] k-means 聚类失败，跳过标签生成")
            return

        sorted_idx = sorted(range(len(centers)), key=lambda i: centers[i])
        cluster_to_rank = {
            cluster_idx: rank for rank, cluster_idx in enumerate(sorted_idx)
        }

        rank_to_label = {
            0: "低反馈敏感/低数据使用型（几乎不查看反馈或不使用解析；反馈后正确率提升不明显）",
            1: "中等反馈敏感/一般数据使用型（偶尔查看反馈；会在部分场景使用解析/示例）",
            2: "高反馈敏感/高数据使用型（频繁查看反馈面板/进度板；积极用解析并能调整策略）",
        }

        label_counts: Dict[str, int] = defaultdict(int)

        for (key, r), cluster_idx in zip(results.items(), assignments):
            rank = cluster_to_rank.get(cluster_idx, 1)
            label = rank_to_label.get(rank, rank_to_label[1])

            r["cluster_index"] = int(cluster_idx)
            r["cluster_rank"] = int(rank)
            r["feedback_label"] = label

            label_counts[label] += 1

        for label, cnt in label_counts.items():
            logger.info(f"[FeedbackOrientationEngine] 标签分布: {label} -> {cnt}")

    # ------------------------------------------------------------------
    # 第四步：聚合到学习者级别
    # ------------------------------------------------------------------
    def _build_learner_summaries(
        self,
        results: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (学习者, 课程) 级别结果聚合为按学习者的结果。

        返回结构：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,          # 多课程 FO_norm 均值
                "overall_label": str 或 None,            # 综合标签
                "overall_cluster_rank": int 或 None,     # 0/1/2
                "per_course_results": [...],             # 每门课程详情
            },
            ...
        }
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), r in results.items():
            if "FO_norm" not in r:
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
                "feedback_view_count": int(r.get("feedback_view_count", 0)),
                "feedback_view_rate": float(r.get("feedback_view_rate", 0.0)),
                "feedback_view_type_dist": r.get("feedback_view_type_dist", {}),
                "support_view_count": int(r.get("support_view_count", 0)),
                "support_view_rate": float(r.get("support_view_rate", 0.0)),
                "improvement_after_feedback": float(
                    r.get("improvement_after_feedback", 0.0)
                ),
                "FO": float(r.get("FO", 0.0)),
                "FO_norm": float(r.get("FO_norm", 0.0)),
                "feedback_label": r.get("feedback_label"),
                "cluster_rank": int(r.get("cluster_rank", 1)),
            }
            learner_data[lrn_uid]["per_course_results"].append(item)

        overall_rank_label = {
            0: "整体反馈敏感度与数据使用能力偏低（整体上很少主动查看反馈或使用解析，反馈后行为调整不明显）",
            1: "整体反馈敏感度与数据使用能力中等（在部分任务中会主动查看反馈和使用解析）",
            2: "整体反馈敏感度与数据使用能力较高（经常查看系统反馈并基于数据调整学习策略）",
        }

        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            scores = [it["FO_norm"] for it in pcs]
            info["overall_score"] = float(sum(scores) / float(len(scores)))

            rank_counts: Dict[int, int] = {}
            for it in pcs:
                rnk = int(it["cluster_rank"])
                rank_counts[rnk] = rank_counts.get(rnk, 0) + 1

            if rank_counts:
                max_count = max(rank_counts.values())
                candidate_ranks = [
                    rnk for rnk, cnt in rank_counts.items() if cnt == max_count
                ]
                best_rank = max(candidate_ranks)  # 并列时选择“更好”的一档

                info["overall_cluster_rank"] = int(best_rank)
                info["overall_label"] = overall_rank_label.get(
                    best_rank,
                    "整体反馈敏感度与数据使用能力中等（默认）",
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
        对多个学习者进行“反馈敏感度与数据使用能力”分析。

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
            events = feedback_orientation_repository.get_feedback_orientation_events(
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

            # 2) 构建 (学习者, 课程) 级别行为指标
            course_results = self._build_course_level_metrics(events)
            if not course_results:
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

            # 3) 计算 FO / FO_norm
            course_results = self._compute_feedback_index(course_results)
            if not course_results:
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
            self._assign_labels(course_results)

            # 5) 聚合为学习者级别结果
            learner_summaries = self._build_learner_summaries(course_results)

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
            logger.error(f"多学习者反馈敏感度分析失败: {e}", exc_info=True)
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
_feedback_engine_instance: Optional[FeedbackOrientationEngine] = None


def get_feedback_orientation_engine() -> FeedbackOrientationEngine:
    global _feedback_engine_instance
    if _feedback_engine_instance is None:
        _feedback_engine_instance = FeedbackOrientationEngine()
    return _feedback_engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_feedback_orientation_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(
    learner_uids: List[str],
) -> Dict[str, Dict[str, Any]]:
    engine = get_feedback_orientation_engine()
    return engine.analyze_multiple_learners(learner_uids)


# 简单本地测试（使用与其它 engine 相同的测试 UID）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = FeedbackOrientationEngine()
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
