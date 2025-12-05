# BackEnd/app/engine/interaction_style_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from math import log
import re
from collections import defaultdict

from app.repositories.interaction_style_repository import (
    interaction_style_repository,
    InteractionStyleRepository,
)

logger = logging.getLogger(__name__)

# 预编译 duration 正则（与脚本保持一致）
DURATION_RE = re.compile(r"^PT(\d+)S$")


class InteractionStyleEngine:
    """
    交互与操作熟练度 / 风格分析引擎

    功能：
    - 给定一个或多个学习者 UID，从 Repository 读取细粒度 xAPI 行为；
    - 以 (学习者, 课程) 为单位计算：
        * freq_per_minute      每分钟对象操作次数（交互强度）
        * step_success_rate    关键步骤成功率
        * unit_success_rate    单元完成成功率
        * performance_score    综合表现分数（0~1）
        * x, y                 用于聚类的二维特征：
                                 x = log(1 + freq_per_minute)
                                 y = 1 - performance_score
        * style_label          三种交互风格标签（多试多练 / 少操作但准确 / 随便乱点）
        * style_index          数值化风格指数（映射到 [0,1]）
        * cluster_index        k-means 原始簇编号
        * cluster_rank         0/1/2，按“好坏”排序后的等级（0=差，2=好）
    - 对单个学习者的多门课程结果做聚合：
        * overall_score        多课程 style_index 的均值
        * overall_cluster_rank 多课程 cluster_rank 的众数（并列时选“更好”的一档）
        * overall_label        依据 overall_cluster_rank 的整体文本描述
    """

    def __init__(self) -> None:
        logger.info("InteractionStyleEngine 初始化完成")

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_iso8601_duration(duration_str: Any) -> Optional[int]:
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
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _kmeans_2d(
        points: List[Tuple[float, float]],
        k: int = 3,
        max_iter: int = 50,
    ) -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        在二维空间对点集执行 k-means 聚类。

        参数：
        - points: [(x, y), ...]
        - k: 簇数，默认 3（对应三种交互风格）
        - max_iter: 最大迭代次数
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

        # 点几乎重合的退化情况
        if abs(x_max - x_min) < 1e-6 and abs(y_max - y_min) < 1e-6:
            centers = [(x_min, y_min) for _ in range(k)]
            assignments = [0 for _ in range(n)]
            return centers, assignments

        # 初始化中心：沿对角线均匀放置
        centers: List[Tuple[float, float]] = []
        for i in range(k):
            alpha = (i + 0.5) / float(k)
            cx = x_min + (x_max - x_min) * alpha
            cy = y_min + (y_max - y_min) * alpha
            centers.append((cx, cy))

        for _ in range(max_iter):
            clusters: List[List[int]] = [[] for _ in range(k)]
            # 分配阶段
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

            # 更新阶段
            new_centers = list(centers)
            for ci in range(k):
                if clusters[ci]:
                    sum_x = sum(points[idx][0] for idx in clusters[ci])
                    sum_y = sum(points[idx][1] for idx in clusters[ci])
                    cnt = float(len(clusters[ci]))
                    new_centers[ci] = (sum_x / cnt, sum_y / cnt)
                else:
                    new_centers[ci] = centers[ci]

            max_shift = 0.0
            for (ox, oy), (nx, ny) in zip(centers, new_centers):
                shift = (ox - nx) ** 2 + (oy - ny) ** 2
                if shift > max_shift:
                    max_shift = shift
            centers = new_centers
            if max_shift < 1e-4:
                break

        # 最终分配
        assignments: List[int] = []
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

    # ------------------------------------------------------------------
    # 第一步：从事件构建 (learner, course) 级别统计
    # ------------------------------------------------------------------
    def _build_stats_from_events(
        self, events: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        对每个 (lrn_uid, crs_uid) 统计：
        - manip_count：manipulated-object 事件总数（对象操作次数）
        - step_total / step_success：procedure-step 总数与成功数
        - unit_total / unit_success：completed 总数与成功数
        - total_interact_duration：完成事件中的总时长（秒）
        """
        verb_dict = InteractionStyleRepository.VERBS

        stats_per_lc: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for ev in events:
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

            verb_id = (ev.get("verb") or {}).get("id") or ev.get("verb.id")
            result = ev.get("result") or {}

            if verb_id == verb_dict["manipulated_object"]:
                st["manip_count"] += 1

            elif verb_id == verb_dict["performed_procedure_step"]:
                st["step_total"] += 1
                if bool(result.get("success")):
                    st["step_success"] += 1

            elif verb_id == verb_dict["completed"]:
                st["unit_total"] += 1
                if bool(result.get("success")):
                    st["unit_success"] += 1
                dur_str = result.get("duration")
                dur_sec = self._parse_iso8601_duration(dur_str)
                if dur_sec is not None:
                    st["total_interact_duration"] += float(dur_sec)

        logger.info(
            f"[InteractionStyleEngine] 汇总得到的 (学习者, 课程) 交互统计条目数：{len(stats_per_lc)}"
        )

        return stats_per_lc

    # ------------------------------------------------------------------
    # 第二步：计算交互强度 & 表现分数 + 聚类特征
    # ------------------------------------------------------------------
    def _compute_style_features(
        self,
        stats_per_lc: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        List[Tuple[float, float]],
        List[Tuple[str, str]],
    ]:
        """
        对每个 (lrn_uid, crs_uid) 计算：
        - freq_per_minute
        - step_success_rate
        - unit_success_rate
        - performance_score
        - x, y（二位聚类特征）
        """
        style_results: Dict[Tuple[str, str], Dict[str, Any]] = {}
        feature_points: List[Tuple[float, float]] = []
        feature_keys: List[Tuple[str, str]] = []

        for key, st in stats_per_lc.items():
            manip = st["manip_count"]
            total_dur = st["total_interact_duration"]
            step_total = st["step_total"]
            step_success = st["step_success"]
            unit_total = st["unit_total"]
            unit_success = st["unit_success"]

            minutes = max(total_dur / 60.0, 1.0)
            freq_per_minute = manip / minutes if minutes > 0 else 0.0

            if step_total > 0:
                step_success_rate = step_success / float(step_total)
            else:
                step_success_rate = 0.5

            if unit_total > 0:
                unit_success_rate = unit_success / float(unit_total)
            else:
                unit_success_rate = 0.5

            performance_score = 0.5 * step_success_rate + 0.5 * unit_success_rate

            x = log(1.0 + freq_per_minute)
            y = 1.0 - performance_score

            style_results[key] = {
                "freq_per_minute": float(freq_per_minute),
                "step_success_rate": float(step_success_rate),
                "unit_success_rate": float(unit_success_rate),
                "performance_score": float(performance_score),
                "x": float(x),
                "y": float(y),
            }
            feature_points.append((x, y))
            feature_keys.append(key)

        logger.info(
            "[InteractionStyleEngine] 已计算交互强度与表现分数，并构造聚类特征。"
        )

        return style_results, feature_points, feature_keys

    # ------------------------------------------------------------------
    # 第三步：基于 (x, y) 聚类并生成风格标签
    # ------------------------------------------------------------------
    def _assign_style_labels(
        self,
        style_results: Dict[Tuple[str, str], Dict[str, Any]],
        feature_points: List[Tuple[float, float]],
        feature_keys: List[Tuple[str, str]],
    ) -> None:
        """
        使用二维 k-means 聚类，对 (x, y) 空间中的点进行聚类，
        并将每个簇映射为三种交互风格：
        - 少操作但准确型
        - 多试多练型
        - 随便乱点型
        """
        if not feature_points:
            logger.warning("[InteractionStyleEngine] 无特征点可用于聚类")
            return

        centers, assignments = self._kmeans_2d(feature_points, k=3, max_iter=50)
        if not centers:
            logger.warning("[InteractionStyleEngine] k-means 聚类未得到有效中心")
            return

        k = len(centers)

        # 1) 找到错误水平最高的中心 → 随便乱点型（y 最大）
        y_values = [c[1] for c in centers]
        idx_random = max(range(k), key=lambda i: y_values[i])

        remain_idx = [i for i in range(k) if i != idx_random]

        idx_precise = remain_idx[0] if remain_idx else idx_random
        idx_practice = remain_idx[1] if len(remain_idx) > 1 else idx_random

        if len(remain_idx) == 2:
            i1, i2 = remain_idx
            x1, y1 = centers[i1]
            x2, y2 = centers[i2]
            # 选择 x + y 更小的作为“少操作但准确型”
            if (x1 + y1) <= (x2 + y2):
                idx_precise = i1
                idx_practice = i2
            else:
                idx_precise = i2
                idx_practice = i1

        cluster_to_label: Dict[int, str] = {}
        cluster_to_style_index: Dict[int, float] = {}

        # 少操作但准确型：操作少但稳定，风格指数最高
        cluster_to_label[idx_precise] = "少操作但准确型（操作次数较少但步骤和任务成功率较高）"
        cluster_to_style_index[idx_precise] = 0.9

        # 多试多练型：频繁尝试，整体表现尚可
        cluster_to_label[idx_practice] = "多试多练型（操作频率较高，通过反复尝试完成任务）"
        cluster_to_style_index[idx_practice] = 0.7

        # 随便乱点型：高频误操作，指数最低
        cluster_to_label[idx_random] = "随便乱点型（操作频率较高但成功率较低，存在较多无效/误操作）"
        cluster_to_style_index[idx_random] = 0.3

        # 依据 style_index 的高低得到 cluster_rank（0=差，2=好）
        unique_indices = sorted(set(cluster_to_style_index.values()))
        index_to_rank: Dict[float, int] = {
            idx: rank for rank, idx in enumerate(unique_indices)
        }

        label_counts: Dict[str, int] = defaultdict(int)

        for (key, cluster_idx) in zip(feature_keys, assignments):
            res = style_results.get(key)
            if res is None:
                continue

            label = cluster_to_label.get(cluster_idx, "多试多练型（默认）")
            s_index = cluster_to_style_index.get(cluster_idx, 0.7)
            rank = index_to_rank.get(s_index, 1)

            res["cluster_index"] = int(cluster_idx)
            res["style_label"] = label
            res["style_index"] = float(s_index)
            res["cluster_rank"] = int(rank)

            label_counts[label] += 1

        for label, cnt in label_counts.items():
            logger.info(f"[InteractionStyleEngine] 风格标签分布: {label} -> {cnt}")

    # ------------------------------------------------------------------
    # 第四步：聚合到学习者级别
    # ------------------------------------------------------------------
    def _build_learner_summaries(
        self,
        style_results: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (学习者, 课程) 级别结果聚合为按学习者的结果。

        返回结构：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,          # 多课程 style_index 均值
                "overall_label": str 或 None,            # 综合交互风格标签
                "overall_cluster_rank": int 或 None,     # 0/1/2
                "per_course_results": [...],             # 每门课程详情
            },
            ...
        }
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), res in style_results.items():
            s_index = res.get("style_index")
            if s_index is None:
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
                "freq_per_minute": float(res.get("freq_per_minute", 0.0)),
                "step_success_rate": float(res.get("step_success_rate", 0.0)),
                "unit_success_rate": float(res.get("unit_success_rate", 0.0)),
                "performance_score": float(res.get("performance_score", 0.0)),
                "x": float(res.get("x", 0.0)),
                "y": float(res.get("y", 0.0)),
                "style_label": res.get("style_label"),
                "style_index": float(res.get("style_index", 0.0)),
                "cluster_index": int(res.get("cluster_index", 0)),
                "cluster_rank": int(res.get("cluster_rank", 1)),
            }
            learner_data[lrn_uid]["per_course_results"].append(item)

        overall_rank_label = {
            0: "整体交互风格偏随意，存在较多无效/误操作，需要进一步提高操作规范性与熟练度。",
            1: "整体交互风格以多试多练为主，能够通过多次尝试逐渐掌握操作。",
            2: "整体交互风格偏稳健准确，操作次数较少但成功率高，步骤执行稳定。",
        }

        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            scores = [it["style_index"] for it in pcs]
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
                    "整体交互风格以多试多练为主（默认）。",
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
        对多个学习者进行“交互与操作熟练度 / 风格”分析。

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
            events = interaction_style_repository.get_interaction_style_events(
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

            # 2) 构建 (learner, course) 统计
            stats_per_lc = self._build_stats_from_events(events)
            if not stats_per_lc:
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

            # 3) 计算行为特征
            style_results, feature_points, feature_keys = self._compute_style_features(
                stats_per_lc
            )

            if not style_results:
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

            # 4) 聚类 + 风格标签
            self._assign_style_labels(style_results, feature_points, feature_keys)

            # 5) 聚合为学习者级别结果
            learner_summaries = self._build_learner_summaries(style_results)

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
            logger.error(f"多学习者交互风格分析失败: {e}", exc_info=True)
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
_interaction_engine_instance: Optional[InteractionStyleEngine] = None


def get_interaction_style_engine() -> InteractionStyleEngine:
    global _interaction_engine_instance
    if _interaction_engine_instance is None:
        _interaction_engine_instance = InteractionStyleEngine()
    return _interaction_engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_interaction_style_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(
    learner_uids: List[str],
) -> Dict[str, Dict[str, Any]]:
    engine = get_interaction_style_engine()
    return engine.analyze_multiple_learners(learner_uids)


# 简单本地测试（使用与其它 engine 相同的测试 UID）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = InteractionStyleEngine()
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
