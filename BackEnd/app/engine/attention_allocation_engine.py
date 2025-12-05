# BackEnd/app/engine/attention_allocation_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from math import sqrt
from datetime import datetime

from app.repositories.attention_allocation_repository import (
    attention_allocation_repository,
)

logger = logging.getLogger(__name__)


# 与分析脚本保持一致的扩展字段 URL
EXT_UNIT_TYPE = "https://legend-meta.com/xapi/ext/unit-type"
EXT_FOCUS_TARGET = "https://legend-meta.com/xapi/ext/focus-target-id"


class AttentionAllocationEngine:
    """
    注意力分配与信息加工方式分析引擎

    功能：
    - 给定一个或多个学习者 UID，从 Repository 读取细粒度 xAPI 行为；
    - 以 (学习者, 课程) 为单位，计算注意力分配比例、首注视比例、表现；
    - 基于规则生成信息加工风格标签（style_label）；
    - 计算注意效率指数 E_att & 归一化到 [0,1] 的 E_att_norm；
    - 基于 E_att_norm 做三档分类（低 / 中 / 高效），并为学习者整体结果聚合：
        * 数值：多课程 E_att_norm 的均值；
        * 分类：多课程效率标签的众数，如有并列则选择“更好”的那一档。
    """

    def __init__(self):
        # 目前不需要复杂初始化，这里保留接口方便以后拓展
        logger.info("AttentionAllocationEngine 初始化完成")

    # ------------------------------------------------------------------
    # 一些工具函数（来自原分析脚本）
    # ------------------------------------------------------------------

    @staticmethod
    def parse_iso8601_duration(duration_str: Optional[str]) -> Optional[int]:
        """
        解析简单形式的 ISO8601 时长字符串 "PT{秒数}S"。
        - 若为空或格式不正确，返回 None。
        """
        if not duration_str:
            return None
        # 简单健壮解析：PT{int}S
        try:
            if not duration_str.startswith("PT") or not duration_str.endswith("S"):
                return None
            num_part = duration_str[2:-1]
            return int(num_part)
        except Exception:
            return None

    @staticmethod
    def compute_mean_std(values: List[float]) -> Tuple[float, float]:
        """
        计算均值和总体标准差：
        - 空列表 => (0.0, 0.0)
        - 单元素 => (value, 0.0)
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
    def categorize_aoi(target_id: Optional[str]) -> str:
        """
        根据 focus-target-id 粗略判断 AOI 类型：
        - 文本（text）
        - 图像/模型（visual）
        - 示例/提示（example）
        - 界面/其他（ui_other）

        规则与原脚本保持一致。
        """
        if not target_id:
            return "ui_other"

        tid = target_id.lower()

        # 文本区域
        if (
            "subtitle" in tid
            or "caption" in tid
            or "text" in tid
            or "label" in tid
            or "title" in tid
        ):
            return "text"

        # 图像 / 示意图 / 3D 模型 / 主屏
        if (
            "diagram" in tid
            or "image" in tid
            or "picture" in tid
            or "screen" in tid
            or "model" in tid
            or tid.startswith("vr-object")
            or tid.startswith("ar-object")
        ):
            return "visual"

        # 提示 / 示例 / 解答 / 演示
        if (
            "hint" in tid
            or "tip" in tid
            or "example" in tid
            or "demo" in tid
            or "solution" in tid
            or "explanation" in tid
        ):
            return "example"

        return "ui_other"

    # ------------------------------------------------------------------
    # 核心内部步骤：从原始事件构造 attention_metrics
    # ------------------------------------------------------------------

    def _build_attention_metrics(
        self,
        focus_events: List[Dict[str, Any]],
        observed_events: List[Dict[str, Any]],
        performance_events: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        根据原始 xAPI 事件，计算 (learner, course) 级别的注意力指标：

        返回：
        attention_metrics[(lrn_uid, crs_uid)] = {
            "text_ratio": ...,
            "visual_ratio": ...,
            "example_ratio": ...,
            "ui_ratio": ...,
            "relevant_ratio": ...,
            "first_text_ratio": ...,
            "first_visual_ratio": ...,
            "first_example_ratio": ...,
            "performance": ... or None,
        }
        """

        # 1) 聚合 focused-on-resource 的 AOI 时长与首注视
        aoi_durations: Dict[Tuple[str, str], Dict[str, float]] = {}
        first_aoi: Dict[Tuple[str, str, str], Tuple[datetime, str]] = {}

        for doc in focus_events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            result = doc.get("result") or {}
            duration_str = result.get("duration")
            duration_sec = self.parse_iso8601_duration(duration_str)
            if duration_sec is None or duration_sec <= 0:
                continue

            context = doc.get("context") or {}
            ctx_ext = context.get("extensions") or {}
            target_id = ctx_ext.get(EXT_FOCUS_TARGET)
            aoi_type = self.categorize_aoi(target_id)

            # 解析 timestamp（用于首注视）
            ts_str = doc.get("timestamp")
            try:
                if ts_str:
                    # 与原脚本兼容：把 Z 结尾处理成 +00:00
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    ts = datetime.utcnow()
            except Exception:
                ts = datetime.utcnow()

            # unit_key：优先用 object.id，否则用 course + unit-type 粗糙表示
            obj = doc.get("object") or {}
            obj_id = obj.get("id")
            if obj_id:
                unit_key = obj_id
            else:
                unit_type = ctx_ext.get(EXT_UNIT_TYPE, "unknown")
                unit_key = f"{crs_uid}:{unit_type}"

            key = (lrn_uid, crs_uid)

            if key not in aoi_durations:
                aoi_durations[key] = {
                    "text": 0.0,
                    "visual": 0.0,
                    "example": 0.0,
                    "ui_other": 0.0,
                }
            aoi_durations[key][aoi_type] += float(duration_sec)

            fu_key = (lrn_uid, crs_uid, unit_key)
            if fu_key not in first_aoi:
                first_aoi[fu_key] = (ts, aoi_type)
            else:
                prev_ts, _ = first_aoi[fu_key]
                if ts < prev_ts:
                    first_aoi[fu_key] = (ts, aoi_type)

        # 2) observed-peer 叠加到 example 类时长
        for doc in observed_events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            result = doc.get("result") or {}
            duration_str = result.get("duration")
            duration_sec = self.parse_iso8601_duration(duration_str)
            if duration_sec is None or duration_sec <= 0:
                continue

            key = (lrn_uid, crs_uid)
            if key not in aoi_durations:
                aoi_durations[key] = {
                    "text": 0.0,
                    "visual": 0.0,
                    "example": 0.0,
                    "ui_other": 0.0,
                }
            aoi_durations[key]["example"] += float(duration_sec)

        # 3) 计算 performance（answered / passed / completed）
        perf_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
        for doc in performance_events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            result = doc.get("result") or {}
            success = result.get("success")
            completion = result.get("completion")
            if success is None and completion is None:
                continue

            if success is None:
                val = 1.0 if completion else 0.0
            else:
                val = 1.0 if bool(success) else 0.0

            key = (lrn_uid, crs_uid)
            if key not in perf_stats:
                perf_stats[key] = {"sum": 0.0, "cnt": 0}
            perf_stats[key]["sum"] += val
            perf_stats[key]["cnt"] += 1

        # 4) 计算各类比例 + 首注视比例
        attention_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # 先统计首注视次数
        first_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
        for (lrn_uid, crs_uid, _unit_key), (_ts, aoi_type) in first_aoi.items():
            key = (lrn_uid, crs_uid)
            if key not in first_counts:
                first_counts[key] = {"text": 0, "visual": 0, "example": 0}
            if aoi_type in ("text", "visual", "example"):
                first_counts[key][aoi_type] += 1

        for key, dur_dict in aoi_durations.items():
            lrn_uid, crs_uid = key
            total_dur = sum(dur_dict.values())
            if total_dur <= 0:
                continue

            text_dur = dur_dict["text"]
            visual_dur = dur_dict["visual"]
            example_dur = dur_dict["example"]
            ui_dur = dur_dict["ui_other"]

            text_ratio = text_dur / total_dur
            visual_ratio = visual_dur / total_dur
            example_ratio = example_dur / total_dur
            ui_ratio = ui_dur / total_dur
            relevant_ratio = (text_dur + visual_dur + example_dur) / total_dur

            fc = first_counts.get(key, {})
            total_first = sum(fc.values())
            if total_first > 0:
                first_text_ratio = fc.get("text", 0) / float(total_first)
                first_visual_ratio = fc.get("visual", 0) / float(total_first)
                first_example_ratio = fc.get("example", 0) / float(total_first)
            else:
                first_text_ratio = first_visual_ratio = first_example_ratio = 0.0

            perf = perf_stats.get(key, {"sum": 0.0, "cnt": 0})
            if perf["cnt"] > 0:
                performance = perf["sum"] / float(perf["cnt"])
            else:
                performance = None

            attention_metrics[key] = {
                "text_ratio": text_ratio,
                "visual_ratio": visual_ratio,
                "example_ratio": example_ratio,
                "ui_ratio": ui_ratio,
                "relevant_ratio": relevant_ratio,
                "first_text_ratio": first_text_ratio,
                "first_visual_ratio": first_visual_ratio,
                "first_example_ratio": first_example_ratio,
                "performance": performance,
            }

        return attention_metrics

    # ------------------------------------------------------------------
    # 加工风格分类 + 注意效率指数
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_style(m: Dict[str, Any]) -> str:
        """
        生成信息加工风格标签（与原脚本逻辑一致）
        """
        tr = m["text_ratio"]
        vr = m["visual_ratio"]
        er = m["example_ratio"]
        rr = m["relevant_ratio"]
        ftr = m["first_text_ratio"]
        fvr = m["first_visual_ratio"]
        fer = m["first_example_ratio"]

        max_ratio = max(tr, vr, er)
        first_max = max(ftr, fvr, fer)

        if (tr == max_ratio and tr >= 0.55) or (ftr == first_max and ftr > 0):
            return "文本优先型加工（进入或整体上更偏向文字信息）"

        if (vr == max_ratio and vr >= 0.55) or (fvr == first_max and fvr > 0):
            return "图像/模型优先型加工（进入或整体上更偏向图像/3D 模型）"

        if (er == max_ratio and er >= 0.45) or (fer == first_max and fer > 0):
            return "示例/演示优先型加工（更偏向提示、示例或同伴演示）"

        if rr >= 0.7 and max_ratio <= 0.6:
            return "均衡整合型加工（在文本/图像/示例之间较为均衡地分配注意）"

        return "加工风格未明（数据不足或注意非常分散）"

    @staticmethod
    def _classify_efficiency(e_norm: float) -> Tuple[int, str]:
        """
        根据 E_att_norm ∈ [0,1] 做简单三档分类：
        - <= 0.33: 低效
        - >= 0.66: 高效
        - 其他：中等

        这样不依赖全局样本的 k-means，单个学习者也有直观的结果。
        """
        if e_norm <= 0.33:
            rank = 0
            label = "低效注意策略（任务表现较低、任务相关注意比例较低且在非任务 UI 区域停留较多）"
        elif e_norm >= 0.66:
            rank = 2
            label = "高效注意策略（在关键资源上集中注意、较少停留在无关 UI，且表现较好）"
        else:
            rank = 1
            label = "中等注意策略（任务相关注意与表现处于中间水平）"
        return rank, label

    def _compute_efficiency_indices(
        self, attention_metrics: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        按课程维度计算注意效率指数 E_att & E_att_norm，并附加到结果中。
        """
        if not attention_metrics:
            return {}

        # 按课程分组
        course_to_entries: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for (lrn_uid, crs_uid), m in attention_metrics.items():
            course_to_entries.setdefault(crs_uid, []).append((lrn_uid, m))

        attention_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for crs_uid, entries in course_to_entries.items():
            if not entries:
                continue

            perf_vals: List[float] = []
            rel_vals: List[float] = []
            ui_vals: List[float] = []

            for (_lrn_uid, m) in entries:
                perf = m["performance"]
                if perf is None:
                    perf = 0.5
                perf_vals.append(perf)
                rel_vals.append(m["relevant_ratio"])
                ui_vals.append(m["ui_ratio"])

            mean_perf, std_perf = self.compute_mean_std(perf_vals)
            mean_rel, std_rel = self.compute_mean_std(rel_vals)
            mean_ui, std_ui = self.compute_mean_std(ui_vals)

            # 计算 E_att
            tmp_store: Dict[Tuple[str, str], Dict[str, float]] = {}
            e_vals: List[float] = []

            for idx, (lrn_uid, m) in enumerate(entries):
                perf = perf_vals[idx]
                rel = rel_vals[idx]
                ui = ui_vals[idx]

                z_perf = (perf - mean_perf) / std_perf if std_perf > 1e-6 else 0.0
                z_rel = (rel - mean_rel) / std_rel if std_rel > 1e-6 else 0.0
                z_ui = (ui - mean_ui) / std_ui if std_ui > 1e-6 else 0.0

                e_att = (z_perf + z_rel - z_ui) / sqrt(3.0)
                tmp_store[(lrn_uid, crs_uid)] = {
                    "z_perf": z_perf,
                    "z_rel": z_rel,
                    "z_ui": z_ui,
                    "E_att": e_att,
                }
                e_vals.append(e_att)

            if not e_vals:
                continue

            e_min = min(e_vals)
            e_max = max(e_vals)
            span = e_max - e_min if e_max > e_min else 0.0

            for (lrn_uid, _m) in entries:
                key = (lrn_uid, crs_uid)
                base = tmp_store[key]
                e_att = base["E_att"]
                if span > 1e-6:
                    e_norm = (e_att - e_min) / span
                else:
                    e_norm = 0.5  # 只有一个样本时，中性值

                m_full = dict(attention_metrics[key])  # 拷贝原 metrics
                m_full["z_perf"] = base["z_perf"]
                m_full["z_rel"] = base["z_rel"]
                m_full["z_ui"] = base["z_ui"]
                m_full["E_att"] = e_att
                m_full["E_att_norm"] = e_norm

                # 加工风格标签
                m_full["style_label"] = self._classify_style(m_full)

                # 效率档位标签
                rank, eff_label = self._classify_efficiency(e_norm)
                m_full["cluster_rank"] = rank
                m_full["efficiency_label"] = eff_label

                attention_results[key] = m_full

        return attention_results

    # ------------------------------------------------------------------
    # 对外公开接口：单个 / 多个学习者
    # ------------------------------------------------------------------

    def _build_learner_summaries(
        self, attention_results: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (lrn, crs) 级别的结果，聚合成按学习者的结果：
        - per_course_results: 列表
        - overall_score: 所有课程 E_att_norm 的均值
        - overall_efficiency_label: 以 cluster_rank 众数为准，冲突时选更高档
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), res in attention_results.items():
            if lrn_uid not in learner_data:
                learner_data[lrn_uid] = {
                    "learner_uid": lrn_uid,
                    "has_data": True,
                    "per_course_results": [],
                    "overall_score": None,
                    "overall_efficiency_label": None,
                    "overall_cluster_rank": None,
                }

            per_course_item = {
                "course_uid": crs_uid,
                "text_ratio": res["text_ratio"],
                "visual_ratio": res["visual_ratio"],
                "example_ratio": res["example_ratio"],
                "ui_ratio": res["ui_ratio"],
                "relevant_ratio": res["relevant_ratio"],
                "first_text_ratio": res["first_text_ratio"],
                "first_visual_ratio": res["first_visual_ratio"],
                "first_example_ratio": res["first_example_ratio"],
                "performance": res["performance"],
                "attention_style_label": res["style_label"],
                "attention_efficiency_index": res["E_att"],
                "attention_efficiency_normalized": res["E_att_norm"],
                "efficiency_label": res["efficiency_label"],
                "cluster_rank": res["cluster_rank"],
            }

            learner_data[lrn_uid]["per_course_results"].append(per_course_item)

        # 聚合 overall_score + overall_efficiency_label
        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            # 数值：E_att_norm 均值
            scores = [it["attention_efficiency_normalized"] for it in pcs]
            info["overall_score"] = sum(scores) / float(len(scores))

            # 分类：按 cluster_rank 众数，若并列则选 rank 较大者（更“好”的那档）
            rank_counts: Dict[int, int] = {}
            for it in pcs:
                r = int(it["cluster_rank"])
                rank_counts[r] = rank_counts.get(r, 0) + 1

            if rank_counts:
                max_count = max(rank_counts.values())
                candidate_ranks = [r for r, c in rank_counts.items() if c == max_count]
                best_rank = max(candidate_ranks)  # 并列时选更高的一档

                # 与 _classify_efficiency 的文案保持一致
                if best_rank == 0:
                    eff_label = (
                        "低效注意策略（任务表现较低、任务相关注意比例较低且在非任务 UI 区域停留较多）"
                    )
                elif best_rank == 2:
                    eff_label = (
                        "高效注意策略（在关键资源上集中注意、较少停留在无关 UI，且表现较好）"
                    )
                else:
                    eff_label = "中等注意策略（任务相关注意与表现处于中间水平）"

                info["overall_cluster_rank"] = best_rank
                info["overall_efficiency_label"] = eff_label
            else:
                info["overall_cluster_rank"] = None
                info["overall_efficiency_label"] = None

        return learner_data

    def analyze_multiple_learners(
        self, learner_uids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        对多个学习者进行注意力分配分析。

        返回：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,
                "overall_efficiency_label": str 或 None,
                "overall_cluster_rank": int 或 None,
                "per_course_results": [...],
            },
            ...
        }
        """
        if not learner_uids:
            return {}

        try:
            # 1) 从 Repository 获取原始事件
            raw = attention_allocation_repository.get_attention_raw_data_for_learners(
                learner_uids
            )
            focus_events = raw.get("focus_events", [])
            observed_events = raw.get("observed_events", [])
            performance_events = raw.get("performance_events", [])

            if not focus_events:
                # 没有 focus 数据，直接标记无数据
                result: Dict[str, Dict[str, Any]] = {}
                for uid in learner_uids:
                    result[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_efficiency_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }
                return result

            # 2) 计算 (lrn, crs) 级别注意力指标
            attention_metrics = self._build_attention_metrics(
                focus_events, observed_events, performance_events
            )

            # 3) 计算注意效率指数和标签
            attention_results = self._compute_efficiency_indices(attention_metrics)

            # 4) 聚合为按学习者的结果
            learner_summaries = self._build_learner_summaries(attention_results)

            # 5) 对于传入但没有任何数据的学习者，也返回空结果
            for uid in learner_uids:
                if uid not in learner_summaries:
                    learner_summaries[uid] = {
                        "learner_uid": uid,
                        "has_data": False,
                        "overall_score": None,
                        "overall_efficiency_label": None,
                        "overall_cluster_rank": None,
                        "per_course_results": [],
                    }

            return learner_summaries

        except Exception as e:
            logger.error(f"多学习者注意力分析失败: {e}", exc_info=True)
            # 出错时，也保证返回结构是按 learner_uid 的 dict
            result: Dict[str, Dict[str, Any]] = {}
            for uid in learner_uids:
                result[uid] = {
                    "learner_uid": uid,
                    "has_data": False,
                    "overall_score": None,
                    "overall_efficiency_label": None,
                    "overall_cluster_rank": None,
                    "per_course_results": [],
                    "error": str(e),
                }
            return result

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
                "overall_score": None,
                "overall_efficiency_label": None,
                "overall_cluster_rank": None,
                "per_course_results": [],
            },
        )


# 全局引擎实例 + 便捷函数，风格与 hgc_engine 对齐
_attention_engine_instance: Optional[AttentionAllocationEngine] = None


def get_attention_allocation_engine() -> AttentionAllocationEngine:
    global _attention_engine_instance
    if _attention_engine_instance is None:
        _attention_engine_instance = AttentionAllocationEngine()
    return _attention_engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_attention_allocation_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(learner_uids: List[str]) -> Dict[str, Dict[str, Any]]:
    engine = get_attention_allocation_engine()
    return engine.analyze_multiple_learners(learner_uids)


# 简单本地测试（可选）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = AttentionAllocationEngine()
    # 真实存在的学习者UID
    test_learner_uids = [
        "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
        "lrn_004a9c3f5bf246faab3d390ce716e658"
    ]

    print("=== 单学习者测试 ===")
    res = engine.analyze_single_learner(test_learner_uids[0])
    print(res)

    print("=== 多学习者测试 ===")
    res = engine.analyze_multiple_learners(test_learner_uids)
    print(res)
