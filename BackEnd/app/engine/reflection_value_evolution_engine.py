# BackEnd/app/engine/reflection_value_evolution_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import re

from app.repositories.reflection_value_evolution_repository import (
    reflection_value_evolution_repository,
    ReflectionValueEvolutionRepository,
)

logger = logging.getLogger(__name__)

# ========== 常量 & 词表 ==========

REFLECTION_ACTION_WINDOW_SECONDS = 30 * 60  # 30 分钟
EXT_REFLECTION_FORMAT = "https://legend-meta.com/xapi/ext/reflection-format"

# 可选 jieba 分词
try:
    import jieba  # type: ignore
except ImportError:  # pragma: no cover
    jieba = None

# 价值/元宇宙相关关键词（与分析脚本保持一致思路）
VALUE_KEYWORDS = {
    # 英文
    "metaverse", "value", "values", "experience", "meaning", "ethics",
    "privacy", "fairness", "community", "identity", "learning", "future",
    # 中文
    "元宇宙", "价值", "体验", "意义", "伦理", "隐私", "公平", "社群",
    "身份", "学习", "未来", "贡献",
    # 与“我们”相关的共同体视角
    "我们", "us", "一起", "集体",
}

# 简单中英停用词
STOPWORDS = {
    # 中文
    "的", "了", "在", "是", "和", "与", "也", "就", "而且", "但是", "因为", "所以", "如果",
    "对", "于", "中", "一个", "自己", "他们", "它", "她", "他",
    # 英文
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on",
    "for", "with", "this", "that", "it", "is", "are", "was", "were",
}


class ReflectionValueEvolutionEngine:
    """
    反思深度与价值观演变分析引擎

    功能：
    - 给定一个或多个学习者 UID，从 Repository 读取 xAPI 行为；
    - 以 (学习者, 课程) 为单位计算：
        * reflection_count                反思次数
        * depth_score_avg                文本深度平均分
        * value_early / value_late       早期/后期价值语汇得分
        * value_evolution_score          价值观演变得分（后期-早期）
        * reflection_to_action_rate      反思后触发拓展的比例
        * freq_norm / depth_norm         频率与深度归一化
        * value_growth_norm / action_norm价值演变与反思驱动行动归一化
        * reflection_index               综合反思指数
        * reflection_index_norm          [0,1] 归一化综合指数
        * reflection_label               文本标签
        * reflection_level               等级（0=浅层/不稳定, 1=稳定深度, 2=成长型）
    - 对单个学习者的多门课程结果做聚合：
        * overall_score                  各课程 reflection_index_norm 均值
        * overall_cluster_rank           各课程 reflection_level 众数（并列时选更高档）
        * overall_label                  综合描述标签
    """

    def __init__(self) -> None:
        logger.info("ReflectionValueEvolutionEngine 初始化完成")

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_min_max(values: List[float], default_mid: bool = False) -> List[float]:
        """
        [0,1] min-max 归一化：
        - 所有值相同或空列表 -> 返回常数：
            * default_mid=True  -> 0.5
            * default_mid=False -> 0.0
        """
        if not values:
            return []
        v_min = min(values)
        v_max = max(values)
        if abs(v_max - v_min) < 1e-8:
            fill = 0.5 if default_mid else 0.0
            return [fill for _ in values]
        span = v_max - v_min
        return [(v - v_min) / span for v in values]

    @staticmethod
    def _parse_iso_timestamp(ts_val: Any) -> Optional[datetime]:
        if isinstance(ts_val, datetime):
            return ts_val
        if not ts_val:
            return None
        ts_str = str(ts_val)
        try:
            if ts_str.endswith("Z"):
                ts_str = ts_str.replace("Z", "+00:00")
            return datetime.fromisoformat(ts_str)
        except Exception:
            return None

    @staticmethod
    def _tokenize_text(text: Any) -> List[str]:
        """
        简单双语分词 + 去停用词
        """
        if text is None:
            return []
        text = str(text).strip()
        if not text:
            return []

        if jieba is not None:
            raw_tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        else:
            text_norm = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
            raw_tokens = text_norm.split()

        tokens: List[str] = []
        for t in raw_tokens:
            t_norm = t.strip().lower()
            if not t_norm or t_norm in STOPWORDS:
                continue
            tokens.append(t_norm)
        return tokens

    def _analyze_single_reflection_text(self, text: Any) -> Dict[str, Any]:
        """
        针对单条反思文本计算：
        - word_count
        - lexical_diversity
        - value_keyword_hits / density / used set
        """
        tokens = self._tokenize_text(text)
        word_count = len(tokens)
        if word_count == 0:
            return {
                "tokens": [],
                "word_count": 0,
                "lexical_diversity": 0.0,
                "value_keyword_hits": 0,
                "value_keyword_density": 0.0,
                "value_keywords_used": set(),
            }

        unique_tokens = set(tokens)
        lexical_diversity = len(unique_tokens) / float(word_count)

        text_lower = str(text).lower()
        value_keywords_used = set()
        for kw in VALUE_KEYWORDS:
            if kw.lower() in text_lower:
                value_keywords_used.add(kw.lower())

        value_keyword_hits = len(value_keywords_used)
        value_keyword_density = value_keyword_hits / float(word_count)

        return {
            "tokens": tokens,
            "word_count": word_count,
            "lexical_diversity": lexical_diversity,
            "value_keyword_hits": value_keyword_hits,
            "value_keyword_density": value_keyword_density,
            "value_keywords_used": value_keywords_used,
        }

    # ------------------------------------------------------------------
    # 核心：从事件计算 (learner, course) 级别的反思指标
    # ------------------------------------------------------------------
    def _compute_reflection_metrics(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        大致对应原脚本的步骤：
        - 事件按 (lrn, course) 聚合为反思列表 & 拓展时间列表；
        - 对每条反思文本做分析，计算深度与价值特征；
        - 聚合到 (lrn, course)，计算：
            reflection_count, depth_score_avg,
            value_early/late, value_evolution_score,
            reflection_to_action_rate；
        - 全局归一化得到 freq_norm / depth_norm / value_growth_norm / action_norm；
        - 综合得到 reflection_index / reflection_index_norm；
        - 按规则生成 reflection_label 与 reflection_level。
        """
        if not events:
            return {}

        verb_dict = ReflectionValueEvolutionRepository.VERBS

        # 1) 按 (learner, course) 聚合反思和拓展事件
        reflections_by_lc: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        extensions_by_lc: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)

        for doc in events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue

            verb_id = (doc.get("verb") or {}).get("id") or doc.get("verb.id")
            ts = self._parse_iso_timestamp(doc.get("timestamp"))
            if ts is None:
                continue

            key = (lrn_uid, crs_uid)

            if verb_id == verb_dict["reflected_on_activity"]:
                result = doc.get("result") or {}
                context = doc.get("context") or {}
                ctx_ext = (context.get("extensions") or {}) if isinstance(context, dict) else {}
                reflection_format = ctx_ext.get(EXT_REFLECTION_FORMAT)
                text = result.get("response") or ""
                reflections_by_lc[key].append(
                    {
                        "timestamp": ts,
                        "text": text,
                        "format": reflection_format,
                    }
                )
            elif verb_id == verb_dict["explored_extension"]:
                extensions_by_lc[key].append(ts)

        for key in reflections_by_lc:
            reflections_by_lc[key].sort(key=lambda r: r["timestamp"])
        for key in extensions_by_lc:
            extensions_by_lc[key].sort()

        if not reflections_by_lc:
            logger.info("[ReflectionValueEvolutionEngine] 无反思事件可分析")
            return {}

        # 2) 逐条反思文本分析 + 全局归一化所需的基础统计
        all_wcs: List[float] = []
        all_lex: List[float] = []
        all_valdens: List[float] = []

        for key, reflist in reflections_by_lc.items():
            for r in reflist:
                analysis = self._analyze_single_reflection_text(r.get("text"))
                r["tokens"] = analysis["tokens"]
                r["word_count"] = analysis["word_count"]
                r["lexical_diversity"] = analysis["lexical_diversity"]
                r["value_keyword_hits"] = analysis["value_keyword_hits"]
                r["value_keyword_density"] = analysis["value_keyword_density"]
                r["value_keywords_used"] = analysis["value_keywords_used"]

                all_wcs.append(analysis["word_count"])
                all_lex.append(analysis["lexical_diversity"])
                all_valdens.append(analysis["value_keyword_density"])

        if not all_wcs:
            logger.info("[ReflectionValueEvolutionEngine] 所有反思文本均为空或无法解析")
            return {}

        wc_norms = self._normalize_min_max(all_wcs, default_mid=True)
        lex_norms = self._normalize_min_max(all_lex, default_mid=True)
        vd_norms = self._normalize_min_max(all_valdens, default_mid=False)

        # 回填归一化结果，并得到 depth_score / value_score
        idx = 0
        total_reflections = sum(len(v) for v in reflections_by_lc.values())
        for key, reflist in reflections_by_lc.items():
            for r in reflist:
                wc_n = wc_norms[idx]
                lv_n = lex_norms[idx]
                vd_n = vd_norms[idx]
                idx += 1

                depth_score = 0.3 * wc_n + 0.3 * lv_n + 0.4 * vd_n
                coverage = (
                    len(r["value_keywords_used"]) / float(len(VALUE_KEYWORDS))
                    if VALUE_KEYWORDS
                    else 0.0
                )
                value_score = 0.5 * vd_n + 0.5 * coverage

                r["wc_norm"] = wc_n
                r["lex_div_norm"] = lv_n
                r["value_density_norm"] = vd_n
                r["depth_score"] = depth_score
                r["value_score"] = value_score

        logger.info(
            f"[ReflectionValueEvolutionEngine] 完成 {total_reflections} 条反思文本的深度与价值特征计算"
        )

        # 3) 聚合到 (learner, course) 级别
        reflection_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for key, reflist in reflections_by_lc.items():
            lrn_uid, crs_uid = key
            if not reflist:
                continue

            n = len(reflist)
            reflection_count = n

            depth_scores = [r["depth_score"] for r in reflist]
            depth_score_avg = (
                sum(depth_scores) / float(len(depth_scores)) if depth_scores else 0.0
            )

            reflist_sorted = sorted(reflist, key=lambda r: r["timestamp"])
            if n >= 4:
                split_idx = n // 2
            elif n >= 2:
                split_idx = 1
            else:
                split_idx = 1

            early_refs = reflist_sorted[:split_idx]
            late_refs = reflist_sorted[split_idx:] if split_idx < n else reflist_sorted

            value_early = (
                sum(r["value_score"] for r in early_refs) / float(len(early_refs))
                if early_refs
                else 0.0
            )
            value_late = (
                sum(r["value_score"] for r in late_refs) / float(len(late_refs))
                if late_refs
                else 0.0
            )
            value_evolution_score = value_late - value_early

            ext_times = extensions_by_lc.get(key, [])
            ext_count = len(ext_times)
            reflection_with_action = 0

            if ext_times:
                j = 0
                for r in reflist_sorted:
                    ts_ref = r["timestamp"]
                    while j < ext_count and ext_times[j] < ts_ref:
                        j += 1
                    has_action = False
                    k = j
                    while k < ext_count:
                        delta = (ext_times[k] - ts_ref).total_seconds()
                        if delta < 0:
                            k += 1
                            continue
                        if delta <= REFLECTION_ACTION_WINDOW_SECONDS:
                            has_action = True
                            break
                        else:
                            break
                    if has_action:
                        reflection_with_action += 1

            reflection_to_action_rate = (
                reflection_with_action / float(reflection_count)
                if reflection_count > 0
                else 0.0
            )

            reflection_metrics[key] = {
                "reflection_count": int(reflection_count),
                "depth_score_avg": float(depth_score_avg),
                "value_early": float(value_early),
                "value_late": float(value_late),
                "value_evolution_score": float(value_evolution_score),
                "reflection_to_action_rate": float(reflection_to_action_rate),
            }

        if not reflection_metrics:
            logger.info("[ReflectionValueEvolutionEngine] 没有可用的 (学习者, 课程) 聚合指标")
            return {}

        logger.info(
            f"[ReflectionValueEvolutionEngine] 完成 {len(reflection_metrics)} 个 (学习者, 课程) 聚合指标计算"
        )

        # 4) 全局归一化 + 综合指数 reflection_index / reflection_index_norm
        freq_vals: List[float] = []
        depth_vals: List[float] = []
        value_growth_vals: List[float] = []
        action_vals: List[float] = []

        keys_list = list(reflection_metrics.keys())
        for key in keys_list:
            m = reflection_metrics[key]
            freq_vals.append(float(m["reflection_count"]))
            depth_vals.append(float(m["depth_score_avg"]))
            value_growth_vals.append(max(float(m["value_evolution_score"]), 0.0))
            action_vals.append(float(m["reflection_to_action_rate"]))

        freq_norm_list = self._normalize_min_max(freq_vals, default_mid=True)
        depth_norm_list = self._normalize_min_max(depth_vals, default_mid=True)
        value_growth_norm_list = self._normalize_min_max(
            value_growth_vals, default_mid=False
        )
        action_norm_list = self._normalize_min_max(action_vals, default_mid=False)

        reflection_index_raw_list: List[float] = []

        for i, key in enumerate(keys_list):
            m = reflection_metrics[key]
            freq_norm = freq_norm_list[i]
            depth_norm = depth_norm_list[i]
            value_growth_norm = value_growth_norm_list[i]
            action_norm = action_norm_list[i]

            value_part = 0.5 * value_growth_norm + 0.5 * action_norm
            reflection_index_raw = (
                0.3 * freq_norm + 0.4 * depth_norm + 0.3 * value_part
            )

            m["freq_norm"] = float(freq_norm)
            m["depth_norm"] = float(depth_norm)
            m["value_growth_norm"] = float(value_growth_norm)
            m["action_norm"] = float(action_norm)
            m["reflection_index_raw"] = float(reflection_index_raw)

            reflection_index_raw_list.append(reflection_index_raw)

        reflection_index_norm_list = self._normalize_min_max(
            reflection_index_raw_list, default_mid=True
        )

        for i, key in enumerate(keys_list):
            m = reflection_metrics[key]
            m["reflection_index"] = float(m["reflection_index_raw"])
            m["reflection_index_norm"] = float(reflection_index_norm_list[i])

        logger.info(
            "[ReflectionValueEvolutionEngine] 已完成 reflection_index 与 reflection_index_norm 计算"
        )

        # 5) 按规则生成标签与等级
        label_counts: Dict[str, int] = defaultdict(int)

        for key, m in reflection_metrics.items():
            reflection_count = m["reflection_count"]
            idx_norm = m["reflection_index_norm"]
            ve = m["value_evolution_score"]

            if reflection_count < 2:
                label = "反思样本不足型学习者（仅有零星反思，暂难判断深度与价值观演变）"
                level = 0
            else:
                if idx_norm >= 0.66 and ve > 0.0:
                    label = (
                        "成长型价值反思者（反思频率与深度较高，且围绕元宇宙/学习价值的语汇在后期明显增强）"
                    )
                    level = 2
                elif idx_norm >= 0.5 and ve >= -0.05:
                    label = (
                        "稳定深度反思者（反思深度较高，但价值相关语汇变化有限或方向较稳定）"
                    )
                    level = 1
                else:
                    label = (
                        "浅层或不稳定反思者（反思频率较低或文本多为描述性，价值相关语汇使用有限/变化不稳定）"
                    )
                    level = 0

            m["reflection_label"] = label
            m["reflection_level"] = int(level)
            label_counts[label] += 1

        for label, cnt in label_counts.items():
            logger.info(f"[ReflectionValueEvolutionEngine] 标签分布: {label} -> {cnt}")

        return reflection_metrics

    # ------------------------------------------------------------------
    # 聚合到学习者级别
    # ------------------------------------------------------------------
    def _build_learner_summaries(
        self,
        reflection_metrics: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (学习者, 课程) 级别结果聚合为按学习者的结果。
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), m in reflection_metrics.items():
            if "reflection_index_norm" not in m:
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
                "reflection_count": int(m.get("reflection_count", 0)),
                "depth_score_avg": float(m.get("depth_score_avg", 0.0)),
                "value_early": float(m.get("value_early", 0.0)),
                "value_late": float(m.get("value_late", 0.0)),
                "value_evolution_score": float(m.get("value_evolution_score", 0.0)),
                "reflection_to_action_rate": float(
                    m.get("reflection_to_action_rate", 0.0)
                ),
                "freq_norm": float(m.get("freq_norm", 0.0)),
                "depth_norm": float(m.get("depth_norm", 0.0)),
                "value_growth_norm": float(m.get("value_growth_norm", 0.0)),
                "action_norm": float(m.get("action_norm", 0.0)),
                "reflection_index": float(m.get("reflection_index", 0.0)),
                "reflection_index_norm": float(
                    m.get("reflection_index_norm", 0.0)
                ),
                "reflection_label": m.get("reflection_label"),
                "reflection_level": int(m.get("reflection_level", 0)),
            }
            learner_data[lrn_uid]["per_course_results"].append(item)

        overall_rank_label = {
            0: "整体反思深度偏低或不稳定（反思次数较少或多为描述性，价值相关语汇使用有限）。",
            1: "整体反思深度较好但价值观演变有限（能够较稳定地进行反思，但价值相关语汇变化不大或较为稳定）。",
            2: "整体为成长型价值反思者（在多门课程中持续进行较深度反思，且围绕元宇宙/学习价值的语汇随时间明显增强）。",
        }

        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            scores = [it["reflection_index_norm"] for it in pcs]
            info["overall_score"] = float(sum(scores) / float(len(scores)))

            rank_counts: Dict[int, int] = {}
            for it in pcs:
                level = int(it["reflection_level"])
                rank_counts[level] = rank_counts.get(level, 0) + 1

            if rank_counts:
                max_count = max(rank_counts.values())
                candidate_levels = [
                    lvl for lvl, cnt in rank_counts.items() if cnt == max_count
                ]
                best_level = max(candidate_levels)  # 并列时选“更好”的一档

                info["overall_cluster_rank"] = int(best_level)
                info["overall_label"] = overall_rank_label.get(
                    best_level,
                    "整体反思深度中等（默认）。",
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
        对多个学习者进行“反思深度与价值观演变”分析。

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
            events = reflection_value_evolution_repository.get_reflection_events(
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

            reflection_metrics = self._compute_reflection_metrics(events)
            if not reflection_metrics:
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

            learner_summaries = self._build_learner_summaries(reflection_metrics)

            # 对于传入但没有任何结果的学习者，也返回结构化空结果
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
                f"多学习者反思深度与价值观演变分析失败: {e}", exc_info=True
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
_reflection_engine_instance: Optional[ReflectionValueEvolutionEngine] = None


def get_reflection_value_evolution_engine() -> ReflectionValueEvolutionEngine:
    global _reflection_engine_instance
    if _reflection_engine_instance is None:
        _reflection_engine_instance = ReflectionValueEvolutionEngine()
    return _reflection_engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_reflection_value_evolution_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(
    learner_uids: List[str],
) -> Dict[str, Dict[str, Any]]:
    engine = get_reflection_value_evolution_engine()
    return engine.analyze_multiple_learners(learner_uids)


# 简单本地测试（使用与其它 engine 相同的测试 UID）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = ReflectionValueEvolutionEngine()
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
