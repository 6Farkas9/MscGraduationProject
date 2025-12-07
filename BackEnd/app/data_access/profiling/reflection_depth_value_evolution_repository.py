# BackEnd/app/data_access/profiling/reflection_depth_value_evolution_repository.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository

logger = logging.getLogger(__name__)

# 可选中文分词：若环境中安装了 jieba，则优先使用；否则使用简单规则分词
try:
    import jieba  # type: ignore
except ImportError:  # pragma: no cover
    jieba = None


class ReflectionDepthValueEvolutionRepository(MongoDBBaseRepository):
    """
    反思深度与价值观演变（reflection_value_evolution）维度的数据仓库。

    职责：
    - 仅负责从 MongoDB.MLS.Interaction 读取与该维度相关的 xAPI 行为事件；
    - 利用现有复合索引：
        * idx_lrn_verb_course: {_lrn_uid, 'verb.id', _course_uid}
        * idx_course_verb_lrn: {_course_uid, 'verb.id', _lrn_uid}
      进行课程发现和按课程分批扫描；
    - 输出每个 (lrn_uid, crs_uid) 的聚合数值指标，不进行聚类和文本标签判定。
    """

    DB_NAME = "MLS"
    INTERACTION_COLLECTION = "Interaction"

    VERB_BASE = "https://legend-meta.com/xapi/verb/"

    VERBS = {
        "reflected_on_activity": VERB_BASE + "reflected-on-activity",
        "explored_extension": VERB_BASE + "explored-extension",
    }

    # 反思之后统计“反思驱动拓展”时使用的时间窗口（秒）
    REFLECTION_ACTION_WINDOW_SECONDS = 30 * 60  # 30 分钟

    # 反思文本中关注的“元宇宙 / 学习价值”相关关键词
    VALUE_KEYWORDS = {
        # 英文
        "metaverse",
        "value",
        "values",
        "experience",
        "meaning",
        "ethics",
        "privacy",
        "fairness",
        "community",
        "identity",
        "learning",
        "future",
        # 中文
        "元宇宙",
        "价值",
        "体验",
        "意义",
        "伦理",
        "隐私",
        "公平",
        "社群",
        "身份",
        "学习",
        "未来",
        "贡献",
        # 与“我们”相关的共同体视角
        "我们",
        "us",
        "一起",
        "集体",
    }

    # 简单中英停用词（减少无信息词对多样性的干扰）
    STOPWORDS = {
        # 中文常见虚词
        "的",
        "了",
        "在",
        "是",
        "和",
        "与",
        "也",
        "就",
        "而且",
        "但是",
        "因为",
        "所以",
        "如果",
        "对",
        "于",
        "中",
        "一个",
        "自己",
        "他们",
        "它",
        "她",
        "他",
        # 英文
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "this",
        "that",
        "it",
        "is",
        "are",
        "was",
        "were",
    }

    def __init__(
        self,
        batch_size: int = 5000,
        course_chunk_size: int = 200,
        mongo_operator: Optional[Any] = None,
    ) -> None:
        """
        Args:
            batch_size: Mongo 游标 batch_size（逻辑参数，当前通过 aggregate 读取）。
            course_chunk_size: 每次查询 _course_uid 的分片大小，避免 $in 过大。
            mongo_operator: 可注入的 MongoDBOperator 实例；不传则由基类内部创建。
        """
        super().__init__(mongo_operator=mongo_operator)
        self.batch_size = batch_size
        self.course_chunk_size = course_chunk_size

    # ------------------------------------------------------------------
    # 对外公共接口
    # ------------------------------------------------------------------
    def load_metrics_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[
        Dict[Tuple[str, str], Dict[str, Any]],
        Dict[str, int],
        Dict[str, Set[str]],
    ]:
        """
        为若干学习者准备“反思深度与价值观演变”分析所需的课程级特征。

        返回:
            metrics_by_lc:
                (lrn_uid, crs_uid) -> {
                    "reflection_count": int,
                    "depth_score_avg": float,
                    "value_early": float,
                    "value_late": float,
                    "value_evolution_score": float,
                    "reflection_to_action_rate": float,
                    # 全局归一化后的特征：
                    "freq_norm": float,
                    "depth_norm": float,
                    "value_growth_norm": float,
                    "action_norm": float,
                    # 综合指数：
                    "reflection_index": float,
                    "reflection_index_norm": float,
                }

            learners_per_course:
                crs_uid -> 该课程内参与相关事件的去重学习者数量（用于课程内聚类）

            learner_courses_map:
                lrn_uid -> set(course_uid)
                仅包含 metrics_by_lc 中存在数据的课程。
        """
        learner_uids = list({uid for uid in (learner_uids or []) if uid})
        if not learner_uids:
            logger.info(
                "ReflectionDepthValueEvolutionRepository.load_metrics_for_learners: 空学习者列表，直接返回。"
            )
            return {}, {}, {}

        logger.info(
            "ReflectionDepthValueEvolutionRepository: 开始准备反思维度原始数据，目标学习者数: %d",
            len(learner_uids),
        )

        learner_courses_map, all_courses = self._get_courses_for_learners(learner_uids)
        logger.info(
            "ReflectionDepthValueEvolutionRepository: 与目标学习者相关的课程数: %d",
            len(all_courses),
        )

        if not all_courses:
            logger.info(
                "ReflectionDepthValueEvolutionRepository: 目标学习者在反思相关事件上没有任何课程记录。"
            )
            return {}, {}, learner_courses_map

        (
            reflections_by_lc,
            extensions_by_lc,
            learners_per_course,
            global_stats,
        ) = self._aggregate_reflections_and_extensions_for_courses(all_courses)

        logger.info(
            "ReflectionDepthValueEvolutionRepository: 已完成反思与拓展事件聚合，(lrn, crs) 有反思记录的条目数: %d",
            len(reflections_by_lc),
        )

        if not reflections_by_lc:
            return {}, learners_per_course, learner_courses_map

        metrics_by_lc = self._compute_metrics_from_reflections(
            reflections_by_lc, extensions_by_lc, global_stats
        )

        logger.info(
            "ReflectionDepthValueEvolutionRepository: 聚合指标与综合指数计算完成，(lrn, crs) 有效条目数: %d",
            len(metrics_by_lc),
        )

        # 按有数据的课程过滤 learner_courses_map
        filtered_map: Dict[str, Set[str]] = {}
        for lrn_uid, courses in learner_courses_map.items():
            valid = {crs for crs in courses if (lrn_uid, crs) in metrics_by_lc}
            if valid:
                filtered_map[lrn_uid] = valid

        return metrics_by_lc, learners_per_course, filtered_map

    # ------------------------------------------------------------------
    # 内部: Mongo 访问与课程发现
    # ------------------------------------------------------------------
    def _get_courses_for_learners(
        self, learner_uids: List[str]
    ) -> Tuple[Dict[str, Set[str]], Set[str]]:
        """
        使用 idx_lrn_verb_course 复合索引：
            key: { _lrn_uid: 1, 'verb.id': 1, _course_uid: 1 }

        pipeline:
            match _lrn_uid in learner_uids
                  AND verb.id in [reflected-on-activity, explored-extension]
            group by (lrn_uid, course_uid)
        """
        verb_list = list(self.VERBS.values())

        pipeline = [
            {
                "$match": {
                    "_lrn_uid": {"$in": learner_uids},
                    "verb.id": {"$in": verb_list},
                }
            },
            {
                "$group": {
                    "_id": {
                        "lrn_uid": "$_lrn_uid",
                        "course_uid": "$_course_uid",
                    }
                }
            },
        ]

        learner_courses_map: Dict[str, Set[str]] = defaultdict(set)
        all_courses: Set[str] = set()

        docs = self.aggregate(self.INTERACTION_COLLECTION, pipeline)
        for doc in docs:
            _id = doc.get("_id") or {}
            lrn_uid = _id.get("lrn_uid")
            crs_uid = _id.get("course_uid")
            if not lrn_uid or not crs_uid:
                continue
            learner_courses_map[lrn_uid].add(crs_uid)
            all_courses.add(crs_uid)

        return learner_courses_map, all_courses

    # ------------------------------------------------------------------
    # 内部: 通用迭代器
    # ------------------------------------------------------------------
    def _iterate_events(
        self,
        match_query: Dict[str, Any],
        projection: Dict[str, int],
        batch_size: Optional[int] = None,
    ) -> Iterable[Dict[str, Any]]:
        """
        通用迭代器：按指定查询和投影，使用 aggregate 管道返回文档。

        为了利用 idx_course_verb_lrn 复合索引，match_query 中应包含：
            - "_course_uid": 某值或 {"$in": [...]}；
            - "verb.id": 某值或 {"$in": [...]}。
        """
        pipeline = [
            {"$match": match_query},
            {"$project": projection},
        ]
        docs = self.aggregate(self.INTERACTION_COLLECTION, pipeline)
        for doc in docs:
            yield doc

    # ------------------------------------------------------------------
    # 文本与时间工具
    # ------------------------------------------------------------------
    @classmethod
    def _parse_iso_timestamp(cls, ts: Any) -> Optional[datetime]:
        if isinstance(ts, datetime):
            return ts
        if not ts:
            return None
        s = str(ts)
        try:
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            return datetime.fromisoformat(s)
        except Exception:
            return None

    def _tokenize_text(self, text: Any) -> List[str]:
        if text is None:
            return []
        text = str(text).strip()
        if not text:
            return []

        if jieba is not None:
            raw_tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
        else:
            # 保留字母、数字和汉字，其余全部视为分隔符
            text_norm = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
            raw_tokens = text_norm.split()

        tokens: List[str] = []
        for t in raw_tokens:
            t_norm = t.strip().lower()
            if not t_norm or t_norm in self.STOPWORDS:
                continue
            tokens.append(t_norm)
        return tokens

    def _analyze_reflection_text(self, text: Any) -> Dict[str, Any]:
        """
        针对单条反思文本，计算“文本深度”与“价值语汇”相关统计。
        这里仅返回后续计算所需的数值，避免存太多中间字段。
        """
        tokens = self._tokenize_text(text)
        word_count = len(tokens)
        if word_count == 0:
            return {
                "word_count": 0,
                "lexical_diversity": 0.0,
                "value_keyword_density": 0.0,
                "coverage": 0.0,
            }

        unique_tokens = set(tokens)
        lexical_diversity = len(unique_tokens) / float(word_count)

        text_lower = str(text).lower()
        used_keywords = set()
        for kw in self.VALUE_KEYWORDS:
            if kw.lower() in text_lower:
                used_keywords.add(kw.lower())

        hits = len(used_keywords)
        value_keyword_density = hits / float(word_count)
        coverage = hits / float(len(self.VALUE_KEYWORDS)) if self.VALUE_KEYWORDS else 0.0

        return {
            "word_count": word_count,
            "lexical_diversity": lexical_diversity,
            "value_keyword_density": value_keyword_density,
            "coverage": coverage,
        }

    @staticmethod
    def _update_min_max(
        value: float,
        current_min: Optional[float],
        current_max: Optional[float],
    ) -> Tuple[float, float]:
        if current_min is None or value < current_min:
            current_min = value
        if current_max is None or value > current_max:
            current_max = value
        return current_min, current_max

    @staticmethod
    def _normalize_scalar(
        value: float,
        v_min: Optional[float],
        v_max: Optional[float],
        default_mid: bool = False,
    ) -> float:
        if v_min is None or v_max is None:
            return 0.0
        if abs(v_max - v_min) < 1e-8:
            return 0.5 if default_mid else 0.0
        return (value - v_min) / (v_max - v_min)

    # ------------------------------------------------------------------
    # 原始反思/拓展数据聚合
    # ------------------------------------------------------------------
    def _aggregate_reflections_and_extensions_for_courses(
        self, course_uids: Set[str]
    ) -> Tuple[
        Dict[Tuple[str, str], List[Dict[str, Any]]],
        Dict[Tuple[str, str], List[datetime]],
        Dict[str, int],
        Dict[str, Any],
    ]:
        """
        对一批课程的反思与拓展事件进行聚合，生成：
            - reflections_by_lc[(lrn_uid, crs_uid)] = [ {timestamp, wc, lexdiv, vd, cov}, ... ]
            - extensions_by_lc[(lrn_uid, crs_uid)] = [ ts1, ts2, ... ]（已排序）
            - learners_per_course[crs_uid] = 课程内去重学习者数量
            - global_stats：反思文本的全局 min/max 统计，用于后续归一化
        """
        if not course_uids:
            return {}, {}, {}, {}

        course_list = list(course_uids)
        total_chunks = int(math.ceil(len(course_list) / float(self.course_chunk_size)))

        reflections_by_lc: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        extensions_by_lc: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)
        learners_per_course_sets: Dict[str, Set[str]] = defaultdict(set)

        projection = {
            "_lrn_uid": 1,
            "_course_uid": 1,
            "verb.id": 1,
            "result": 1,
            "context": 1,
            "timestamp": 1,
        }
        verb_list = list(self.VERBS.values())

        # 全局 min/max
        wc_min = wc_max = None
        lv_min = lv_max = None
        vd_min = vd_max = None

        for chunk_idx in range(total_chunks):
            sub_courses = course_list[
                chunk_idx * self.course_chunk_size : (chunk_idx + 1) * self.course_chunk_size
            ]

            logger.info(
                "ReflectionDepthValueEvolutionRepository: 读取事件，课程分片 %d/%d，课程数: %d",
                chunk_idx + 1,
                total_chunks,
                len(sub_courses),
            )

            match_query = {
                "_course_uid": {"$in": sub_courses},
                "verb.id": {"$in": verb_list},
            }

            event_cnt = 0
            for doc in self._iterate_events(match_query, projection):
                event_cnt += 1

                lrn_uid = doc.get("_lrn_uid")
                crs_uid = doc.get("_course_uid")
                if not lrn_uid or not crs_uid:
                    continue

                learners_per_course_sets[crs_uid].add(lrn_uid)

                verb_id = (doc.get("verb") or {}).get("id") or doc.get("verb.id")
                ts = self._parse_iso_timestamp(doc.get("timestamp"))
                if ts is None:
                    continue

                key = (lrn_uid, crs_uid)

                if verb_id == self.VERBS["reflected_on_activity"]:
                    result = doc.get("result") or {}
                    text = result.get("response") or ""
                    t_stats = self._analyze_reflection_text(text)
                    wc = float(t_stats["word_count"])
                    lv = float(t_stats["lexical_diversity"])
                    vd = float(t_stats["value_keyword_density"])
                    cov = float(t_stats["coverage"])

                    wc_min, wc_max = self._update_min_max(wc, wc_min, wc_max)
                    lv_min, lv_max = self._update_min_max(lv, lv_min, lv_max)
                    vd_min, vd_max = self._update_min_max(vd, vd_min, vd_max)

                    reflections_by_lc[key].append(
                        {
                            "timestamp": ts,
                            "word_count": wc,
                            "lexical_diversity": lv,
                            "value_keyword_density": vd,
                            "coverage": cov,
                        }
                    )

                elif verb_id == self.VERBS["explored_extension"]:
                    extensions_by_lc[key].append(ts)

            logger.info(
                "ReflectionDepthValueEvolutionRepository: 完成事件读取，课程分片 %d/%d，事件数: %d",
                chunk_idx + 1,
                total_chunks,
                event_cnt,
            )

        # 排序
        for key in reflections_by_lc:
            reflections_by_lc[key].sort(key=lambda r: r["timestamp"])
        for key in extensions_by_lc:
            extensions_by_lc[key].sort()

        learners_per_course = {
            crs_uid: len(uids) for crs_uid, uids in learners_per_course_sets.items()
        }

        global_stats = {
            "wc_min": wc_min,
            "wc_max": wc_max,
            "lv_min": lv_min,
            "lv_max": lv_max,
            "vd_min": vd_min,
            "vd_max": vd_max,
        }

        logger.info(
            "ReflectionDepthValueEvolutionRepository: 聚合完成，课程数: %d，(lrn, crs) 反思条目数: %d",
            len(learners_per_course),
            len(reflections_by_lc),
        )

        return reflections_by_lc, extensions_by_lc, learners_per_course, global_stats

    # ------------------------------------------------------------------
    # 从反思记录派生 (lrn, crs) 指标 + 综合指数
    # ------------------------------------------------------------------
    def _compute_metrics_from_reflections(
        self,
        reflections_by_lc: Dict[Tuple[str, str], List[Dict[str, Any]]],
        extensions_by_lc: Dict[Tuple[str, str], List[datetime]],
        global_stats: Dict[str, Any],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        输入按 (学习者, 课程) 聚合的反思记录，输出每个 (lrn, crs) 的聚合指标与综合指数。
        """
        wc_min = global_stats.get("wc_min")
        wc_max = global_stats.get("wc_max")
        lv_min = global_stats.get("lv_min")
        lv_max = global_stats.get("lv_max")
        vd_min = global_stats.get("vd_min")
        vd_max = global_stats.get("vd_max")

        # 先为每条反思计算 depth_score 和 value_score
        total_reflections = 0
        for key, reflist in reflections_by_lc.items():
            for r in reflist:
                total_reflections += 1
                wc = float(r["word_count"])
                lv = float(r["lexical_diversity"])
                vd = float(r["value_keyword_density"])
                cov = float(r["coverage"])

                wc_norm = self._normalize_scalar(wc, wc_min, wc_max, default_mid=True)
                lv_norm = self._normalize_scalar(lv, lv_min, lv_max, default_mid=True)
                vd_norm = self._normalize_scalar(vd, vd_min, vd_max, default_mid=False)

                depth_score = 0.3 * wc_norm + 0.3 * lv_norm + 0.4 * vd_norm
                value_score = 0.5 * vd_norm + 0.5 * cov

                r["wc_norm"] = wc_norm
                r["lex_div_norm"] = lv_norm
                r["value_density_norm"] = vd_norm
                r["depth_score"] = depth_score
                r["value_score"] = value_score

        logger.info(
            "ReflectionDepthValueEvolutionRepository: 完成 %d 条反思文本的深度与价值特征计算。",
            total_reflections,
        )

        # 针对每个 (lrn, crs) 计算聚合指标
        metrics_by_lc: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for key, reflist in reflections_by_lc.items():
            if not reflist:
                continue

            reflection_count = len(reflist)
            depth_scores = [r["depth_score"] for r in reflist]
            depth_score_avg = (
                sum(depth_scores) / float(len(depth_scores)) if depth_scores else 0.0
            )

            reflist_sorted = sorted(reflist, key=lambda r: r["timestamp"])
            n = reflection_count
            if n >= 4:
                split_idx = n // 2
            elif n >= 2:
                split_idx = 1
            else:
                split_idx = 1

            early_refs = reflist_sorted[:split_idx]
            late_refs = reflist_sorted[split_idx:] if split_idx < n else reflist_sorted

            if early_refs:
                value_early = sum(r["value_score"] for r in early_refs) / float(
                    len(early_refs)
                )
            else:
                value_early = 0.0

            if late_refs:
                value_late = sum(r["value_score"] for r in late_refs) / float(
                    len(late_refs)
                )
            else:
                value_late = 0.0

            value_evolution_score = value_late - value_early

            # 计算“反思驱动拓展”的比例
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
                        if delta <= self.REFLECTION_ACTION_WINDOW_SECONDS:
                            has_action = True
                            break
                        else:
                            break
                    if has_action:
                        reflection_with_action += 1

            if reflection_count > 0:
                reflection_to_action_rate = (
                    reflection_with_action / float(reflection_count)
                )
            else:
                reflection_to_action_rate = 0.0

            metrics_by_lc[key] = {
                "reflection_count": reflection_count,
                "depth_score_avg": depth_score_avg,
                "value_early": value_early,
                "value_late": value_late,
                "value_evolution_score": value_evolution_score,
                "reflection_to_action_rate": reflection_to_action_rate,
            }

        logger.info(
            "ReflectionDepthValueEvolutionRepository: 完成 %d 个 (lrn, crs) 的反思聚合指标计算。",
            len(metrics_by_lc),
        )

        # 在 (lrn, crs) 级别上做全局归一化和综合指数
        if not metrics_by_lc:
            return {}

        keys_list = list(metrics_by_lc.keys())
        freq_vals: List[float] = []
        depth_vals: List[float] = []
        value_growth_vals: List[float] = []
        action_vals: List[float] = []

        for key in keys_list:
            m = metrics_by_lc[key]
            freq_vals.append(float(m["reflection_count"]))
            depth_vals.append(float(m["depth_score_avg"]))
            value_growth_vals.append(max(float(m["value_evolution_score"]), 0.0))
            action_vals.append(float(m["reflection_to_action_rate"]))

        def min_max(vals: List[float], default_mid: bool) -> List[float]:
            if not vals:
                return []
            v_min = min(vals)
            v_max = max(vals)
            if abs(v_max - v_min) < 1e-8:
                fill = 0.5 if default_mid else 0.0
                return [fill for _ in vals]
            return [(v - v_min) / (v_max - v_min) for v in vals]

        freq_norm_list = min_max(freq_vals, default_mid=True)
        depth_norm_list = min_max(depth_vals, default_mid=True)
        value_growth_norm_list = min_max(value_growth_vals, default_mid=False)
        action_norm_list = min_max(action_vals, default_mid=False)

        reflection_index_raw_list: List[float] = []

        for i, key in enumerate(keys_list):
            freq_norm = freq_norm_list[i]
            depth_norm = depth_norm_list[i]
            value_growth_norm = value_growth_norm_list[i]
            action_norm = action_norm_list[i]

            value_part = 0.5 * value_growth_norm + 0.5 * action_norm
            reflection_index_raw = 0.3 * freq_norm + 0.4 * depth_norm + 0.3 * value_part
            reflection_index_raw_list.append(reflection_index_raw)

            m = metrics_by_lc[key]
            m["freq_norm"] = freq_norm
            m["depth_norm"] = depth_norm
            m["value_growth_norm"] = value_growth_norm
            m["action_norm"] = action_norm
            m["reflection_index"] = reflection_index_raw

        # 对 reflection_index_raw 再做一次 [0,1] 归一化
        index_norm_list = min_max(reflection_index_raw_list, default_mid=True)
        for i, key in enumerate(keys_list):
            metrics_by_lc[key]["reflection_index_norm"] = index_norm_list[i]

        logger.info(
            "ReflectionDepthValueEvolutionRepository: 完成综合指数 reflection_index_norm 的全局归一化。"
        )

        return metrics_by_lc
