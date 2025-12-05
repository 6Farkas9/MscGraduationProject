# BackEnd/app/engine/srl_helpseeking_engine.py
import logging
from typing import Dict, Any, List, Tuple, Optional
from math import sqrt, exp
from collections import defaultdict

from app.repositories.srl_helpseeking_repository import (
    srl_helpseeking_repository,
    SrlHelpSeekingRepository,
)

logger = logging.getLogger(__name__)

ACTIVITY_TYPE_BASE = "https://legend-meta.com/xapi/activity-type/"
QUESTION_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "item"
COURSE_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "course"


class SrlHelpSeekingEngine:
    """
    自我调节与求助策略分析引擎

    功能：
    - 给定一个或多个学习者 UID，从 Repository 读取细粒度 xAPI 行为；
    - 以 (学习者, 课程) 为单位计算：
        * SRL_help_index              自我调节与求助策略指数 [0,1]
        * help_need_rate              错误后求助比例
        * early_help_ratio            过早求助比例（首次作答前求助）
        * no_help_success_ratio       无帮助成功比例
        * feedback_density            反馈查看密度（按题目数归一）
        * extension_flag              是否使用扩展/补救单元
        * reflection_flag             是否提交过反思
        * cluster_rank                0/1/2，按 SRL_help_index 从低到高排序后的等级
        * cluster_label               "low" / "medium" / "high"
    - 对单个学习者的多门课程结果做聚合：
        * overall_score               多课程 SRL_help_index 的均值
        * overall_cluster_rank        多课程 cluster_rank 的众数，并列时选“更好”的一档
        * overall_label               综合中文描述标签
    """

    def __init__(self) -> None:
        logger.info("SrlHelpSeekingEngine 初始化完成")

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_ratio(num: float, den: float) -> float:
        if not den:
            return 0.0
        return float(num) / float(den)

    @staticmethod
    def _kmeans_1d(
        values: List[float],
        k: int = 3,
        max_iter: int = 50,
    ) -> Tuple[List[float], List[int]]:
        """
        一维 k-means 聚类（与分析脚本保持一致的实现思路）
        """
        if not values:
            return [], []

        unique_vals = sorted(set(values))
        if len(unique_vals) <= k:
            centers = unique_vals[:]
            while len(centers) < k:
                centers.append(unique_vals[-1])
        else:
            import random

            centers = random.sample(unique_vals, k)

        for _ in range(max_iter):
            clusters = [[] for _ in range(k)]
            for v in values:
                dists = [abs(v - c) for c in centers]
                idx = dists.index(min(dists))
                clusters[idx].append(v)

            new_centers: List[float] = []
            for i in range(k):
                if clusters[i]:
                    new_centers.append(sum(clusters[i]) / float(len(clusters[i])))
                else:
                    import random

                    new_centers.append(random.choice(values))

            if all(abs(a - b) < 1e-6 for a, b in zip(centers, new_centers)):
                centers = new_centers
                break
            centers = new_centers

        labels: List[int] = []
        for v in values:
            dists = [abs(v - c) for c in centers]
            idx = dists.index(min(dists))
            labels.append(idx)

        return centers, labels

    @staticmethod
    def _is_question_object(obj: Dict[str, Any]) -> bool:
        """
        判断 xAPI 对象是否为题目级活动。
        """
        if not obj:
            return False
        obj_id = obj.get("id") or ""
        if obj_id.startswith("https://legend-meta.com/item/"):
            return True
        definition = obj.get("definition") or {}
        tp = definition.get("type")
        if tp == QUESTION_ACTIVITY_TYPE:
            return True
        return False

    @staticmethod
    def _is_course_object(obj: Dict[str, Any]) -> bool:
        """
        判断 xAPI 对象是否为课程级活动。
        """
        if not obj:
            return False
        obj_id = obj.get("id") or ""
        if obj_id.startswith("https://legend-meta.com/course/"):
            return True
        definition = obj.get("definition") or {}
        tp = definition.get("type")
        if tp == COURSE_ACTIVITY_TYPE:
            return True
        return False

    # ------------------------------------------------------------------
    # 单课程 SRL_help_index 计算（移植自脚本 compute_srl_help_index_for_course）
    # ------------------------------------------------------------------
    def _compute_srl_help_index_for_course(
        self,
        course_stat: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        questions: Dict[str, Dict[str, Any]] = course_stat.get("questions") or {}
        feedback_question = course_stat.get("feedback_question", 0)
        feedback_course = course_stat.get("feedback_course", 0)
        extension_cnt = course_stat.get("extension_cnt", 0)
        reflection_cnt = course_stat.get("reflection_cnt", 0)

        # 没有任何题目行为时，给一个中性偏低的默认值
        if not questions:
            return 0.4, {
                "help_need_rate": 0.0,
                "early_help_ratio": 0.0,
                "no_help_success_ratio": 0.0,
                "feedback_density": 0.0,
                "extension_flag": 1 if extension_cnt > 0 else 0,
                "reflection_flag": 1 if reflection_cnt > 0 else 0,
            }

        num_questions = 0
        num_error_questions = 0
        total_wrong_attempts = 0
        total_help_events = 0
        total_help_after_error = 0
        total_help_before_attempt = 0

        num_success_questions = 0
        num_success_no_help_questions = 0
        num_error_then_success_no_help_questions = 0

        for q_id, qstat in questions.items():
            attempts = qstat.get("attempts", 0)
            wrong = qstat.get("wrong", 0)
            correct = qstat.get("correct", 0)
            help_total = qstat.get("help_total", 0)
            help_after_error = qstat.get("help_after_error", 0)
            help_before_attempt = qstat.get("help_before_attempt", 0)

            if attempts <= 0:
                continue

            num_questions += 1
            total_wrong_attempts += wrong
            total_help_events += help_total
            total_help_after_error += help_after_error
            total_help_before_attempt += help_before_attempt

            if wrong > 0:
                num_error_questions += 1

            if correct > 0:
                num_success_questions += 1
                if help_total == 0:
                    num_success_no_help_questions += 1
                    if wrong > 0:
                        num_error_then_success_no_help_questions += 1

        # 1) 错误后的求助比例
        if num_error_questions > 0:
            help_need_rate = self._safe_ratio(total_help_after_error, num_error_questions)
        else:
            help_need_rate = 0.5  # 几乎没有错误，用中性值

        # 2) 过早求助比例
        if total_help_events > 0:
            early_help_ratio = self._safe_ratio(
                total_help_before_attempt, total_help_events
            )
        else:
            early_help_ratio = 0.0

        # 3) 无帮助成功比例
        if num_success_questions > 0:
            no_help_success_ratio = self._safe_ratio(
                num_success_no_help_questions, num_success_questions
            )
        else:
            no_help_success_ratio = 0.0

        # 4) 反馈密度
        total_feedback = feedback_question + feedback_course
        if num_questions > 0:
            feedback_density = self._safe_ratio(total_feedback, num_questions)
        else:
            feedback_density = 0.0

        extension_flag = 1 if extension_cnt > 0 else 0
        reflection_flag = 1 if reflection_cnt > 0 else 0

        # ---------- 将上述特征映射到 [0,1] 子分数 ----------

        # 1) 适应性求助：help_need_rate 理想值 0.6，0.4～0.8 视为较好
        x = help_need_rate
        adaptivity_score = exp(-((x - 0.6) ** 2) / (2 * (0.3 ** 2)))
        adaptivity_score = max(0.0, min(1.0, adaptivity_score))

        # 2) 过早求助：越低越好
        early_score = 1.0 - early_help_ratio
        early_score = max(0.0, min(1.0, early_score))

        # 3) 独立解决：对 no_help_success_ratio 做轻微向 0.5 收缩
        independent_score = 0.5 + 0.5 * (no_help_success_ratio - 0.5)
        independent_score = max(0.0, min(1.0, independent_score))

        # 4) 反馈得分：以 0.5 次/题为上限线性归一
        feedback_score = min(feedback_density / 0.5, 1.0)

        # 5) 补救/反思：0/1
        extension_score = float(extension_flag)
        reflection_score = float(reflection_flag)

        # ---------- 综合加权 ----------
        w1 = 0.25  # 适应性求助
        w2 = 0.15  # 不过早依赖
        w3 = 0.25  # 独立解决
        w4 = 0.15  # 反馈使用
        w5 = 0.10  # 补救资源
        w6 = 0.10  # 反思行为

        total_weight = w1 + w2 + w3 + w4 + w5 + w6
        srl_index = (
            w1 * adaptivity_score
            + w2 * early_score
            + w3 * independent_score
            + w4 * feedback_score
            + w5 * extension_score
            + w6 * reflection_score
        ) / float(total_weight)

        srl_index = max(0.0, min(1.0, srl_index))

        feature_summary = {
            "help_need_rate": float(help_need_rate),
            "early_help_ratio": float(early_help_ratio),
            "no_help_success_ratio": float(no_help_success_ratio),
            "feedback_density": float(feedback_density),
            "extension_flag": int(extension_flag),
            "reflection_flag": int(reflection_flag),
        }

        return srl_index, feature_summary

    # ------------------------------------------------------------------
    # 从事件构建 (learner, course) 统计
    # ------------------------------------------------------------------
    def _build_course_stats(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        按 (learner, course) 聚合，并在组内按题目粒度统计求助模式与反馈/扩展/反思。
        """
        if not events:
            return {}

        verb_dict = SrlHelpSeekingRepository.VERBS

        events_by_lc: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for doc in events:
            lrn_uid = doc.get("_lrn_uid")
            crs_uid = doc.get("_course_uid")
            if not lrn_uid or not crs_uid:
                continue
            events_by_lc[(lrn_uid, crs_uid)].append(doc)

        # 每个 (learner, course) 按时间排序（用 timestamp 字符串即可）
        for key, evs in events_by_lc.items():
            evs.sort(key=lambda d: str(d.get("timestamp") or ""))

        course_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), evs in events_by_lc.items():
            stat = {
                "questions": defaultdict(
                    lambda: {
                        "attempts": 0,
                        "wrong": 0,
                        "correct": 0,
                        "help_total": 0,
                        "help_after_error": 0,
                        "help_before_attempt": 0,
                    }
                ),
                "feedback_question": 0,
                "feedback_course": 0,
                "extension_cnt": 0,
                "reflection_cnt": 0,
            }

            for doc in evs:
                verb = (doc.get("verb") or {}).get("id")
                obj = doc.get("object") or {}
                result = doc.get("result") or {}

                is_q = self._is_question_object(obj)
                is_c = self._is_course_object(obj)
                obj_id = obj.get("id") or ""

                # 1) 题目作答
                if verb == verb_dict["answered"] and is_q:
                    qstat = stat["questions"][obj_id]
                    qstat["attempts"] += 1
                    success = result.get("success")
                    if success is True:
                        qstat["correct"] += 1
                    else:
                        qstat["wrong"] += 1

                # 2) 求助事件
                elif verb == verb_dict["requested_support"] and is_q:
                    qstat = stat["questions"][obj_id]
                    qstat["help_total"] += 1
                    if qstat["attempts"] == 0:
                        qstat["help_before_attempt"] += 1
                    else:
                        if qstat["wrong"] > 0 and qstat["correct"] == 0:
                            qstat["help_after_error"] += 1

                # 3) 查看反馈（题目级/课程级）
                elif verb == verb_dict["reviewed_feedback"]:
                    if is_q:
                        stat["feedback_question"] += 1
                    elif is_c:
                        stat["feedback_course"] += 1
                    else:
                        stat["feedback_course"] += 1

                # 4) 使用扩展/补救单元
                elif verb == verb_dict["explored_extension"]:
                    stat["extension_cnt"] += 1

                # 5) 提交反思
                elif verb == verb_dict["reflected_on_activity"]:
                    stat["reflection_cnt"] += 1

            course_stats[(lrn_uid, crs_uid)] = stat

        logger.info(
            f"[SrlHelpSeekingEngine] 构建 (学习者, 课程) 粗粒度统计条目数: {len(course_stats)}"
        )
        return course_stats

    # ------------------------------------------------------------------
    # 计算 SRL_help_index 并聚类
    # ------------------------------------------------------------------
    def _compute_course_indices_and_cluster(
        self,
        course_stats: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        对每个 (learner, course) 计算 SRL_help_index 与特征，
        然后在全局上使用一维 k-means 聚类，得到 cluster_rank & cluster_label。
        """
        if not course_stats:
            return {}

        keys_list: List[Tuple[str, str]] = list(course_stats.keys())
        srl_indices: List[float] = []
        srl_results: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for key in keys_list:
            stat = course_stats[key]
            srl_index, feature_summary = self._compute_srl_help_index_for_course(stat)
            srl_results[key] = {
                "SRL_help_index": float(srl_index),
                "features": feature_summary,
                "cluster_rank": None,
                "cluster_label": None,
            }
            srl_indices.append(float(srl_index))

        if not srl_indices:
            logger.warning(
                "[SrlHelpSeekingEngine] 在聚合后没有可用的 SRL_help_index"
            )
            return srl_results

        centers, labels = self._kmeans_1d(srl_indices, k=3, max_iter=50)
        if not centers:
            logger.warning("[SrlHelpSeekingEngine] k-means 聚类失败，仅保留连续指数")
            return srl_results

        # 按中心从低到高排序 -> rank_map: 原始簇编号 -> 0/1/2
        sorted_centers = sorted(
            [(c, idx) for idx, c in enumerate(centers)], key=lambda x: x[0]
        )
        rank_map = {orig_idx: rank for rank, (c, orig_idx) in enumerate(sorted_centers)}

        def label_from_rank(rank: int) -> str:
            if rank == 0:
                return "low"
            if rank == 1:
                return "medium"
            return "high"

        # 按 keys_list 的顺序同步 labels
        for i, key in enumerate(keys_list):
            res = srl_results[key]
            if i >= len(labels):
                continue
            cluster_idx = labels[i]
            rank = rank_map.get(cluster_idx, 1)
            res["cluster_rank"] = int(rank)
            res["cluster_label"] = label_from_rank(rank)

        return srl_results

    # ------------------------------------------------------------------
    # 聚合到学习者级别
    # ------------------------------------------------------------------
    def _build_learner_summaries(
        self,
        srl_results: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        把 (学习者, 课程) 级别结果聚合为按学习者的结果。

        返回结构：
        {
            learner_uid: {
                "learner_uid": "...",
                "has_data": bool,
                "overall_score": float 或 None,
                "overall_label": str 或 None,
                "overall_cluster_rank": int 或 None,
                "per_course_results": [...],
            },
            ...
        }
        """
        learner_data: Dict[str, Dict[str, Any]] = {}

        for (lrn_uid, crs_uid), res in srl_results.items():
            if "SRL_help_index" not in res:
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

            feat = res.get("features") or {}

            item = {
                "course_uid": crs_uid,
                "SRL_help_index": float(res.get("SRL_help_index", 0.0)),
                "cluster_rank": int(res.get("cluster_rank", 1))
                if res.get("cluster_rank") is not None
                else None,
                "cluster_label": res.get("cluster_label"),
                "help_need_rate": float(feat.get("help_need_rate", 0.0)),
                "early_help_ratio": float(feat.get("early_help_ratio", 0.0)),
                "no_help_success_ratio": float(
                    feat.get("no_help_success_ratio", 0.0)
                ),
                "feedback_density": float(feat.get("feedback_density", 0.0)),
                "extension_flag": int(feat.get("extension_flag", 0)),
                "reflection_flag": int(feat.get("reflection_flag", 0)),
            }
            learner_data[lrn_uid]["per_course_results"].append(item)

        overall_rank_label = {
            0: "整体自我调节与求助策略水平偏低（在需要时较少求助或求助模式不够合理，反馈/补救/反思使用有限）。",
            1: "整体自我调节与求助策略水平中等（在部分困难情境下能适度求助，并有一定程度的反馈与补救使用）。",
            2: "整体自我调节与求助策略水平较高（能在遇到困难时适度求助，同时主动使用反馈、补救资源与反思工具）。",
        }

        for lrn_uid, info in learner_data.items():
            pcs = info["per_course_results"]
            if not pcs:
                info["has_data"] = False
                continue

            scores = [it["SRL_help_index"] for it in pcs]
            info["overall_score"] = float(sum(scores) / float(len(scores)))

            rank_counts: Dict[int, int] = {}
            for it in pcs:
                rnk = it.get("cluster_rank")
                if rnk is None:
                    continue
                rnk = int(rnk)
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
                    "整体自我调节与求助策略水平中等（默认）。",
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
        对多个学习者进行“自我调节与求助策略”分析。

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
            events = srl_helpseeking_repository.get_srl_helpseeking_events(
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

            course_stats = self._build_course_stats(events)
            if not course_stats:
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

            srl_results = self._compute_course_indices_and_cluster(course_stats)
            if not srl_results:
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

            learner_summaries = self._build_learner_summaries(srl_results)

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
            logger.error(f"多学习者自我调节与求助策略分析失败: {e}", exc_info=True)
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
_srl_help_engine_instance: Optional[SrlHelpSeekingEngine] = None


def get_srl_helpseeking_engine() -> SrlHelpSeekingEngine:
    global _srl_help_engine_instance
    if _srl_help_engine_instance is None:
        _srl_help_engine_instance = SrlHelpSeekingEngine()
    return _srl_help_engine_instance


def analyze_single_learner(learner_uid: str) -> Dict[str, Any]:
    engine = get_srl_helpseeking_engine()
    return engine.analyze_single_learner(learner_uid)


def analyze_multiple_learners(
    learner_uids: List[str],
) -> Dict[str, Dict[str, Any]]:
    engine = get_srl_helpseeking_engine()
    return engine.analyze_multiple_learners(learner_uids)


# 简单本地测试（使用与其它 engine 相同的测试 UID）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = SrlHelpSeekingEngine()
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
