# -*- coding: utf-8 -*-
"""
脚本功能：
1. 从 MySQL.mls 读取粗粒度行为数据及相关表。
2. 根据粗粒度行为，按课程窗口为每个学习者倒推一部分画像维度，
   其他维度采用可复现随机+轻微调节的人设。
3. 根据人设，将粗粒度 Interaction 数据拓展为细粒度的 xAPI 行为序列。
4. 将人设写入 MongoDB.MLS.LearnerProfile，将细粒度行为写入 MongoDB.MLS.Interaction。
5. 使用 tqdm 显示学习者处理进度。
6. 增加防“卡死”机制：
   - 异常大数据的学习者进行抽样/舍弃；
   - 对单条记录的时长/次数做清洗；
   - 分批写入 Mongo，避免一次性堆积。
"""

import pymysql
from pymongo import MongoClient, ASCENDING
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
import random

# ===================== 配置区域 =====================

# MySQL Interaction 表主键，用作会话 ID（bigint）
INTERACTION_PK_FIELD = "id"

MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "db": "mls",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

MONGO_CONFIG = {
    "host": "localhost",
    "port": 27017,
    "db_name": "MLS",
    "profile_collection": "LearnerProfile",
    "xapi_collection": "Interaction",
}

# 批量写入大小
PROFILE_BATCH_SIZE = 1000
XAPI_BATCH_SIZE = 50000

# 学习者粗粒度交互数量阈值
MAX_INTERACTIONS_SOFT = 50000   # 超过此值开始抽样
MAX_INTERACTIONS_HARD = 200000  # 超过此值直接舍弃该学习者

# 单条记录时长裁剪上限（秒）
MAX_DURATION_SECONDS = 8 * 3600

# xAPI 常量（使用最终版语义体系）
VERB_BASE = "https://legend-meta.com/xapi/verb/"
ACTIVITY_TYPE_BASE = "https://legend-meta.com/xapi/activity-type/"

VERBS = {
    # 任务与结果行为
    "experienced": VERB_BASE + "experienced",
    "initialized": VERB_BASE + "initialized",
    "completed": VERB_BASE + "completed",
    "answered": VERB_BASE + "answered",
    "passed": VERB_BASE + "passed",
    "failed": VERB_BASE + "failed",

    # 空间行为
    "navigated_to_space": VERB_BASE + "navigated-to-space",
    "teleported_to_space": VERB_BASE + "teleported-to-space",

    # 对象操作行为
    "manipulated_object": VERB_BASE + "manipulated-object",
    "performed_procedure_step": VERB_BASE + "performed-procedure-step",
    "contributed_resource": VERB_BASE + "contributed-resource",
    "exchanged_value": VERB_BASE + "exchanged-value",

    # 注意、状态与认知加工行为
    "focused_on_resource": VERB_BASE + "focused-on-resource",
    "reviewed_feedback": VERB_BASE + "reviewed-feedback",
    "explored_extension": VERB_BASE + "explored-extension",
    "reflected_on_activity": VERB_BASE + "reflected-on-activity",
    "remained_idle": VERB_BASE + "remained-idle",

    # 协作与社会交互行为
    "collaborated_on_activity": VERB_BASE + "collaborated-on-activity",
    "co_edited_artifact": VERB_BASE + "co-edited-artifact",
    "observed_peer": VERB_BASE + "observed-peer",
    "requested_support": VERB_BASE + "requested-support",
}

UNIT_ACTIVITY_TYPES = {
    "video": ACTIVITY_TYPE_BASE + "unit/video",
    "vr": ACTIVITY_TYPE_BASE + "unit/vr",
    "ar": ACTIVITY_TYPE_BASE + "unit/ar",
    "interact": ACTIVITY_TYPE_BASE + "unit/interact",
    "cooperate": ACTIVITY_TYPE_BASE + "unit/cooperate",
}

QUESTION_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "item"
COURSE_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "course"
KP_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "knowledge-point"
TOPIC_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "topic"
DOMAIN_ACTIVITY_TYPE = ACTIVITY_TYPE_BASE + "domain"

# 画像维度（不包含知识掌握）
PROFILE_DIMENSIONS = [
    "task_efficiency",
    "attention_allocation",
    "engagement",
    "perseverance",
    "collaboration",
    "feedback_orientation",
    "exploration_orientation",
    "social_learning",
    "value_contribution",
    "reflection_depth",
    "interaction_style",
]


# ===================== 工具函数 =====================

def get_mysql_connection():
    return pymysql.connect(**MYSQL_CONFIG)


def get_mongo_db():
    client = MongoClient(MONGO_CONFIG["host"], MONGO_CONFIG["port"])
    return client[MONGO_CONFIG["db_name"]]


def ratio_safe(n, d):
    if not d:
        return 0.0
    return float(n) / float(d)


def categorize_score(r):
    if r < 0.4:
        return "low"
    elif r < 0.7:
        return "medium"
    else:
        return "high"


def split_duration(total_seconds, n_parts):
    """用固定权重拆时长，避免强随机。"""
    if total_seconds is None or total_seconds <= 0 or n_parts <= 0:
        return [0] * n_parts

    base_weights = [0.4, 0.3, 0.2, 0.1]
    if n_parts <= len(base_weights):
        weights = base_weights[:n_parts]
    else:
        extra = n_parts - len(base_weights)
        weights = base_weights + [0.0] * extra
        remaining = 1.0 - sum(base_weights)
        if remaining < 0:
            remaining = 0.0
        for i in range(extra):
            weights[len(base_weights) + i] = remaining / extra

    total_weight = sum(weights)
    if total_weight <= 0:
        return [0] * n_parts

    weights = [w / total_weight for w in weights]
    parts = [total_seconds * w for w in weights]
    parts = [int(round(p)) for p in parts]
    diff = int(total_seconds) - sum(parts)
    if diff != 0 and parts:
        parts[0] += diff
    return parts


def get_rng_for_learner(lrn_uid):
    seed = abs(hash(lrn_uid)) % (2 ** 32)
    return random.Random(seed)


def sanitize_duration(value, max_seconds=MAX_DURATION_SECONDS):
    if value is None:
        return 0.0
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0:
        v = 0.0
    if v > max_seconds:
        v = float(max_seconds)
    return v


def sanitize_attempt_index(value, min_attempt=1, max_attempt=20):
    if value is None:
        return min_attempt
    try:
        v = int(value)
    except Exception:
        return min_attempt
    if v < min_attempt:
        v = min_attempt
    if v > max_attempt:
        v = max_attempt
    return v


# ===================== 数据加载 =====================

def load_basic_data(conn):
    """
    加载：
      - BasicLearners
      - Units
      - Questions
      - Courses
      - Course_Unit
      - Interaction
    并建立必要映射。
    """
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM BasicLearners")
        learners = cursor.fetchall()

        cursor.execute("SELECT * FROM Units")
        units = cursor.fetchall()

        cursor.execute("SELECT * FROM Questions")
        questions = cursor.fetchall()

        cursor.execute("SELECT * FROM Courses")
        courses = cursor.fetchall()

        cursor.execute("SELECT * FROM Course_Unit")
        course_units = cursor.fetchall()

        cursor.execute("SELECT * FROM Interaction")
        interactions = cursor.fetchall()

    learners_by_uid = {row["uid"]: row for row in learners}
    units_by_uid = {row["uid"]: row for row in units}
    units_by_oid = {row["oid"]: row for row in units}
    questions_by_uid = {row["uid"]: row for row in questions}
    courses_by_uid = {row["uid"]: row for row in courses}

    # 单元 -> 课程
    unit_to_course = {}
    for cu in course_units:
        unit_to_course[cu["unt_uid"]] = cu["crs_uid"]

    # 题目 -> 课程（通过 oid 去掉 _qus 再匹配 Units）
    question_to_course = {}
    for q in questions:
        oid = q["oid"]
        if oid and oid.endswith("_qus"):
            base_oid = oid[:-4]
        else:
            base_oid = oid
        unit = units_by_oid.get(base_oid)
        if unit:
            unit_uid = unit["uid"]
            crs_uid = unit_to_course.get(unit_uid)
            if crs_uid:
                question_to_course[q["uid"]] = crs_uid

    return {
        "learners_by_uid": learners_by_uid,
        "units_by_uid": units_by_uid,
        "questions_by_uid": questions_by_uid,
        "courses_by_uid": courses_by_uid,
        "unit_to_course": unit_to_course,
        "question_to_course": question_to_course,
        "interactions": interactions,
    }


# ===================== 粗粒度统计 =====================

def build_stats_from_interactions(interactions, units_by_uid, question_to_course, unit_to_course):
    stats_per_learner_course = defaultdict(lambda: {
        "video_total_len": 0.0,
        "video_watch": 0.0,
        "vr_total": 0.0,
        "vr_focus": 0.0,
        "ar_total": 0.0,
        "ar_focus": 0.0,
        "interact_total": 0.0,
        "interact_correct": 0.0,
        "cooperate_total": 0.0,
        "cooperate_effective": 0.0,
        "question_attempts": 0,
        "question_correct": 0,
        "question_wrong": 0,
        "question_retry_after_wrong": 0,
        "unit_counts": defaultdict(int),
        "total_interactions": 0,
    })

    question_records = defaultdict(list)
    interactions_by_learner = defaultdict(list)
    learner_course_units = defaultdict(set)

    for row in interactions:
        lrn_uid = row["lrn_uid"]
        unt_uid = row["unt_uid"]
        add1 = row["additioninfo1"]
        add2 = row["additioninfo2"]
        ctime = row["create_time"]

        interactions_by_learner[lrn_uid].append(row)

        is_unit = unt_uid in units_by_uid
        if is_unit:
            unit = units_by_uid[unt_uid]
            utype = (unit["type"] or "").lower()
            crs_uid = unit_to_course.get(unt_uid)
        else:
            utype = "question"
            crs_uid = question_to_course.get(unt_uid)

        if not crs_uid:
            continue

        stats = stats_per_learner_course[(lrn_uid, crs_uid)]
        stats["total_interactions"] += 1

        if is_unit:
            learner_course_units[(lrn_uid, crs_uid)].add(unt_uid)
            stats["unit_counts"][utype] += 1

            dur1 = sanitize_duration(add1)
            dur2 = sanitize_duration(add2)

            if utype == "video":
                stats["video_total_len"] += dur1
                stats["video_watch"] += min(dur2, dur1)
            elif utype == "vr":
                stats["vr_total"] += dur1
                stats["vr_focus"] += min(dur2, dur1)
            elif utype == "ar":
                stats["ar_total"] += dur1
                stats["ar_focus"] += min(dur2, dur1)
            elif utype == "interact":
                stats["interact_total"] += dur1
                stats["interact_correct"] += min(dur2, dur1)
            elif utype == "cooperate":
                stats["cooperate_total"] += dur1
                stats["cooperate_effective"] += min(dur2, dur1)
        else:
            attempt_idx = sanitize_attempt_index(add1)
            stats["question_attempts"] += 1
            if add2 and add2 > 0:
                stats["question_correct"] += 1
            else:
                stats["question_wrong"] += 1

            question_records[(lrn_uid, unt_uid)].append((ctime, attempt_idx, add2))

    # 处理 wrong -> later correct
    for (lrn_uid, q_uid), recs in question_records.items():
        recs_sorted = sorted(recs, key=lambda x: x[0])
        had_wrong = False
        ever_correct_after_wrong = False
        for (ctime, attempt_index, correct_flag) in recs_sorted:
            if not correct_flag or correct_flag <= 0:
                had_wrong = True
            elif correct_flag > 0 and had_wrong:
                ever_correct_after_wrong = True
                break
        if ever_correct_after_wrong:
            crs_uid = question_to_course.get(q_uid)
            if crs_uid:
                stats_per_learner_course[(lrn_uid, crs_uid)]["question_retry_after_wrong"] += 1

    return stats_per_learner_course, interactions_by_learner, learner_course_units


# ===================== 人设推断 =====================

def infer_persona_for_course(stats, rng):
    interact_ratio = ratio_safe(stats["interact_correct"], stats["interact_total"])
    vr_ratio = ratio_safe(stats["vr_focus"], stats["vr_total"])
    ar_ratio = ratio_safe(stats["ar_focus"], stats["ar_total"])
    video_ratio = ratio_safe(stats["video_watch"], stats["video_total_len"])

    efficiency_components = [x for x in [interact_ratio, vr_ratio, ar_ratio, video_ratio] if x > 0]
    if efficiency_components:
        task_eff_score = sum(efficiency_components) / len(efficiency_components)
    else:
        task_eff_score = 0.5

    att_components = [x for x in [vr_ratio, ar_ratio] if x > 0]
    if att_components:
        att_score = sum(att_components) / len(att_components)
    else:
        att_score = video_ratio if video_ratio > 0 else 0.5

    time_sum = (
        stats["video_watch"]
        + stats["vr_total"]
        + stats["ar_total"]
        + stats["interact_total"]
        + stats["cooperate_total"]
    )
    engagement_score = min(time_sum / 3600.0, 1.0) if time_sum > 0 else 0.5

    perseverance_score = ratio_safe(stats["question_retry_after_wrong"], stats["question_wrong"])
    if stats["question_wrong"] == 0:
        perseverance_score = 0.6

    collaboration_score = ratio_safe(stats["cooperate_effective"], stats["cooperate_total"])
    if stats["cooperate_total"] == 0:
        collaboration_score = 0.4

    persona = {}

    persona["task_efficiency"] = {
        "score": min(max(task_eff_score, 0.0), 1.0),
        "level": categorize_score(task_eff_score),
        "source": "inferred"
    }
    persona["attention_allocation"] = {
        "score": min(max(att_score, 0.0), 1.0),
        "level": categorize_score(att_score),
        "source": "inferred"
    }
    persona["engagement"] = {
        "score": min(max(engagement_score, 0.0), 1.0),
        "level": categorize_score(engagement_score),
        "source": "inferred"
    }
    persona["perseverance"] = {
        "score": min(max(perseverance_score, 0.0), 1.0),
        "level": categorize_score(perseverance_score),
        "source": "inferred"
    }
    persona["collaboration"] = {
        "score": min(max(collaboration_score, 0.0), 1.0),
        "level": categorize_score(collaboration_score),
        "source": "inferred"
    }

    def rand_score(base=None):
        s = rng.uniform(0.25, 0.85)
        if base is not None:
            s = 0.7 * s + 0.3 * base
        return min(max(s, 0.0), 1.0)

    fb_score = rand_score((task_eff_score + perseverance_score) / 2.0)
    persona["feedback_orientation"] = {
        "score": fb_score,
        "level": categorize_score(fb_score),
        "source": "random"
    }

    exp_score = rand_score(engagement_score)
    persona["exploration_orientation"] = {
        "score": exp_score,
        "level": categorize_score(exp_score),
        "source": "random"
    }

    soc_score = rand_score(collaboration_score)
    persona["social_learning"] = {
        "score": soc_score,
        "level": categorize_score(soc_score),
        "source": "random"
    }

    vc_base = (engagement_score + collaboration_score) / 2.0
    vc_score = rand_score(vc_base)
    persona["value_contribution"] = {
        "score": vc_score,
        "level": categorize_score(vc_score),
        "source": "random"
    }

    ref_score = rand_score(perseverance_score)
    persona["reflection_depth"] = {
        "score": ref_score,
        "level": categorize_score(ref_score),
        "source": "random"
    }

    interact_sum = stats["interact_total"]
    other_sum = (
        stats["video_watch"]
        + stats["vr_total"]
        + stats["ar_total"]
        + stats["cooperate_total"]
    )
    inter_ratio = ratio_safe(interact_sum, interact_sum + other_sum)
    style_score = rand_score(inter_ratio)
    persona["interaction_style"] = {
        "score": style_score,
        "level": categorize_score(style_score),
        "source": "mixed"
    }

    for dim in PROFILE_DIMENSIONS:
        if dim not in persona:
            s = rand_score()
            persona[dim] = {
                "score": s,
                "level": categorize_score(s),
                "source": "random"
            }

    return persona


def aggregate_global_profile(course_profiles):
    if not course_profiles:
        return {}

    dim_scores = defaultdict(list)
    for cp in course_profiles:
        persona = cp["persona"]
        for dim_name, val in persona.items():
            dim_scores[dim_name].append(val["score"])

    global_profile = {}
    for dim_name, scores in dim_scores.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        global_profile[dim_name] = {
            "score": avg,
            "level": categorize_score(avg)
        }

    return global_profile


# ===================== xAPI 帮助函数 =====================

def make_actor(lrn_uid):
    return {
        "objectType": "Agent",
        "account": {
            "homePage": "https://legend-meta.com/learner",
            "name": lrn_uid,
        },
    }


def make_unit_activity(unit_uid, unit_type, name=None):
    return {
        "objectType": "Activity",
        "id": f"https://legend-meta.com/unit/{unit_uid}",
        "definition": {
            "name": {"zh-CN": name or unit_uid},
            "type": UNIT_ACTIVITY_TYPES.get(unit_type, ACTIVITY_TYPE_BASE + "unit/other"),
        },
    }


def make_question_activity(question_uid):
    return {
        "objectType": "Activity",
        "id": f"https://legend-meta.com/item/{question_uid}",
        "definition": {
            "name": {"zh-CN": question_uid},
            "type": QUESTION_ACTIVITY_TYPE,
        },
    }


def make_course_activity(course_uid, name=None):
    return {
        "objectType": "Activity",
        "id": f"https://legend-meta.com/course/{course_uid}",
        "definition": {
            "name": {"zh-CN": name or course_uid},
            "type": COURSE_ACTIVITY_TYPE,
        },
    }


def make_context(course_uid=None, unit_type=None):
    parent = []
    if course_uid:
        parent.append({
            "id": f"https://legend-meta.com/course/{course_uid}",
            "definition": {
                "type": COURSE_ACTIVITY_TYPE
            }
        })

    ctx = {
        "contextActivities": {
            "parent": parent,
            "grouping": [],
        },
        "extensions": {}
    }
    if unit_type:
        ctx["extensions"]["https://legend-meta.com/xapi/ext/unit-type"] = unit_type

    return ctx


# ===================== 细粒度行为生成 =====================

def generate_xapi_for_video(row, unit, course_uid, persona_course):
    lrn_uid = row["lrn_uid"]
    unit_uid = row["unt_uid"]
    total_len_raw = row["additioninfo1"]
    watch_len_raw = row["additioninfo2"]

    total_len = sanitize_duration(total_len_raw)
    watch_len = sanitize_duration(watch_len_raw, max_seconds=total_len or MAX_DURATION_SECONDS)

    base_time = row["create_time"] or datetime.utcnow()

    actor = make_actor(lrn_uid)
    activity = make_unit_activity(unit_uid, "video", unit["name"])
    context = make_context(course_uid, "video")

    events = []

    # experienced
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["experienced"], "display": {"en": "experienced"}},
        "object": activity,
        "context": context,
        "timestamp": (base_time - timedelta(seconds=int(watch_len) + 10)).isoformat(),
        "result": {}
    })

    # initialized
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["initialized"], "display": {"en": "initialized"}},
        "object": activity,
        "context": context,
        "timestamp": (base_time - timedelta(seconds=int(watch_len))).isoformat(),
        "result": {}
    })

    # remained_idle（低注意力 + 有明显空闲）
    attention_score = persona_course.get("attention_allocation", {}).get("score", 0.5)
    idle_time = max(total_len - watch_len, 0.0)
    if idle_time > 5 and attention_score < 0.4:
        ctx_idle = make_context(course_uid, "video")
        ctx_idle["extensions"]["https://legend-meta.com/xapi/ext/idle-threshold-seconds"] = 5
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["remained_idle"], "display": {"en": "remained idle"}},
            "object": activity,
            "context": ctx_idle,
            "timestamp": (base_time - timedelta(seconds=int(watch_len) // 2)).isoformat(),
            "result": {
                "duration": f"PT{int(idle_time)}S"
            }
        })

    # focused_on_resource（分段聚焦 AOI）
    n_focus = 3
    parts = split_duration(watch_len, n_focus)
    aoi_ids = ["main-screen", "subtitle-area", "diagram-area"]
    cursor_time = base_time - timedelta(seconds=int(watch_len))

    for i in range(n_focus):
        dur = parts[i]
        if dur <= 0:
            continue
        end_time = cursor_time + timedelta(seconds=dur)
        cursor_time = end_time
        ctx_focus = make_context(course_uid, "video")
        ctx_focus["extensions"]["https://legend-meta.com/xapi/ext/focus-target-id"] = aoi_ids[i % len(aoi_ids)]
        ctx_focus["extensions"]["https://legend-meta.com/xapi/ext/focus-method"] = "viewport"

        events.append({
            "actor": actor,
            "verb": {"id": VERBS["focused_on_resource"], "display": {"en": "focused on"}},
            "object": activity,
            "context": ctx_focus,
            "timestamp": end_time.isoformat(),
            "result": {
                "duration": f"PT{int(dur)}S"
            }
        })

    # completed
    ctx_completed = make_context(course_uid, "video")
    ctx_completed["extensions"]["https://legend-meta.com/xapi/ext/video-total-length"] = total_len
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["completed"], "display": {"en": "completed"}},
        "object": activity,
        "context": ctx_completed,
        "timestamp": base_time.isoformat(),
        "result": {
            "completion": True,
            "duration": f"PT{int(watch_len)}S"
        }
    })

    return events


def generate_xapi_for_vr_ar(row, unit, course_uid, persona_course, unit_type):
    lrn_uid = row["lrn_uid"]
    unit_uid = row["unt_uid"]
    total_raw = row["additioninfo1"]
    focus_raw = row["additioninfo2"]

    total = sanitize_duration(total_raw)
    focus = sanitize_duration(focus_raw, max_seconds=total or MAX_DURATION_SECONDS)

    base_time = row["create_time"] or datetime.utcnow()

    actor = make_actor(lrn_uid)
    activity = make_unit_activity(unit_uid, unit_type, unit["name"])
    context = make_context(course_uid, unit_type)

    events = []

    # experienced
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["experienced"], "display": {"en": "experienced"}},
        "object": activity,
        "context": context,
        "timestamp": (base_time - timedelta(seconds=int(total) + 20)).isoformat(),
        "result": {}
    })

    # initialized
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["initialized"], "display": {"en": "initialized"}},
        "object": activity,
        "context": context,
        "timestamp": (base_time - timedelta(seconds=int(total))).isoformat(),
        "result": {}
    })

    # 探索倾向决定是否使用 teleported-to-space
    exp_score = persona_course.get("exploration_orientation", {}).get("score", 0.5)
    n_spaces = 2 if exp_score < 0.5 else 3
    space_ids = [f"{unit_type}-space-{i+1}" for i in range(n_spaces)]
    nav_parts = split_duration(total, n_spaces)
    cursor_time = base_time - timedelta(seconds=int(total))
    for i, sid in enumerate(space_ids):
        dur = nav_parts[i]
        if dur <= 0:
            continue
        end_time = cursor_time + timedelta(seconds=dur)
        cursor_time = end_time
        ctx_nav = make_context(course_uid, unit_type)
        ctx_nav["extensions"]["https://legend-meta.com/xapi/ext/space-id"] = sid

        # ★ 新逻辑：高探索倾向时，第一个空间跳转采用 teleported-to-space
        if exp_score >= 0.5 and i == 0:
            verb_id = VERBS["teleported_to_space"]
            display = {"en": "teleported to"}
            ctx_nav["extensions"]["https://legend-meta.com/xapi/ext/navigation-mode"] = "teleport"
        else:
            verb_id = VERBS["navigated_to_space"]
            display = {"en": "navigated to"}
            ctx_nav["extensions"]["https://legend-meta.com/xapi/ext/navigation-mode"] = "walk"

        events.append({
            "actor": actor,
            "verb": {"id": verb_id, "display": display},
            "object": activity,
            "context": ctx_nav,
            "timestamp": end_time.isoformat(),
            "result": {}
        })

    # focused_on_resource
    n_focus = 3
    parts_focus = split_duration(focus, n_focus)
    cursor_time = base_time - timedelta(seconds=int(total))
    for i in range(n_focus):
        dur = parts_focus[i]
        if dur <= 0:
            continue
        end_time = cursor_time + timedelta(seconds=dur)
        cursor_time = end_time
        ctx_focus = make_context(course_uid, unit_type)
        ctx_focus["extensions"]["https://legend-meta.com/xapi/ext/focus-target-id"] = f"{unit_type}-object-{i+1}"
        ctx_focus["extensions"]["https://legend-meta.com/xapi/ext/focus-method"] = "gaze"
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["focused_on_resource"], "display": {"en": "focused on"}},
            "object": activity,
            "context": ctx_focus,
            "timestamp": end_time.isoformat(),
            "result": {
                "duration": f"PT{int(dur)}S"
            }
        })

    # completed
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["completed"], "display": {"en": "completed"}},
        "object": activity,
        "context": context,
        "timestamp": base_time.isoformat(),
        "result": {
            "completion": True,
            "duration": f"PT{int(total)}S"
        }
    })

    return events


def generate_xapi_for_interact(row, unit, course_uid, persona_course):
    lrn_uid = row["lrn_uid"]
    unit_uid = row["unt_uid"]
    total_raw = row["additioninfo1"]
    correct_raw = row["additioninfo2"]

    total = sanitize_duration(total_raw)
    correct_time = sanitize_duration(correct_raw, max_seconds=total or MAX_DURATION_SECONDS)

    base_time = row["create_time"] or datetime.utcnow()

    actor = make_actor(lrn_uid)
    activity = make_unit_activity(unit_uid, "interact", unit["name"])
    context = make_context(course_uid, "interact")

    events = []

    # experienced
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["experienced"], "display": {"en": "experienced"}},
        "object": activity,
        "context": context,
        "timestamp": (base_time - timedelta(seconds=int(total) + 20)).isoformat(),
        "result": {}
    })
    # initialized
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["initialized"], "display": {"en": "initialized"}},
        "object": activity,
        "context": context,
        "timestamp": (base_time - timedelta(seconds=int(total))).isoformat(),
        "result": {}
    })

    eff_score = persona_course.get("task_efficiency", {}).get("score", 0.5)

    n_steps = 4
    parts = split_duration(total, n_steps)
    cursor_time = base_time - timedelta(seconds=int(total))
    for i in range(n_steps):
        dur = parts[i]
        if dur <= 0:
            continue
        end_time = cursor_time + timedelta(seconds=dur)
        cursor_time = end_time

        ctx_step = make_context(course_uid, "interact")
        ctx_step["extensions"]["https://legend-meta.com/xapi/ext/step-id"] = f"step-{i+1}"
        ctx_step["extensions"]["https://legend-meta.com/xapi/ext/step-order"] = i + 1

        success_flag = eff_score >= 0.4

        events.append({
            "actor": actor,
            "verb": {"id": VERBS["performed_procedure_step"], "display": {"en": "performed step"}},
            "object": activity,
            "context": ctx_step,
            "timestamp": end_time.isoformat(),
            "result": {
                "success": success_flag,
                "duration": f"PT{int(dur)}S"
            }
        })

        inter_style = persona_course.get("interaction_style", {}).get("score", 0.5)
        if inter_style > 0.6:
            ctx_obj = make_context(course_uid, "interact")
            ctx_obj["extensions"]["https://legend-meta.com/xapi/ext/object-id"] = f"tool-{i+1}"
            ctx_obj["extensions"]["https://legend-meta.com/xapi/ext/object-action"] = "activate"
            events.append({
                "actor": actor,
                "verb": {"id": VERBS["manipulated_object"], "display": {"en": "manipulated"}},
                "object": activity,
                "context": ctx_obj,
                "timestamp": (end_time - timedelta(seconds=1)).isoformat(),
                "result": {}
            })

    success_flag = eff_score >= 0.5
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["completed"], "display": {"en": "completed"}},
        "object": activity,
        "context": context,
        "timestamp": base_time.isoformat(),
        "result": {
            "completion": True,
            "success": success_flag,
            "duration": f"PT{int(total)}S"
        }
    })

    return events


def generate_xapi_for_cooperate(row, unit, course_uid, persona_course):
    lrn_uid = row["lrn_uid"]
    unit_uid = row["unt_uid"]
    total_raw = row["additioninfo1"]
    effective_raw = row["additioninfo2"]

    total = sanitize_duration(total_raw)
    effective = sanitize_duration(effective_raw, max_seconds=total or MAX_DURATION_SECONDS)

    base_time = row["create_time"] or datetime.utcnow()

    actor = make_actor(lrn_uid)
    activity = make_unit_activity(unit_uid, "cooperate", unit["name"])
    context = make_context(course_uid, "cooperate")

    events = []

    # experienced
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["experienced"], "display": {"en": "experienced"}},
        "object": activity,
        "context": context,
        "timestamp": (base_time - timedelta(seconds=int(total) + 20)).isoformat(),
        "result": {}
    })
    # initialized
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["initialized"], "display": {"en": "initialized"}},
        "object": activity,
        "context": context,
        "timestamp": (base_time - timedelta(seconds=int(total))).isoformat(),
        "result": {}
    })

    coll_score = persona_course.get("collaboration", {}).get("score", 0.5)

    # collaborated_on_activity
    ctx_collab = make_context(course_uid, "cooperate")
    collab_time = base_time - timedelta(seconds=int(total) // 2)
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["collaborated_on_activity"], "display": {"en": "collaborated on"}},
        "object": activity,
        "context": ctx_collab,
        "timestamp": collab_time.isoformat(),
        "result": {
            "duration": f"PT{int(effective)}S"
        }
    })

    # ★ 新逻辑：高协作 / 社会学习 / 贡献倾向时，补 co-edited-artifact
    co_edit_score = max(
        coll_score,
        persona_course.get("social_learning", {}).get("score", 0.5),
        persona_course.get("value_contribution", {}).get("score", 0.5),
    )
    if co_edit_score >= 0.5:
        ctx_edit = make_context(course_uid, "cooperate")
        ctx_edit["extensions"]["https://legend-meta.com/xapi/ext/artifact-id"] = "shared-artifact-1"
        ctx_edit["extensions"]["https://legend-meta.com/xapi/ext/artifact-type"] = "co-edited-document"
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["co_edited_artifact"], "display": {"en": "co-edited artifact"}},
            "object": activity,
            "context": ctx_edit,
            # 共同编辑通常发生在协作过程稍后
            "timestamp": (collab_time + timedelta(seconds=10)).isoformat(),
            "result": {}
        })

    # remained_idle
    idle = total - effective
    if idle > 10 and coll_score < 0.6:
        ctx_idle = make_context(course_uid, "cooperate")
        ctx_idle["extensions"]["https://legend-meta.com/xapi/ext/idle-threshold-seconds"] = 10
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["remained_idle"], "display": {"en": "remained idle"}},
            "object": activity,
            "context": ctx_idle,
            "timestamp": (base_time - timedelta(seconds=5)).isoformat(),
            "result": {
                "duration": f"PT{int(idle)}S"
            }
        })

    # completed
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["completed"], "display": {"en": "completed"}},
        "object": activity,
        "context": context,
        "timestamp": base_time.isoformat(),
        "result": {
            "completion": True,
            "duration": f"PT{int(total)}S"
        }
    })

    return events


def generate_xapi_for_question(row, course_uid, persona_course):
    lrn_uid = row["lrn_uid"]
    q_uid = row["unt_uid"]

    attempt_index = sanitize_attempt_index(row["additioninfo1"])
    is_correct = bool(row["additioninfo2"] and row["additioninfo2"] > 0)
    base_time = row["create_time"] or datetime.utcnow()

    actor = make_actor(lrn_uid)
    activity = make_question_activity(q_uid)
    context = make_context(course_uid)

    events = []

    # 首次尝试前的 experienced & initialized（只在 attempt_index==1）
    if attempt_index == 1:
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["experienced"], "display": {"en": "experienced"}},
            "object": activity,
            "context": context,
            "timestamp": (base_time - timedelta(seconds=20)).isoformat(),
            "result": {}
        })
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["initialized"], "display": {"en": "initialized"}},
            "object": activity,
            "context": context,
            "timestamp": (base_time - timedelta(seconds=10)).isoformat(),
            "result": {}
        })

    # answered
    response_str = "correct" if is_correct else "incorrect"
    events.append({
        "actor": actor,
        "verb": {"id": VERBS["answered"], "display": {"en": "answered"}},
        "object": activity,
        "context": context,
        "timestamp": base_time.isoformat(),
        "result": {
            "response": response_str,
            "success": is_correct
        }
    })

    # ★ 新逻辑：根据答题结果，补 passed / failed
    perseverance_score = persona_course.get("perseverance", {}).get("score", 0.5)
    assess_ctx = make_context(course_uid)
    assess_ctx["extensions"]["https://legend-meta.com/xapi/ext/attempt-index"] = attempt_index
    assess_ctx["extensions"]["https://legend-meta.com/xapi/ext/perseverance-score"] = perseverance_score

    if is_correct:
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["passed"], "display": {"en": "passed"}},
            "object": activity,
            "context": assess_ctx,
            "timestamp": (base_time + timedelta(seconds=2)).isoformat(),
            "result": {
                "success": True,
                "score": {"scaled": 1.0}
            }
        })
    else:
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["failed"], "display": {"en": "failed"}},
            "object": activity,
            "context": assess_ctx,
            "timestamp": (base_time + timedelta(seconds=2)).isoformat(),
            "result": {
                "success": False,
                "score": {"scaled": 0.0}
            }
        })

    # 请求帮助 + 查看反馈（原有逻辑）
    perseverance_score = persona_course.get("perseverance", {}).get("score", 0.5)
    feedback_score = persona_course.get("feedback_orientation", {}).get("score", 0.5)
    if not is_correct and (perseverance_score + feedback_score) / 2.0 >= 0.5:
        ctx_support = make_context(course_uid)
        ctx_support["extensions"]["https://legend-meta.com/xapi/ext/support-type"] = "hint"
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["requested_support"], "display": {"en": "requested support"}},
            "object": activity,
            "context": ctx_support,
            "timestamp": (base_time + timedelta(seconds=5)).isoformat(),
            "result": {}
        })

        ctx_feedback = make_context(course_uid)
        ctx_feedback["extensions"]["https://legend-meta.com/xapi/ext/feedback-view-type"] = "unit-dashboard"
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["reviewed_feedback"], "display": {"en": "reviewed feedback"}},
            "object": activity,
            "context": ctx_feedback,
            "timestamp": (base_time + timedelta(seconds=15)).isoformat(),
            "result": {}
        })

    return events


def generate_persona_driven_extra_events(
    lrn_uid,
    course_uid,
    course_name,
    persona_course,
    unit_uids_for_course
):
    events = []
    actor = make_actor(lrn_uid)
    course_activity = make_course_activity(course_uid, course_name)
    base_time = datetime.utcnow()

    some_unit_uid = next(iter(unit_uids_for_course), None)
    some_unit_activity = None
    if some_unit_uid:
        some_unit_activity = {
            "objectType": "Activity",
            "id": f"https://legend-meta.com/unit/{some_unit_uid}",
            "definition": {
                "name": {"zh-CN": some_unit_uid},
                "type": ACTIVITY_TYPE_BASE + "unit/other"
            }
        }

    fb_score = persona_course.get("feedback_orientation", {}).get("score", 0.5)
    if fb_score >= 0.5:
        ctx = make_context(course_uid)
        ctx["extensions"]["https://legend-meta.com/xapi/ext/feedback-view-type"] = "course-dashboard"
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["reviewed_feedback"], "display": {"en": "reviewed feedback"}},
            "object": course_activity,
            "context": ctx,
            "timestamp": (base_time + timedelta(seconds=5)).isoformat(),
            "result": {
                "duration": "PT30S"
            }
        })

    exp_score = persona_course.get("exploration_orientation", {}).get("score", 0.5)
    if exp_score >= 0.4 and some_unit_activity is not None:
        ctx = make_context(course_uid)
        ctx["extensions"]["https://legend-meta.com/xapi/ext/unit-optional"] = True
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["explored_extension"], "display": {"en": "explored extension"}},
            "object": some_unit_activity,
            "context": ctx,
            "timestamp": (base_time + timedelta(seconds=15)).isoformat(),
            "result": {
                "duration": "PT60S"
            }
        })

    soc_score = persona_course.get("social_learning", {}).get("score", 0.5)
    if soc_score >= 0.5 and some_unit_activity is not None:
        ctx = make_context(course_uid)
        ctx["extensions"]["https://legend-meta.com/xapi/ext/observed-learner-id"] = "peer-placeholder"
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["observed_peer"], "display": {"en": "observed peer"}},
            "object": some_unit_activity,
            "context": ctx,
            "timestamp": (base_time + timedelta(seconds=25)).isoformat(),
            "result": {
                "duration": "PT45S"
            }
        })

    vc_score = persona_course.get("value_contribution", {}).get("score", 0.5)
    if vc_score >= 0.4:
        ctx_val = make_context(course_uid)
        ctx_val["extensions"]["https://legend-meta.com/xapi/ext/value-token-type"] = "learning-token"
        ctx_val["extensions"]["https://legend-meta.com/xapi/ext/value-change"] = 5
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["exchanged_value"], "display": {"en": "exchanged value"}},
            "object": course_activity,
            "context": ctx_val,
            "timestamp": (base_time + timedelta(seconds=35)).isoformat(),
            "result": {
                "response": "reward-for-contribution"
            }
        })

        ctx_res = make_context(course_uid)
        ctx_res["extensions"]["https://legend-meta.com/xapi/ext/resource-id"] = "res-note-1"
        ctx_res["extensions"]["https://legend-meta.com/xapi/ext/resource-type"] = "note"
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["contributed_resource"], "display": {"en": "contributed resource"}},
            "object": course_activity,
            "context": ctx_res,
            "timestamp": (base_time + timedelta(seconds=40)).isoformat(),
            "result": {}
        })

    ref_score = persona_course.get("reflection_depth", {}).get("score", 0.5)
    if ref_score >= 0.4:
        ctx_ref = make_context(course_uid)
        ctx_ref["extensions"]["https://legend-meta.com/xapi/ext/reflection-format"] = "text"
        base_words = 20 + int(ref_score * 30)
        text = " ".join(["reflection"] * base_words)
        events.append({
            "actor": actor,
            "verb": {"id": VERBS["reflected_on_activity"], "display": {"en": "reflected on"}},
            "object": course_activity,
            "context": ctx_ref,
            "timestamp": (base_time + timedelta(seconds=50)).isoformat(),
            "result": {
                "response": text
            }
        })

    return events


# ===================== 主流程 =====================

def main():
    print("连接 MySQL...")
    conn = get_mysql_connection()
    try:
        print("加载基础数据...")
        data = load_basic_data(conn)
    finally:
        conn.close()

    learners_by_uid = data["learners_by_uid"]
    units_by_uid = data["units_by_uid"]
    questions_by_uid = data["questions_by_uid"]
    courses_by_uid = data["courses_by_uid"]
    unit_to_course = data["unit_to_course"]
    question_to_course = data["question_to_course"]
    interactions = data["interactions"]

    # 计算已有粗粒度交互主键的最大值，用于为课程级虚拟会话分配不冲突的会话ID
    existing_ids = [
        row.get(INTERACTION_PK_FIELD)
        for row in interactions
        if row.get(INTERACTION_PK_FIELD) is not None
    ]
    max_session_id = max(existing_ids) if existing_ids else 0
    next_persona_session_id = max_session_id + 1

    print("根据粗粒度行为构建统计数据...")
    stats_per_learner_course, interactions_by_learner, learner_course_units = build_stats_from_interactions(
        interactions, units_by_uid, question_to_course, unit_to_course
    )

    print("连接 MongoDB...")
    mongo_db = get_mongo_db()
    profile_col = mongo_db[MONGO_CONFIG["profile_collection"]]
    xapi_col = mongo_db[MONGO_CONFIG["xapi_collection"]]

    print("删除已有的 LearnerProfile 和 Interaction 集合（如果存在）...")
    mongo_db.drop_collection(MONGO_CONFIG["profile_collection"])
    mongo_db.drop_collection(MONGO_CONFIG["xapi_collection"])

    # ================== 新建集合并建立索引 ==================
    print("创建 LearnerProfile 和 Interaction 集合，并建立索引...")

    profile_col = mongo_db[MONGO_CONFIG["profile_collection"]]
    xapi_col = mongo_db[MONGO_CONFIG["xapi_collection"]]

    # LearnerProfile 索引
    profile_col.create_index(
        [("learner_uid", ASCENDING)],
        name="idx_learner_uid",
        unique=True,
    )

    # Interaction 索引（整合 rebuild 中的 3 个 + 会话相关 2 个）
    print("[index] create idx_lrn_verb_course")
    xapi_col.create_index(
        [("_lrn_uid", ASCENDING), ("verb.id", ASCENDING), ("_course_uid", ASCENDING)],
        name="idx_lrn_verb_course",
        background=False,
    )

    print("[index] create idx_course_verb_lrn")
    xapi_col.create_index(
        [("_course_uid", ASCENDING), ("verb.id", ASCENDING), ("_lrn_uid", ASCENDING)],
        name="idx_course_verb_lrn",
        background=False,
    )

    print("[index] create idx_course_lrn")
    xapi_col.create_index(
        [("_course_uid", ASCENDING), ("_lrn_uid", ASCENDING)],
        name="idx_course_lrn",
        background=False,
    )

    print("[index] create idx_lrn_session_time")
    xapi_col.create_index(
        [("_lrn_uid", ASCENDING), ("_session_id", ASCENDING), ("timestamp", ASCENDING)],
        name="idx_lrn_session_time",
        background=False,
    )

    print("[index] create idx_session_id")
    xapi_col.create_index(
        [("_session_id", ASCENDING)],
        name="idx_session_id",
        background=False,
    )

    print("集合与索引创建完成。继续生成细粒度行为数据...")

    learner_profiles_docs = []
    xapi_docs = []

    interactions_count_per_learner = {
        lrn_uid: len(rows) for lrn_uid, rows in interactions_by_learner.items()
    }

    hard_discard_learners = {
        lrn_uid for lrn_uid, cnt in interactions_count_per_learner.items()
        if cnt > MAX_INTERACTIONS_HARD
    }
    if hard_discard_learners:
        print("将被直接舍弃的学习者（数据量超出硬上限）数量：", len(hard_discard_learners))

    all_learners = list(interactions_by_learner.keys())
    print("开始为每个学习者生成画像和细粒度 xAPI 行为...")

    for lrn_uid in tqdm(all_learners):
        if lrn_uid in hard_discard_learners:
            continue

        rng = get_rng_for_learner(lrn_uid)

        course_profiles = []
        persona_by_course = {}

        # 构建课程级 persona
        for (key_lrn, crs_uid), stats in stats_per_learner_course.items():
            if key_lrn != lrn_uid:
                continue
            persona = infer_persona_for_course(stats, rng)
            persona_by_course[crs_uid] = persona
            course_profiles.append({
                "course_uid": crs_uid,
                "persona": persona,
                "stats": stats
            })

        global_profile = aggregate_global_profile(course_profiles)

        learner_doc = {
            "learner_uid": lrn_uid,
            "basic": learners_by_uid.get(lrn_uid, {}),
            "global_profile": global_profile,
            "course_profiles": course_profiles,
            "generated_at": datetime.utcnow(),
        }
        learner_profiles_docs.append(learner_doc)

        if len(learner_profiles_docs) >= PROFILE_BATCH_SIZE:
            profile_col.insert_many(learner_profiles_docs)
            learner_profiles_docs = []

        learner_interactions_full = sorted(
            interactions_by_learner[lrn_uid],
            key=lambda r: r["create_time"] or datetime.utcnow()
        )

        total_cnt = len(learner_interactions_full)
        if total_cnt > MAX_INTERACTIONS_SOFT:
            step = max(total_cnt // MAX_INTERACTIONS_SOFT, 1)
            learner_interactions = learner_interactions_full[::step]
        else:
            learner_interactions = learner_interactions_full

        # 生成单元/题目级会话
        for row in learner_interactions:
            unt_uid = row["unt_uid"]
            is_unit = unt_uid in units_by_uid

            if is_unit:
                unit = units_by_uid[unt_uid]
                utype = (unit["type"] or "").lower()
                crs_uid = unit_to_course.get(unt_uid)
            else:
                utype = "question"
                unit = None
                crs_uid = question_to_course.get(unt_uid)

            if not crs_uid:
                continue

            persona_course = persona_by_course.get(crs_uid)
            if not persona_course:
                stats_dummy = {
                    "video_total_len": 0.0,
                    "video_watch": 0.0,
                    "vr_total": 0.0,
                    "vr_focus": 0.0,
                    "ar_total": 0.0,
                    "ar_focus": 0.0,
                    "interact_total": 0.0,
                    "interact_correct": 0.0,
                    "cooperate_total": 0.0,
                    "cooperate_effective": 0.0,
                    "question_attempts": 0,
                    "question_correct": 0,
                    "question_wrong": 0,
                    "question_retry_after_wrong": 0,
                    "unit_counts": defaultdict(int),
                    "total_interactions": 0,
                }
                persona_course = infer_persona_for_course(stats_dummy, rng)
                persona_by_course[crs_uid] = persona_course

            events = []
            if is_unit:
                if utype == "video":
                    events = generate_xapi_for_video(row, unit, crs_uid, persona_course)
                elif utype == "vr":
                    events = generate_xapi_for_vr_ar(row, unit, crs_uid, persona_course, "vr")
                elif utype == "ar":
                    events = generate_xapi_for_vr_ar(row, unit, crs_uid, persona_course, "ar")
                elif utype == "interact":
                    events = generate_xapi_for_interact(row, unit, crs_uid, persona_course)
                elif utype == "cooperate":
                    events = generate_xapi_for_cooperate(row, unit, crs_uid, persona_course)
                else:
                    events = generate_xapi_for_video(row, unit, crs_uid, persona_course)
            else:
                events = generate_xapi_for_question(row, crs_uid, persona_course)

            session_id = row.get(INTERACTION_PK_FIELD)

            for ev in events:
                ev["_session_id"] = session_id
                ev["_lrn_uid"] = lrn_uid
                ev["_unt_uid"] = unt_uid
                ev["_course_uid"] = crs_uid
                ev["_type"] = utype
                xapi_docs.append(ev)

                if len(xapi_docs) >= XAPI_BATCH_SIZE:
                    xapi_col.insert_many(xapi_docs)
                    xapi_docs = []

        # 课程级 persona 补充会话
        for crs_uid, persona_course in persona_by_course.items():
            course = courses_by_uid.get(crs_uid, {})
            course_name = course.get("name", crs_uid)
            unit_uids = learner_course_units.get((lrn_uid, crs_uid), set())

            persona_session_id = next_persona_session_id
            next_persona_session_id += 1

            extra_events = generate_persona_driven_extra_events(
                lrn_uid,
                crs_uid,
                course_name,
                persona_course,
                unit_uids
            )
            for ev in extra_events:
                ev["_session_id"] = persona_session_id
                ev["_lrn_uid"] = lrn_uid
                ev["_unt_uid"] = None
                ev["_course_uid"] = crs_uid
                ev["_type"] = "course-level"
                xapi_docs.append(ev)
                if len(xapi_docs) >= XAPI_BATCH_SIZE:
                    xapi_col.insert_many(xapi_docs)
                    xapi_docs = []

    if learner_profiles_docs:
        profile_col.insert_many(learner_profiles_docs)
    if xapi_docs:
        xapi_col.insert_many(xapi_docs)

    print("处理完成。")


if __name__ == "__main__":
    main()
