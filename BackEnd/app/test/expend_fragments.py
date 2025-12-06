# -*- coding: utf-8 -*-
"""
从 MongoDB.MLS.Fragments_bak 读取原始分段数据，
结合 MySQL.mls 的 Units / Concepts / Unit_Concept，
生成拓展后的分段资源文档并写入 MongoDB.MLS.Fragments。

特性：
- 自动删除并重建 Fragments 集合
- 自动创建高效索引
- 忽略 Fragments_bak 中无关字段
- 使用 tqdm 进度条，不会刷屏
- 批处理数据，适合大型数据集
"""

import pymysql
from pymongo import MongoClient, ASCENDING, TEXT
from tqdm import tqdm
from typing import Dict, List, Any
import math


# =====================
# 配置区
# =====================

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "mls",
    "charset": "utf8mb4",
}

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "MLS"
FRAGMENTS_BAK_COLLECTION = "Fragments_bak"
FRAGMENTS_COLLECTION = "Fragments"

BATCH_SIZE = 1000  # 批处理大小


# =====================
# MySQL 数据读取
# =====================

def load_units_and_concepts():
    """
    读取 Units + Unit_Concept + Concepts
    返回:
        units_by_uid: {unit_uid → {"uid","oid","name","type"}}
        concepts_by_unit: {unit_uid → [{"uid","name"}, ...]}
    """
    conn = pymysql.connect(**MYSQL_CONFIG, cursorclass=pymysql.cursors.DictCursor)
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT uid, oid, name, type FROM Units")
            units_by_uid = {row["uid"]: row for row in cursor.fetchall()}

            sql = """
                SELECT uc.unt_uid AS unt_uid,
                       c.uid AS cpt_uid,
                       c.name AS cpt_name
                FROM Unit_Concept uc
                JOIN Concepts c ON uc.cpt_uid = c.uid
            """
            cursor.execute(sql)
            concepts_by_unit = {}
            for r in cursor.fetchall():
                concepts_by_unit.setdefault(r["unt_uid"], []).append(
                    {"uid": r["cpt_uid"], "name": r["cpt_name"]}
                )

        return units_by_uid, concepts_by_unit

    finally:
        conn.close()


# =====================
# 启发式标签推断函数
# =====================

def infer_pedagogical_function_from_content(content: str) -> str:
    if not content:
        return "concept_explanation"
    c = content.lower()
    if any(k in c for k in ["练习", "试一试", "任务"]):
        return "practice"
    if any(k in c for k in ["例子", "示例", "举例"]):
        return "example"
    if any(k in c for k in ["演示", "操作演示"]):
        return "demonstration"
    if any(k in c for k in ["总结", "回顾", "小结"]):
        return "summary"
    if any(k in c for k in ["激励", "鼓励"]):
        return "motivation"
    return "concept_explanation"


def infer_difficulty_from_concepts(concepts: List[Dict[str, Any]]) -> str:
    if not concepts:
        return "basic"
    names = " ".join([c.get("name", "") for c in concepts])
    if any(k in names for k in ["高级", "进阶", "优化"]):
        return "advanced"
    if len(concepts) == 1:
        return "basic"
    return "intermediate"


def default_labels_by_type(unit_type: str) -> Dict[str, Any]:
    """
    统一标签取值，带默认值。
    """
    unit_type = (unit_type or "").lower()
    labels = {
        "visual_richness": "medium",
        "audio_richness": "medium",
        "structure_level": "medium",
        "guidance_level": "medium",
        "example_included": False,
        "difficulty_level": "basic",
        "cognitive_load": "medium",

        "interaction_level": "none",
        "exploration_freedom": "low",
        "task_steps": 0,
        "error_feedback": "none",

        "collaboration_mode": "group",
        "social_intensity": "none",
        "role_requirement": [],
        "communication_format": "text",

        "environment_complexity": "simple",
        "spatial_navigation_demand": "low",
        "immersion_level": "low",

        "pedagogical_function": "concept_explanation",
        "attention_demand": "medium",
        "time_estimate": 10,
    }

    # 针对类别自定义
    if unit_type == "video":
        labels.update(
            audio_richness="high",
            communication_format="voice",
            immersion_level="medium",
        )
    elif unit_type == "vr":
        labels.update(
            visual_richness="high",
            interaction_level="medium",
            exploration_freedom="high",
            error_feedback="implicit",
            environment_complexity="moderate",
            spatial_navigation_demand="medium",
            immersion_level="high",
        )
    elif unit_type == "ar":
        labels.update(
            visual_richness="high",
            exploration_freedom="medium",
            error_feedback="implicit",
            environment_complexity="moderate",
            spatial_navigation_demand="medium",
        )
    elif unit_type == "interact":
        labels.update(
            structure_level="high",
            guidance_level="high",
            interaction_level="high",
            error_feedback="explicit",
        )
    elif unit_type == "cooperate":
        labels.update(
            collaboration_mode="group",
            social_intensity="high",
            communication_format="mixed",
            role_requirement=["coordinator", "executor"],
        )

    return labels


# =====================
# 分段扩展操作
# =====================

def enrich_fragment_doc(raw_doc, unit_type, concepts):
    uid = raw_doc.get("UID")
    oid = raw_doc.get("OID")
    raw_type = raw_doc.get("Type") or unit_type
    unit_type = (unit_type or raw_type or "").lower()

    location = raw_doc.get("位置")
    content = raw_doc.get("具体内容") or ""

    labels = default_labels_by_type(unit_type)
    labels["pedagogical_function"] = infer_pedagogical_function_from_content(content)
    labels["difficulty_level"] = infer_difficulty_from_concepts(concepts)

    # 如果是视频，则根据时间范围估算长度
    time_estimate = labels["time_estimate"]
    if unit_type == "video" and isinstance(location, str) and "-" in location and "秒" in location:
        try:
            rng = location.replace("秒", "")
            s, e = rng.split("-")
            d = max(1, int(round(float(e) - float(s))))
            time_estimate = d
        except:
            pass
    labels["time_estimate"] = time_estimate

    return {
        "uid": uid,
        "oid": oid,
        "type": unit_type,
        "location": location,
        "content": content,
        "concepts": concepts or [],
        **labels
    }


# =====================
# MongoDB 集合初始化 + 索引
# =====================

def setup_fragments_collection(db):
    if FRAGMENTS_COLLECTION in db.list_collection_names():
        print(f"Deleting existing collection: {FRAGMENTS_COLLECTION}")
        db.drop_collection(FRAGMENTS_COLLECTION)

    coll = db.create_collection(FRAGMENTS_COLLECTION)
    print("Creating indexes...")

    coll.create_index([("uid", ASCENDING)], unique=True, name="idx_uid")
    coll.create_index([("oid", ASCENDING)], name="idx_oid")

    coll.create_index([("concepts.uid", ASCENDING), ("type", ASCENDING)], name="idx_concept_type")

    # 常用标签索引
    tag_fields = [
        "type", "pedagogical_function", "difficulty_level", "interaction_level",
        "social_intensity", "environment_complexity", "cognitive_load",
        "guidance_level", "visual_richness"
    ]
    for f in tag_fields:
        coll.create_index([(f, ASCENDING)], name=f"idx_{f}")

    # 组合索引
    coll.create_index(
        [("type", ASCENDING), ("pedagogical_function", ASCENDING), ("difficulty_level", ASCENDING)],
        name="idx_type_pedagogical_difficulty"
    )

    # 文本索引
    coll.create_index([("content", TEXT)], name="idx_text_content")

    print("Indexes created.")
    return coll


# =====================
# 主程序
# =====================

def main():
    print("Loading Units and Concepts from MySQL...")
    units_by_uid, concepts_by_unit = load_units_and_concepts()

    # MongoDB 连接
    mongo = MongoClient(MONGO_URI)[MONGO_DB_NAME]
    fragments_bak = mongo[FRAGMENTS_BAK_COLLECTION]
    fragments = setup_fragments_collection(mongo)

    total = fragments_bak.count_documents({})
    print(f"Total source docs: {total}")

    cursor = fragments_bak.find({}, no_cursor_timeout=True).batch_size(BATCH_SIZE)

    batch = []
    processed = 0

    try:
        for raw_doc in tqdm(cursor, total=total, desc="Processing fragments"):
            oid = raw_doc.get("OID")
            unit = units_by_uid.get(oid, {})
            unit_type = unit.get("type", raw_doc.get("Type"))

            concepts = concepts_by_unit.get(oid, [])

            enriched = enrich_fragment_doc(raw_doc, unit_type, concepts)
            batch.append(enriched)

            if len(batch) >= BATCH_SIZE:
                fragments.insert_many(batch)
                processed += len(batch)
                batch.clear()

        if batch:
            fragments.insert_many(batch)
            processed += len(batch)

    finally:
        cursor.close()

    print(f"Completed. Inserted {processed} documents.")


if __name__ == "__main__":
    main()
