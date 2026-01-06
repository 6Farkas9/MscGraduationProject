import os
from functools import lru_cache
from typing import Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_DB_URL = "mysql+pymysql://root:123456@localhost:3306/mls_sample?charset=utf8mb4"

def get_db_url() -> str:
    """
    可选覆盖：
      MLS_DB_URL="mysql+pymysql://root:123456@localhost:3306/mls_sample?charset=utf8mb4"
    """
    return os.getenv("MLS_DB_URL", DEFAULT_DB_URL)

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(get_db_url(), pool_pre_ping=True, future=True)

@lru_cache(maxsize=1)
def load_topic_concept_map() -> Dict[str, str]:
    """
    返回：concept_uid -> topic_uid
    """
    engine = get_engine()
    mapping: Dict[str, str] = {}
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT tpc_uid, cpt_uid FROM Topic_Concept")).fetchall()
        for tpc_uid, cpt_uid in rows:
            if cpt_uid and tpc_uid:
                mapping[str(cpt_uid)] = str(tpc_uid)
    return mapping

@lru_cache(maxsize=1)
def load_topics() -> Dict[str, Dict[str, Any]]:
    """
    返回：topic_uid -> {uid, name, explanation}
    """
    engine = get_engine()
    topics: Dict[str, Dict[str, Any]] = {}
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT uid, name, explanation FROM Topics")).fetchall()
        for uid, name, explanation in rows:
            topics[str(uid)] = {"uid": str(uid), "name": name, "explanation": explanation}
    return topics

@lru_cache(maxsize=1)
def load_concepts() -> Dict[str, Dict[str, Any]]:
    """
    返回：concept_uid -> {uid, name, explanation}
    """
    engine = get_engine()
    concepts: Dict[str, Dict[str, Any]] = {}
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT uid, name, explanation FROM Concepts")).fetchall()
        for uid, name, explanation in rows:
            concepts[str(uid)] = {"uid": str(uid), "name": name, "explanation": explanation}
    return concepts
