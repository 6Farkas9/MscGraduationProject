from __future__ import annotations
from typing import Any, Dict, List, Optional
import os

try:
    import pymysql
except Exception:  # pragma: no cover
    pymysql = None

DEFAULT_DB = {
    "host": os.environ.get("MLS_DB_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MLS_DB_PORT", "3306")),
    "user": os.environ.get("MLS_DB_USER", "root"),
    "password": os.environ.get("MLS_DB_PASSWORD", "123456"),
    "database": os.environ.get("MLS_DB_NAME", "mls_sample"),
    "charset": "utf8mb4",
}

def _conn(conf: Optional[dict] = None):
    if pymysql is None:
        raise RuntimeError("pymysql not installed")
    c = dict(DEFAULT_DB)
    if conf:
        c.update({k:v for k,v in conf.items() if v not in (None, "")})
    return pymysql.connect(
        host=c["host"], port=c["port"], user=c["user"], password=c["password"],
        database=c["database"], charset=c["charset"],
        cursorclass=pymysql.cursors.DictCursor,
    )

def fetch_basic_learners(limit: int = 120, conf: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Only fetch uid and name from BasicLearners (per requirement)."""
    conn = _conn(conf)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT uid, name FROM BasicLearners "
                "WHERE uid IS NOT NULL AND name IS NOT NULL LIMIT %s",
                (int(limit),),
            )
            return list(cur.fetchall())
    finally:
        conn.close()

def fetch_concepts(limit: int = 500, conf: Optional[dict] = None) -> List[str]:
    conn = _conn(conf)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT uid FROM Concepts LIMIT %s", (int(limit),))
            rows = cur.fetchall()
            return [str(r["uid"]) for r in rows if r.get("uid")]
    finally:
        conn.close()
