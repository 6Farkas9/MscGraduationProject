from __future__ import annotations
from typing import Dict, Optional
import os

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover
    MongoClient = None

DEFAULT_MONGO_URI = os.environ.get("MLS_MONGO_URI", "mongodb://127.0.0.1:27017")
DEFAULT_DB = os.environ.get("MLS_MONGO_DB", "MLS")
DEFAULT_COL = os.environ.get("MLS_MONGO_COLLECTION", "Learners")
DEFAULT_UID_FIELD = os.environ.get("MLS_MONGO_UID_FIELD", "uid")
DEFAULT_KT_FIELD = os.environ.get("MLS_MONGO_KT_FIELD", "KT")

def fetch_kt_for_learner(learner_uid: str,
                         mongo_uri: Optional[str] = None,
                         db_name: Optional[str] = None,
                         col_name: Optional[str] = None,
                         uid_field: Optional[str] = None,
                         kt_field: Optional[str] = None) -> Dict[str, float]:
    """Fetch KT dict from MongoDB:
    db=MLS, collection=Learners, match by uid field, return KT dict.
    """
    if MongoClient is None:
        raise RuntimeError("pymongo not installed")

    uri = mongo_uri or DEFAULT_MONGO_URI
    dbn = db_name or DEFAULT_DB
    coln = col_name or DEFAULT_COL
    uf = uid_field or DEFAULT_UID_FIELD
    kf = kt_field or DEFAULT_KT_FIELD

    client = MongoClient(uri)
    try:
        col = client[dbn][coln]
        doc = col.find_one({uf: learner_uid}, {kf: 1, "_id": 0})
        if not doc or kf not in doc or not isinstance(doc.get(kf), dict):
            return {}
        raw = doc.get(kf) or {}
        kt: Dict[str, float] = {}
        for k, v in raw.items():
            try:
                kt[str(k)] = float(v)
            except Exception:
                continue
        return kt
    finally:
        client.close()
