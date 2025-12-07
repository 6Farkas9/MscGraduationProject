# BackEnd/app/shared/utils/mongo_utils.py
from typing import Dict, Any, Iterable, List
from bson import ObjectId
from bson.errors import InvalidId


def is_valid_object_id(value: Any) -> bool:
    """
    判断一个值是否是合法的 ObjectId 字符串
    """
    try:
        ObjectId(str(value))
        return True
    except (InvalidId, TypeError):
        return False


def convert_id_in_query(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将查询 / 文档中的 `_id` 字段从 str 转为 ObjectId（如果合法）
    该方法本身与具体集合无关，可以被任何 Mongo 仓库重用
    """
    result = dict(data)
    if "_id" in result and isinstance(result["_id"], str) and is_valid_object_id(result["_id"]):
        result["_id"] = ObjectId(result["_id"])
    return result


def normalize_mongo_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    将文档中的 ObjectId 转为字符串，方便向上层返回
    """
    if doc is None:
        return doc
    if "_id" in doc and isinstance(doc["_id"], ObjectId):
        doc = dict(doc)
        doc["_id"] = str(doc["_id"])
    return doc


def normalize_mongo_documents(docs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    批量规范化文档列表
    """
    return [normalize_mongo_document(d) for d in docs]
