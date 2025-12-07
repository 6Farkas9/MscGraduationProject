# BackEnd/app/shared/utils/repository_mixins.py
import re
from typing import List, Dict, Any, Optional

class UIDRepositoryMixin:
    """
    与 UID 列表处理相关的通用方法
    - 从结果集中提取 uid
    - 多个 uid 列表去重
    - 从列表中删除指定 uid
    """

    @staticmethod
    def extract_uids_from_results(results: List[Dict[str, Any]], uid_field: str = "uid") -> List[str]:
        return [row[uid_field] for row in results if uid_field in row]

    @staticmethod
    def filter_unique_uids(uid_lists: List[List[str]]) -> List[str]:
        unique_uids: set[str] = set()
        for uid_list in uid_lists:
            unique_uids.update(uid_list)
        return list(unique_uids)

    @staticmethod
    def remove_uid_from_list(uid_list: List[str], uid_to_remove: str) -> List[str]:
        return [uid for uid in uid_list if uid != uid_to_remove]


class MappingRepositoryMixin:
    """
    通用的“从行记录构建字典映射”工具
    典型用法：Concepts 表的 uid -> id 映射 等
    """

    @staticmethod
    def build_mapping_from_rows(
        rows: List[Dict[str, Any]],
        key_field: str,
        value_field: str,
    ) -> Dict[Any, Any]:
        mapping: Dict[Any, Any] = {}
        for row in rows:
            if key_field in row and value_field in row:
                mapping[row[key_field]] = row[value_field]
        return mapping
    
_DURATION_SECONDS_RE = re.compile(r"^PT(\d+)S$")

def parse_iso8601_duration_seconds(duration_str: Optional[str]) -> Optional[int]:
    """
    解析简化的 ISO8601 时长字符串 "PT{n}S" -> n（秒）.
    不符合格式或解析失败时返回 None。
    """
    if not duration_str:
        return None
    m = _DURATION_SECONDS_RE.match(duration_str)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法：当分母接近 0 时返回 default（默认 0.0），避免 ZeroDivisionError。
    """
    if abs(denominator) <= 1e-9:
        return float(default)
    return float(numerator) / float(denominator)
