# mysql_operator.py
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
# 修改为相对导入
from ...core.database import MySQLManager

logger = logging.getLogger(__name__)

class MySQLOperator:
    """MySQL通用操作类 - 只包含基础数据库操作，不包含业务逻辑"""
    
    def __init__(self):
        self.mysql_manager = MySQLManager()
    
    def execute_query(self, query: str, params: Tuple = None) -> List[Dict[str, Any]]:
        """执行查询并返回结果列表"""
        return self.mysql_manager.execute_query(query, params)
    
    def execute_single_query(self, query: str, params: Tuple = None) -> Optional[Dict[str, Any]]:
        """执行查询并返回单条结果"""
        return self.mysql_manager.execute_single_query(query, params)
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """执行批量操作"""
        connection = self.mysql_manager.get_connection()
        try:
            with connection.cursor() as cursor:
                affected_rows = cursor.executemany(query, params_list)
                return affected_rows
        except Exception as e:
            logger.error(f"批量操作失败: {e}, SQL: {query}")
            raise
    
    def fetch_by_uid(self, table: str, uid: str) -> Optional[Dict[str, Any]]:
        """根据UID从指定表获取记录"""
        query = f"SELECT * FROM {table} WHERE uid = %s"
        return self.execute_single_query(query, (uid,))
    
    def fetch_all(self, table: str, columns: List[str] = None) -> List[Dict[str, Any]]:
        """获取指定表的所有记录"""
        if columns:
            columns_str = ', '.join(columns)
            query = f"SELECT {columns_str} FROM {table}"
        else:
            query = f"SELECT * FROM {table}"
        return self.execute_query(query)
    
    def fetch_by_field(self, table: str, field: str, value: Any) -> List[Dict[str, Any]]:
        """根据字段值获取记录"""
        query = f"SELECT * FROM {table} WHERE {field} = %s"
        return self.execute_query(query, (value,))
    
    def fetch_by_fields(self, table: str, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据多个字段条件获取记录"""
        where_clause = ' AND '.join([f"{key} = %s" for key in conditions.keys()])
        query = f"SELECT * FROM {table} WHERE {where_clause}"
        return self.execute_query(query, tuple(conditions.values()))
    
    def fetch_in_list(self, table: str, field: str, values: List[Any]) -> List[Dict[str, Any]]:
        """根据字段值列表获取记录"""
        if not values:
            return []
        
        placeholders = ', '.join(['%s'] * len(values))
        query = f"SELECT * FROM {table} WHERE {field} IN ({placeholders})"
        return self.execute_query(query, tuple(values))
    
    def fetch_with_order(self, table: str, order_by: str, limit: int = None) -> List[Dict[str, Any]]:
        """获取记录并排序"""
        query = f"SELECT * FROM {table} ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"
        return self.execute_query(query)
    
    def fetch_by_field_with_order(self, table: str, field: str, value: Any, 
                                order_by: str, limit: int = None) -> List[Dict[str, Any]]:
        """根据字段值获取记录并排序"""
        query = f"SELECT * FROM {table} WHERE {field} = %s ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"
        return self.execute_query(query, (value,))
    
    def count_records(self, table: str, conditions: Dict[str, Any] = None) -> int:
        """统计记录数量"""
        if conditions:
            where_clause = ' AND '.join([f"{key} = %s" for key in conditions.keys()])
            query = f"SELECT COUNT(*) as count FROM {table} WHERE {where_clause}"
            result = self.execute_single_query(query, tuple(conditions.values()))
        else:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = self.execute_single_query(query)
        
        return result['count'] if result else 0
    
    def execute_custom_query(self, query: str, params: Tuple = None) -> List[Dict[str, Any]]:
        """执行自定义查询"""
        return self.execute_query(query, params)
    
    def execute_custom_single_query(self, query: str, params: Tuple = None) -> Optional[Dict[str, Any]]:
        """执行自定义查询返回单条记录"""
        return self.execute_single_query(query, params)

# 全局操作器实例
mysql_operator = MySQLOperator()