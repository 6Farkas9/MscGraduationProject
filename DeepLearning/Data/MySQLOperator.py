import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import mysql
import mysql.connector

class BaseMySQLRepository:

    def __init__(self):
        # self.con = mysql.connector.connect(
        #     host="localhost",  # MySQL服务器地址
        #     user="root",   # 用户名
        #     password="123456",  # 密码
        #     database="MLS"  # 数据库名称
        # )
        self.con = mysql.connector.connect(
            host="localhost",  # MySQL服务器地址
            user="root",   # 用户名
            password="123456",  # 密码
            database="mls_sample"  # 数据库名称
        )

    def execute_query(self, sql, params=None):
        """执行查询并返回所有结果"""
        cursor = self.con.cursor()
        try:
            cursor.execute(sql, params or [])
            return cursor.fetchall()
        finally:
            cursor.close()
    
    def execute_query_one(self, sql, params=None):
        """执行查询并返回单个结果"""
        cursor = self.con.cursor()
        try:
            cursor.execute(sql, params or [])
            return cursor.fetchone()
        finally:
            cursor.close()
    
    def execute_query_scalar(self, sql, params=None):
        """执行查询并返回标量值（第一行的第一列）"""
        result = self.execute_query_one(sql, params)
        return result[0] if result else None
    
    def execute_query_dict(self, sql, params=None, key_column=None):
        """执行查询并返回字典形式的结果"""
        results = self.execute_query(sql, params)
        if key_column is None:
            return {item[0]: item[1] for item in results}
        else:
            # 需要根据实际情况调整
            return {item[key_column]: item for item in results}
    
    def execute_update(self, sql, params=None):
        """执行更新操作"""
        cursor = self.con.cursor()
        try:
            cursor.execute(sql, params or [])
            self.con.commit()
            return cursor.rowcount
        finally:
            cursor.close()
    
    def execute_batch(self, sql, params_list):
        """批量执行操作"""
        cursor = self.con.cursor()
        try:
            cursor.executemany(sql, params_list)
            self.con.commit()
            return cursor.rowcount
        finally:
            cursor.close()

class GenericRepository(BaseMySQLRepository):
    
    def get_count(self, table_name, where_conditions=None, params=None):
        """获取表中记录数量"""
        sql = f"SELECT COUNT(*) FROM {table_name}"
        if where_conditions:
            sql += f" WHERE {where_conditions}"
        return self.execute_query_scalar(sql, params)
    
    def get_all_records(self, table_name, columns="*", order_by=None, limit=None):
        """获取表中所有记录"""
        sql = f"SELECT {columns} FROM {table_name}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"
        return self.execute_query(sql)
    
    def get_records_by_condition(self, table_name, where_conditions, params=None, 
                               columns="*", order_by=None, limit=None):
        """根据条件查询记录"""
        sql = f"SELECT {columns} FROM {table_name} WHERE {where_conditions}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"
        return self.execute_query(sql, params)
    
    def get_record_by_id(self, table_name, id_value, id_column="uid", columns="*"):
        """根据ID获取单条记录"""
        sql = f"SELECT {columns} FROM {table_name} WHERE {id_column} = %s"
        return self.execute_query_one(sql, [id_value])
    
    def insert_record(self, table_name, data_dict):
        """插入单条记录"""
        columns = ", ".join(data_dict.keys())
        placeholders = ", ".join(["%s"] * len(data_dict))
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        return self.execute_update(sql, list(data_dict.values()))
    
    def update_record(self, table_name, data_dict, where_conditions, params=None):
        """更新记录"""
        set_clause = ", ".join([f"{key} = %s" for key in data_dict.keys()])
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_conditions}"
        all_params = list(data_dict.values()) + (params or [])
        return self.execute_update(sql, all_params)
    
    def delete_records(self, table_name, where_conditions, params=None):
        """删除记录"""
        sql = f"DELETE FROM {table_name} WHERE {where_conditions}"
        return self.execute_update(sql, params)
    
    def get_distinct_values(self, table_name, column_name, where_conditions=None, params=None):
        """获取某列的唯一值"""
        sql = f"SELECT DISTINCT {column_name} FROM {table_name}"
        if where_conditions:
            sql += f" WHERE {where_conditions}"
        results = self.execute_query(sql, params)
        return [item[0] for item in results]
    
mysqlop = GenericRepository()
