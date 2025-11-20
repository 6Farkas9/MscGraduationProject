# database.py
import logging
from typing import Optional, Dict
import pymysql
from pymysql.connections import Connection
from pymysql.cursors import DictCursor
import pymongo
from pymongo.database import Database as MongoDatabase
from pymongo.errors import ConnectionFailure
# 修改为相对导入
from .config import db_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MySQLManager:
    """MySQL数据库连接管理器"""
    
    _instance: Optional['MySQLManager'] = None
    _connection: Optional[Connection] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_connection(self) -> Connection:
        """获取MySQL数据库连接"""
        if self._connection is None or not self._connection.open:
            try:
                # 获取基础配置
                config = db_config.get_mysql_config()
                
                # 移除可能存在的autocommit配置，因为我们会在连接后单独设置
                config.pop('autocommit', None)
                
                self._connection = pymysql.connect(
                    **config,
                    cursorclass=DictCursor
                )
                
                # 单独设置autocommit
                self._connection.autocommit(True)
                
                logger.info("MySQL连接已建立")
            except Exception as e:
                logger.error(f"MySQL连接失败: {e}")
                raise
        return self._connection
    
    def close_connection(self):
        """关闭MySQL连接"""
        if self._connection and self._connection.open:
            self._connection.close()
            self._connection = None
            logger.info("MySQL连接已关闭")
    
    def execute_query(self, query: str, params: tuple = None) -> list:
        """执行查询语句"""
        connection = self.get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params or ())
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"查询执行失败: {e}, SQL: {query}")
            raise
    
    def execute_single_query(self, query: str, params: tuple = None) -> Optional[Dict]:
        """执行查询语句，返回单条记录"""
        connection = self.get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params or ())
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"单条查询执行失败: {e}, SQL: {query}")
            raise

class MongoDBManager:
    """MongoDB数据库连接管理器"""
    
    _instance: Optional['MongoDBManager'] = None
    _client: Optional[pymongo.MongoClient] = None
    _database: Optional[MongoDatabase] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_database(self) -> MongoDatabase:
        """获取MongoDB数据库实例"""
        if self._client is None or self._database is None:
            try:
                connection_string = db_config.get_mongodb_connection_string()
                self._client = pymongo.MongoClient(connection_string)
                self._database = self._client[db_config.mongodb_config['database']]
                
                # 测试连接
                self._client.admin.command('ismaster')
                logger.info("MongoDB连接已建立")
            except ConnectionFailure as e:
                logger.error(f"MongoDB连接失败: {e}")
                raise
        return self._database
    
    def close_connection(self):
        """关闭MongoDB连接"""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("MongoDB连接已关闭")
    
    def get_collection(self, collection_name: str):
        """获取集合对象"""
        database = self.get_database()
        return database[collection_name]

class DatabaseManager:
    """统一的数据库管理器"""
    
    def __init__(self):
        self.mysql_manager = MySQLManager()
        self.mongodb_manager = MongoDBManager()
    
    def get_mysql_connection(self) -> Connection:
        """获取MySQL连接"""
        return self.mysql_manager.get_connection()
    
    def get_mongodb_database(self) -> MongoDatabase:
        """获取MongoDB数据库"""
        return self.mongodb_manager.get_database()
    
    def close_all_connections(self):
        """关闭所有数据库连接"""
        self.mysql_manager.close_connection()
        self.mongodb_manager.close_connection()

# 全局数据库管理器实例
db_manager = DatabaseManager()

# 便捷函数
def get_mysql_connection() -> Connection:
    """便捷函数：获取MySQL连接"""
    return db_manager.get_mysql_connection()

def get_mongodb_database() -> MongoDatabase:
    """便捷函数：获取MongoDB数据库"""
    return db_manager.get_mongodb_database()

def close_db_connections():
    """便捷函数：关闭所有数据库连接"""
    db_manager.close_all_connections()