# config.py
import os
from typing import Dict, Any

class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self):
        # MySQL 配置
        self.mysql_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': '123456',
            'database': 'mls_sample',
            'charset': 'utf8mb4',
            'autocommit': True
        }
        
        # MongoDB 配置
        self.mongodb_config = {
            'host': 'localhost',
            'port': 27017,
            'database': 'MLS',
            'username': None,
            'password': None,
            'auth_source': 'admin'
        }
    
    def get_mysql_config(self) -> Dict[str, Any]:
        """获取MySQL配置"""
        return self.mysql_config.copy()
    
    def get_mongodb_config(self) -> Dict[str, Any]:
        """获取MongoDB配置"""
        return self.mongodb_config.copy()
    
    def get_mongodb_connection_string(self) -> str:
        """获取MongoDB连接字符串"""
        config = self.mongodb_config
        if config['username'] and config['password']:
            return f"mongodb://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?authSource={config['auth_source']}"
        else:
            return f"mongodb://{config['host']}:{config['port']}/{config['database']}"


# 全局配置实例
db_config = DatabaseConfig()