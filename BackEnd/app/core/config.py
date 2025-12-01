# config.py
import os
import sys
from typing import Dict, Any

class PathConfig:
    """路径配置类"""
    
    def __init__(self):
        # 获取当前config.py文件的目录
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 计算项目根路径：BackEnd的父目录（包含BackEnd和DeepLearning的目录）
        # 假设config.py在 BackEnd/app/config/ 目录下
        backend_dir = os.path.dirname(os.path.dirname(current_file_dir))
        self.project_root = os.path.dirname(backend_dir)  # BackEnd的父目录
        
        # 重要目录路径
        self.backend_dir = backend_dir
        self.deep_learning_dir = os.path.join(self.project_root, 'DeepLearning')
        
        # 添加项目根路径到Python路径（这样可以直接导入DeepLearning）
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
    
    def get_project_root(self) -> str:
        """获取项目根路径（包含BackEnd和DeepLearning的目录）"""
        return self.project_root
    
    def get_backend_dir(self) -> str:
        """获取BackEnd目录路径"""
        return self.backend_dir
    
    def get_deep_learning_dir(self) -> str:
        """获取DeepLearning目录路径"""
        return self.deep_learning_dir

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
path_config = PathConfig()
db_config = DatabaseConfig()