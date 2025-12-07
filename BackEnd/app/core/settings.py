# BackEnd/app/core/settings.py
import os
import sys
from typing import Dict, Any


class PathSettings:
    """
    路径配置：
    - project_root: 包含 BackEnd 与 DeepLearning 的父目录
    - backend_dir: BackEnd 目录
    - deep_learning_dir: DeepLearning 目录
    """

    def __init__(self) -> None:
        # 当前文件所在目录：BackEnd/app/core/
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # BackEnd 目录：core 的上两级
        backend_dir = os.path.dirname(os.path.dirname(current_dir))
        project_root = os.path.dirname(backend_dir)

        self._project_root = project_root
        self._backend_dir = backend_dir
        self._deep_learning_dir = os.path.join(project_root, "DeepLearning")

        # 将 project_root 加入 sys.path，方便直接导入 DeepLearning 下模块
        if self._project_root not in sys.path:
            sys.path.insert(0, self._project_root)

    @property
    def project_root(self) -> str:
        return self._project_root

    @property
    def backend_dir(self) -> str:
        return self._backend_dir

    @property
    def deep_learning_dir(self) -> str:
        return self._deep_learning_dir


class DatabaseSettings:
    """
    数据库配置：
    - MySQL 配置
    - MongoDB 配置
    """

    def __init__(self) -> None:
        # 实际使用时建议从环境变量或配置文件中读取
        self._mysql_config: Dict[str, Any] = {
            "host": "localhost",
            "port": 3306,
            "user": "root",
            "password": "123456",
            "database": "mls_sample",
            "charset": "utf8mb4",
            "autocommit": True,
        }

        self._mongodb_config: Dict[str, Any] = {
            "host": "localhost",
            "port": 27017,
            "database": "MLS",
            "username": None,
            "password": None,
            "auth_source": "admin",
        }

    @property
    def mysql_config(self) -> Dict[str, Any]:
        # 返回副本防止外部修改
        return dict(self._mysql_config)

    @property
    def mongodb_config(self) -> Dict[str, Any]:
        return dict(self._mongodb_config)

    def build_mongodb_uri(self) -> str:
        cfg = self._mongodb_config
        if cfg["username"] and cfg["password"]:
            return (
                f"mongodb://{cfg['username']}:{cfg['password']}"
                f"@{cfg['host']}:{cfg['port']}/{cfg['database']}?authSource={cfg['auth_source']}"
            )
        return f"mongodb://{cfg['host']}:{cfg['port']}/{cfg['database']}"


# 全局配置实例（这里是“配置对象”，不会直接产生数据库连接）
path_settings = PathSettings()
db_settings = DatabaseSettings()
