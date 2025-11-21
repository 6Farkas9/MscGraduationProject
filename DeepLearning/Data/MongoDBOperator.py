# MongoDBOperator.py 修复版
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymongo
from pymongo import MongoClient
from bson import Binary
import pickle
import torch
import numpy as np
from typing import Dict, List, Optional

class BaseMongoDBRepository:
    """MongoDB基础操作类 - 修复序列化方式"""
    
    def __init__(self, database_name="MLS", host="localhost", port=27017):
        self.client = MongoClient(host=host, port=port)
        self.db = self.client[database_name]
    
    def insert_one(self, collection_name: str, document: Dict) -> str:
        """插入单个文档"""
        collection = self.db[collection_name]
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    def insert_many(self, collection_name: str, documents: List[Dict]) -> List[str]:
        """批量插入文档"""
        collection = self.db[collection_name]
        result = collection.insert_many(documents)
        return [str(id) for id in result.inserted_ids]
    
    def upsert_one(self, collection_name: str, query: Dict, update: Dict) -> str:
        """更新或插入单个文档"""
        collection = self.db[collection_name]
        result = collection.update_one(query, {'$set': update}, upsert=True)
        if result.upserted_id:
            return str(result.upserted_id)
        return str(result.modified_count)
    
    def upsert_many(self, collection_name: str, documents: List[Dict], key_fields: List[str]) -> Dict:
        """批量更新或插入文档"""
        collection = self.db[collection_name]
        bulk_operations = []
        
        for doc in documents:
            # 构建查询条件
            query = {field: doc[field] for field in key_fields if field in doc}
            operation = pymongo.UpdateOne(query, {'$set': doc}, upsert=True)
            bulk_operations.append(operation)
        
        if bulk_operations:
            result = collection.bulk_write(bulk_operations)
            return {
                'upserted_count': result.upserted_count,
                'modified_count': result.modified_count,
                'matched_count': result.matched_count
            }
        return {'upserted_count': 0, 'modified_count': 0, 'matched_count': 0}
    
    def find_one(self, collection_name: str, query: Dict) -> Optional[Dict]:
        """查找单个文档"""
        collection = self.db[collection_name]
        return collection.find_one(query)
    
    def find_many(self, collection_name: str, query: Dict = None, 
                 projection: Dict = None, limit: int = 0) -> List[Dict]:
        """查找多个文档"""
        collection = self.db[collection_name]
        if query is None:
            query = {}
        cursor = collection.find(query, projection)
        if limit > 0:
            cursor = cursor.limit(limit)
        return list(cursor)
    
    def update_one(self, collection_name: str, query: Dict, update: Dict) -> int:
        """更新单个文档"""
        collection = self.db[collection_name]
        result = collection.update_one(query, {'$set': update})
        return result.modified_count
    
    def update_many(self, collection_name: str, query: Dict, update: Dict) -> int:
        """更新多个文档"""
        collection = self.db[collection_name]
        result = collection.update_many(query, {'$set': update})
        return result.modified_count
    
    def delete_one(self, collection_name: str, query: Dict) -> int:
        """删除单个文档"""
        collection = self.db[collection_name]
        result = collection.delete_one(query)
        return result.deleted_count
    
    def delete_many(self, collection_name: str, query: Dict) -> int:
        """删除多个文档"""
        collection = self.db[collection_name]
        result = collection.delete_many(query)
        return result.deleted_count
    
    def count_documents(self, collection_name: str, query: Dict = None) -> int:
        """统计文档数量"""
        collection = self.db[collection_name]
        if query is None:
            query = {}
        return collection.count_documents(query)
    
    def create_index(self, collection_name: str, keys: List, unique: bool = False) -> str:
        """创建索引"""
        collection = self.db[collection_name]
        return collection.create_index(keys, unique=unique)
    
    def tensor_to_list(self, tensor: torch.Tensor) -> List:
        """将PyTorch Tensor转换为Python列表 - 修复序列化方式"""
        if tensor is None:
            return None
        # 转换为numpy数组然后转换为Python列表
        numpy_array = tensor.detach().cpu().numpy()
        return numpy_array.tolist()
    
    def list_to_tensor(self, data_list: List, device: str = 'cpu') -> torch.Tensor:
        """将Python列表转换为PyTorch Tensor"""
        if data_list is None:
            return None
        numpy_array = np.array(data_list)
        return torch.from_numpy(numpy_array).to(device)
    
    # 保留旧方法用于兼容性，但不再使用
    def tensor_to_binary(self, tensor: torch.Tensor) -> Binary:
        """将PyTorch Tensor转换为Binary格式（不再使用）"""
        if tensor is None:
            return None
        numpy_array = tensor.detach().cpu().numpy()
        return Binary(pickle.dumps(numpy_array, protocol=2))
    
    def binary_to_tensor(self, binary_data: Binary, device: str = 'cpu') -> torch.Tensor:
        """将Binary格式转换为PyTorch Tensor（不再使用）"""
        if binary_data is None:
            return None
        numpy_array = pickle.loads(binary_data)
        return torch.from_numpy(numpy_array).to(device)
    
    def close_connection(self):
        """关闭数据库连接"""
        self.client.close()

# 创建全局实例
mongodb_op = BaseMongoDBRepository()