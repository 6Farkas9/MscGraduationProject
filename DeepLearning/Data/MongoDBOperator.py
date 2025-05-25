import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

from pymongo import MongoClient
from pymongo import UpdateOne

class MongoDB:
    def __init__(self):
        self.mongo_client = MongoClient("mongodb://localhost:27017/")
        self.mongo_db = self.mongo_client["MLS_db"]

    def save_kt_final_data(self, final_data):
        collection = self.mongo_db["learners"]

        operations = [
            UpdateOne(
                {"_id": lrn_uid},  # 查询条件
                {"$set": {
                    "KT" : data
                }},    # 更新内容（完全替换匹配字段）
                upsert=True        # 有则更新无则插入
            )
            for lrn_uid, data in final_data.items()
        ]

        result = collection.bulk_write(operations)

        print(f"""
        匹配: {result.matched_count} 条
        修改: {result.modified_count} 条
        新增: {len(result.upserted_ids)} 条
        """)


mongodb = MongoDB()
    
