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

        operations = []
        for lrn_uid, cpt_values in final_data.items():
            # 为每个cpt_uid创建更新路径
            update_fields = {f"KT.{cpt_uid}": value for cpt_uid, value in cpt_values.items()}
            
            # 添加更新操作到批量列表
            operations.append(
                UpdateOne(
                    {"_id": lrn_uid},  # 匹配条件
                    {"$set": update_fields},  # 更新操作
                    upsert=True  # 插入新文档，更新现有文档
                )
            )

        result = collection.bulk_write(operations)

    def save_final_lrn_emb(self, lrn_emb_dict):
        collection = self.mongo_db["learners"]

        operations = [
            UpdateOne(
                {"_id": lrn_uid},  # 查询条件
                {"$set": {
                    "HGC_Emb" : data
                }},    # 更新内容（完全替换匹配字段）
                upsert=True        # 有则更新无则插入
            )
            for lrn_uid, data in lrn_emb_dict.items()
        ]

        result = collection.bulk_write(operations)

    def save_final_scn_emb(self, scn_emb_dict):
        collection = self.mongo_db["scenes"]

        operations = [
            UpdateOne(
                {"_id": scn_uid},  # 查询条件
                {"$set": {
                    "HGC_Emb" : data
                }},    # 更新内容（完全替换匹配字段）
                upsert=True        # 有则更新无则插入
            )
            for scn_uid, data in scn_emb_dict.items()
        ]

        result = collection.bulk_write(operations)

    def save_final_cpt_emb(self, cpt_emb_dict):
        collection = self.mongo_db["concepts"]

        operations = [
            UpdateOne(
                {"_id": cpt_uid},  # 查询条件
                {"$set": {
                    "HGC_Emb" : data
                }},    # 更新内容（完全替换匹配字段）
                upsert=True        # 有则更新无则插入
            )
            for cpt_uid, data in cpt_emb_dict.items()
        ]

        result = collection.bulk_write(operations)

    def save_rr_final_r_pred_emb(self, r_pred_dict):
        collection = self.mongo_db["learners"]

        operations = []
        for lrn_uid, cpt_values in r_pred_dict.items():
            # 为每个cpt_uid创建更新路径
            update_fields = {f"RR.{cpt_uid}": value for cpt_uid, value in cpt_values.items()}
            
            # 添加更新操作到批量列表
            operations.append(
                UpdateOne(
                    {"_id": lrn_uid},  # 匹配条件
                    {"$set": update_fields},  # 更新操作
                    upsert=True  # 插入新文档，更新现有文档
                )
            )

        result = collection.bulk_write(operations)

    def save_cd_final_r_pred_emb(self, r_pred_dict):
        collection = self.mongo_db["learners"]

        operations = []
        for lrn_uid, cpt_values in r_pred_dict.items():
            # 为每个cpt_uid创建更新路径
            update_fields = {f"CD.{cpt_uid}": value for cpt_uid, value in cpt_values.items()}
            
            # 添加更新操作到批量列表
            operations.append(
                UpdateOne(
                    {"_id": lrn_uid},  # 匹配条件
                    {"$set": update_fields},  # 更新操作
                    upsert=True  # 插入新文档，更新现有文档
                )
            )

        result = collection.bulk_write(operations)

    def save_kcge_final_scn_emb(self, scn_emb_dict):
        collection = self.mongo_db["scenes"]

        operations = [
            UpdateOne(
                {"_id": cpt_uid},  # 查询条件
                {"$set": {
                    "KCGE_Emb" : data
                }},    # 更新内容（完全替换匹配字段）
                upsert=True        # 有则更新无则插入
            )
            for cpt_uid, data in scn_emb_dict.items()
        ]

        result = collection.bulk_write(operations)

    def save_kcge_final_cpt_emb(self, cpt_emb_dict):
        collection = self.mongo_db["concepts"]

        operations = [
            UpdateOne(
                {"_id": cpt_uid},  # 查询条件
                {"$set": {
                    "KCGE_Emb" : data
                }},    # 更新内容（完全替换匹配字段）
                upsert=True        # 有则更新无则插入
            )
            for cpt_uid, data in cpt_emb_dict.items()
        ]

        result = collection.bulk_write(operations)

    def get_cd_emb_of_scn_uids(self, scn_uids):
        collection = self.mongo_db["scnenes"]


        return 0
    
    def get_cd_emb_of_cpt_uids(self, cpt_uids):
        collection = self.mongo_db["concepts"]

        cursor = collection.find(
            {"_id": {"$in": cpt_uids}},
            {"_id": 1, "cd": 1}  # 只返回_id和cd字段
        )
        
        # 将结果转换为列表
        results = list(cursor)

        return 0

mongodb = MongoDB()
    
