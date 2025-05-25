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

    def save_rr_final_lrn_emb(self, lrn_emb_dict):
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

    def save_rr_final_scn_emb(self, scn_emb_dict):
        collection = self.mongo_db["scnenes"]

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

    def save_rr_final_cpt_emb(self, cpt_emb_dict):
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

        operations = [
            UpdateOne(
                {"_id": lrn_uid},  # 查询条件
                {"$set": {
                    "RR_PRED" : data
                }},    # 更新内容（完全替换匹配字段）
                upsert=True        # 有则更新无则插入
            )
            for lrn_uid, data in r_pred_dict.items()
        ]

        result = collection.bulk_write(operations)


mongodb = MongoDB()
    
