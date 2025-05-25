import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
import pandas as pd

import numpy as np 

from Data.MySQLOperator import mysqldb
from Data.MongoDBOperator import mongodb
from HGC.Dataset.HGCDataReader import HGCDataReader

class CDDataReader():
    def __init__(self):
        self.hgcdr = HGCDataReader()

    def get_interacts_with_scn_greater_4(self, lrn_uids):
        # lrn_scn = [[] for _ in range(len(lrn_uids))]
        lrn_scn = {lrn_uid : [[], []] for lrn_uid in lrn_uids.keys()}
        interacts = mysqldb.get_interacts_with_scn_greater_4()
        for onedata in interacts:
            lrn_scn[onedata[0]][0].append(onedata[1])
            lrn_scn[onedata[0]][1].append(onedata[2])
        return lrn_scn
    
    def get_concepts_of_scenes(self, scn_uids):
        return mysqldb.get_concepts_of_scenes(scn_uids)

    def load_Data_from_db(self):
        uids, inits, p_matrixes = self.hgcdr.load_data_from_db()

        # 用来计算h_lrn
        lrn_scn = self.get_interacts_with_scn_greater_4(uids[0])
        # scn_cpt = self.get_concepts_of_scenes(list(uids[1].keys()))

        # 按照8:2的比例获取train和master数据
        train_data = {lrn_uid : [[], []] for lrn_uid in uids[0].keys()}
        master_data = {lrn_uid : [[], []] for lrn_uid in uids[0].keys()}
        for lrn_uid in lrn_scn:
            scn_uids = list(lrn_scn[lrn_uid][0])
            results = list(lrn_scn[lrn_uid][1])

            scn_num = len(scn_uids)
            
            train_num = max(1, int(scn_num * 0.8))

            # 训练集
            for i in range(train_num):
                train_data[lrn_uid][0].append(scn_uids[i])
                train_data[lrn_uid][1].append(results[i])
            
            # 测试集
            for i in range(train_num, scn_num):
                master_data[lrn_uid][0].append(scn_uids[i])
                master_data[lrn_uid][1].append(results[i])
        
        # 实际上返回的这些值都是不变的，dataset中需要根据lrn_uid来获取对应的子集
        return train_data, master_data, uids, inits, p_matrixes#, lrn_scn_mat, scn_cpt_mat
    
    def get_final_lrn_scn_index(self, lrn_uids, scn_uids):

        # 获取所有的特殊课程的scn_uid和cpt_uid
        # 通过uid和scn_uids获取每个特殊课程的id
        # 然后获取有序的cpt_uid_list

        special_scn_cpt_uids =  mysqldb.get_all_special_scn_cpt_uid()

        cpt_num = len(special_scn_cpt_uids)
        scn_mask  = torch.ones(len(lrn_uids), cpt_num, dtype=torch.float32)

        scn_seq = [scn_uids[scn_uid] for scn_uid, _ in special_scn_cpt_uids]
        cpt_uids_list_orderd = [cpt_uid for _, cpt_uid in special_scn_cpt_uids]
        scn_index_oneline = torch.tensor(scn_seq, dtype=torch.long)
        
        scn_index = scn_index_oneline.expand(len(lrn_uids), -1).contiguous()

        return scn_index, scn_mask, cpt_uids_list_orderd
    
    def save_final_data(self, lrn_uids, scn_uids, cpt_uids, lrn_emb, scn_emb, cpt_emb, r_pred, ordered_cpt_uids):

        lrn_emb_dict = {
            lrn_uid : c_lrn_emb.tolist() for lrn_uid, c_lrn_emb in zip(lrn_uids, lrn_emb)
        }

        mongodb.save_final_lrn_emb(lrn_emb_dict)

        scn_emb_dict = {
            scn_uid : c_scn_emb.tolist() for scn_uid, c_scn_emb in zip(scn_uids, scn_emb)
        }

        mongodb.save_final_scn_emb(scn_emb_dict)

        cpt_emb_dict = {
            cpt_uid : c_cpt_emb.tolist() for cpt_uid, c_cpt_emb in zip(cpt_uids, cpt_emb)
        }

        mongodb.save_final_cpt_emb(cpt_emb_dict)

        # cd不同于其他的模型，最终的结果计算要计算学习者关于特定场景的正确概率
        r_pred_dict = {
            lrn_uid: {
                cpt_uid: float(r_pred[i, j])  # 显式转换为Python float
                for j, cpt_uid in enumerate(ordered_cpt_uids)
            }
            for i, lrn_uid in enumerate(lrn_uids)
        }
        mongodb.save_cd_final_r_pred_emb(r_pred_dict)
    
if __name__ == '__main__':
    cddr = CDDataReader('are_3fee9e47d0f3428382f4afbcb1004117')
    
    td, md, uids, inits, p_matrixes = cddr.load_Data_from_db()

    test_key = list(md.keys())[0]
    print(md[test_key][1])
