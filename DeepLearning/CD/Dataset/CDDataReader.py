import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch

from datetime import datetime, timedelta

from Data.MySQLOperator import mysqldb
from Data.MongoDBOperator import mongodb
from KCGE.DataSet.KCGEDataReader import KCGEDataReader

class CDDataReader():
    def load_area_uids(self):
        return mysqldb.get_areas_uid()

    def set_are_uid(self, are_uid):
        self.are_uid = are_uid
        self.kcgedr = KCGEDataReader(are_uid)

    def get_all_recordings_with_result(self, limit = -1):
        time_start = self.get_30days_before()
        result = mysqldb.get_interacts_with_cpt_in_are_from_with_result(
                self.are_uid,
                time_start,
                limit
            )
        return result
    
    def get_30days_before(self):
        return (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    
    # def get_cd_emb_of_scn_uids(self, scn_uids):
    #     return mongodb.get_cd_emb_of_scn_uids(scn_uids)

    def load_Data_from_db(self):
        cpt_uids, scn_uids, cpt_idx, scn_idx, edge_index, edge_attr, edge_type = self.kcgedr.load_data_from_db()

        # 用来计算h_lrn
        interacts = self.get_all_recordings_with_result()

        lrn_scn = {}
        for interact in interacts:
            lrn_uid = interact[0]
            if lrn_uid not in lrn_scn:
                lrn_scn[lrn_uid] = [[], []]
            lrn_scn[lrn_uid][0].append(interact[1])
            lrn_scn[lrn_uid][1].append(interact[2])
        
        lrn_uids_list = list(lrn_scn.keys())
        lrn_uids = {lrn_uid : idx for idx, lrn_uid in enumerate(lrn_uids_list)}

        # print(lrn_uids)

        # 按照8:2的比例获取train和master数据
        train_data = {lrn_uid : [[], []] for lrn_uid in lrn_uids}
        master_data = {lrn_uid : [[], []] for lrn_uid in lrn_uids}

        for lrn_uid in lrn_scn:
            interact_scn_uids = list(lrn_scn[lrn_uid][0])
            results = list(lrn_scn[lrn_uid][1])

            scn_num = len(interact_scn_uids)
            
            train_num = max(1, int(scn_num * 0.8))

            # 训练集
            for i in range(train_num):
                train_data[lrn_uid][0].append(interact_scn_uids[i])
                train_data[lrn_uid][1].append(results[i])
            
            # 测试集
            for i in range(train_num, scn_num):
                master_data[lrn_uid][0].append(interact_scn_uids[i])
                master_data[lrn_uid][1].append(results[i])
        
        # 实际上返回的这些值都是不变的，dataset中需要根据lrn_uid来获取对应的子集
        return train_data, master_data, lrn_uids, cpt_uids, scn_uids, cpt_idx, scn_idx, edge_index, edge_attr, edge_type
    
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
    
    def save_final_data(self, lrn_uids, r_pred, ordered_cpt_uids):

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
    # cddr = CDDataReader('are_3fee9e47d0f3428382f4afbcb1004117')
    cddr = CDDataReader()
    cddr.set_are_uid('are_3fee9e47d0f3428382f4afbcb1004117')
    
    train_data, master_data, lrn_uids, cpt_uids, scn_uids, cpt_idx, scn_idx, edge_index, edge_attr, edge_type = cddr.load_Data_from_db()

    print(train_data)

    print(master_data)
