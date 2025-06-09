import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch

from datetime import datetime, timedelta
from collections import defaultdict

from Data.MySQLOperator import mysqldb
from Data.MongoDBOperator import mongodb
from KCGE.DataSet.KCGEDataReader import KCGEDataReader

class CDDataReader():
    def load_area_uids(self):
        return mysqldb.get_areas_uid()

    def set_are_uid(self, are_uid):
        self.are_uid = are_uid
        self.kcgedr = KCGEDataReader(are_uid)

    def get_all_recording(self, limit = -1):
        time_start = self.get_30days_before()
        result = mysqldb.get_interacts_of_are(
                self.are_uid,
                time_start,
                limit
            )
        return result
    
    def get_30days_before(self):
        return (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    
    def get_scn_of_are_with_result(self):
        return mysqldb.get_scn_of_are_with_result(self.are_uid)

    def load_Data_from_db(self):
        cpt_uids, scn_uids, cpt_idx, scn_idx, edge_index, edge_attr, edge_type = self.kcgedr.load_data_from_db()

        self.scn_uids = scn_uids
        self.scn_idx = scn_idx
        self.cpt_uids = cpt_uids
        self.cpt_idx = cpt_idx

        # 用来计算h_lrn

        # TODO:
        # 这里要改为获取所有的交互数据，然后根据是否有result来修改对应的mask
        interacts = self.get_all_recording()
        print(1)
        # 获取当前are下的所有有result的scn，不对，mask不是在这里搞的，实在dataset里搞的，dataset哪里默认是0，那么一个取巧的方法是让interacts里的默认值是0
        # 理论上special_scene下的所有scn都是没有result的
        # 还是要获取所有有或者所有没有result的scn，然后将对应的lrn_scn的对应位置的结果置为-1
        # 然后在dataset根据这个-1的位置来对对应的mask位置置0
        scn_uids_withresult = self.get_scn_of_are_with_result()
        print(2)

        lrn_scn = {}
        self.max_scn_num = 0
        for interact in interacts:
            lrn_uid = interact[0]
            if lrn_uid not in lrn_scn:
                lrn_scn[lrn_uid] = [[], []]
            lrn_scn[lrn_uid][0].append(interact[1])
            if interact[1] in scn_uids_withresult:
                lrn_scn[lrn_uid][1].append(interact[2])
            else:
                lrn_scn[lrn_uid][1].append(-1)

        self.lrn_scn = lrn_scn
        
        lrn_uids_list = list(lrn_scn.keys())
        lrn_uids = {lrn_uid : idx for idx, lrn_uid in enumerate(lrn_uids_list)}
        self.lrn_uids = lrn_uids

        for lrn_uid in lrn_uids:
            self.max_scn_num = max(self.max_scn_num, len(lrn_scn[lrn_uid][0]))

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
        
    
    def get_final_Data(self):

        # 获取所有的特殊课程的scn_uid和cpt_uid
        # 通过uid和scn_uids获取每个特殊课程的id
        # 然后获取有序的cpt_uid_list

        # 根据这个获取scn_index，scn_mask因为每个lrn的交互个数不同，还是要设置一下的
        num_learners = len(self.lrn_scn)
        scn_index = torch.zeros((num_learners, self.max_scn_num), dtype=torch.long)
        scn_mask = torch.zeros((num_learners, self.max_scn_num), dtype=torch.float)
        
        # 预先将scn_uids转换为defaultdict提高查找效率
        scn_uids_default = defaultdict(int, self.scn_uids)  # 不存在的key返回0
        
        for i, (lrn_uid, (scn_list, _)) in enumerate(self.lrn_scn.items()):
            seq_len = min(len(scn_list), self.max_scn_num)
            
            # 一次性处理所有场景
            scn_ids = [scn_uids_default[scn_uid] for scn_uid in scn_list[:seq_len]]
            scn_index[i, :seq_len] = torch.tensor(scn_ids, dtype=torch.long)
            scn_mask[i, :seq_len] = 1.0

        # 这个变量用来获取之后的h_scn和h_cpt
        special_scn_cpt_uids = mysqldb.get_special_scn_cpt_uid_of_are(self.are_uid)

        cpt_num = len(special_scn_cpt_uids)
        scn_mask_special  = torch.ones(len(self.lrn_uids), cpt_num, dtype=torch.float32)

        scn_seq = [self.scn_uids[scn_uid] for scn_uid, _ in special_scn_cpt_uids]
        self.cpt_uids_list_orderd = [cpt_uid for _, cpt_uid in special_scn_cpt_uids]
        scn_index_oneline = torch.tensor(scn_seq, dtype=torch.long)
        
        scn_index_special = scn_index_oneline.expand(len(self.lrn_uids), -1).contiguous()

        return scn_index, scn_mask, scn_index_special, scn_mask_special, self.scn_idx ,self.cpt_idx
    
    def save_final_data(self, r_pred, h_scn, h_cpt):

        # cd不同于其他的模型，最终的结果计算要计算学习者关于特定场景的正确概率
        r_pred_dict = {
            lrn_uid: {
                cpt_uid: float(r_pred[i, j])  # 显式转换为Python float
                for j, cpt_uid in enumerate(self.cpt_uids_list_orderd)
            }
            for i, lrn_uid in enumerate(list(self.lrn_uids.keys()))
        }
        mongodb.save_cd_final_r_pred_emb(r_pred_dict)

        scn_emb_dict = {
            scn_uid : h_scn[self.scn_uids[scn_uid]].tolist() for scn_uid in self.scn_uids
        }
        mongodb.save_kcge_final_scn_emb(scn_emb_dict)

        cpt_emb_dict = {
            cpt_uid : h_cpt[self.cpt_uids[cpt_uid]].tolist() for cpt_uid in self.cpt_uids
        }
        mongodb.save_kcge_final_cpt_emb(cpt_emb_dict)

    
if __name__ == '__main__':
    # cddr = CDDataReader('are_3fee9e47d0f3428382f4afbcb1004117')
    cddr = CDDataReader()
    cddr.set_are_uid('are_3fee9e47d0f3428382f4afbcb1004117')
    
    train_data, master_data, lrn_uids, cpt_uids, scn_uids, cpt_idx, scn_idx, edge_index, edge_attr, edge_type = cddr.load_Data_from_db()

    print(train_data)

    print(master_data)
