import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
from tqdm import tqdm

from datetime import datetime, timedelta

from Data.MySQLOperator import mysqldb
from Data.MongoDBOperator import mongodb

class IPDKTDataReader():

    def __init__(self):
        self.are_uid = ""

    def set_are_uid(self, are_uid):
        self.are_uid = are_uid

    # def __init__(self, are_uid):
    #     self.are_uid = are_uid
    #     print(self.are_uid)

    # 这里是要准备KT所需要的交互数据
    # 学习者uid - 数字id
    # 场景uid   - 数字id
    # 学习者数字id    场景数字id  知识点数字id，数字id...     正确与否

    # 获得只包含当前领域相关知识点的场景的交互信息 - 添加了has_result属性，要获取has_result为1的场景
    def get_all_recordings_with_result(self, limit = -1):
        time_start = self.get_30days_before()
        result = mysqldb.get_interacts_with_cpt_in_are_with_result(
                self.are_uid,
                time_start,
                limit
            )
        return result
    
    # 获得当前领域的所有知识点，cpt_uid, inner_id
    def get_all_concepts_of_area(self):
        cpt_uids = mysqldb.get_all_concepts_uid_and_id_of_area(self.are_uid)
        return cpt_uids

    def get_30days_before(self):
        return (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    
    def get_concepts_of_unit(self, unt_uids):
        return mysqldb.get_concepts_of_units(unt_uids)
    
    def get_cpt_num_of_area(self):
        return self.cpt_num
    
    def get_cpt_uid_of_unit(self, unt_uids):
        return mysqldb.get_concepts_uid_of_units(unt_uids)
    
    def load_area_uids(self):
        return mysqldb.get_areas_uid()
    
    def get_lrn_uids(self):
        return self.lrn_uids

    # 从数据库中获取所有数据
    def load_data_from_db(self):
        # 获取当前领域所有知识点相关的场景的交互信息
        # result = [(lrn_uid, unt_uid, correct),...]
        interacts = self.get_all_recordings_with_result()
        cpt_uids = self.get_all_concepts_of_area()
        self.cpt_uids = cpt_uids
        self.cpt_num = len(cpt_uids)
        self.cpt_id2uid = {cpt_uids[cpt_uid] : cpt_uid for cpt_uid in cpt_uids}
        # print(cpt_uids)
        # cpt_num = self.get_concept_num_of_area()

        # 构建lrn和unt的数字id -- 貌似，转化就行
        lrn_uids_set = set()
        unt_uids_set = set()

        for interact in interacts:
            lrn_uids_set.add(interact[0])
            unt_uids_set.add(interact[1])

        lrn_uids_list = list(lrn_uids_set)
        unt_uids = list(unt_uids_set)

        lrn_uids = {lrn_uid : idx for idx, lrn_uid in enumerate(lrn_uids_list)}
        self.lrn_uids = lrn_uids
        self.lrn_id2uid = {idx : lrn_uid for idx, lrn_uid in enumerate(lrn_uids_list)}
        
        unt_cpts = self.get_cpt_uid_of_unit(list(unt_uids))
        
        data = [[lrn_uid, [], []] for lrn_uid in lrn_uids]
        for interact in interacts:
            lrn_idx = lrn_uids[interact[0]]
            # unt_id = unt_uids[interact[1]]
            correct = float(interact[2])
            data[lrn_idx][1].append([cpt_uids[cpt_uid] for cpt_uid in unt_cpts[interact[1]]])
            data[lrn_idx][2].append(correct)

        self.data = data

        train_data = []
        master_data = []
        pos = -1
        for onelrndata in data:
            unt_num = len(onelrndata[2])
            if unt_num == 1:
                continue
            unt_train_num = int(unt_num * 0.8)
            # unt_master_num = unt_num - unt_train_num
            pos += 1
            train_data.append([0, [] ,[]])
            master_data.append([0, [] ,[]])
            train_data[pos][0] = onelrndata[0]
            master_data[pos][0] = onelrndata[0]
            
            for i in range(unt_train_num):
                train_data[pos][1].append(onelrndata[1][i])
                train_data[pos][2].append(onelrndata[2][i])
                # train_data[pos][3].append(onelrndata[3][i])

            for i in range(unt_train_num, unt_num):
                master_data[pos][1].append(onelrndata[1][i])
                master_data[pos][2].append(onelrndata[2][i])
                # master_data[pos][3].append(onelrndata[3][i])
        
        self.cpt_uids_list = list(cpt_uids.keys())

        return train_data, master_data, cpt_uids
    
    # # 应对KT的要求，将此次参与训练的cpt置为trained，之后在使用的时候kt只能预测这些知识点
    def make_cpt_trained(self):
        mysqldb.make_cpt_trained(self.cpt_uids_list)

    # 获取当前时间前推30天内的所有interacts数据
    def load_final_data(self, lrn_uid, device):
        # shape: [[lrn_uid, [[cpt_uid, cpt_uid], [...],  ...], [correct, correct]], [....], ...]

        the_lrn_data = self.data[self.lrn_uids[lrn_uid]]
        interact_num = len(the_lrn_data[1])

        final_data = torch.zeros(interact_num, self.cpt_num * 2, dtype=torch.float32, device=device)

        row = []
        col = []

        for i in range(interact_num):
            skip = self.cpt_num * (1 - the_lrn_data[2][i])
            row.extend([i] * len(the_lrn_data[1][i]))
            col.extend([cpt_id + skip for cpt_id in the_lrn_data[1][i]])

        final_data[row, col] = 1.0

        return final_data, self.cpt_id2uid

    def save_final_data(self, final_data):
        mongodb.save_kt_final_data(final_data)

if __name__ == '__main__':
    dr = IPDKTDataReader('are_3fee9e47d0f3428382f4afbcb1004117')
    # are_3fee9e47d0f3428382f4afbcb1004117
    data, num = dr.load_data_from_db()
    print(num)
    print(data)