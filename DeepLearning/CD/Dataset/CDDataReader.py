import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
import pandas as pd

import numpy as np 

from Data.MySQLOperator import mysqldb
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

            if scn_num == 0:
                print('shit')
            
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

    def load_Data(self):
        reader_file = pd.read_csv(self.path, header=0)

        num_stu = 0
        stu_all = {} # {stu_origin_id:newid,....}
        stu_exer = []
        len_row, _ = reader_file.shape

        res_data = []

        for i in range(len_row):
            stu_name = reader_file.iloc[i,0]
            exer_id = reader_file.iloc[i,1]
            correct = reader_file.iloc[i,2]

            if stu_name not in stu_all.keys():
                stu_all[stu_name] = num_stu
                stu_exer.append([])
                num_stu += 1
            
            stu_id = stu_all[stu_name]
            if exer_id not in stu_exer[stu_id]:
                stu_exer[stu_id].append(exer_id)
            res_data.append([stu_id,exer_id,correct])

        res_data = torch.tensor(res_data, dtype=torch.long).to(self.device)

        return res_data, stu_exer
    
if __name__ == '__main__':
    cddr = CDDataReader('are_3fee9e47d0f3428382f4afbcb1004117')
    
    td, md, uids, inits, p_matrixes = cddr.load_Data_from_db()

    test_key = list(md.keys())[0]
    print(md[test_key][1])
