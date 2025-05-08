import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
import numpy as np
from datetime import datetime, timedelta
from torch_geometric.data import Data

from Data.DBOperator import db
from HGC.Dataset.HGCDataReader import HGCDataReader


class RRDataReader():
    
    def __init__(self, sample_num):
        self.sample_num = sample_num
        self.hgcdr = HGCDataReader()

    def get_concepts_of_scenes(self, scn_uids):
        return db.get_concepts_of_scenes(scn_uids)
    
    def get_interacts_with_scn_greater_2(self, lrn_uids):
        # lrn_scn = [[] for _ in range(len(lrn_uids))]
        lrn_scn = {lrn_uid : [] for lrn_uid in lrn_uids.keys()}
        interacts = db.get_interacts_with_scn_greater_2()
        for onedata in interacts:
            lrn_scn[onedata[0]].append(onedata[1])
        return lrn_scn

    def load_data_from_db(self):
        uids, inits, p_matrixes = self.hgcdr.load_data_from_db()
        
        lrn_scn = self.get_interacts_with_scn_greater_2(uids[0])
        scn_cpt = self.get_concepts_of_scenes(list(uids[1].keys()))

        # 为方便之后的计算，这里要计算出scn_cpt的二维矩阵，之后可以直接使用矩阵运算
        # 不用在这运算
        
        # 获取学习者和知识点之间的交互情况
        # 这里要改，改成最后一个统计01情况，前面的统计学习次数，直接构建训练集和测试集
        
        # lrn_cpt：一个两行n列的tensor，其中第一行是训练集的知识点学习次数，第二行是测试集的学习与否
        # lrn_cpt = {lrn_uid : np.zeros((2, len(uids[2])), dtype=np.float32) for lrn_uid in uids[0].keys()}

        train_data = {lrn_uid : [[], np.zeros(len(uids[2]), dtype=np.float32)] for lrn_uid in uids[0].keys()}
        master_data = {lrn_uid : [[], np.zeros(len(uids[2]), dtype=np.float32)] for lrn_uid in uids[0].keys()}
        for lrn_uid in lrn_scn:
            scn_uids = list(lrn_scn[lrn_uid])
            # 训练集
            for i in range(len(scn_uids) - 1):
                train_data[lrn_uid][0].append(scn_uids[i])
                for cpt_uid in scn_cpt[scn_uids[i]]:
                    train_data[lrn_uid][1][uids[2][cpt_uid]] += 1
            # 添加训练集负采样
            non_zero_num = np.count_nonzero(train_data[lrn_uid][1] != 0)
            neg_num = max(0, self.sample_num - non_zero_num)
            zero_indexes = np.where(train_data[lrn_uid][1] == 0)[0]
            select_indexes = np.random.choice(zero_indexes, neg_num, replace=True)
            for idx in select_indexes:
                train_data[lrn_uid][1][idx] += 1
            # 测试集
            master_data[lrn_uid][0].append(scn_uids[-1])
            for cpt_uid in scn_cpt[scn_uids[-1]]:
                master_data[lrn_uid][1][uids[2][cpt_uid]] += 1
            non_zero_num = np.count_nonzero(master_data[lrn_uid][1] != 0)
            neg_num = min(non_zero_num, len(uids[2]) - non_zero_num)
            zero_indexes = np.where(master_data[lrn_uid][1] == 0)[0]
            select_indexes = np.random.choice(zero_indexes, neg_num, replace=True)
            for idx in select_indexes:
                master_data[lrn_uid][1][idx] += 1

        # 计算动态场景嵌入的知识点索引
        rows = []
        cols = []
        for scn_uid, cpt_uids in scn_cpt.items():
            scn_idx = uids[1][scn_uid]
            for cpt_uid in cpt_uids:
                cpt_idx = uids[2][cpt_uid]
                rows.append(scn_idx)
                cols.append(cpt_idx)

        index_matrix = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(index_matrix.shape[1])

        dynamic_scn_mat = torch.sparse_coo_tensor(
            indices=index_matrix,
            values=values,
            size=(len(uids[1]), len(uids[2])),
        )

        return train_data, master_data, uids, inits, p_matrixes, dynamic_scn_mat
    
if __name__ == '__main__':
    rrdr = RRDataReader(128)
    td, md, uids, inits, p_matrixes, dsm = rrdr.load_data_from_db()
    print(dsm)
    # print(len(md))
    # print(len(ls))
    # print(len(sc))
    # print(len(lc))
    # print(lc[list(uids[0].keys())[0]])