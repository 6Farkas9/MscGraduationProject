import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
from datetime import datetime, timedelta
from torch_geometric.data import Data

from Data.DBOperator import db

# DataReader的职责应该是从数据库中读取数据，构建出图
# 返回的结果传给Dataset

class HGCDataReader():
    
    def get_cpt_are(self):
        return db.get_cpt_are()

    def get_cpt_cpt(self):
        return db.get_cpt_cpt()
    
    def get_areas_uid(self):
        uids = db.get_areas_uid()
        return {uid : uids.index(uid) for uid in uids}
    
    def get_learners_uid_with_scn_greater_4(self):
        uids = db.get_learners_uid_with_scn_greater_4()
        self.lrn_uids = {uid : idx for idx, uid in enumerate(uids)}
        self.lrn_num = len(uids)

    def get_scenes_uid(self):
        uids = db.get_scenes_uid()
        self.scn_uids = {uid : idx for idx, uid in enumerate(uids)}
        self.scn_num = len(uids)

    def get_concepts_uid(self):
        uids = db.get_concepts_uid()
        self.cpt_uids = {uid : idx for idx, uid in enumerate(uids)}
        self.cpt_num = len(uids)

    def get_lrn_scn_num_with_scn_greater_4(self):
        return db.get_lrn_scn_num_with_scn_greater_4()
    
    def get_scn_cpt_dif(self):
        return db.get_scn_cpt_dif()
    
    def get_cpt_uid_name(self):
        return db.get_cpt_uid_name()

    def get_P_lsl(self):
        A = self.learners_init.clone()
        A_T = A.t()
        A = torch.matmul(A, A_T)
        A_I = torch.eye(A.size(0), A.size(1), dtype = torch.float)
        A = A + A_I
        line = torch.ones(A.size(0), 1, dtype = torch.float)
        D = torch.diag(torch.matmul(A, line).squeeze())
        D = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
        p = torch.matmul(torch.matmul(D, A), D)
        row, col = p.nonzero(as_tuple = True)
        weight = p[row, col]
        # self.p_lsl = Data(edge_index = torch.stack([row, col], dim=0), edge_attr = weight)
        self.p_lsl = (torch.stack([row, col], dim=0), weight)

    def get_P_cc(self):
        A = torch.zeros(self.cpt_num, self.cpt_num, dtype = torch.float)
        cpt_cpt_data = self.get_cpt_cpt()
        for onedata in cpt_cpt_data:
            cpt_id_pre = self.cpt_uids[onedata[0]]
            cpt_id_aft = self.cpt_uids[onedata[1]]
            A[cpt_id_pre][cpt_id_aft] = 1
        A_I = torch.eye(A.size(0), A.size(1), dtype = torch.float)
        A = A + A_I
        line = torch.ones(A.size(0), 1, dtype = torch.float)
        D = torch.diag(torch.matmul(A, line).squeeze())
        D = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
        p = torch.matmul(torch.matmul(D, A), D)
        row, col = p.nonzero(as_tuple = True)
        weight = p[row, col]
        # self.p_cc = Data(edge_index = torch.stack([row, col], dim=0), edge_attr = weight)
        self.p_cc = (torch.stack([row, col], dim=0), weight)
    
    def get_P_cac(self):
        are_uids = self.get_areas_uid()
        A = torch.zeros(self.cpt_num, len(are_uids), dtype = torch.float)
        cpt_are_data = self.get_cpt_are()
        for onedata in cpt_are_data:
            cpt_id = self.cpt_uids[onedata[0]]
            are_id = are_uids[onedata[1]]
            A[cpt_id][are_id] = 1
        A_T = A.t()
        A = torch.matmul(A, A_T)
        A_I = torch.eye(A.size(0), A.size(1), dtype = torch.float)
        A = A + A_I
        line = torch.ones(A.size(0), 1, dtype = torch.float)
        D = torch.diag(torch.matmul(A, line).squeeze())
        D = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
        p = torch.matmul(torch.matmul(D, A), D)
        row, col = p.nonzero(as_tuple = True)
        weight = p[row, col]
        # self.p_cac = Data(edge_index = torch.stack([row, col], dim=0), edge_attr = weight)
        self.p_cac = (torch.stack([row, col], dim=0), weight)
    
    def get_P_csc(self):
        A = self.scenes_init.clone().t()
        A_T = A.t()
        A = torch.matmul(A, A_T)
        A_I = torch.eye(A.size(0), A.size(1), dtype = torch.float)
        A = A + A_I
        line = torch.ones(A.size(0), 1, dtype = torch.float)
        D = torch.diag(torch.matmul(A, line).squeeze())
        D = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
        p = torch.matmul(torch.matmul(D, A), D)
        row, col = p.nonzero(as_tuple = True)
        weight = p[row, col]
        # self.p_csc = Data(edge_index = torch.stack([row, col], dim=0), edge_attr = weight)
        self.p_csc = (torch.stack([row, col], dim=0), weight)
    
    def get_P_scs(self):
        A = self.scenes_init.clone()
        A_T = A.t()
        A = torch.matmul(A, A_T)
        A_I = torch.eye(A.size(0), A.size(1), dtype = torch.float)
        A = A + A_I
        line = torch.ones(A.size(0), 1, dtype = torch.float)
        D = torch.diag(torch.matmul(A, line).squeeze())
        D = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
        p = torch.matmul(torch.matmul(D, A), D)
        row, col = p.nonzero(as_tuple = True)
        weight = p[row, col]
        # self.p_scs = Data(edge_index = torch.stack([row, col], dim=0), edge_attr = weight)
        self.p_scs = (torch.stack([row, col], dim=0), weight)
    
    def get_P_sls(self):
        A = self.learners_init.clone().t()
        A_T = A.t()
        A = torch.matmul(A, A_T)
        A_I = torch.eye(A.size(0), A.size(1), dtype = torch.float)
        A = A + A_I
        line = torch.ones(A.size(0), 1, dtype = torch.float)
        D = torch.diag(torch.matmul(A, line).squeeze())
        D = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
        p = torch.matmul(torch.matmul(D, A), D)
        row, col = p.nonzero(as_tuple = True)
        weight = p[row, col]
        # self.p_sls = Data(edge_index = torch.stack([row, col], dim=0), edge_attr = weight)
        self.p_sls = (torch.stack([row, col], dim=0), weight)

    def learner_init_embedding(self):
        # 使用ls初始化
        # 获取所有学习者uid - 数量
        # 获取所有场景uid - 数量
        # 构建ls矩阵
        # 计算出学习者的初始嵌入表达
        # 返回初始嵌入的结果
        self.learners_init = torch.zeros(self.lrn_num, self.scn_num, dtype=torch.float)
        lrn_scn_num = self.get_lrn_scn_num_with_scn_greater_4()
        for onedata in lrn_scn_num:
            lrn_pos = self.lrn_uids[onedata[0]]
            scn_pos = self.scn_uids[onedata[1]]
            times   = onedata[2]
            self.learners_init[lrn_pos][scn_pos] += times
        
        self.get_P_lsl()
        self.get_P_sls()

        # 计算度矩阵 D：每个学习者与不同场景的交互次数（非零元素的个数）
        # 计算每个学习者与不同场景的交互次数（非零元素的个数）
        D_diag = (self.learners_init > 0).sum(dim=1).float() 

        # 计算度矩阵 D 的逆（每个对角元素取倒数）
        D_inv_diag = torch.where(
            D_diag != 0,
            1.0 / D_diag.clamp(min = 1e-6),
            torch.zeros_like(D_diag, dtype=torch.float)
        )
        
        # D_inv = torch.diag(D_inv_diag)
        # self.learners_init = torch.matmul(D_inv, self.learners_init)
        self.learners_init = self.learners_init * D_inv_diag.unsqueeze(1)
    
    def concept_init_embedding(self):
        # 比较特殊
        # 这里没办法直接获取初始嵌入表达
        # 需要使用word2vec计算
        # 返回每个知识点的ascii码的tensor，不足128的填充0
        self.concepts_init = torch.zeros(self.cpt_num, 128, dtype=torch.float)
        cpt_uid_name = self.get_cpt_uid_name()
        for onedata in cpt_uid_name:
            cpt_pos = self.cpt_uids[onedata[0]]
            for i in range(len(onedata[1])):
                self.concepts_init[cpt_pos][i] = ord(onedata[1][i])
    
    def scene_init_embedding(self):
        # 使用sc初始化
        # 获取所有场景uid - 数量
        # 获取所有知识点uid - 数量
        # 构建sc矩阵
        # 计算出场景的初始嵌入表达
        # 返回初始嵌入的结果
        self.scenes_init = torch.zeros(self.scn_num, self.cpt_num, dtype=torch.float)
        scn_cpt_dif = self.get_scn_cpt_dif()
        for onedata in scn_cpt_dif:
            scn_pos = self.scn_uids[onedata[0]]
            cpt_pos = self.cpt_uids[onedata[1]]
            difficulty = onedata[2]
            self.scenes_init[scn_pos][cpt_pos] += difficulty
        
        self.get_P_scs()
        self.get_P_csc()

        # 计算度矩阵 D：每个学习者与不同场景的交互次数（非零元素的个数）
        # 计算每个学习者与不同场景的交互次数（非零元素的个数）
        D_diag = (self.scenes_init > 0).sum(dim=1) 

        # 计算度矩阵 D 的逆（每个对角元素取倒数）
        D_inv_diag = torch.where(
            D_diag != 0,
            1.0 / D_diag,
            torch.zeros_like(D_diag, dtype=torch.float)
        )
        
        # D_inv = torch.diag(D_inv_diag)
        # self.learners_init = torch.matmul(D_inv, self.scenes_init)
        self.scenes_init = self.scenes_init * D_inv_diag.unsqueeze(1)

    def load_data_from_db(self):
        self.get_learners_uid_with_scn_greater_4()
        self.get_scenes_uid()
        self.get_concepts_uid()
        # 学习者初始嵌入
        # 知识点的初始tensor
        # 场景的初始嵌入
        # 以上三个单独返回
        self.learner_init_embedding()
        self.scene_init_embedding()
        self.concept_init_embedding()

        # l-s-l（两个学习者和同一个场景交互过）      2 - 2   
        # c-c（知识点前修后继的关系）                        1
        # c-a-c（两个知识点属于一个领域）                 0 - 0
        # c-s-c（两个知识点属于同一个场景）              3 - 3
        # s-c-s（两个场景涉及同一个知识点）              3 - 3
        # s-l-s（两个场景被同一个学习者互动过）       2 - 2
        # 以上的可以作为图的data返回
        self.get_P_cc()
        self.get_P_cac()

        return  (self.lrn_uids, self.scn_uids, self.cpt_uids), \
                (self.learners_init, self.scenes_init, self.concepts_init), \
                (self.p_lsl, self.p_scs, self.p_sls, self.p_cc, self.p_cac, self.p_csc,)
    
    
if __name__ == '__main__':
    datareader =  HGCDataReader()
    # datareader.get_learners_uid()
    # datareader.get_scenes_uid()
    # datareader.get_concepts_uid()
    # print(datareader.lrn_num, datareader.scn_num, datareader.cpt_num)

    # res = datareader.get_lrn_scn_num()
    # print(type(res), res)

    ids, inits, ps =  datareader.load_data_from_db()
    
    print(ps)

    # datareader.learner_init_embedding()

    # print(datareader.learners_init.shape, datareader.learners_init.sum())
    # print(datareader.learners_init[0][0])

    