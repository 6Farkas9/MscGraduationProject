import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer

from DataReader.BasicDataReader import basicdr
from Data.HGCRepository import hgcrepo

# DataReader的职责应该是从数据库中读取数据，构建出图
# 返回的结果传给Dataset

class HGCDataReader():

    def __init__(self):

        self.lrn_uid = basicdr.lrn_uid
        self.lrn_num = basicdr.lrn_num
        self.untqus_uid = basicdr.untqus_uid
        self.untqus_num = basicdr.untqus_num
        self.tpc_uid = basicdr.tpc_uid
        self.tpc_num = basicdr.tpc_num
        self.crs_uid = basicdr.crs_uid
        self.crs_num = basicdr.crs_num
        self.cpt_uid = basicdr.cpt_uid
        self.cpt_num = basicdr.cpt_num
        self.qus_start = basicdr.unt_num

        self.getCptUid2Name()

        self.lrn_untqus_count = hgcrepo.getLrnUntQusCount()
        self.lrn_crs_count = hgcrepo.getLrnCrsCount()
        self.lrn_tpc_count = hgcrepo.getLrnTpcCount()

        self.untqus_cpt = hgcrepo.getUntQusCpt()
        self.unt_crs = hgcrepo.getUntCrs()
        self.unt_unt = hgcrepo.getUntUnt()

        self.cpt_tpc = hgcrepo.getCptTpc()
        self.cpt_cpt = hgcrepo.getCptCpt()


    def computeMetaPathMatrix(self, A1, A2=None):
        """
        计算元路径邻接矩阵
        
        Args:
            A1: 第一个邻接矩阵
            A2: 第二个邻接矩阵（如果为None，则使用A1的转置）
        
        Returns:
            A_meta: 元路径邻接矩阵
        """
        if A2 is None:
            A2 = A1.t()
        
        A_meta = torch.matmul(A1, A2)
        return A_meta
    
    def computeNormalizedAdjacencyMatrix(self, A, add_self_loop=True):
        """
        计算归一化邻接矩阵 P = D^(-1) * (A + I) * D^(-1)
        
        Args:
            A: 原始邻接矩阵 (torch.Tensor)
            add_self_loop: 是否添加自连接
        
        Returns:
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges]
        """
        # 1. 添加自连接
        if add_self_loop:
            I = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
            A = A + I
        
        # 2. 计算度矩阵 D
        row_sum = A.sum(dim=1)  # 每行的和
        D_diag = torch.where(
            row_sum != 0,
            1.0 / torch.sqrt(row_sum.clamp(min=1e-6)),
            torch.zeros_like(row_sum, dtype=A.dtype)
        )
        D_inv_sqrt = torch.diag(D_diag)
        
        # 3. 计算归一化邻接矩阵 P = D^(-1) * A * D^(-1)
        P = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
        
        # 4. 提取非零元素作为边
        row, col = P.nonzero(as_tuple=True)
        edge_index = torch.stack([row, col], dim=0)
        edge_weight = P[row, col]
        
        return edge_index, edge_weight

    def getCptUid2Name(self):
        uid_name = hgcrepo.getCptUidName()
        self.cpt_uid2name = {uid : name for (uid, name) in uid_name}

    def getP_lul(self):
        A_lu = torch.zeros(self.lrn_num, self.untqus_num, dtype=torch.float)
        lrn_untqus_count = self.lrn_untqus_count
        for onedata in lrn_untqus_count:
            lrn_uid, untqus_uid, count = onedata
            lrn_idx = self.lrn_uid.get(lrn_uid)
            untqus_idx = self.untqus_uid.get(untqus_uid)
            
            if lrn_idx is not None and untqus_idx is not None:
                A_lu[lrn_idx, untqus_idx] = count
        A_lul = self.computeMetaPathMatrix(A_lu)
        edge_index, edge_weight = self.computeNormalizedAdjacencyMatrix(A_lul, True)
        self.p_lul = (edge_index, edge_weight)

    def getP_lcl(self):
        A_lc = torch.zeros(self.lrn_num, self.crs_num, dtype=torch.float)
        lrn_crs_count = self.lrn_crs_count
        for onedata in lrn_crs_count:
            lrn_uid, crs_uid, count = onedata
            lrn_idx = self.lrn_uid.get(lrn_uid)
            crs_idx = self.crs_uid.get(crs_uid)
            
            if lrn_idx is not None and crs_idx is not None:
                A_lc[lrn_idx, crs_idx] = count
        A_lcl = self.computeMetaPathMatrix(A_lc)
        edge_index, edge_weight = self.computeNormalizedAdjacencyMatrix(A_lcl, True)
        self.p_lcl = (edge_index, edge_weight)

    def getP_ltl(self):
        A_lt = torch.zeros(self.lrn_num, self.tpc_num, dtype=torch.float)
        lrn_tpc_count = self.lrn_tpc_count
        for onedata in lrn_tpc_count:
            lrn_uid, tpc_uid, count = onedata
            lrn_idx = self.lrn_uid.get(lrn_uid)
            tpc_idx = self.tpc_uid.get(tpc_uid)
            
            if lrn_idx is not None and tpc_idx is not None:
                A_lt[lrn_idx, tpc_idx] = count
        A_ltl = self.computeMetaPathMatrix(A_lt)
        edge_index, edge_weight = self.computeNormalizedAdjacencyMatrix(A_ltl, True)
        self.p_ltl = (edge_index, edge_weight)

    def getInit(self, init):
        D_diag = (init > 0).sum(dim=1).float()
        
        # 归一化
        D_inv_diag = torch.where(
            D_diag != 0,
            1.0 / D_diag.clamp(min=1e-6),
            torch.zeros_like(D_diag, dtype=torch.float)
        )
        init = init * D_inv_diag.unsqueeze(1)

    def getLearnerInit(self):
        self.lrn_init = torch.zeros(self.lrn_num, self.untqus_num, dtype=torch.float)
        # 获取交互数据
        lrn_untqus_count = self.lrn_untqus_count
        
        for onedata in lrn_untqus_count:
            lrn_uid, untqus_uid, _ = onedata
            lrn_idx = self.lrn_uid.get(lrn_uid)
            untqus_idx = self.untqus_uid.get(untqus_uid)
            if lrn_idx is not None and untqus_idx is not None:
                self.lrn_init[lrn_idx, untqus_idx] = 1.0
        
        self.getInit(self.lrn_init)
        
        # return self.lrn_init

    def getUnitInit(self):
        self.untqus_init = torch.zeros(self.untqus_num, self.cpt_num, dtype=torch.float)

        untqus_cpt = self.untqus_cpt

        for onedata in untqus_cpt:
            untqus_uid, cpt_uid = onedata
            untqus_idx = self.untqus_uid.get(untqus_uid)
            cpt_idx = self.cpt_uid.get(cpt_uid)
            if untqus_idx is not None and cpt_idx is not None:
                self.untqus_init[untqus_idx, cpt_idx] = 1.0
        
        self.getInit(self.untqus_init)

    def getP_ulu(self):
        A_ul = torch.zeros(self.untqus_num, self.lrn_num, dtype=torch.float)
        lrn_untqus_count = self.lrn_untqus_count
        for onedata in lrn_untqus_count:
            lrn_uid, untqus_uid, count = onedata
            lrn_idx = self.lrn_uid.get(lrn_uid)
            untqus_idx = self.untqus_uid.get(untqus_uid)
            
            if lrn_idx is not None and untqus_idx is not None:
                A_ul[untqus_idx, lrn_idx] = count
        A_ulu = self.computeMetaPathMatrix(A_ul)
        edge_index, edge_weight = self.computeNormalizedAdjacencyMatrix(A_ulu, True)
        self.p_ulu = (edge_index, edge_weight)

    def getP_ucrsu(self):
        A_ucrs = torch.zeros(self.untqus_num, self.lrn_num, dtype=torch.float)
        unt_crs = self.unt_crs
        for onedata in unt_crs:
            unt_uid, crs_uid = onedata
            unt_idx = self.untqus_uid.get(unt_uid)
            crs_idx = self.crs_uid.get(crs_uid)

            if unt_idx is not None and crs_idx is not None:
                A_ucrs[unt_idx, crs_idx] = 1.0
        A_ucrsu = self.computeMetaPathMatrix(A_ucrs)
        edge_index, edge_weight = self.computeNormalizedAdjacencyMatrix(A_ucrsu, True)
        self.p_ucrsu = (edge_index, edge_weight)

    def getP_ucptu(self):
        A_ucpt = torch.zeros(self.untqus_num, self.cpt_num, dtype=torch.float)
        untqus_cpt = self.untqus_cpt
        for onedata in untqus_cpt:
            untqus_uid, cpt_uid = onedata
            untqus_idx = self.untqus_uid.get(untqus_uid)
            cpt_idx = self.cpt_uid.get(cpt_uid)

            if untqus_idx is not None and cpt_idx is not None:
                A_ucpt[untqus_idx, cpt_idx] = 1.0
        A_ucptu = self.computeMetaPathMatrix(A_ucpt)
        edge_index, edge_weight = self.computeNormalizedAdjacencyMatrix(A_ucptu, True)
        self.p_ucptu = (edge_index, edge_weight)

    def getP_uu(self):
        A_uu = torch.zeros(self.untqus_num, self.untqus_num, dtype=torch.float)
        unt_unt = self.unt_unt
        for onedata in unt_unt:
            # print(onedata)
            uid1, uid2 = onedata
            uid1_idx = self.untqus_uid.get(uid1)
            uid2_idx = self.untqus_uid.get(uid2)

            if uid1_idx is not None and uid2_idx is not None:
                A_uu[uid1_idx, uid2_idx] = 1.0
        
        for i in range(self.qus_start, self.untqus_num):
            A_uu[i, i] = 1.0
        
        edge_index, edge_weight = self.computeNormalizedAdjacencyMatrix(A_uu, False)
        self.p_uu = (edge_index, edge_weight)

    def getCptInit(self, model_name='all-MiniLM-L6-v2'):
        # 获取项目根目录
        deeplearningroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 构建模型保存路径：DeepLearning/Model/all-MiniLM-L6-v2
        model_path = os.path.join(deeplearningroot, "Model", model_name)
        
        # 如果本地不存在，先下载
        if not os.path.exists(model_path):
            print(f"下载模型中到: {model_path}...")
            model = SentenceTransformer(model_name)
            model.save(model_path)
            print(f"模型已保存到: {model_path}")
        else:
            print(f"从本地加载模型: {model_path}")
            model = SentenceTransformer(model_path)
        
        # 直接按照idx顺序构建名称列表
        cpt_names = [""] * self.cpt_num
        for uid, idx in self.cpt_uid.items():
            cpt_names[idx] = self.cpt_uid2name.get(uid)
        
        with torch.no_grad():
            self.cpt_init = model.encode(cpt_names, convert_to_tensor=True, device='cpu')
    
    def getP_ctc(self):
        A_ct = torch.zeros(self.cpt_num, self.tpc_num, dtype=torch.float)
        cpt_tpc = self.cpt_tpc
        for onedata in cpt_tpc:
            cpt_uid, tpc_uid = onedata
            cpt_idx = self.lrn_uid.get(cpt_uid)
            tpc_idx = self.untqus_uid.get(tpc_uid)
            
            if cpt_idx is not None and tpc_idx is not None:
                A_ct[cpt_idx, tpc_idx] = 1.0
        A_ctc = self.computeMetaPathMatrix(A_ct)
        edge_index, edge_weight = self.computeNormalizedAdjacencyMatrix(A_ctc, True)
        self.p_ctc = (edge_index, edge_weight)

    def getP_cc(self):
        A_cc = torch.zeros(self.cpt_num, self.cpt_num, dtype=torch.float)
        cpt_cpt = self.cpt_cpt
        for onedata in cpt_cpt:
            pre_uid, aft_uid = onedata
            pre_uid = self.lrn_uid.get(pre_uid)
            aft_uid = self.untqus_uid.get(aft_uid)
            
            if pre_uid is not None and aft_uid is not None:
                A_cc[pre_uid, aft_uid] = 1.0
        edge_index, edge_weight = self.computeNormalizedAdjacencyMatrix(A_cc, True)
        self.p_cc = (edge_index, edge_weight)

    def getP_cuc(self):
        A_cu = torch.zeros(self.cpt_num, self.untqus_num, dtype=torch.float)
        untqus_cpt = self.untqus_cpt
        for onedata in untqus_cpt:
            untqus_uid, cpt_uid = onedata
            untqus_idx = self.lrn_uid.get(untqus_uid)
            cpt_idx = self.untqus_uid.get(cpt_uid)
            
            if cpt_idx is not None and untqus_idx is not None:
                A_cu[cpt_idx, untqus_idx] = 1.0
        A_cuc = self.computeMetaPathMatrix(A_cu)
        edge_index, edge_weight = self.computeNormalizedAdjacencyMatrix(A_cuc, True)
        self.p_cuc = (edge_index, edge_weight)

    def loadDatafromSql(self):
        self.getLearnerInit()
        self.getP_lul()
        self.getP_lcl()
        self.getP_ltl()

        self.getUnitInit()
        self.getP_ulu()
        self.getP_ucrsu()
        self.getP_ucptu()
        self.getP_uu()

        self.getCptInit()
        self.getP_ctc()
        self.getP_cc()
        self.getP_cuc()

hgcdr = HGCDataReader()

if __name__ == '__main__':

    hgcdr.loadDatafromSql()

    # dr.getLearnerInit()
    # dr.getP_lul()
    # dr.getP_lcl()
    # dr.getP_ltl()
    # print(dr.lrn_init)
    # print(dr.p_lul)
    # print(dr.p_lcl)
    # print(dr.p_ltl)

    # dr.getUnitInit()
    # dr.getP_ulu()
    # dr.getP_ucrsu()
    # dr.getP_ucptu()
    # dr.getP_uu()
    # print(dr.untqus_init)
    # print(dr.p_ulu)
    # print(dr.p_ucrsu)
    # print(dr.p_ucptu)
    # print(dr.p_uu)

    # dr.getCptInit()
    # dr.getP_ctc()
    # dr.getP_cc()
    # dr.getP_cuc()
    # print(dr.cpt_init)
    # print(dr.p_ctc)
    # print(dr.p_cc)
    # print(dr.p_cuc)

    
    


    