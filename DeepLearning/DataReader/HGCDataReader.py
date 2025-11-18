import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy import sparse
import time
from collections import defaultdict

from DataReader.BasicDataReader import basicdr
from Data.HGCRepository import hgcrepo
from hyperparams.hyperparameter import hyperparams

class HGCDataReader():

    def __init__(self):
        self.lrn_uid = basicdr.lrn_uid
        self.lrn_num = basicdr.lrn_num
        self.qusunt_uid = basicdr.qusunt_uid
        self.qusunt_num = basicdr.qusunt_num
        self.tpc_uid = basicdr.tpc_uid
        self.tpc_num = basicdr.tpc_num
        self.crs_uid = basicdr.crs_uid
        self.crs_num = basicdr.crs_num
        self.cpt_uid = basicdr.cpt_uid
        self.cpt_num = basicdr.cpt_num
        self.qus_start = basicdr.unt_num

        self.getCptUid2Name()

        # 预加载数据
        print("预加载基础数据...")
        self.lrn_qusunt_count = hgcrepo.getLrnUntQusCount()
        self.lrn_crs_count = hgcrepo.getLrnCrsCount()
        self.lrn_tpc_count = hgcrepo.getLrnTpcCount()
        self.qusunt_cpt = hgcrepo.getUntQusCpt()
        self.unt_crs = hgcrepo.getUntCrs()
        self.unt_unt = hgcrepo.getUntUnt()
        self.cpt_tpc = hgcrepo.getCptTpc()
        self.cpt_cpt = hgcrepo.getCptCpt()

        # 缓存计算结果
        self._computed_matrices = {}
        self._sparse_matrices = {}

    def buildSparseMatrixFast(self, data_list, shape, row_mapping, col_mapping, value_func=lambda x: 1.0, symmetric=False):
        """
        快速构建稀疏矩阵 - 使用批量处理
        """
        if not data_list:
            return sparse.csr_matrix(shape)
            
        # 预分配数组
        n_edges = len(data_list)
        rows = np.zeros(n_edges, dtype=np.int32)
        cols = np.zeros(n_edges, dtype=np.int32)
        data = np.zeros(n_edges, dtype=np.float32)
        
        valid_count = 0
        for i, item in enumerate(data_list):
            if len(item) >= 2:
                row_uid, col_uid = item[0], item[1]
                row_idx = row_mapping.get(row_uid)
                col_idx = col_mapping.get(col_uid)
                
                if row_idx is not None and col_idx is not None:
                    rows[valid_count] = row_idx
                    cols[valid_count] = col_idx
                    value = value_func(item) if len(item) > 2 else 1.0
                    data[valid_count] = value
                    valid_count += 1
        
        # 只使用有效数据创建稀疏矩阵
        if valid_count > 0:
            sparse_matrix = sparse.coo_matrix((data[:valid_count], (rows[:valid_count], cols[:valid_count])), shape=shape)
            if symmetric:
                # 对于对称关系，确保矩阵对称
                sparse_matrix = sparse_matrix + sparse_matrix.T
                sparse_matrix.data = np.minimum(sparse_matrix.data, 1.0)  # 二值化
            return sparse_matrix.tocsr()
        else:
            return sparse.csr_matrix(shape)

    def computeMetaPathMatrixFast(self, A1_sparse, A2_sparse=None):
        """
        快速计算元路径邻接矩阵 - 使用优化的稀疏矩阵乘法
        """
        if A2_sparse is None:
            A2_sparse = A1_sparse.T
        
        # 使用稀疏矩阵乘法
        A_meta_sparse = A1_sparse.dot(A2_sparse)
        
        # 二值化处理，避免权重过大
        A_meta_sparse.data = np.ones_like(A_meta_sparse.data)
        
        return A_meta_sparse
    
    def normalizeSparseMatrix(self, sparse_matrix, add_self_loop=True):
        """
        快速归一化稀疏矩阵
        """
        if add_self_loop:
            n = sparse_matrix.shape[0]
            identity = sparse.identity(n, format='csr')
            sparse_matrix = sparse_matrix + identity
        
        # 计算度矩阵
        row_sum = np.array(sparse_matrix.sum(axis=1)).flatten()
        
        # 避免除零
        row_sum = np.maximum(row_sum, 1e-6)
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(row_sum))
        
        # 归一化: D^(-1/2) * A * D^(-1/2)
        normalized = D_inv_sqrt.dot(sparse_matrix).dot(D_inv_sqrt)
        
        return normalized

    def sparseToEdgeIndexWeightFast(self, normalized_sparse):
        """
        从归一化稀疏矩阵快速提取边索引和权重
        """
        coo = normalized_sparse.tocoo()
        
        # 直接转换为torch tensor，避免中间转换
        edge_index = torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long)
        edge_weight = torch.tensor(coo.data, dtype=torch.float)
        
        return edge_index, edge_weight

    def getCptUid2Name(self):
        uid_name = hgcrepo.getCptUidName()
        self.cpt_uid2name = {uid: name for (uid, name) in uid_name}

    def getP_lul(self):
        """优化版本：L-U-L元路径"""
        cache_key = 'p_lul'
        if cache_key in self._computed_matrices:
            self.p_lul = self._computed_matrices[cache_key]
            return
            
        # print("  计算L-U-L元路径...")
        # start_time = time.time()
        
        # 构建L-U稀疏矩阵
        A_lu_sparse = self.buildSparseMatrixFast(
            self.lrn_qusunt_count,
            (self.lrn_num, self.qusunt_num),
            self.lrn_uid, self.qusunt_uid,
            value_func=lambda x: min(x[2], 1.0) if len(x) > 2 else 1.0  # 限制最大值
        )
        
        # 计算L-U-L元路径
        A_lul_sparse = self.computeMetaPathMatrixFast(A_lu_sparse, A_lu_sparse.T)
        
        # 归一化
        A_lul_normalized = self.normalizeSparseMatrix(A_lul_sparse, True)
        edge_index, edge_weight = self.sparseToEdgeIndexWeightFast(A_lul_normalized)
        
        self.p_lul = (edge_index, edge_weight)
        self._computed_matrices[cache_key] = self.p_lul
        
        # print(f"    完成: {time.time() - start_time:.2f}秒, {edge_index.shape[1]}条边")

    def getP_lcl(self):
        """优化版本：L-C-L元路径"""
        cache_key = 'p_lcl'
        if cache_key in self._computed_matrices:
            self.p_lcl = self._computed_matrices[cache_key]
            return
            
        # print("  计算L-C-L元路径...")
        # start_time = time.time()
        
        A_lc_sparse = self.buildSparseMatrixFast(
            self.lrn_crs_count,
            (self.lrn_num, self.crs_num),
            self.lrn_uid, self.crs_uid,
            value_func=lambda x: min(x[2], 1.0) if len(x) > 2 else 1.0
        )
        
        A_lcl_sparse = self.computeMetaPathMatrixFast(A_lc_sparse, A_lc_sparse.T)
        A_lcl_normalized = self.normalizeSparseMatrix(A_lcl_sparse, True)
        edge_index, edge_weight = self.sparseToEdgeIndexWeightFast(A_lcl_normalized)
        
        self.p_lcl = (edge_index, edge_weight)
        self._computed_matrices[cache_key] = self.p_lcl
        
        # print(f"    完成: {time.time() - start_time:.2f}秒, {edge_index.shape[1]}条边")

    def getP_ltl(self):
        """优化版本：L-T-L元路径"""
        cache_key = 'p_ltl'
        if cache_key in self._computed_matrices:
            self.p_ltl = self._computed_matrices[cache_key]
            return
            
        # print("  计算L-T-L元路径...")
        # start_time = time.time()
        
        A_lt_sparse = self.buildSparseMatrixFast(
            self.lrn_tpc_count,
            (self.lrn_num, self.tpc_num),
            self.lrn_uid, self.tpc_uid,
            value_func=lambda x: min(x[2], 1.0) if len(x) > 2 else 1.0
        )
        
        A_ltl_sparse = self.computeMetaPathMatrixFast(A_lt_sparse, A_lt_sparse.T)
        A_ltl_normalized = self.normalizeSparseMatrix(A_ltl_sparse, True)
        edge_index, edge_weight = self.sparseToEdgeIndexWeightFast(A_ltl_normalized)
        
        self.p_ltl = (edge_index, edge_weight)
        self._computed_matrices[cache_key] = self.p_ltl
        
        # print(f"    完成: {time.time() - start_time:.2f}秒, {edge_index.shape[1]}条边")

    def getInit(self, init):
        """初始化矩阵归一化"""
        D_diag = (init > 0).sum(dim=1).float()
        D_inv_diag = torch.where(
            D_diag != 0,
            1.0 / D_diag.clamp(min=1e-6),
            torch.zeros_like(D_diag, dtype=torch.float)
        )
        init = init * D_inv_diag.unsqueeze(1)

    def getLearnerInit(self):
        """学习者初始化矩阵 - 优化版本"""
        # print("  构建学习者初始化矩阵...")
        # start_time = time.time()
        
        # 使用稀疏矩阵构建，然后转换为稠密矩阵
        lrn_init_sparse = self.buildSparseMatrixFast(
            self.lrn_qusunt_count,
            (self.lrn_num, self.qusunt_num),
            self.lrn_uid, self.qusunt_uid
        )
        
        # 转换为稠密矩阵并归一化
        self.lrn_init = torch.tensor(lrn_init_sparse.toarray(), dtype=torch.float)
        self.getInit(self.lrn_init)
        
        # print(f"    完成: {time.time() - start_time:.2f}秒")

    def getUnitInit(self):
        """学习单元初始化矩阵 - 优化版本"""
        # print("  构建学习单元初始化矩阵...")
        # start_time = time.time()
        
        qusunt_init_sparse = self.buildSparseMatrixFast(
            self.qusunt_cpt,
            (self.qusunt_num, self.cpt_num),
            self.qusunt_uid, self.cpt_uid
        )
        
        self.qusunt_init = torch.tensor(qusunt_init_sparse.toarray(), dtype=torch.float)
        self.getInit(self.qusunt_init)
        
        # print(f"    完成: {time.time() - start_time:.2f}秒")

    def getP_ulu(self):
        """优化版本：U-L-U元路径"""
        cache_key = 'p_ulu'
        if cache_key in self._computed_matrices:
            self.p_ulu = self._computed_matrices[cache_key]
            return
            
        # print("  计算U-L-U元路径...")
        # start_time = time.time()
        
        A_ul_sparse = self.buildSparseMatrixFast(
            self.lrn_qusunt_count,
            (self.qusunt_num, self.lrn_num),
            self.qusunt_uid, self.lrn_uid,
            value_func=lambda x: min(x[2], 1.0) if len(x) > 2 else 1.0
        )
        
        A_ulu_sparse = self.computeMetaPathMatrixFast(A_ul_sparse, A_ul_sparse.T)
        A_ulu_normalized = self.normalizeSparseMatrix(A_ulu_sparse, True)
        edge_index, edge_weight = self.sparseToEdgeIndexWeightFast(A_ulu_normalized)
        
        self.p_ulu = (edge_index, edge_weight)
        self._computed_matrices[cache_key] = self.p_ulu
        
        # print(f"    完成: {time.time() - start_time:.2f}秒, {edge_index.shape[1]}条边")

    def getP_ucrsu(self):
        """优化版本：U-Crs-U元路径"""
        cache_key = 'p_ucrsu'
        if cache_key in self._computed_matrices:
            self.p_ucrsu = self._computed_matrices[cache_key]
            return
            
        # print("  计算U-Crs-U元路径...")
        # start_time = time.time()
        
        A_ucrs_sparse = self.buildSparseMatrixFast(
            self.unt_crs,
            (self.qusunt_num, self.crs_num),
            self.qusunt_uid, self.crs_uid
        )
        
        A_ucrsu_sparse = self.computeMetaPathMatrixFast(A_ucrs_sparse, A_ucrs_sparse.T)
        A_ucrsu_normalized = self.normalizeSparseMatrix(A_ucrsu_sparse, True)
        edge_index, edge_weight = self.sparseToEdgeIndexWeightFast(A_ucrsu_normalized)
        
        self.p_ucrsu = (edge_index, edge_weight)
        self._computed_matrices[cache_key] = self.p_ucrsu
        
        # print(f"    完成: {time.time() - start_time:.2f}秒, {edge_index.shape[1]}条边")

    def getP_ucptu(self):
        """优化版本：U-Cpt-U元路径"""
        cache_key = 'p_ucptu'
        if cache_key in self._computed_matrices:
            self.p_ucptu = self._computed_matrices[cache_key]
            return
            
        # print("  计算U-Cpt-U元路径...")
        # start_time = time.time()
        
        A_ucpt_sparse = self.buildSparseMatrixFast(
            self.qusunt_cpt,
            (self.qusunt_num, self.cpt_num),
            self.qusunt_uid, self.cpt_uid
        )
        
        A_ucptu_sparse = self.computeMetaPathMatrixFast(A_ucpt_sparse, A_ucpt_sparse.T)
        A_ucptu_normalized = self.normalizeSparseMatrix(A_ucptu_sparse, True)
        edge_index, edge_weight = self.sparseToEdgeIndexWeightFast(A_ucptu_normalized)
        
        self.p_ucptu = (edge_index, edge_weight)
        self._computed_matrices[cache_key] = self.p_ucptu
        
        # print(f"    完成: {time.time() - start_time:.2f}秒, {edge_index.shape[1]}条边")

    # def getP_uu(self):
    #     """U-U关系 - 注释掉，暂时不参与运算"""
    #     print("  跳过U-U关系计算（性能优化）...")
    #     # 创建一个空的边索引和权重
    #     self.p_uu = (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.float))
    #     self._computed_matrices['p_uu'] = self.p_uu
    #     return
        
    #     # 以下是原始实现，暂时注释掉
    #     """
    #     cache_key = 'p_uu'
    #     if cache_key in self._computed_matrices:
    #         self.p_uu = self._computed_matrices[cache_key]
    #         return
            
    #     print("  计算U-U关系...")
    #     start_time = time.time()
        
    #     # 构建U-U稀疏矩阵
    #     A_uu_sparse = self.buildSparseMatrixFast(
    #         self.unt_unt,
    #         (self.qusunt_num, self.qusunt_num),
    #         self.qusunt_uid, self.qusunt_uid,
    #         symmetric=True  # U-U关系是对称的
    #     )
        
    #     # 为题目添加自连接 - 使用更高效的方式
    #     if self.qus_start < self.qusunt_num:
    #         # 使用lil格式进行高效修改
    #         A_uu_lil = A_uu_sparse.tolil()
    #         for i in range(self.qus_start, min(self.qusunt_num, A_uu_lil.shape[0])):
    #             A_uu_lil[i, i] = 1.0
    #         A_uu_sparse = A_uu_lil.tocsr()
        
    #     A_uu_normalized = self.normalizeSparseMatrix(A_uu_sparse, False)
    #     edge_index, edge_weight = self.sparseToEdgeIndexWeightFast(A_uu_normalized)
        
    #     self.p_uu = (edge_index, edge_weight)
    #     self._computed_matrices[cache_key] = self.p_uu
        
    #     print(f"    完成: {time.time() - start_time:.2f}秒, {edge_index.shape[1]}条边")
    #     """

    def getCptInit(self, model_name=None):
        """知识点初始化 - 使用超参数配置"""
        if model_name is None:
            model_name = hyperparams.data_sentence_transformer_model
            
        # print("  构建知识点初始化矩阵...")
        # start_time = time.time()
        
        # 获取项目根目录
        deeplearningroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(deeplearningroot, "Model", model_name)
        
        # 如果本地不存在，先下载
        if not os.path.exists(model_path):
            print(f"    下载模型中到: {model_path}...")
            model = SentenceTransformer(model_name)
            model.save(model_path)
            print(f"    模型已保存到: {model_path}")
        else:
            print(f"    从本地加载模型: {model_path}")
            model = SentenceTransformer(model_path)
        
        # 直接按照idx顺序构建名称列表
        cpt_names = [""] * self.cpt_num
        for uid, idx in self.cpt_uid.items():
            cpt_names[idx] = self.cpt_uid2name.get(uid, "")
        
        with torch.no_grad():
            self.cpt_init = model.encode(cpt_names, convert_to_tensor=True, device='cpu')
        
        # print(f"    完成: {time.time() - start_time:.2f}秒")

    def getP_ctc(self):
        """优化版本：C-T-C元路径"""
        cache_key = 'p_ctc'
        if cache_key in self._computed_matrices:
            self.p_ctc = self._computed_matrices[cache_key]
            return
            
        # print("  计算C-T-C元路径...")
        # start_time = time.time()
        
        A_ct_sparse = self.buildSparseMatrixFast(
            self.cpt_tpc,
            (self.cpt_num, self.tpc_num),
            self.cpt_uid, self.tpc_uid
        )
        
        A_ctc_sparse = self.computeMetaPathMatrixFast(A_ct_sparse, A_ct_sparse.T)
        A_ctc_normalized = self.normalizeSparseMatrix(A_ctc_sparse, True)
        edge_index, edge_weight = self.sparseToEdgeIndexWeightFast(A_ctc_normalized)
        
        self.p_ctc = (edge_index, edge_weight)
        self._computed_matrices[cache_key] = self.p_ctc
        
        # print(f"    完成: {time.time() - start_time:.2f}秒, {edge_index.shape[1]}条边")

    def getP_cc(self):
        """优化版本：C-C关系"""
        cache_key = 'p_cc'
        if cache_key in self._computed_matrices:
            self.p_cc = self._computed_matrices[cache_key]
            return
            
        # print("  计算C-C关系...")
        # start_time = time.time()
        
        A_cc_sparse = self.buildSparseMatrixFast(
            self.cpt_cpt,
            (self.cpt_num, self.cpt_num),
            self.cpt_uid, self.cpt_uid,
            symmetric=True  # C-C关系是对称的
        )
        
        A_cc_normalized = self.normalizeSparseMatrix(A_cc_sparse, True)
        edge_index, edge_weight = self.sparseToEdgeIndexWeightFast(A_cc_normalized)
        
        self.p_cc = (edge_index, edge_weight)
        self._computed_matrices[cache_key] = self.p_cc
        
        # print(f"    完成: {time.time() - start_time:.2f}秒, {edge_index.shape[1]}条边")

    def getP_cuc(self):
        """优化版本：C-U-C元路径"""
        cache_key = 'p_cuc'
        if cache_key in self._computed_matrices:
            self.p_cuc = self._computed_matrices[cache_key]
            return
            
        # print("  计算C-U-C元路径...")
        # start_time = time.time()
        
        A_cu_sparse = self.buildSparseMatrixFast(
            self.qusunt_cpt,
            (self.cpt_num, self.qusunt_num),
            self.cpt_uid, self.qusunt_uid
        )
        
        A_cuc_sparse = self.computeMetaPathMatrixFast(A_cu_sparse, A_cu_sparse.T)
        A_cuc_normalized = self.normalizeSparseMatrix(A_cuc_sparse, True)
        edge_index, edge_weight = self.sparseToEdgeIndexWeightFast(A_cuc_normalized)
        
        self.p_cuc = (edge_index, edge_weight)
        self._computed_matrices[cache_key] = self.p_cuc
        
        # print(f"    完成: {time.time() - start_time:.2f}秒, {edge_index.shape[1]}条边")

    def loadDatafromSql(self, use_cache=True):
        """加载所有数据 - 优化版本"""
        print("=== 加载HGC数据 ===")
        total_start_time = time.time()
        
        # 学习者相关
        print("1. 处理学习者数据...")
        self.getLearnerInit()
        self.getP_lul()
        self.getP_lcl()
        self.getP_ltl()

        # 学习单元相关
        print("2. 处理学习单元数据...")
        self.getUnitInit()
        self.getP_ulu()
        self.getP_ucrsu()
        self.getP_ucptu()
        # self.getP_uu()  # 注释掉的版本

        # 知识点相关
        print("3. 处理知识点数据...")
        self.getCptInit()
        self.getP_ctc()
        self.getP_cc()
        self.getP_cuc()
        
        total_time = time.time() - total_start_time
        print(f"✓ HGC数据加载完成，总耗时: {total_time:.2f}秒")

    def get_data_info(self):
        """返回数据统计信息"""
        edge_counts = {}
        for key in ['p_lul', 'p_lcl', 'p_ltl', 'p_ulu', 'p_ucrsu', 'p_ucptu', 'p_ctc', 'p_cc', 'p_cuc']:
            if hasattr(self, key):
                edge_index, _ = getattr(self, key)
                edge_counts[key] = edge_index.shape[1]
        
        return {
            'learners': self.lrn_num,
            'units_questions': self.qusunt_num,
            'concepts': self.cpt_num,
            'courses': self.crs_num,
            'topics': self.tpc_num,
            'edge_counts': edge_counts
        }

hgcdr = HGCDataReader()

if __name__ == '__main__':
    def test_hgc_data_reader():
        """测试HGC数据读取器"""
        print("=== HGCDataReader 测试 ===")
        
        # 加载数据
        hgcdr.loadDatafromSql()
        
        # 显示数据信息
        info = hgcdr.get_data_info()
        print(f"\n数据统计:")
        print(f"  学习者: {info['learners']}")
        print(f"  学习单元+题目: {info['units_questions']}")
        print(f"  知识点: {info['concepts']}")
        print(f"  课程: {info['courses']}")
        print(f"  主题: {info['topics']}")
        
        print(f"\n边数量统计:")
        for path, count in info['edge_counts'].items():
            print(f"  {path}: {count}条边")
        
        # 测试关键数据结构
        print(f"\n关键数据结构:")
        print(f"  学习者初始化矩阵: {hgcdr.lrn_init.shape}")
        print(f"  学习单元初始化矩阵: {hgcdr.qusunt_init.shape}")
        print(f"  知识点初始化矩阵: {hgcdr.cpt_init.shape}")
        
        # 验证数据范围
        print(f"\n数据范围验证:")
        print(f"  学习者嵌入范围: [{hgcdr.lrn_init.min():.3f}, {hgcdr.lrn_init.max():.3f}]")
        print(f"  学习单元嵌入范围: [{hgcdr.qusunt_init.min():.3f}, {hgcdr.qusunt_init.max():.3f}]")
        print(f"  知识点嵌入范围: [{hgcdr.cpt_init.min():.3f}, {hgcdr.cpt_init.max():.3f}]")
        
        print("\n✓ HGCDataReader 测试完成")

    test_hgc_data_reader()