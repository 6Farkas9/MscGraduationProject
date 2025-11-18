import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from DataReader.CDDataReader import cddr
from DataReader.BasicDataReader import basicdr
from hyperparams.hyperparameter import hyperparams

class CDDataset(Dataset):

    def __init__(self, static_data, h_lrn, qusunt_emb, h_cpt, data_type='train', max_seq_len=None):
        """
        Args:
            static_data: 从CDDataReader获取的静态数据，包含train_data和test_data
            h_lrn: (lrn_num, emb_dim) - HGC计算的学习者嵌入
            qusunt_emb: (qusunt_num, emb_dim) - HGC计算的题目+学习单元嵌入（新顺序：前半部分qus，后半部分unt）
            h_cpt: (cpt_num, emb_dim) - HGC计算的知识点嵌入
            data_type: 'train' 或 'test'
            max_seq_len: 最大序列长度
        """
        super(CDDataset, self).__init__()
        
        if max_seq_len is None:
            max_seq_len = hyperparams.data_max_seq_len
        
        # 交互数据
        self.data = static_data[f'{data_type}_data']
        self.lrn_uid = static_data['lrn_uid']
        self.qus_uid = static_data['qus_uid']  # 直接使用qus_uid，不需要转换
        self.cpt_uid = static_data['cpt_uid']
        
        # 数量统计
        self.lrn_num = len(self.lrn_uid)
        self.qus_num = len(self.qus_uid)
        self.cpt_num = len(self.cpt_uid)

        # HGC计算的嵌入
        self.h_lrn = h_lrn
        self.h_cpt = h_cpt
        
        # 新逻辑：qusunt_emb的前半部分是题目，直接使用
        # 因为CD只使用题目，不需要学习单元部分
        self.h_qus = qusunt_emb[:self.qus_num]  # 直接取前半部分作为题目嵌入
        
        self.max_seq_len = max_seq_len
        
        # 创建反向映射
        self.idx2lrn = {idx: uid for uid, idx in self.lrn_uid.items()}
        self.idx2qus = {idx: uid for uid, idx in self.qus_uid.items()}
        self.idx2cpt = {idx: uid for uid, idx in self.cpt_uid.items()}
        
        # 预计算序列数据
        self._precompute_sequences()

    def _precompute_sequences(self):
        """预计算所有学习者的序列数据 - 优化版本"""
        print(f"  预计算序列数据，最大长度: {self.max_seq_len}")
        
        # 初始化张量
        self.qus_seq_indices = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.long, device='cpu')
        self.qus_seq_masks = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.results = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        
        # 统计信息
        total_records = 0
        max_actual_len = 0
        
        # 遍历defaultdict中的数据
        for lrn_uid, (qus_seq, result_seq) in self.data.items():
            if lrn_uid not in self.lrn_uid:
                continue
                
            lrn_idx = self.lrn_uid[lrn_uid]
            
            # 直接使用qus_uid映射，不需要转换
            qus_indices = []
            for qus_uid in qus_seq:
                if qus_uid in self.qus_uid:
                    qus_indices.append(self.qus_uid[qus_uid])
            
            valid_len = min(len(qus_indices), self.max_seq_len)
            
            if valid_len == 0:
                continue
                
            # 更新统计
            total_records += valid_len
            max_actual_len = max(max_actual_len, valid_len)
            
            # 填充序列数据 - 从序列开头开始填充
            self.qus_seq_indices[lrn_idx, :valid_len] = torch.tensor(
                qus_indices[:valid_len], dtype=torch.long, device='cpu'
            )
            self.results[lrn_idx, :valid_len] = torch.tensor(
                result_seq[:valid_len], dtype=torch.float32, device='cpu'
            )
            self.qus_seq_masks[lrn_idx, :valid_len] = 1.0
        
        print(f"    总记录数: {total_records}, 最大实际序列长度: {max_actual_len}")

    def __len__(self):
        return self.lrn_num

    def __getitem__(self, idx):
        """返回单个学习者的数据"""
        lrn_uid = self.idx2lrn[idx]
        
        return {
            'lrn_idx': idx,
            'lrn_uid': lrn_uid,
            'qus_seq_index': self.qus_seq_indices[idx],
            'qus_seq_mask': self.qus_seq_masks[idx],
            'result': self.results[idx]
        }

    def collate_fn(self, batch):
        """批次处理函数 - 优化版本"""
        lrn_indices = torch.tensor([item['lrn_idx'] for item in batch])
        qus_seq_indices = torch.stack([item['qus_seq_index'] for item in batch])
        qus_seq_masks = torch.stack([item['qus_seq_mask'] for item in batch])
        results = torch.stack([item['result'] for item in batch])
        
        # 获取批次对应的嵌入 - 使用clone确保梯度安全
        h_lrn_batch = self.h_lrn[lrn_indices].clone()
        
        return {
            'lrn_indices': lrn_indices,
            'qus_seq_indices': qus_seq_indices,
            'qus_seq_masks': qus_seq_masks,
            'results': results,
            'h_lrn_batch': h_lrn_batch,
        }

    def get_data_statistics(self):
        """返回数据集统计信息"""
        total_records = self.qus_seq_masks.sum().item()
        avg_seq_len = total_records / len(self) if len(self) > 0 else 0
        
        return {
            'total_learners': len(self),
            'total_records': int(total_records),
            'average_sequence_length': round(avg_seq_len, 2),
            'max_sequence_length': self.max_seq_len,
            'question_count': self.qus_num,
            'concept_count': self.cpt_num
        }

    def get_embedding_info(self):
        """返回嵌入维度信息"""
        return {
            'lrn_emb_dim': self.h_lrn.shape[1],
            'qus_emb_dim': self.h_qus.shape[1],
            'cpt_emb_dim': self.h_cpt.shape[1],
            'question_count': self.qus_num,
            'concept_count': self.cpt_num
        }

def test_cd_dataset():
    """测试CD数据集"""
    print("=== CDDataSet 测试 ===")
    
    # 模拟数据
    import torch.nn as nn
    from Model.HGC import HGC
    from DataReader.HGCDataReader import hgcdr
    from torch.utils.data import DataLoader

    # 1. 加载HGC数据并计算嵌入
    print("1. 加载HGC数据...")
    hgcdr.loadDatafromSql()
    device = hyperparams.device
    
    # 动态获取输入维度
    lrn_input_dim = hgcdr.lrn_init.shape[1]
    unt_input_dim = hgcdr.qusunt_init.shape[1]
    cpt_input_dim = hgcdr.cpt_init.shape[1]
    
    model_hgc = HGC(
        embedding_dim=hyperparams.hgc_embedding_dim,
        lrn_input_dim=lrn_input_dim,
        unt_input_dim=unt_input_dim,
        cpt_input_dim=cpt_input_dim
    ).to(device)

    with torch.no_grad():
        lrn_emb, qusunt_emb, cpt_emb = model_hgc(hgcdr, device)

    print("✓ HGC嵌入计算完成")
    print(f"  学习者嵌入: {lrn_emb.shape}")
    print(f"  题目+学习单元嵌入: {qusunt_emb.shape} (新顺序: 前半部分qus)")
    print(f"  知识点嵌入: {cpt_emb.shape}")

    # 2. 加载CD数据
    print("\n2. 加载CD数据...")
    cddata = cddr.loadDatafromSql()

    # 3. 创建数据集
    print("\n3. 创建CD数据集...")
    train_dataset = CDDataset(cddata, lrn_emb, qusunt_emb, cpt_emb, 'train')
    test_dataset = CDDataset(cddata, lrn_emb, qusunt_emb, cpt_emb, 'test')

    # 4. 查看统计信息
    print("\n4. 数据集统计信息:")
    print("训练集统计:", train_dataset.get_data_statistics())
    print("测试集统计:", test_dataset.get_data_statistics())
    print("嵌入信息:", train_dataset.get_embedding_info())

    # 5. 测试单个样本
    print("\n5. 单个样本测试:")
    if len(train_dataset) > 0:
        single_sample = train_dataset[0]
        print("单个样本键值:", single_sample.keys())
        for key, value in single_sample.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape} (dtype: {value.dtype})")
            else:
                print(f"  {key}: {value}")

    # 6. 测试batch功能
    print("\n6. Batch功能测试:")
    batch_size = min(4, len(train_dataset))
    if batch_size > 0:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=train_dataset.collate_fn
        )

        # 检查一个batch
        for i, batch in enumerate(train_loader):
            if i >= 1:  # 只检查第一个batch
                break
                
            print(f"Batch {i + 1}:")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape} (dtype: {value.dtype})")
            
            # 验证关键数据一致性
            lrn_indices = batch['lrn_indices']
            qus_seq_indices = batch['qus_seq_indices']
            qus_seq_masks = batch['qus_seq_masks']
            results = batch['results']
            h_lrn_batch = batch['h_lrn_batch']
            
            # 检查维度一致性
            assert lrn_indices.shape[0] == qus_seq_indices.shape[0], "batch_size不一致"
            assert qus_seq_indices.shape == qus_seq_masks.shape == results.shape, "序列维度不一致"
            assert h_lrn_batch.shape[0] == lrn_indices.shape[0], "学习者嵌入batch_size不一致"
            
            print(f"  ✓ 维度一致性检查通过")
            print(f"  ✓ 有效序列位置: {qus_seq_masks.sum().item():.0f}")
            print(f"  ✓ 平均正确率: {results.mean().item():.3f}")
            
            # 验证题目索引范围
            max_qus_idx = qus_seq_indices.max().item()
            if max_qus_idx < train_dataset.qus_num:
                print(f"  ✓ 题目索引范围验证通过 [0, {max_qus_idx}] < {train_dataset.qus_num}")
            else:
                print(f"  ✗ 题目索引超出范围: {max_qus_idx} >= {train_dataset.qus_num}")
    else:
        print("  没有足够的数据进行batch测试")

    # 7. 数据完整性检查
    print(f"\n7. 数据完整性检查:")
    has_nan = False
    has_inf = False
    
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        for key, value in sample.items():
            if torch.is_tensor(value):
                if torch.isnan(value).any():
                    print(f"✗ {key} 包含NaN值")
                    has_nan = True
                if torch.isinf(value).any():
                    print(f"✗ {key} 包含Inf值")
                    has_inf = True
    
    if not has_nan and not has_inf:
        print("✓ 数据完整性检查通过 - 无NaN和Inf值")
    
    print(f"\n=== CDDataSet测试完成 ===")

if __name__ == '__main__':
    test_cd_dataset()