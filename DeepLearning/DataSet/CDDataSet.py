import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from DataReader.CDDataReader import cddr
from DataReader.BasicDataReader import basicdr

class CDDataset(Dataset):

    def __init__(self, static_data, h_lrn, unt_emb, h_cpt, data_type='train', max_seq_len=128):
        """
        Args:
            static_data: 从CDDataReader获取的静态数据，包含train_data和test_data
            h_lrn: (lrn_num, emb_dim) - HGC计算的学习者嵌入
            h_qus: (qus_num, emb_dim) - HGC计算的题目嵌入
            h_cpt: (cpt_num, emb_dim) - HGC计算的知识点嵌入
            data_type: 'train' 或 'test'
            max_seq_len: 最大序列长度
        """
        super(CDDataset, self).__init__()
        
        # 交互数据 - 现在static_data中的train_data和test_data是defaultdict
        self.data = static_data[f'{data_type}_data']
        self.lrn_uid = static_data['lrn_uid']
        self.qus_uid = static_data['qus_uid']
        self.cpt_uid = static_data['cpt_uid']
        
        # 数量统计
        self.lrn_num = len(self.lrn_uid)
        self.qus_num = len(self.qus_uid)
        self.cpt_num = len(self.cpt_uid)

        # HGC计算的嵌入
        # 从unt_emb中提取题目嵌入
        # 假设unt_emb的前unt_num个是学习单元，后面qus_num个是题目
        self.unt_num = basicdr.unt_num

        # 创建题目索引映射：将原始unt_emb中的题目索引映射到0~qus_num-1
        self.qus_index_mapping = {}
        for qus_uid, original_idx in self.qus_uid.items():
            if original_idx >= self.unt_num:  # 确保是题目索引
                new_idx = original_idx - self.unt_num
                self.qus_index_mapping[qus_uid] = new_idx
            else:
                print(f"警告: 题目 {qus_uid} 的索引 {original_idx} 不在题目范围内")
        
        # 从unt_emb中提取题目嵌入，并保持梯度传递
        self.h_qus = unt_emb[self.unt_num:]  # 提取题目部分，保持梯度

        self.h_lrn = h_lrn
        self.h_cpt = h_cpt
        
        self.max_seq_len = max_seq_len
        
        # 创建反向映射
        self.idx2lrn = {idx: uid for uid, idx in self.lrn_uid.items()}
        self.idx2qus = {idx: uid for uid, idx in self.qus_uid.items()}
        self.idx2cpt = {idx: uid for uid, idx in self.cpt_uid.items()}
        
        # 预计算序列数据
        self._precompute_sequences()

    def _precompute_sequences(self):
        """预计算所有学习者的序列数据"""
        # 初始化张量
        self.qus_seq_indices = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.long, device='cpu')
        self.qus_seq_masks = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.results = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        
        # 遍历defaultdict中的数据
        for lrn_uid, (qus_seq, result_seq) in self.data.items():
            if lrn_uid not in self.lrn_uid:
                continue
                
            lrn_idx = self.lrn_uid[lrn_uid]
            # 使用新的题目索引映射
            qus_indices = [self.qus_index_mapping[qus_uid] for qus_uid in qus_seq 
                          if qus_uid in self.qus_index_mapping]
            
            valid_len = min(len(qus_indices), self.max_seq_len)
            
            if valid_len == 0:
                continue
                
            # 填充序列数据 - 从序列开头开始填充
            self.qus_seq_indices[lrn_idx, :valid_len] = torch.tensor(
                qus_indices[:valid_len], dtype=torch.long, device='cpu'
            )
            self.results[lrn_idx, :valid_len] = torch.tensor(
                result_seq[:valid_len], dtype=torch.float32, device='cpu'
            )
            self.qus_seq_masks[lrn_idx, :valid_len] = 1.0
    
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
        """批次处理函数 - 简化版本"""
        lrn_indices = torch.tensor([item['lrn_idx'] for item in batch])
        qus_seq_indices = torch.stack([item['qus_seq_index'] for item in batch])
        qus_seq_masks = torch.stack([item['qus_seq_mask'] for item in batch])
        results = torch.stack([item['result'] for item in batch])
        
        # 获取批次对应的嵌入
        h_lrn_batch = self.h_lrn[lrn_indices]
        
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
            'max_sequence_length': self.max_seq_len
        }

    def get_embedding_info(self):
        """返回嵌入维度信息"""
        return {
            'lrn_emb_dim': self.h_lrn.shape[1],
            'qus_emb_dim': self.h_qus.shape[1],
            'cpt_emb_dim': self.h_cpt.shape[1],
            'cpt_num': self.cpt_num
        }
    
if __name__ == '__main__':
    import torch.nn as nn
    from Model.HGC import HGC
    from DataReader.HGCDataReader import hgcdr
    from torch.utils.data import DataLoader

    # 1. 加载HGC数据并计算嵌入
    hgcdr.loadDatafromSql()
    device = 'cpu'
    
    # 动态获取输入维度
    lrn_input_dim = hgcdr.lrn_init.shape[1]
    unt_input_dim = hgcdr.untqus_init.shape[1]
    cpt_input_dim = hgcdr.cpt_init.shape[1]
    
    model_hgc = HGC(
        embedding_dim=64,
        lrn_input_dim=lrn_input_dim,
        unt_input_dim=unt_input_dim,
        cpt_input_dim=cpt_input_dim
    ).to(device)

    lrn_emb, unt_emb, cpt_emb = model_hgc(hgcdr, device)

    print("=== 嵌入维度信息 ===")
    print("Learner Embedding:", lrn_emb.shape)
    print("Unit Embedding:", unt_emb.shape)
    print("Concept Embedding:", cpt_emb.shape)

    # 2. 加载CD数据
    cddata = cddr.loadDatafromSql()

    # 3. 创建数据集
    train_dataset = CDDataset(cddata, lrn_emb, unt_emb, cpt_emb, 'train', max_seq_len=128)
    test_dataset = CDDataset(cddata, lrn_emb, unt_emb, cpt_emb, 'test', max_seq_len=128)

    # 4. 查看统计信息
    print("\n=== 数据集统计信息 ===")
    print("训练集统计:", train_dataset.get_data_statistics())
    print("测试集统计:", test_dataset.get_data_statistics())

    # 5. 测试单个样本
    print("\n=== 单个样本测试 ===")
    if len(train_dataset) > 0:
        single_sample = train_dataset[0]
        print("单个样本键值:", single_sample.keys())
        for key, value in single_sample.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape} (dtype: {value.dtype})")
            else:
                print(f"  {key}: {value}")

    # 6. 测试batch功能
    print("\n=== Batch功能测试 ===")
    batch_size = 4
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=train_dataset.collate_fn
    )

    # 检查几个batch
    batch_sizes = []
    for i, batch in enumerate(train_loader):
        if i >= 3:  # 只检查前3个batch
            break
            
        print(f"\nBatch {i + 1}:")
        batch_sizes.append(len(batch['lrn_indices']))
        
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape} (dtype: {value.dtype})")
            else:
                print(f"  {key}: {value}")
        
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

    # 7. 验证batch大小一致性
    print(f"\n=== Batch大小验证 ===")
    if len(batch_sizes) > 0:
        unique_sizes = set(batch_sizes)
        if len(unique_sizes) == 1:
            print(f"✓ 所有batch大小一致: {batch_sizes[0]}")
        else:
            print(f"✗ batch大小不一致: {batch_sizes}")
    else:
        print("没有获取到batch数据")

    # 8. 测试模型输入兼容性
    print(f"\n=== 模型输入兼容性测试 ===")
    if len(train_dataset) > 0:
        # 创建简化模型进行测试
        class TestModel(nn.Module):
            def __init__(self, embedding_dim, concept_num):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.concept_num = concept_num
            
            def forward(self, h_lrn_batch, qus_seq_indices, qus_seq_masks):
                batch_size, seq_len = qus_seq_indices.shape
                print(f"  模型接收 - batch_size: {batch_size}, seq_len: {seq_len}")
                print(f"  模型接收 - h_lrn_batch: {h_lrn_batch.shape}")
                return torch.randn(batch_size, seq_len)
        
        # 测试一个batch
        test_batch = next(iter(train_loader))
        test_model = TestModel(embedding_dim=64, concept_num=len(cddata['cpt_uid']))
        
        with torch.no_grad():
            output = test_model(
                test_batch['h_lrn_batch'],
                test_batch['qus_seq_indices'],
                test_batch['qus_seq_masks']
            )
            print(f"  模型输出: {output.shape}")
            print("✓ 模型输入输出兼容性测试通过")

    # 9. 数据完整性检查
    print(f"\n=== 数据完整性检查 ===")
    # 检查是否有无效数据
    has_nan = False
    has_inf = False
    
    for i, batch in enumerate(train_loader):
        if i >= 2:  # 只检查前2个batch
            break
            
        for key, value in batch.items():
            if torch.is_tensor(value):
                if torch.isnan(value).any():
                    print(f"✗ {key} 包含NaN值")
                    has_nan = True
                if torch.isinf(value).any():
                    print(f"✗ {key} 包含Inf值")
                    has_inf = True
    
    if not has_nan and not has_inf:
        print("✓ 数据完整性检查通过 - 无NaN和Inf值")
    
    print(f"\n=== 测试完成 ===")
