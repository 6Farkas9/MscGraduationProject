import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from DataReader.KTDataReader import ktdr
from DataReader.BasicDataReader import basicdr

class KTDataSet(Dataset):
    """
    知识追踪数据集 - 处理动态的、涉及梯度的数据
    """

    def __init__(self, static_data, h_lrn, unt_emb, h_cpt, data_type='train', max_seq_len=128):
        """
        Args:
            static_data: 从KTDataReader获取的静态数据
            h_lrn: (lrn_num, emb_dim) - HGC计算的学习者嵌入
            unt_emb: (untqus_num, emb_dim) - HGC计算的学习单元+题目嵌入
            h_cpt: (cpt_num, emb_dim) - HGC计算的知识点嵌入
            data_type: 'train' 或 'test'
            max_seq_len: 最大序列长度
        """
        super(KTDataSet, self).__init__()
        
        # 交互数据
        self.data = static_data[f'{data_type}_data']
        self.lrn_uid = static_data['lrn_uid']
        self.untqus_uid = static_data['untqus_uid']
        self.cpt_uid = static_data['cpt_uid']
        self.unit_types = static_data['unit_types']
        self.question_concepts = static_data['question_concepts']
        self.unit_concepts = static_data['unit_concepts']
        
        # 数量统计
        self.lrn_num = len(self.lrn_uid)
        self.untqus_num = len(self.untqus_uid)
        self.cpt_num = len(self.cpt_uid)

        # HGC计算的嵌入
        self.h_lrn = h_lrn
        self.h_unt = unt_emb  # 学习单元+题目嵌入
        self.h_cpt = h_cpt
        
        self.max_seq_len = max_seq_len
        
        # 创建反向映射
        self.idx2lrn = {idx: uid for uid, idx in self.lrn_uid.items()}
        self.idx2untqus = {idx: uid for uid, idx in self.untqus_uid.items()}
        self.idx2cpt = {idx: uid for uid, idx in self.cpt_uid.items()}
        
        # 预计算序列数据
        self._precompute_sequences()

    def _precompute_sequences(self):
        """预计算所有学习者的序列数据"""
        # 初始化张量
        self.unt_seq_indices = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.long, device='cpu')
        self.add1_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.add2_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.type_indices_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.long, device='cpu')
        self.is_question_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.results_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.next_is_question_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.next_results_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.seq_masks = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.next_question_masks = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        
        # 遍历数据
        for lrn_uid, (unt_seq, add1_seq, add2_seq, is_question_seq, results_seq, next_is_question_seq) in self.data.items():
            if lrn_uid not in self.lrn_uid:
                continue
                
            lrn_idx = self.lrn_uid[lrn_uid]
            valid_len = min(len(unt_seq), self.max_seq_len)
            
            if valid_len == 0:
                continue
                
            # 填充序列数据
            for i in range(valid_len):
                unt_uid = unt_seq[i]
                if unt_uid in self.untqus_uid:
                    self.unt_seq_indices[lrn_idx, i] = self.untqus_uid[unt_uid]
                
                self.add1_seq[lrn_idx, i] = add1_seq[i]
                self.add2_seq[lrn_idx, i] = add2_seq[i]
                self.type_indices_seq[lrn_idx, i] = self.unit_types.get(unt_uid, 5)  # 默认question类型
                self.is_question_seq[lrn_idx, i] = is_question_seq[i]
                self.results_seq[lrn_idx, i] = results_seq[i] if results_seq[i] != -1 else 0
                self.next_is_question_seq[lrn_idx, i] = next_is_question_seq[i]
                
                # 下一个结果（用于训练）
                if i < valid_len - 1:
                    self.next_results_seq[lrn_idx, i] = results_seq[i+1] if results_seq[i+1] != -1 else 0
                else:
                    self.next_results_seq[lrn_idx, i] = 0
            
            # 填充掩码
            self.seq_masks[lrn_idx, :valid_len] = 1.0
            self.next_question_masks[lrn_idx, :valid_len] = torch.tensor(
                next_is_question_seq[:valid_len], dtype=torch.float32, device='cpu'
            )
    
    def __len__(self):
        return self.lrn_num

    def __getitem__(self, idx):
        """返回单个学习者的数据"""
        lrn_uid = self.idx2lrn[idx]
        
        return {
            'lrn_idx': idx,
            'lrn_uid': lrn_uid,
            'unt_seq_index': self.unt_seq_indices[idx],
            'add1': self.add1_seq[idx],
            'add2': self.add2_seq[idx],
            'type_indices': self.type_indices_seq[idx],
            'is_question': self.is_question_seq[idx],
            'results': self.results_seq[idx],
            'next_is_question': self.next_is_question_seq[idx],
            'next_results': self.next_results_seq[idx],
            'seq_mask': self.seq_masks[idx],
            'next_question_mask': self.next_question_masks[idx]
        }

    def collate_fn(self, batch):
        """批次处理函数"""
        lrn_indices = torch.tensor([item['lrn_idx'] for item in batch])
        unt_seq_indices = torch.stack([item['unt_seq_index'] for item in batch])
        add1 = torch.stack([item['add1'] for item in batch])
        add2 = torch.stack([item['add2'] for item in batch])
        type_indices = torch.stack([item['type_indices'] for item in batch])
        is_question = torch.stack([item['is_question'] for item in batch])
        results = torch.stack([item['results'] for item in batch])
        next_is_question = torch.stack([item['next_is_question'] for item in batch])
        next_results = torch.stack([item['next_results'] for item in batch])
        seq_masks = torch.stack([item['seq_mask'] for item in batch])
        next_question_masks = torch.stack([item['next_question_mask'] for item in batch])
        
        # 获取批次对应的学习者嵌入
        h_lrn_batch = self.h_lrn[lrn_indices]
        
        return {
            'lrn_indices': lrn_indices,
            'unt_seq_indices': unt_seq_indices,
            'add1': add1,
            'add2': add2,
            'type_indices': type_indices,
            'is_question': is_question,
            'results': results,
            'next_is_question': next_is_question,
            'next_results': next_results,
            'seq_masks': seq_masks,
            'next_question_masks': next_question_masks,
            'h_lrn_batch': h_lrn_batch,
        }

    def get_data_statistics(self):
        """返回数据集统计信息"""
        total_records = self.seq_masks.sum().item()
        total_next_questions = self.next_question_masks.sum().item()
        avg_seq_len = total_records / len(self) if len(self) > 0 else 0
        
        return {
            'total_learners': len(self),
            'total_records': int(total_records),
            'total_next_questions': int(total_next_questions),
            'average_sequence_length': round(avg_seq_len, 2),
            'max_sequence_length': self.max_seq_len
        }

    def get_embedding_info(self):
        """返回嵌入维度信息"""
        return {
            'lrn_emb_dim': self.h_lrn.shape[1],
            'unt_emb_dim': self.h_unt.shape[1],
            'cpt_emb_dim': self.h_cpt.shape[1],
            'cpt_num': self.cpt_num
        }

if __name__ == '__main__':
    import torch.nn as nn
    from Model.HGC import HGC
    from DataReader.HGCDataReader import hgcdr
    from torch.utils.data import DataLoader

    # 1. 加载HGC数据并计算嵌入
    print("1. 加载HGC数据并计算嵌入...")
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

    with torch.no_grad():
        lrn_emb, unt_emb, cpt_emb = model_hgc(hgcdr, device)

    print("✓ HGC嵌入计算完成")
    print(f"  学习者嵌入: {lrn_emb.shape}")
    print(f"  单元+题目嵌入: {unt_emb.shape}")
    print(f"  知识点嵌入: {cpt_emb.shape}")

    # 2. 加载KT数据
    print("\n2. 加载KT数据...")
    ktdata = ktdr.loadDatafromSql()

    # 3. 创建数据集
    print("\n3. 创建数据集...")
    train_dataset = KTDataSet(ktdata, lrn_emb, unt_emb, cpt_emb, 'train', max_seq_len=128)
    test_dataset = KTDataSet(ktdata, lrn_emb, unt_emb, cpt_emb, 'test', max_seq_len=128)

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
        unt_seq_indices = batch['unt_seq_indices']
        seq_masks = batch['seq_masks']
        next_question_masks = batch['next_question_masks']
        h_lrn_batch = batch['h_lrn_batch']
        
        # 检查维度一致性
        assert lrn_indices.shape[0] == unt_seq_indices.shape[0], "batch_size不一致"
        assert unt_seq_indices.shape == seq_masks.shape == next_question_masks.shape, "序列维度不一致"
        assert h_lrn_batch.shape[0] == lrn_indices.shape[0], "学习者嵌入batch_size不一致"
        
        print(f"  ✓ 维度一致性检查通过")
        print(f"  ✓ 有效序列位置: {seq_masks.sum().item():.0f}")
        print(f"  ✓ 下一个是题目的位置: {next_question_masks.sum().item():.0f}")

    # 7. 验证batch大小一致性
    print(f"\n7. Batch大小验证:")
    if len(batch_sizes) > 0:
        unique_sizes = set(batch_sizes)
        if len(unique_sizes) == 1:
            print(f"✓ 所有batch大小一致: {batch_sizes[0]}")
        else:
            print(f"✗ batch大小不一致: {batch_sizes}")
    else:
        print("没有获取到batch数据")

    # 8. 数据完整性检查
    print(f"\n8. 数据完整性检查:")
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
    
    print(f"\n=== KTDataSet测试完成 ===")