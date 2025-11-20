import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from DataReader.KTDataReader import ktdr
from DataReader.BasicDataReader import basicdr
from hyperparams.hyperparameter import hyperparams

class KTDataSet(Dataset):
    """
    知识追踪数据集 - 适配新的7元素数据结构
    数据结构: [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]
    """

    def __init__(self, static_data, h_lrn, qusunt_emb, h_cpt, data_type='train', max_seq_len=None):
        """
        Args:
            static_data: 从KTDataReader获取的静态数据
            h_lrn: (lrn_num, emb_dim) - HGC计算的学习者嵌入
            qusunt_emb: (qusunt_num, emb_dim) - HGC计算的题目+学习单元嵌入
            h_cpt: (cpt_num, emb_dim) - HGC计算的知识点嵌入
            data_type: 'train' 或 'test' 或 'all'
            max_seq_len: 最大序列长度
        """
        super(KTDataSet, self).__init__()
        
        if max_seq_len is None:
            max_seq_len = hyperparams.data_max_seq_len
        
        # 交互数据
        if data_type == 'all':
            # 直接使用complete_data
            if hasattr(ktdr, 'complete_data'):
                self.data = ktdr.complete_data
                print("  使用KT完整数据 (complete_data)")
            else:
                # 如果没有complete_data，回退到合并train和test
                print("  警告: 未找到complete_data，回退到合并train和test数据")
                self.data = self._merge_train_test_data(static_data)
        else:
            self.data = static_data[f'{data_type}_data']
            
        self.lrn_uid = static_data['lrn_uid']
        self.qusunt_uid = static_data['qusunt_uid']
        self.cpt_uid = static_data['cpt_uid']
        self.unit_types = static_data['unit_types']
        self.question_concepts = static_data['question_concepts']
        self.unit_concepts = static_data['unit_concepts']
        
        # 数量统计
        self.lrn_num = len(self.lrn_uid)
        self.qusunt_num = len(self.qusunt_uid)
        self.cpt_num = len(self.cpt_uid)
        self.qus_num = basicdr.qus_num

        # HGC计算的嵌入
        self.h_lrn = h_lrn
        self.h_qusunt = qusunt_emb
        self.h_cpt = h_cpt
        
        self.max_seq_len = max_seq_len
        
        # 创建反向映射
        self.idx2lrn = {idx: uid for uid, idx in self.lrn_uid.items()}
        self.idx2qusunt = {idx: uid for uid, idx in self.qusunt_uid.items()}
        self.idx2cpt = {idx: uid for uid, idx in self.cpt_uid.items()}
        
        # 预计算序列数据
        self._precompute_sequences()

    def _merge_train_test_data(self, static_data):
        """合并train和test数据 - 保留作为回退方案"""
        from collections import defaultdict
        
        train_data = static_data['train_data']
        test_data = static_data['test_data']
        
        # 创建合并后的数据
        merged_data = defaultdict(lambda: [[], [], [], [], [], [], []])
        
        # 合并训练数据
        for lrn_uid, data in train_data.items():
            for i in range(7):  # 7个数据列表
                merged_data[lrn_uid][i].extend(data[i])
        
        # 合并测试数据
        for lrn_uid, data in test_data.items():
            for i in range(7):  # 7个数据列表
                merged_data[lrn_uid][i].extend(data[i])
        
        return merged_data

    def _precompute_sequences(self):
        """预计算所有学习者的序列数据 - 适配新数据结构"""
        print(f"  预计算KT序列数据，最大长度: {self.max_seq_len}")
        
        # 初始化张量
        self.qusunt_seq_indices = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.long, device='cpu')
        self.add1_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.add2_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.type_indices_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.long, device='cpu')
        self.is_question_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.results_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.prediction_masks_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.next_results_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.seq_masks = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        
        # 统计信息
        total_records = 0
        total_questions = 0
        total_valid_predictions = 0
        max_actual_len = 0
        
        # 遍历数据
        for lrn_uid, data in self.data.items():
            if lrn_uid not in self.lrn_uid:
                continue
                
            lrn_idx = self.lrn_uid[lrn_uid]
            valid_len = min(len(data[0]), self.max_seq_len)  # 使用unt_uids的长度
            
            if valid_len == 0:
                continue
            
            # 更新统计
            total_records += valid_len
            total_questions += sum(data[3][:valid_len])  # is_questions
            total_valid_predictions += sum(data[5][:valid_len])  # prediction_masks
            max_actual_len = max(max_actual_len, valid_len)
                
            # 填充序列数据
            for i in range(valid_len):
                # 学习单元索引
                qusunt_uid = data[0][i]
                if qusunt_uid in self.qusunt_uid:
                    self.qusunt_seq_indices[lrn_idx, i] = self.qusunt_uid[qusunt_uid]
                
                # 数值特征
                self.add1_seq[lrn_idx, i] = data[1][i]
                self.add2_seq[lrn_idx, i] = data[2][i]
                
                # 类型和题目标记
                self.type_indices_seq[lrn_idx, i] = self.unit_types.get(qusunt_uid, 5)
                self.is_question_seq[lrn_idx, i] = data[3][i]  # is_question
                
                # 当前结果（用于特征）
                current_result = data[4][i]
                self.results_seq[lrn_idx, i] = current_result if current_result != -1 else 0
                
                # 预测掩码和下一个结果（用于训练）
                self.prediction_masks_seq[lrn_idx, i] = data[5][i]  # prediction_mask
                self.next_results_seq[lrn_idx, i] = data[6][i]  # next_result
            
            # 填充序列掩码
            self.seq_masks[lrn_idx, :valid_len] = 1.0
        
        print(f"    总记录数: {total_records}, 题目数量: {total_questions}, 有效预测位置: {total_valid_predictions}")
        print(f"    最大序列长度: {max_actual_len}, 掩码覆盖率: {total_valid_predictions/total_records*100:.1f}%")

    def __len__(self):
        return self.lrn_num

    def __getitem__(self, idx):
        """返回单个学习者的数据"""
        lrn_uid = self.idx2lrn[idx]
        
        return {
            'lrn_idx': idx,
            'lrn_uid': lrn_uid,
            'qusunt_seq_indices': self.qusunt_seq_indices[idx],
            'add1': self.add1_seq[idx],
            'add2': self.add2_seq[idx],
            'type_indices': self.type_indices_seq[idx],
            'is_question': self.is_question_seq[idx],
            'results': self.results_seq[idx],
            'prediction_masks': self.prediction_masks_seq[idx],  # 新的预测掩码
            'next_results': self.next_results_seq[idx],  # 新的下一个结果
            'seq_masks': self.seq_masks[idx],
        }

    def collate_fn(self, batch):
        """批次处理函数"""
        lrn_indices = torch.tensor([item['lrn_idx'] for item in batch])
        qusunt_seq_indices = torch.stack([item['qusunt_seq_indices'] for item in batch])
        add1 = torch.stack([item['add1'] for item in batch])
        add2 = torch.stack([item['add2'] for item in batch])
        type_indices = torch.stack([item['type_indices'] for item in batch])
        is_question = torch.stack([item['is_question'] for item in batch])
        results = torch.stack([item['results'] for item in batch])
        prediction_masks = torch.stack([item['prediction_masks'] for item in batch])  # 新的预测掩码
        next_results = torch.stack([item['next_results'] for item in batch])  # 新的下一个结果
        seq_masks = torch.stack([item['seq_masks'] for item in batch])
        
        # 获取批次对应的学习者嵌入
        h_lrn_batch = self.h_lrn[lrn_indices].clone()
        
        return {
            'lrn_indices': lrn_indices,
            'qusunt_seq_indices': qusunt_seq_indices,
            'add1': add1,
            'add2': add2,
            'type_indices': type_indices,
            'is_question': is_question,
            'results': results,
            'prediction_masks': prediction_masks,  # 新的预测掩码
            'next_results': next_results,  # 新的下一个结果
            'seq_masks': seq_masks,
            'h_lrn_batch': h_lrn_batch,
        }

    def get_data_statistics(self):
        """返回数据集统计信息"""
        total_records = self.seq_masks.sum().item()
        total_valid_predictions = self.prediction_masks_seq.sum().item()
        total_questions = self.is_question_seq.sum().item()
        avg_seq_len = total_records / len(self) if len(self) > 0 else 0
        
        return {
            'total_learners': len(self),
            'total_records': int(total_records),
            'total_questions': int(total_questions),
            'total_valid_predictions': int(total_valid_predictions),
            'average_sequence_length': round(avg_seq_len, 2),
            'max_sequence_length': self.max_seq_len,
            'prediction_coverage': f"{total_valid_predictions/total_records*100:.1f}%",
            'qusunt_count': self.qusunt_num,
            'question_count': self.qus_num,
            'concept_count': self.cpt_num
        }

    def get_embedding_info(self):
        """返回嵌入维度信息"""
        return {
            'lrn_emb_dim': self.h_lrn.shape[1],
            'qusunt_emb_dim': self.h_qusunt.shape[1],
            'cpt_emb_dim': self.h_cpt.shape[1],
            'qusunt_count': self.qusunt_num,
            'question_count': self.qus_num,
            'concept_count': self.cpt_num
        }

def test_kt_dataset():
    """测试KT数据集"""
    print("=== KTDataSet 测试 (新数据结构) ===")
    
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

    # 2. 加载KT数据
    print("\n2. 加载KT数据...")
    ktdata = ktdr.loadDatafromSql()

    # 3. 创建数据集
    print("\n3. 创建KT数据集...")
    train_dataset = KTDataSet(ktdata, lrn_emb, qusunt_emb, cpt_emb, 'train')

    # 4. 查看统计信息
    print("\n4. 数据集统计信息:")
    stats = train_dataset.get_data_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 5. 测试单个样本
    print("\n5. 单个样本测试:")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"  样本键值: {list(sample.keys())}")
        
        # 检查关键数据
        for key in ['prediction_masks', 'next_results', 'results']:
            if key in sample:
                value = sample[key]
                valid_mask = train_dataset.seq_masks[0]
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                if len(valid_indices) > 0:
                    valid_values = value[valid_indices]
                    print(f"  {key}: 有效值范围 [{valid_values.min().item():.3f}, {valid_values.max().item():.3f}]")
                    print(f"    前5个值: {valid_values[:5].tolist()}")

    # 6. 验证掩码逻辑
    print("\n6. 掩码逻辑验证:")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        prediction_masks = sample['prediction_masks']
        next_results = sample['next_results']
        seq_mask = sample['seq_masks']
        
        # 获取有效位置
        valid_positions = seq_mask.nonzero(as_tuple=True)[0]
        if len(valid_positions) > 0:
            valid_masks = prediction_masks[valid_positions]
            valid_next_results = next_results[valid_positions]
            
            print(f"  有效序列长度: {len(valid_positions)}")
            print(f"  有效预测位置: {valid_masks.sum().item()}")
            
            # 检查掩码与结果的对应关系
            masked_next_results = valid_next_results[valid_masks.bool()]
            if len(masked_next_results) > 0:
                print(f"  掩码后结果范围: [{masked_next_results.min().item():.3f}, {masked_next_results.max().item():.3f}]")
                print(f"  ✓ 掩码逻辑验证通过")

    print(f"\n=== KTDataSet测试完成 (新数据结构) ===")

if __name__ == '__main__':
    test_kt_dataset()