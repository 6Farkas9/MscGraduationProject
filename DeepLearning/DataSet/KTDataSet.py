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
    知识追踪数据集 - 适配新顺序：qusunt（前半部分qus，后半部分unt）
    """

    def __init__(self, static_data, h_lrn, qusunt_emb, h_cpt, data_type='train', max_seq_len=None):
        """
        Args:
            static_data: 从KTDataReader获取的静态数据
            h_lrn: (lrn_num, emb_dim) - HGC计算的学习者嵌入
            qusunt_emb: (qusunt_num, emb_dim) - HGC计算的题目+学习单元嵌入（新顺序：前半部分qus，后半部分unt）
            h_cpt: (cpt_num, emb_dim) - HGC计算的知识点嵌入
            data_type: 'train' 或 'test' 或 'all'
            max_seq_len: 最大序列长度
        """
        super(KTDataSet, self).__init__()
        
        if max_seq_len is None:
            max_seq_len = hyperparams.data_max_seq_len
        
        # 交互数据
        if data_type == 'all':
            # 合并train和test数据
            self.data = self._merge_train_test_data(static_data)
        else:
            self.data = static_data[f'{data_type}_data']
            
        self.lrn_uid = static_data['lrn_uid']
        self.qusunt_uid = static_data['qusunt_uid']  # 新顺序：前半部分qus，后半部分unt
        self.cpt_uid = static_data['cpt_uid']
        self.unit_types = static_data['unit_types']
        self.question_concepts = static_data['question_concepts']
        self.unit_concepts = static_data['unit_concepts']
        
        # 数量统计
        self.lrn_num = len(self.lrn_uid)
        self.qusunt_num = len(self.qusunt_uid)
        self.cpt_num = len(self.cpt_uid)
        self.qus_num = basicdr.qus_num  # 题目数量

        # HGC计算的嵌入
        self.h_lrn = h_lrn
        self.h_qusunt = qusunt_emb  # 题目+学习单元嵌入（新顺序）
        self.h_cpt = h_cpt
        
        self.max_seq_len = max_seq_len
        
        # 创建反向映射
        self.idx2lrn = {idx: uid for uid, idx in self.lrn_uid.items()}
        self.idx2qusunt = {idx: uid for uid, idx in self.qusunt_uid.items()}
        self.idx2cpt = {idx: uid for uid, idx in self.cpt_uid.items()}
        
        # 预计算序列数据
        self._precompute_sequences()

    def _merge_train_test_data(self, static_data):
        """合并train和test数据"""
        from collections import defaultdict
        
        train_data = static_data['train_data']
        test_data = static_data['test_data']
        
        # 创建合并后的数据
        merged_data = defaultdict(lambda: [[], [], [], [], [], []])  # [unt_uids, add1s, add2s, is_questions, results, next_is_questions]
        
        # 合并训练数据
        for lrn_uid, (unt_seq, add1_seq, add2_seq, is_question_seq, results_seq, next_is_question_seq) in train_data.items():
            merged_data[lrn_uid][0].extend(unt_seq)
            merged_data[lrn_uid][1].extend(add1_seq)
            merged_data[lrn_uid][2].extend(add2_seq)
            merged_data[lrn_uid][3].extend(is_question_seq)
            merged_data[lrn_uid][4].extend(results_seq)
            merged_data[lrn_uid][5].extend(next_is_question_seq)
        
        # 合并测试数据
        for lrn_uid, (unt_seq, add1_seq, add2_seq, is_question_seq, results_seq, next_is_question_seq) in test_data.items():
            merged_data[lrn_uid][0].extend(unt_seq)
            merged_data[lrn_uid][1].extend(add1_seq)
            merged_data[lrn_uid][2].extend(add2_seq)
            merged_data[lrn_uid][3].extend(is_question_seq)
            merged_data[lrn_uid][4].extend(results_seq)
            merged_data[lrn_uid][5].extend(next_is_question_seq)
        
        return merged_data

    def _precompute_sequences(self):
        """预计算所有学习者的序列数据 - 适配新顺序"""
        print(f"  预计算KT序列数据，最大长度: {self.max_seq_len}")
        
        # 初始化张量
        self.qusunt_seq_indices = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.long, device='cpu')
        self.add1_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.add2_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.type_indices_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.long, device='cpu')
        self.is_question_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.results_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.next_is_question_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.next_results_seq = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.seq_masks = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        self.next_question_masks = torch.zeros(self.lrn_num, self.max_seq_len, dtype=torch.float32, device='cpu')
        
        # 统计信息
        total_records = 0
        total_questions = 0
        max_actual_len = 0
        
        # 遍历数据
        for lrn_uid, (qusunt_seq, add1_seq, add2_seq, is_question_seq, results_seq, next_is_question_seq) in self.data.items():
            if lrn_uid not in self.lrn_uid:
                continue
                
            lrn_idx = self.lrn_uid[lrn_uid]
            valid_len = min(len(qusunt_seq), self.max_seq_len)
            
            if valid_len == 0:
                continue
            
            # 更新统计
            total_records += valid_len
            total_questions += sum(is_question_seq[:valid_len])
            max_actual_len = max(max_actual_len, valid_len)
                
            # 填充序列数据
            for i in range(valid_len):
                qusunt_uid = qusunt_seq[i]
                if qusunt_uid in self.qusunt_uid:
                    self.qusunt_seq_indices[lrn_idx, i] = self.qusunt_uid[qusunt_uid]
                
                self.add1_seq[lrn_idx, i] = add1_seq[i]
                self.add2_seq[lrn_idx, i] = add2_seq[i]
                self.type_indices_seq[lrn_idx, i] = self.unit_types.get(qusunt_uid, 5)  # 默认question类型
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
        
        print(f"    总记录数: {total_records}, 题目数量: {total_questions}, 最大序列长度: {max_actual_len}")

    def __len__(self):
        return self.lrn_num

    def __getitem__(self, idx):
        """返回单个学习者的数据"""
        lrn_uid = self.idx2lrn[idx]
        
        return {
            'lrn_idx': idx,
            'lrn_uid': lrn_uid,
            'qusunt_seq_index': self.qusunt_seq_indices[idx],  # 新名称
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
        """批次处理函数 - 适配新顺序"""
        lrn_indices = torch.tensor([item['lrn_idx'] for item in batch])
        qusunt_seq_indices = torch.stack([item['qusunt_seq_index'] for item in batch])  # 新名称
        add1 = torch.stack([item['add1'] for item in batch])
        add2 = torch.stack([item['add2'] for item in batch])
        type_indices = torch.stack([item['type_indices'] for item in batch])
        is_question = torch.stack([item['is_question'] for item in batch])
        results = torch.stack([item['results'] for item in batch])
        next_is_question = torch.stack([item['next_is_question'] for item in batch])
        next_results = torch.stack([item['next_results'] for item in batch])
        seq_masks = torch.stack([item['seq_mask'] for item in batch])
        next_question_masks = torch.stack([item['next_question_mask'] for item in batch])
        
        # 获取批次对应的学习者嵌入 - 使用clone确保梯度安全
        h_lrn_batch = self.h_lrn[lrn_indices].clone()
        
        return {
            'lrn_indices': lrn_indices,
            'qusunt_seq_indices': qusunt_seq_indices,  # 新名称
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
        total_questions = self.is_question_seq.sum().item()
        avg_seq_len = total_records / len(self) if len(self) > 0 else 0
        
        return {
            'total_learners': len(self),
            'total_records': int(total_records),
            'total_questions': int(total_questions),
            'total_next_questions': int(total_next_questions),
            'average_sequence_length': round(avg_seq_len, 2),
            'max_sequence_length': self.max_seq_len,
            'qusunt_count': self.qusunt_num,
            'question_count': self.qus_num,
            'concept_count': self.cpt_num
        }

    def get_embedding_info(self):
        """返回嵌入维度信息"""
        return {
            'lrn_emb_dim': self.h_lrn.shape[1],
            'qusunt_emb_dim': self.h_qusunt.shape[1],  # 新名称
            'cpt_emb_dim': self.h_cpt.shape[1],
            'qusunt_count': self.qusunt_num,
            'question_count': self.qus_num,
            'concept_count': self.cpt_num
        }

def test_kt_dataset():
    """测试KT数据集"""
    print("=== KTDataSet 测试 (适配新顺序) ===")
    
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

    # 2. 加载KT数据
    print("\n2. 加载KT数据...")
    ktdata = ktdr.loadDatafromSql()

    # 3. 创建数据集
    print("\n3. 创建KT数据集...")
    train_dataset = KTDataSet(ktdata, lrn_emb, qusunt_emb, cpt_emb, 'train')
    test_dataset = KTDataSet(ktdata, lrn_emb, qusunt_emb, cpt_emb, 'test')
    all_dataset = KTDataSet(ktdata, lrn_emb, qusunt_emb, cpt_emb, 'all')  # 新增all模式

    # 4. 查看统计信息
    print("\n4. 数据集统计信息:")
    print("训练集统计:", train_dataset.get_data_statistics())
    print("测试集统计:", test_dataset.get_data_statistics())
    print("全数据集统计:", all_dataset.get_data_statistics())  # 新增all模式统计
    
    # 验证all模式数据量是否正确
    train_stats = train_dataset.get_data_statistics()
    test_stats = test_dataset.get_data_statistics()
    all_stats = all_dataset.get_data_statistics()
    
    expected_total_records = train_stats['total_records'] + test_stats['total_records']
    expected_total_questions = train_stats['total_questions'] + test_stats['total_questions']
    expected_total_next_questions = train_stats['total_next_questions'] + test_stats['total_next_questions']
    
    actual_total_records = all_stats['total_records']
    actual_total_questions = all_stats['total_questions']
    actual_total_next_questions = all_stats['total_next_questions']
    
    records_match = expected_total_records == actual_total_records
    questions_match = expected_total_questions == actual_total_questions
    next_questions_match = expected_total_next_questions == actual_total_next_questions
    
    if records_match and questions_match and next_questions_match:
        print("✓ all模式数据合并验证通过:")
        print(f"  记录数: {train_stats['total_records']} + {test_stats['total_records']} = {actual_total_records}")
        print(f"  题目数: {train_stats['total_questions']} + {test_stats['total_questions']} = {actual_total_questions}")
        print(f"  下一题目数: {train_stats['total_next_questions']} + {test_stats['total_next_questions']} = {actual_total_next_questions}")
    else:
        print("✗ all模式数据合并验证失败")
        if not records_match:
            print(f"  记录数不匹配: {expected_total_records} ≠ {actual_total_records}")
        if not questions_match:
            print(f"  题目数不匹配: {expected_total_questions} ≠ {actual_total_questions}")
        if not next_questions_match:
            print(f"  下一题目数不匹配: {expected_total_next_questions} ≠ {actual_total_next_questions}")
    
    print("嵌入信息:", train_dataset.get_embedding_info())

    # 5. 测试单个样本
    print("\n5. 单个样本测试:")
    datasets = [('train', train_dataset), ('test', test_dataset), ('all', all_dataset)]
    
    for data_type, dataset in datasets:
        if len(dataset) > 0:
            print(f"\n  {data_type.upper()}模式单个样本:")
            single_sample = dataset[0]
            print(f"    样本键值: {list(single_sample.keys())}")
            for key, value in single_sample.items():
                if torch.is_tensor(value):
                    print(f"    {key}: {value.shape} (dtype: {value.dtype})")
                    # 显示部分序列内容
                    if key in ['qusunt_seq_index', 'is_question', 'results']:
                        valid_mask = dataset.seq_masks[0] if hasattr(dataset, 'seq_masks') else None
                        if valid_mask is not None:
                            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                            if len(valid_indices) > 0:
                                valid_values = value[valid_indices]
                                print(f"      有效值示例: {valid_values[:3].tolist()}{'...' if len(valid_values) > 3 else ''}")

    # 6. 测试batch功能
    print("\n6. Batch功能测试:")
    for data_type, dataset in datasets:
        batch_size = min(4, len(dataset))
        if batch_size > 0:
            print(f"\n  {data_type.upper()}模式Batch测试:")
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=dataset.collate_fn
            )

            # 检查一个batch
            for i, batch in enumerate(data_loader):
                if i >= 1:  # 只检查第一个batch
                    break
                    
                print(f"    Batch {i + 1}:")
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        print(f"      {key}: {value.shape} (dtype: {value.dtype})")
                
                # 验证关键数据一致性
                lrn_indices = batch['lrn_indices']
                qusunt_seq_indices = batch['qusunt_seq_indices']
                seq_masks = batch['seq_masks']
                next_question_masks = batch['next_question_masks']
                h_lrn_batch = batch['h_lrn_batch']
                
                # 检查维度一致性
                assert lrn_indices.shape[0] == qusunt_seq_indices.shape[0], "batch_size不一致"
                assert qusunt_seq_indices.shape == seq_masks.shape == next_question_masks.shape, "序列维度不一致"
                assert h_lrn_batch.shape[0] == lrn_indices.shape[0], "学习者嵌入batch_size不一致"
                
                print(f"      ✓ 维度一致性检查通过")
                print(f"      ✓ 有效序列位置: {seq_masks.sum().item():.0f}")
                print(f"      ✓ 下一个是题目的位置: {next_question_masks.sum().item():.0f}")
                
                # 验证索引范围
                max_idx = qusunt_seq_indices.max().item()
                if max_idx < dataset.qusunt_num:
                    print(f"      ✓ 索引范围验证通过 [0, {max_idx}] < {dataset.qusunt_num}")
                else:
                    print(f"      ✗ 索引超出范围: {max_idx} >= {dataset.qusunt_num}")
        else:
            print(f"  {data_type.upper()}模式: 没有足够的数据进行batch测试")

    # 7. 数据完整性检查
    print(f"\n7. 数据完整性检查:")
    for data_type, dataset in datasets:
        has_nan = False
        has_inf = False
        
        if len(dataset) > 0:
            sample = dataset[0]
            for key, value in sample.items():
                if torch.is_tensor(value):
                    if torch.isnan(value).any():
                        print(f"✗ {data_type}模式 {key} 包含NaN值")
                        has_nan = True
                    if torch.isinf(value).any():
                        print(f"✗ {data_type}模式 {key} 包含Inf值")
                        has_inf = True
        
        if not has_nan and not has_inf:
            print(f"✓ {data_type}模式数据完整性检查通过 - 无NaN和Inf值")
    
    # 8. 测试all模式特有功能
    print(f"\n8. all模式特有功能测试:")
    if len(all_dataset) > 0:
        # 验证all模式是否包含了所有学习者的数据
        all_learners = set(all_dataset.data.keys())
        train_learners = set(train_dataset.data.keys())
        test_learners = set(test_dataset.data.keys())
        
        expected_learners = train_learners.union(test_learners)
        
        if all_learners == expected_learners:
            print(f"✓ all模式学习者合并验证通过: {len(all_learners)} 个学习者")
        else:
            print(f"✗ all模式学习者合并验证失败")
            print(f"  缺失的学习者: {expected_learners - all_learners}")
            print(f"  多余的学习者: {all_learners - expected_learners}")
        
        # 验证序列长度和数据类型
        print("  all模式数据类型验证:")
        sample_learner = list(all_learners)[0] if all_learners else None
        if sample_learner:
            all_data = all_dataset.data[sample_learner]
            print(f"    示例学习者 {sample_learner} 的数据结构:")
            print(f"    学习单元序列长度: {len(all_data[0])}")
            print(f"    add1序列长度: {len(all_data[1])}")
            print(f"    add2序列长度: {len(all_data[2])}")
            print(f"    题目标记序列长度: {len(all_data[3])}")
            print(f"    结果序列长度: {len(all_data[4])}")
            print(f"    下一题目标记序列长度: {len(all_data[5])}")
            
            # 检查数据类型
            if len(all_data[0]) > 0:
                print(f"    学习单元类型示例: {all_data[0][:3]}{'...' if len(all_data[0]) > 3 else ''}")
                print(f"    题目标记示例: {all_data[3][:3]}{'...' if len(all_data[3]) > 3 else ''}")
                print(f"    结果示例: {all_data[4][:3]}{'...' if len(all_data[4]) > 3 else ''}")
    
    print(f"\n=== KTDataSet测试完成 (适配新顺序) ===")

if __name__ == '__main__':
    test_kt_dataset()