# Test_CD_KT_DataSet.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 导入数据读取器
from DataReader.CDDataReader import cddr
from DataReader.KTDataReader import ktdr
from DataReader.HGCDataReader import hgcdr
from DataReader.BasicDataReader import basicdr

# 导入数据合并器
from DataService.CD_KTDataMerger import cd_kt_merger

# 导入数据集
from DataSet.CDDataSet import CDDataset
from DataSet.KTDataSet import KTDataSet

# 导入模型
from Model.HGC import HGC

# 导入超参数
from hyperparams.hyperparameter import hyperparams

class CD_KT_DataPipelineTest:
    """CD和KT数据管道完整测试 - 适配新数据结构"""
    
    def __init__(self):
        self.device = hyperparams.device
        # self.setup_directories()
    
    def setup_directories(self):
        """创建测试输出目录"""
        self.test_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_outputs")
        os.makedirs(self.test_output_dir, exist_ok=True)
        print(f"测试输出目录: {self.test_output_dir}")
    
    def test_data_reader_initialization(self):
        """测试DataReader初始化 - 适配新数据结构"""
        print("\n" + "="*60)
        print("1. 测试DataReader初始化 (新数据结构)")
        print("="*60)
        
        # 测试基础数据读取器
        print("✓ BasicDataReader 初始化完成")
        print(f"  学习者数量: {len(basicdr.lrn_uid)}")
        print(f"  题目数量: {len(basicdr.qus_uid)}")
        print(f"  知识点数量: {len(basicdr.cpt_uid)}")
        print(f"  学习单元+题目数量: {len(basicdr.qusunt_uid)}")
        
        # 测试CD数据读取器
        cd_data = cddr.loadDatafromSql()
        print("✓ CDDataReader 初始化完成")
        print(f"  CD训练数据学习者: {len(cddr.train_data)}")
        print(f"  CD测试数据学习者: {len(cddr.test_data)}")
        
        # 测试KT数据读取器 - 适配新数据结构
        kt_data = ktdr.loadDatafromSql()
        print("✓ KTDataReader 初始化完成 (新数据结构)")
        print(f"  KT训练数据学习者: {len(ktdr.train_data)}")
        print(f"  KT测试数据学习者: {len(ktdr.test_data)}")
        
        # 检查KT数据结构
        if ktdr.train_data:
            sample_lrn = list(ktdr.train_data.keys())[0]
            sample_data = ktdr.train_data[sample_lrn]
            print(f"  KT数据结构验证: {len(sample_data)}个数据列表")
            print(f"    学习单元序列: {len(sample_data[0])}")
            print(f"    预测掩码序列: {len(sample_data[5])}")
            print(f"    下一个结果序列: {len(sample_data[6])}")
        
        # 测试HGC数据读取器
        hgcdr.loadDatafromSql()
        print("✓ HGCDataReader 初始化完成")
        print(f"  学习者初始特征维度: {hgcdr.lrn_init.shape}")
        print(f"  学习单元初始特征维度: {hgcdr.qusunt_init.shape}")
        print(f"  知识点初始特征维度: {hgcdr.cpt_init.shape}")
        
        return True
    
    def test_complete_data_function(self):
        """测试完整数据功能 - 适配新数据结构"""
        print("\n" + "="*60)
        print("2. 测试完整数据功能 (新数据结构)")
        print("="*60)
        
        # 测试CD完整数据
        cd_complete_data = cddr.getCompleteData()
        print("✓ CD完整数据获取完成")
        print(f"  CD完整数据学习者数量: {len(cd_complete_data)}")
        cd_total_records = sum(len(data[0]) for data in cd_complete_data.values())
        print(f"  CD完整数据记录数量: {cd_total_records}")
        
        # 测试KT完整数据 - 适配新数据结构
        kt_complete_data = ktdr.getCompleteData()
        print("✓ KT完整数据获取完成 (新数据结构)")
        print(f"  KT完整数据学习者数量: {len(kt_complete_data)}")
        kt_total_records = sum(len(data[0]) for data in kt_complete_data.values())
        kt_valid_predictions = sum(sum(data[5]) for data in kt_complete_data.values())  # 预测掩码统计
        print(f"  KT完整数据记录数量: {kt_total_records}")
        print(f"  KT有效预测位置数量: {kt_valid_predictions}")
        print(f"  KT掩码覆盖率: {kt_valid_predictions/kt_total_records*100:.1f}%")
        
        # 检查数据格式
        if cd_complete_data:
            sample_lrn = list(cd_complete_data.keys())[0]
            sample_data = cd_complete_data[sample_lrn]
            print(f"  CD数据格式: {type(sample_data)}, 题目序列长度: {len(sample_data[0])}, 结果序列长度: {len(sample_data[1])}")
        
        if kt_complete_data:
            sample_lrn = list(kt_complete_data.keys())[0]
            sample_data = kt_complete_data[sample_lrn]
            print(f"  KT数据格式: {type(sample_data)}, 序列长度: {len(sample_data[0])}")
            print(f"    预测掩码数量: {sum(sample_data[5])}/{len(sample_data[5])}")
            print(f"    下一个结果范围: [{min([r for r in sample_data[6] if r != 0])}, {max(sample_data[6])}]")
        
        return True
    
    def test_data_merger(self):
        """测试数据合并器 - 适配新数据结构"""
        print("\n" + "="*60)
        print("3. 测试数据合并器 (新数据结构)")
        print("="*60)
        
        # 测试训练测试数据合并
        print("3.1 训练测试数据合并测试")
        cd_kt_merger.merge_and_filter_train_test()
        
        # 获取统计信息
        stats = cd_kt_merger.get_merged_statistics()
        if stats and 'train_test' in stats:
            train_test_stats = stats['train_test']
            print("✓ 训练测试数据合并统计 (新数据结构):")
            for key, value in train_test_stats.items():
                print(f"  {key}: {value}")
        
        # 验证训练测试数据一致性
        print("\n3.2 训练测试数据一致性验证")
        train_test_consistent = cd_kt_merger.verify_consistency('train_test')
        print(f"  训练测试数据一致性: {'通过' if train_test_consistent else '失败'}")
        
        # 测试完整数据合并
        print("\n3.3 完整数据合并测试")
        cd_kt_merger.merge_and_filter_complete_data()
        
        # 获取统计信息
        stats = cd_kt_merger.get_merged_statistics()
        if stats and 'complete' in stats:
            complete_stats = stats['complete']
            print("✓ 完整数据合并统计 (新数据结构):")
            for key, value in complete_stats.items():
                print(f"  {key}: {value}")
        
        # 验证完整数据一致性
        print("\n3.4 完整数据一致性验证")
        complete_consistent = cd_kt_merger.verify_consistency('complete')
        print(f"  完整数据一致性: {'通过' if complete_consistent else '失败'}")
        
        return train_test_consistent and complete_consistent
    
    def test_hgc_embeddings(self):
        """测试HGC嵌入计算"""
        print("\n" + "="*60)
        print("4. 测试HGC嵌入计算")
        print("="*60)
        
        # 动态获取输入维度
        lrn_input_dim = hgcdr.lrn_init.shape[1]
        unt_input_dim = hgcdr.qusunt_init.shape[1]
        cpt_input_dim = hgcdr.cpt_init.shape[1]
        
        print(f"  输入维度 - lrn: {lrn_input_dim}, unt: {unt_input_dim}, cpt: {cpt_input_dim}")
        
        # 初始化HGC模型
        model_hgc = HGC(
            embedding_dim=hyperparams.hgc_embedding_dim
            # 新版HGC不再需要输入维度参数
        ).to(self.device)
        
        # 计算嵌入
        with torch.no_grad():
            input_data = {
                'lrn_init': hgcdr.lrn_init,
                'unt_init': hgcdr.qusunt_init,
                'cpt_init': hgcdr.cpt_init,
                'p_lul': (hgcdr.p_lul[0], hgcdr.p_lul[1]),
                'p_lcl': (hgcdr.p_lcl[0], hgcdr.p_lcl[1]),
                'p_ltl': (hgcdr.p_ltl[0], hgcdr.p_ltl[1]),
                'p_ulu': (hgcdr.p_ulu[0], hgcdr.p_ulu[1]),
                'p_ucrsu': (hgcdr.p_ucrsu[0], hgcdr.p_ucrsu[1]),
                'p_ucptu': (hgcdr.p_ucptu[0], hgcdr.p_ucptu[1]),
                'p_cc': (hgcdr.p_cc[0], hgcdr.p_cc[1]),
                'p_cuc': (hgcdr.p_cuc[0], hgcdr.p_cuc[1]),
                'p_ctc': (hgcdr.p_ctc[0], hgcdr.p_ctc[1])
            }
            lrn_emb, qusunt_emb, cpt_emb = model_hgc(input_data=input_data, device=self.device, return_dict=False)
        
        print("✓ HGC嵌入计算完成")
        print(f"  学习者嵌入维度: {lrn_emb.shape}")
        print(f"  学习单元+题目嵌入维度: {qusunt_emb.shape}")
        print(f"  知识点嵌入维度: {cpt_emb.shape}")
        
        return lrn_emb, qusunt_emb, cpt_emb
    
    def test_dataset_creation(self):
        """测试数据集创建 - 适配新数据结构"""
        print("\n" + "="*60)
        print("5. 测试数据集创建 (新数据结构)")
        print("="*60)
        
        # 首先计算HGC嵌入
        lrn_emb, qusunt_emb, cpt_emb = self.test_hgc_embeddings()
        
        # 加载数据
        cd_data = cddr.loadDatafromSql()
        kt_data = ktdr.loadDatafromSql()
        
        print("5.1 训练数据集测试")
        cd_train_dataset = CDDataset(cd_data, lrn_emb, qusunt_emb, cpt_emb, 'train')
        kt_train_dataset = KTDataSet(kt_data, lrn_emb, qusunt_emb, cpt_emb, 'train')
        
        print("✓ 训练数据集创建完成")
        cd_train_stats = cd_train_dataset.get_data_statistics()
        kt_train_stats = kt_train_dataset.get_data_statistics()
        
        print(f"  CD训练数据集统计:")
        for key, value in cd_train_stats.items():
            print(f"    {key}: {value}")
        
        print(f"  KT训练数据集统计 (新数据结构):")
        for key, value in kt_train_stats.items():
            print(f"    {key}: {value}")
        
        print("\n5.2 测试数据集测试")
        cd_test_dataset = CDDataset(cd_data, lrn_emb, qusunt_emb, cpt_emb, 'test')
        kt_test_dataset = KTDataSet(kt_data, lrn_emb, qusunt_emb, cpt_emb, 'test')
        
        print("✓ 测试数据集创建完成")
        cd_test_stats = cd_test_dataset.get_data_statistics()
        kt_test_stats = kt_test_dataset.get_data_statistics()
        
        print(f"  CD测试数据集统计:")
        for key, value in cd_test_stats.items():
            print(f"    {key}: {value}")
        
        print(f"  KT测试数据集统计 (新数据结构):")
        for key, value in kt_test_stats.items():
            print(f"    {key}: {value}")
        
        print("\n5.3 完整数据集测试")
        cd_all_dataset = CDDataset(cd_data, lrn_emb, qusunt_emb, cpt_emb, 'all')
        kt_all_dataset = KTDataSet(kt_data, lrn_emb, qusunt_emb, cpt_emb, 'all')
        
        print("✓ 完整数据集创建完成")
        cd_all_stats = cd_all_dataset.get_data_statistics()
        kt_all_stats = kt_all_dataset.get_data_statistics()
        
        print(f"  CD完整数据集统计:")
        for key, value in cd_all_stats.items():
            print(f"    {key}: {value}")
        
        print(f"  KT完整数据集统计 (新数据结构):")
        for key, value in kt_all_stats.items():
            print(f"    {key}: {value}")
        
        return {
            'cd_train': cd_train_dataset,
            'cd_test': cd_test_dataset,
            'cd_all': cd_all_dataset,
            'kt_train': kt_train_dataset,
            'kt_test': kt_test_dataset,
            'kt_all': kt_all_dataset
        }
    
    def test_data_loader(self, datasets):
        """测试数据加载器 - 适配新数据结构"""
        print("\n" + "="*60)
        print("6. 测试数据加载器 (新数据结构)")
        print("="*60)
        
        test_results = {}
        
        for data_type, dataset in datasets.items():
            print(f"\n6.1 测试 {data_type} 数据加载器")
            
            if len(dataset) == 0:
                print(f"  ⚠ {data_type} 数据集为空，跳过测试")
                continue
            
            # 创建数据加载器
            batch_size = min(4, len(dataset))
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=dataset.collate_fn
            )
            
            # 测试批次加载
            batch_count = 0
            for i, batch in enumerate(data_loader):
                if i >= 2:  # 只测试前2个批次
                    break
                
                print(f"    批次 {i+1}:")
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        print(f"      {key}: {value.shape} (dtype: {value.dtype})")
                
                # 验证关键数据 - 适配新数据结构
                if 'lrn_indices' in batch and 'h_lrn_batch' in batch:
                    lrn_indices = batch['lrn_indices']
                    h_lrn_batch = batch['h_lrn_batch']
                    assert lrn_indices.shape[0] == h_lrn_batch.shape[0], "学习者索引和嵌入批次大小不一致"
                    print(f"      ✓ 学习者嵌入批次验证通过")
                
                # 检查KT特定字段
                if data_type.startswith('kt'):
                    if 'prediction_masks' in batch and 'next_results' in batch:
                        prediction_masks = batch['prediction_masks']
                        next_results = batch['next_results']
                        print(f"      ✓ KT新数据结构验证通过")
                        print(f"        预测掩码形状: {prediction_masks.shape}")
                        print(f"        下一个结果形状: {next_results.shape}")
                        print(f"        有效预测位置: {prediction_masks.sum().item()}/{prediction_masks.numel()}")
                
                batch_count += 1
            
            test_results[data_type] = {
                'batch_count': batch_count,
                'dataset_size': len(dataset),
                'success': batch_count > 0
            }
            
            print(f"  ✓ {data_type} 数据加载器测试完成: {batch_count} 个批次")
        
        return test_results
    
    def test_data_integrity(self, datasets):
        """测试数据完整性 - 适配新数据结构"""
        print("\n" + "="*60)
        print("7. 测试数据完整性 (新数据结构)")
        print("="*60)
        
        integrity_results = {}
        
        for data_type, dataset in datasets.items():
            print(f"\n7.1 测试 {data_type} 数据完整性")
            
            if len(dataset) == 0:
                print(f"  ⚠ {data_type} 数据集为空，跳过完整性检查")
                integrity_results[data_type] = {'has_nan': False, 'has_inf': False, 'empty': True}
                continue
            
            # 检查NaN和Inf值
            has_nan = False
            has_inf = False
            
            # 检查第一个样本
            sample = dataset[0]
            for key, value in sample.items():
                if torch.is_tensor(value):
                    if torch.isnan(value).any():
                        print(f"    ✗ {key} 包含NaN值")
                        has_nan = True
                    if torch.isinf(value).any():
                        print(f"    ✗ {key} 包含Inf值")
                        has_inf = True
            
            # 检查序列掩码有效性 - 适配新数据结构
            if 'seq_masks' in sample:
                mask = sample['seq_masks']
                valid_positions = mask.sum().item()
                total_positions = mask.numel()
                print(f"    ✓ 序列掩码有效性: {valid_positions}/{total_positions} ({valid_positions/total_positions:.1%})")
            
            # 检查预测掩码有效性 (KT特有)
            if data_type.startswith('kt') and 'prediction_masks' in sample:
                prediction_masks = sample['prediction_masks']
                valid_predictions = prediction_masks.sum().item()
                total_predictions = prediction_masks.numel()
                print(f"    ✓ 预测掩码有效性: {valid_predictions}/{total_predictions} ({valid_predictions/total_predictions:.1%})")
                
                # 检查下一个结果的范围
                if 'next_results' in sample:
                    next_results = sample['next_results']
                    valid_next_results = next_results[prediction_masks.bool()]
                    if valid_next_results.numel() > 0:
                        print(f"    ✓ 下一个结果范围: [{valid_next_results.min().item():.3f}, {valid_next_results.max().item():.3f}]")
            
            integrity_results[data_type] = {
                'has_nan': has_nan,
                'has_inf': has_inf,
                'empty': False
            }
            
            if not has_nan and not has_inf:
                print(f"  ✓ {data_type} 数据完整性检查通过")
            else:
                print(f"  ✗ {data_type} 数据完整性检查失败")
        
        return integrity_results
    
    def test_complete_data_consistency(self):
        """测试完整数据一致性 - 适配新数据结构"""
        print("\n" + "="*60)
        print("8. 测试完整数据一致性 (新数据结构)")
        print("="*60)
        
        # 检查CD和KT完整数据的学习者一致性
        cd_complete_data = getattr(cddr, 'complete_data', {})
        kt_complete_data = getattr(ktdr, 'complete_data', {})
        
        cd_learners = set(cd_complete_data.keys())
        kt_learners = set(kt_complete_data.keys())
        
        print(f"CD完整数据学习者数量: {len(cd_learners)}")
        print(f"KT完整数据学习者数量: {len(kt_learners)}")
        print(f"共同学习者数量: {len(cd_learners.intersection(kt_learners))}")
        
        # 检查索引顺序一致性
        consistent = True
        for lrn_uid in cd_learners.intersection(kt_learners):
            if cddr.lrn_uid[lrn_uid] != ktdr.lrn_uid[lrn_uid]:
                print(f"✗ 学习者 {lrn_uid} 索引不一致: CD={cddr.lrn_uid[lrn_uid]}, KT={ktdr.lrn_uid[lrn_uid]}")
                consistent = False
        
        if consistent:
            print("✓ CD和KT完整数据学习者索引顺序一致")
        else:
            print("✗ CD和KT完整数据学习者索引顺序不一致")
        
        # 检查KT数据结构完整性
        if kt_complete_data:
            sample_lrn = list(kt_complete_data.keys())[0]
            sample_data = kt_complete_data[sample_lrn]
            print(f"KT数据结构验证: {len(sample_data)}个数据列表")
            for i, data_list in enumerate(sample_data):
                print(f"  列表{i}: {len(data_list)}个元素")
        
        return consistent
    
    def run_complete_test(self):
        """运行完整测试"""
        print("="*80)
        print("CD和KT数据管道完整测试 (适配新数据结构)")
        print("="*80)
        
        test_results = {}
        
        try:
            # 1. 测试DataReader初始化
            test_results['data_reader'] = self.test_data_reader_initialization()
            
            # 2. 测试完整数据功能
            test_results['complete_data'] = self.test_complete_data_function()
            
            # 3. 测试数据合并器
            test_results['merger'] = self.test_data_merger()
            
            # 4. 测试数据集创建
            datasets = self.test_dataset_creation()
            test_results['datasets'] = datasets is not None
            
            # 5. 测试数据加载器
            test_results['data_loader'] = self.test_data_loader(datasets)
            
            # 6. 测试数据完整性
            test_results['data_integrity'] = self.test_data_integrity(datasets)
            
            # 7. 测试完整数据一致性
            test_results['complete_consistency'] = self.test_complete_data_consistency()
            
            # 生成测试报告
            self.generate_test_report(test_results)
            
            return True
            
        except Exception as e:
            print(f"\n💥 测试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_test_report(self, test_results):
        """生成测试报告"""
        print("\n" + "="*80)
        print("测试报告总结 (适配新数据结构)")
        print("="*80)
        
        success_count = 0
        total_count = 0
        
        for test_name, result in test_results.items():
            if isinstance(result, bool):
                status = "通过" if result else "失败"
                success_count += 1 if result else 0
                total_count += 1
                print(f"{test_name:.<30} {status}")
            elif isinstance(result, dict):
                print(f"{test_name}:")
                for sub_test, sub_result in result.items():
                    if isinstance(sub_result, bool):
                        status = "通过" if sub_result else "失败"
                        success_count += 1 if sub_result else 0
                        total_count += 1
                        print(f"  {sub_test:.<28} {status}")
                    elif isinstance(sub_result, dict):
                        # 处理嵌套字典
                        pass
        
        success_rate = success_count / total_count if total_count > 0 else 0
        print(f"\n总体成功率: {success_count}/{total_count} ({success_rate:.1%})")
        
        if success_rate == 1.0:
            print("🎉 所有测试通过！数据管道工作正常。")
        elif success_rate >= 0.8:
            print("✅ 大部分测试通过，数据管道基本正常。")
        else:
            print("⚠ 部分测试失败，请检查相关组件。")

def main():
    """主函数"""
    print("CD和KT数据管道完整测试脚本 (适配新数据结构)")
    print("测试流程: DataReader → Merger → DataSet → DataLoader")
    print("KT数据结构: [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]")
    
    try:
        tester = CD_KT_DataPipelineTest()
        success = tester.run_complete_test()
        
        if success:
            print("\n✅ 测试完成！")
            return True
        else:
            print("\n❌ 测试失败！")
            return False
            
    except Exception as e:
        print(f"\n💥 测试执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)