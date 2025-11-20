# Debug_KT_Data_Pipeline.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from collections import defaultdict

# 导入数据读取器和合并器
from DataReader.KTDataReader import ktdr
from DataService.CD_KTDataMerger import cd_kt_merger

class KTDataPipelineDebugger:
    """KT数据管道调试器 - 适配新数据结构"""
    
    def __init__(self):
        self.verbose = True
    
    def debug_raw_data_from_repository(self):
        """调试从Repository获取的原始数据"""
        print("\n" + "="*80)
        print("1. 调试Repository原始数据")
        print("="*80)
        
        from Data.KTRepository import ktrepo
        
        # 获取原始交互数据
        raw_interactions = ktrepo.getLrnInteractions()
        print(f"原始交互数据数量: {len(raw_interactions)}")
        
        # 采样显示一些原始数据
        print("\n原始数据采样 (前5条):")
        for i in range(min(5, len(raw_interactions))):
            lrn_uid, unt_uid, add1, add2, create_time = raw_interactions[i]
            print(f"  {i+1}: lrn={lrn_uid}, unt={unt_uid}, add1={add1}, add2={add2}")
        
        # 获取学习单元类型
        unit_types = ktrepo.getUntTypes()
        print(f"\n学习单元类型数量: {len(unit_types)}")
        
        # 获取知识点映射
        qus_cpts = ktrepo.getQusCpts()
        unt_cpts = ktrepo.getUntCpts()
        print(f"题目-知识点映射数量: {len(qus_cpts)}")
        print(f"学习单元-知识点映射数量: {len(unt_cpts)}")
        
        return raw_interactions
    
    def debug_kt_data_reader(self):
        """调试KTDataReader处理后的数据 - 适配新数据结构"""
        print("\n" + "="*80)
        print("2. 调试KTDataReader处理后的数据 (新数据结构)")
        print("="*80)
        
        # 初始化KTDataReader
        ktdr_instance = ktdr
        
        print("KTDataReader初始化完成")
        print(f"学习者数量: {len(ktdr_instance.lrn_uid)}")
        print(f"学习单元+题目数量: {len(ktdr_instance.qusunt_uid)}")
        
        # 检查交互数据
        print(f"\n交互数据数量: {len(ktdr_instance.interactions)}")
        
        # 分析交互数据类型分布
        interaction_types = defaultdict(int)
        question_results = []
        
        for interaction in ktdr_instance.interactions[:100]:  # 采样100条
            lrn_uid, unt_uid, add1, add2, create_time, is_question, result = interaction
            interaction_types[is_question] += 1
            if is_question == 1:
                question_results.append(result)
        
        print(f"交互类型分布: 题目={interaction_types[1]}, 非题目={interaction_types[0]}")
        if question_results:
            print(f"题目结果范围: min={min(question_results)}, max={max(question_results)}, 均值={np.mean(question_results):.3f}")
        
        # 检查训练测试数据
        kt_data = ktdr_instance.loadDatafromSql()
        
        train_data = kt_data['train_data']
        test_data = kt_data['test_data']
        
        print(f"\n训练数据学习者数量: {len(train_data)}")
        print(f"测试数据学习者数量: {len(test_data)}")
        
        # 详细检查一个学习者的数据 - 适配新数据结构
        if train_data:
            sample_lrn = list(train_data.keys())[0]
            sample_data = train_data[sample_lrn]
            print(f"\n样例学习者 {sample_lrn} 的数据结构 (新7元素格式):")
            print(f"  学习单元序列长度: {len(sample_data[0])}")
            print(f"  add1序列长度: {len(sample_data[1])}")
            print(f"  add2序列长度: {len(sample_data[2])}")
            print(f"  题目标记序列长度: {len(sample_data[3])}")
            print(f"  结果序列长度: {len(sample_data[4])}")
            print(f"  预测掩码序列长度: {len(sample_data[5])}")  # 新的预测掩码
            print(f"  下一个结果序列长度: {len(sample_data[6])}")  # 新的下一个结果
            
            # 检查数据内容
            print(f"\n样例数据内容 (前10个位置):")
            for i in range(min(10, len(sample_data[0]))):
                print(f"  位置{i}: unt={sample_data[0][i]}, add1={sample_data[1][i]:.2f}, add2={sample_data[2][i]:.2f}, "
                      f"is_q={sample_data[3][i]}, result={sample_data[4][i]}, "
                      f"pred_mask={sample_data[5][i]}, next_result={sample_data[6][i]}")
        
        return kt_data
    
    def debug_complete_data(self):
        """调试完整数据 - 适配新数据结构"""
        print("\n" + "="*80)
        print("3. 调试完整数据 (新数据结构)")
        print("="*80)
        
        complete_data = ktdr.getCompleteData()
        print(f"完整数据学习者数量: {len(complete_data)}")
        
        if complete_data:
            sample_lrn = list(complete_data.keys())[0]
            sample_data = complete_data[sample_lrn]
            
            print(f"\n完整数据样例 (学习者 {sample_lrn}):")
            print(f"  总序列长度: {len(sample_data[0])}")
            
            # 统计掩码信息 - 适配新数据结构
            total_positions = len(sample_data[5])  # prediction_masks
            valid_prediction_positions = sum(sample_data[5])  # 有效预测位置
            question_positions = sum(sample_data[3])  # is_questions
            
            print(f"  题目位置数量: {question_positions}")
            print(f"  有效预测位置数量: {valid_prediction_positions}")
            print(f"  掩码覆盖率: {valid_prediction_positions}/{total_positions} ({valid_prediction_positions/total_positions:.1%})")
            
            # 检查结果数据的范围
            results = [r for r in sample_data[4] if r != -1]  # 过滤掉非题目的-1
            if results:
                print(f"  结果数据范围: [{min(results)}, {max(results)}]")
                print(f"  结果数据类型: {type(results[0])}")
            
            # 检查下一个结果的范围
            next_results = [r for r in sample_data[6] if r != 0]  # 过滤掉无效的0
            if next_results:
                print(f"  下一个结果范围: [{min(next_results)}, {max(next_results)}]")
            
            # 检查add1/add2的范围
            add1_values = sample_data[1]
            add2_values = sample_data[2]
            print(f"  add1范围: [{min(add1_values):.2f}, {max(add1_values):.2f}]")
            print(f"  add2范围: [{min(add2_values):.2f}, {max(add2_values):.2f}]")
        
        return complete_data
    
    def debug_merger_processing(self):
        """调试Merger处理后的数据 - 适配新数据结构"""
        print("\n" + "="*80)
        print("4. 调试Merger处理 (新数据结构)")
        print("="*80)
        
        # 执行合并
        cd_kt_merger.merge_and_filter_train_test()
        
        # 获取合并统计
        stats = cd_kt_merger.get_merged_statistics()
        if stats and 'train_test' in stats:
            print("合并后统计:")
            for key, value in stats['train_test'].items():
                print(f"  {key}: {value}")
        
        # 检查合并后的数据
        kt_train_data = ktdr.train_data
        kt_test_data = ktdr.test_data
        
        print(f"\nMerger处理后:")
        print(f"  KT训练数据学习者: {len(kt_train_data)}")
        print(f"  KT测试数据学习者: {len(kt_test_data)}")
        
        # 检查一个样例学习者的掩码数据 - 适配新数据结构
        if kt_train_data:
            sample_lrn = list(kt_train_data.keys())[0]
            sample_data = kt_train_data[sample_lrn]
            
            prediction_masks = sample_data[5]  # 预测掩码
            next_results = sample_data[6]  # 下一个结果
            
            print(f"\n样例学习者 {sample_lrn} 的掩码分析 (新逻辑):")
            print(f"  总序列长度: {len(prediction_masks)}")
            print(f"  有效预测位置: {sum(prediction_masks)}")
            
            # 检查掩码与结果的对应关系
            valid_indices = [i for i, mask in enumerate(prediction_masks) if mask == 1]
            if valid_indices:
                print(f"  前5个有效掩码位置的结果:")
                for i in valid_indices[:5]:
                    if i < len(next_results):
                        print(f"    位置{i}: 下一个结果={next_results[i]}")
        
        return kt_train_data, kt_test_data
    
    def debug_kt_dataset(self, kt_data):
        """调试KTDataSet处理后的数据 - 适配新数据结构"""
        print("\n" + "="*80)
        print("5. 调试KTDataSet (新数据结构)")
        print("="*80)
        
        # 创建模拟的HGC嵌入（用于测试）
        lrn_num = len(kt_data['lrn_uid'])
        qusunt_num = len(kt_data['qusunt_uid'])
        cpt_num = len(kt_data['cpt_uid'])
        
        # 创建随机嵌入（仅用于测试）
        lrn_emb = torch.randn(lrn_num, 64)
        qusunt_emb = torch.randn(qusunt_num, 64)
        cpt_emb = torch.randn(cpt_num, 64)
        
        # 创建数据集
        from DataSet.KTDataSet import KTDataSet
        
        train_dataset = KTDataSet(kt_data, lrn_emb, qusunt_emb, cpt_emb, 'train')
        
        print("KTDataSet创建完成")
        stats = train_dataset.get_data_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 检查单个样本的数据 - 适配新数据结构
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\n单个样本数据结构 (新字段):")
            for key, value in sample.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape} {value.dtype}")
                    
                    # 检查数据范围
                    if value.numel() > 0:
                        valid_values = value[train_dataset.seq_masks[0].bool()] if 'seq_masks' in sample else value
                        if valid_values.numel() > 0:
                            print(f"    范围: [{valid_values.min().item():.3f}, {valid_values.max().item():.3f}]")
                            
                            # 特别检查关键数据
                            if key in ['next_results', 'results']:
                                print(f"    值示例: {valid_values[:5].tolist()}")
                            elif key == 'prediction_masks':  # 新的预测掩码
                                print(f"    掩码有效性: {valid_values.sum().item()}/{valid_values.numel()}")
        
        return train_dataset
    
    def debug_training_inputs(self, dataset):
        """调试训练输入数据 - 适配新数据结构"""
        print("\n" + "="*80)
        print("6. 调试训练输入数据 (新数据结构)")
        print("="*80)
        
        from torch.utils.data import DataLoader
        
        if len(dataset) == 0:
            print("数据集为空，跳过训练输入调试")
            return
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        
        # 检查第一个批次
        for i, batch in enumerate(data_loader):
            if i >= 1:  # 只检查第一个批次
                break
                
            print(f"批次 {i+1} 数据 (新数据结构):")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape} {value.dtype}")
                    
                    # 特别检查可能出问题的数据
                    if key in ['next_results', 'results']:
                        # 检查值范围
                        print(f"    值范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
                        
                        # 检查是否有NaN或Inf
                        if torch.isnan(value).any():
                            print(f"    ⚠ 包含NaN值!")
                        if torch.isinf(value).any():
                            print(f"    ⚠ 包含Inf值!")
                    
                    elif key == 'prediction_masks':  # 新的预测掩码
                        # 检查掩码
                        mask_sum = value.sum().item()
                        mask_total = value.numel()
                        print(f"    掩码覆盖率: {mask_sum}/{mask_total} ({mask_sum/mask_total:.1%})")
        
        return True
    
    def check_bceloss_inputs(self, dataset):
        """专门检查BCELoss输入数据 - 适配新数据结构"""
        print("\n" + "="*80)
        print("7. 专门检查BCELoss输入数据 (新数据结构)")
        print("="*80)
        
        from torch.utils.data import DataLoader
        
        if len(dataset) == 0:
            print("数据集为空，跳过BCELoss检查")
            return
        
        data_loader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        
        criterion = torch.nn.BCELoss()
        
        for i, batch in enumerate(data_loader):
            if i >= 1:  # 只检查第一个批次
                break
                
            print(f"批次 {i+1} BCELoss输入检查 (新数据结构):")
            
            # 使用新的数据结构
            prediction_masks = batch['prediction_masks']  # 新的预测掩码
            next_results = batch['next_results']  # 新的下一个结果
            
            print(f"  输入张量形状:")
            print(f"    prediction_masks: {prediction_masks.shape}")
            print(f"    next_results: {next_results.shape}")
            
            # 创建模拟预测（范围在0-1之间）
            # 根据next_results的形状创建匹配的模拟预测
            if next_results.dim() == 1:
                # next_results: [batch_size]
                simulated_predictions = torch.rand_like(next_results) * 0.5 + 0.25
            elif next_results.dim() == 2:
                # next_results: [batch_size, seq_len]
                simulated_predictions = torch.rand_like(next_results) * 0.5 + 0.25
            else:
                print(f"  不支持的next_results维度: {next_results.dim()}")
                continue
            
            print(f"  模拟预测范围: [{simulated_predictions.min().item():.3f}, {simulated_predictions.max().item():.3f}]")
            print(f"  真实结果范围: [{next_results.min().item():.3f}, {next_results.max().item():.3f}]")
            print(f"  掩码覆盖率: {prediction_masks.sum().item()}/{prediction_masks.numel()}")
            
            # 应用掩码 - 简化处理，只检查有效位置
            valid_mask = prediction_masks.bool()
            
            if valid_mask.any():
                # 获取有效位置的预测和标签
                if next_results.dim() == 1:
                    # 对于[batch_size]的情况，直接使用掩码
                    valid_predictions = simulated_predictions[valid_mask.any(dim=1)]
                    valid_targets = next_results[valid_mask.any(dim=1)]
                elif next_results.dim() == 2:
                    # 对于[batch_size, seq_len]的情况，展平处理
                    valid_predictions = simulated_predictions[valid_mask]
                    valid_targets = next_results[valid_mask]
                
                print(f"  有效预测数量: {valid_predictions.numel()}")
                print(f"  有效目标数量: {valid_targets.numel()}")
                
                if valid_predictions.numel() > 0 and valid_targets.numel() > 0:
                    try:
                        # 确保形状匹配
                        if valid_predictions.shape != valid_targets.shape:
                            print(f"  形状不匹配: predictions {valid_predictions.shape}, targets {valid_targets.shape}")
                            # 尝试调整形状
                            min_len = min(valid_predictions.numel(), valid_targets.numel())
                            if min_len > 0:
                                valid_predictions = valid_predictions[:min_len]
                                valid_targets = valid_targets[:min_len]
                                print(f"  调整后形状: predictions {valid_predictions.shape}, targets {valid_targets.shape}")
                        
                        loss = criterion(valid_predictions, valid_targets)
                        print(f"  ✓ BCELoss计算成功: {loss.item():.4f}")
                        
                        # 计算准确率
                        pred_binary = (valid_predictions > 0.5).float()
                        accuracy = (pred_binary == valid_targets).float().mean()
                        print(f"  ✓ 模拟准确率: {accuracy.item():.4f}")
                        
                    except Exception as e:
                        print(f"  ✗ BCELoss计算失败: {e}")
                        
                        # 详细调试
                        print(f"    预测数据范围: [{valid_predictions.min().item():.3f}, {valid_predictions.max().item():.3f}]")
                        print(f"    目标数据范围: [{valid_targets.min().item():.3f}, {valid_targets.max().item():.3f}]")
                        print(f"    预测数据类型: {valid_predictions.dtype}")
                        print(f"    目标数据类型: {valid_targets.dtype}")
                else:
                    print("  ⚠ 没有有效的预测或目标数据")
            else:
                print("  ⚠ 没有有效掩码位置")
        
        return True
    
    def debug_prediction_mask_logic(self):
        """专门调试预测掩码逻辑"""
        print("\n" + "="*80)
        print("8. 专门调试预测掩码逻辑")
        print("="*80)
        
        # 获取完整数据
        complete_data = ktdr.getCompleteData()
        
        if not complete_data:
            print("没有完整数据可用")
            return
        
        sample_lrn = list(complete_data.keys())[0]
        sample_data = complete_data[sample_lrn]
        
        print(f"样例学习者 {sample_lrn} 的预测掩码逻辑分析:")
        
        unt_uids = sample_data[0]
        is_questions = sample_data[3]
        prediction_masks = sample_data[5]
        next_results = sample_data[6]
        
        # 分析掩码逻辑
        total_positions = len(prediction_masks)
        valid_positions = sum(prediction_masks)
        
        print(f"  总位置数: {total_positions}")
        print(f"  有效预测位置: {valid_positions}")
        print(f"  掩码覆盖率: {valid_positions/total_positions:.1%}")
        
        # 检查掩码与题目类型的对应关系
        print(f"\n  掩码与题目类型对应关系:")
        for i in range(min(10, len(prediction_masks))):
            current_is_question = is_questions[i]
            current_pred_mask = prediction_masks[i]
            current_next_result = next_results[i]
            
            print(f"    位置{i}: 当前是题目={current_is_question}, 预测掩码={current_pred_mask}, 下一个结果={current_next_result}")
        
        # 统计掩码逻辑
        mask_with_question = 0
        mask_without_question = 0
        
        for i in range(len(prediction_masks)):
            if prediction_masks[i] == 1:
                if i < len(is_questions) - 1:  # 不是最后一个位置
                    next_is_question = is_questions[i + 1] if i + 1 < len(is_questions) else 0
                    if next_is_question == 1:
                        mask_with_question += 1
                    else:
                        mask_without_question += 1
        
        print(f"\n  掩码逻辑统计:")
        print(f"    掩码且下一步是题目: {mask_with_question}")
        print(f"    掩码但下一步不是题目: {mask_without_question}")
        if mask_with_question + mask_without_question > 0:
            print(f"    掩码逻辑正确率: {mask_with_question/(mask_with_question + mask_without_question):.1%}")
        else:
            print("    没有有效掩码位置")
    
    def run_complete_debug(self):
        """运行完整调试"""
        print("KT数据管道完整调试 (适配新数据结构)")
        print("="*80)
        
        try:
            # 1. 调试原始数据
            self.debug_raw_data_from_repository()
            
            # 2. 调试KTDataReader
            kt_data = self.debug_kt_data_reader()
            
            # 3. 调试完整数据
            self.debug_complete_data()
            
            # 4. 调试Merger处理
            self.debug_merger_processing()
            
            # 5. 调试KTDataSet
            dataset = self.debug_kt_dataset(kt_data)
            
            # 6. 调试训练输入
            success_6 = self.debug_training_inputs(dataset)
            if not success_6:
                print("⚠ 训练输入调试失败")
            
            # 7. 专门检查BCELoss输入
            success_7 = self.check_bceloss_inputs(dataset)
            if not success_7:
                print("⚠ BCELoss输入检查失败")
            
            # 8. 专门调试预测掩码逻辑
            self.debug_prediction_mask_logic()
            
            print("\n" + "="*80)
            print("调试完成!")
            print("="*80)
            
        except Exception as e:
            print(f"\n💥 调试过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    debugger = KTDataPipelineDebugger()
    debugger.run_complete_debug()

if __name__ == '__main__':
    main()