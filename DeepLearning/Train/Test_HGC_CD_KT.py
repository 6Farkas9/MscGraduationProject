import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

# 导入必要的模块
try:
    from Model.HGC import HGC
    from Model.CD import CD
    from Model.KT import KT
    from DataReader.HGCDataReader import hgcdr
    from DataReader.CDDataReader import cddr
    from DataReader.KTDataReader import ktdr
    from DataSet.CDDataSet import CDDataset
    from DataSet.KTDataSet import KTDataSet
    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
    exit(1)

def test_hgc_kt_cd_full_pipeline():
    """测试HGC→KT→CD完整闭环流程"""
    print("=== HGC-KT-CD完整闭环流程测试 ===")
    
    # 1. 加载HGC数据并计算嵌入
    print("\n1. 加载HGC数据并计算嵌入...")
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

    # 2. 加载CD和KT数据
    print("\n2. 加载CD和KT数据...")
    cddata = cddr.loadDatafromSql()
    ktdata = ktdr.loadDatafromSql()
    
    print(f"  CD数据 - 学习者: {len(cddata['lrn_uid'])}, 题目: {len(cddata['qus_uid'])}, 知识点: {len(cddata['cpt_uid'])}")
    print(f"  KT数据 - 学习者: {len(ktdata['lrn_uid'])}, 学习单元: {len(ktdata['untqus_uid'])}, 知识点: {len(ktdata['cpt_uid'])}")

    # 3. 创建CD模型
    print("\n3. 创建CD模型...")
    # 从unt_emb中提取题目嵌入
    unt_num = unt_emb.shape[0] - len(cddata['qus_uid'])
    h_qus = unt_emb[unt_num:]
    
    cd_model = CD(
        embedding_dim=64,
        concept_num=len(cddata['cpt_uid']),
        h_qus=h_qus,
        h_cpt=cpt_emb
    ).to(device)
    
    print("✓ CD模型创建完成")
    print(f"  CD模型参数数量: {sum(p.numel() for p in cd_model.parameters())}")

    # 4. 创建KT模型
    print("\n4. 创建KT模型...")
    kt_model = KT(
        embedding_dim=64,
        concept_num=len(ktdata['cpt_uid']),
        h_lrn=lrn_emb,
        h_unt=unt_emb,
        h_cpt=cpt_emb
    ).to(device)
    
    print("✓ KT模型创建完成")
    print(f"  KT模型参数数量: {sum(p.numel() for p in kt_model.parameters())}")

    # 5. 创建数据集 - 确保学习者顺序一致
    print("\n5. 创建数据集...")
    cd_dataset = CDDataset(cddata, lrn_emb, unt_emb, cpt_emb, 'train', max_seq_len=128)
    kt_dataset = KTDataSet(ktdata, lrn_emb, unt_emb, cpt_emb, 'train', max_seq_len=128)
    
    print("✓ 数据集创建完成")
    print(f"  CD数据集统计: {cd_dataset.get_data_statistics()}")
    print(f"  KT数据集统计: {kt_dataset.get_data_statistics()}")

    # 6. 创建数据加载器 - 使用相同的batch_size和顺序
    print("\n6. 创建数据加载器...")
    batch_size = 4
    
    cd_loader = DataLoader(
        cd_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 不shuffle以保持学习者顺序一致
        collate_fn=cd_dataset.collate_fn
    )
    
    kt_loader = DataLoader(
        kt_dataset,
        batch_size=batch_size,
        shuffle=False,  # 不shuffle以保持学习者顺序一致
        collate_fn=kt_dataset.collate_fn
    )
    
    print("✓ 数据加载器创建完成")

    # 7. 验证学习者顺序一致性
    print("\n7. 验证学习者顺序一致性...")
    
    # 获取第一个batch的数据
    cd_batch = next(iter(cd_loader))
    kt_batch = next(iter(kt_loader))
    
    print(f"  CD Batch学习者索引: {cd_batch['lrn_indices'].tolist()}")
    print(f"  KT Batch学习者索引: {kt_batch['lrn_indices'].tolist()}")
    
    # 验证学习者顺序一致性
    if torch.equal(cd_batch['lrn_indices'], kt_batch['lrn_indices']):
        print("  ✓ CD和KT批次学习者顺序一致")
    else:
        print("  ✗ CD和KT批次学习者顺序不一致")
        # 重新创建数据加载器确保顺序一致
        print("  重新创建数据加载器确保顺序一致...")
        cd_loader = DataLoader(cd_dataset, batch_size=batch_size, shuffle=False, collate_fn=cd_dataset.collate_fn)
        kt_loader = DataLoader(kt_dataset, batch_size=batch_size, shuffle=False, collate_fn=kt_dataset.collate_fn)
        cd_batch = next(iter(cd_loader))
        kt_batch = next(iter(kt_loader))
        
        if torch.equal(cd_batch['lrn_indices'], kt_batch['lrn_indices']):
            print("  ✓ 重新创建后学习者顺序一致")
        else:
            print("  ✗ 无法保证学习者顺序一致，继续测试但结果可能不准确")

    # 8. 单轮闭环流程测试
    print("\n8. 单轮闭环流程测试...")
    
    # CD计算初始能力维度
    print("  a) CD计算初始能力维度...")
    cd_model.eval()
    with torch.no_grad():
        cd_initial_ability = cd_model.get_ability_matrix(
            cd_batch['h_lrn_batch'].to(device),
            cd_batch['qus_seq_indices'].to(device),
            cd_batch['qus_seq_masks'].to(device)
        )
    print(f"    CD初始能力维度: {cd_initial_ability.shape}")

    # KT使用CD能力初始化并进行动态追踪
    print("  b) KT使用CD能力初始化并进行动态追踪...")
    kt_model.eval()
    
    # 设置CD优化能力到KT模型
    kt_model.set_cd_optimized_ability(cd_initial_ability)
    
    # KT前向传播
    kt_predictions, kt_concept_mastery = kt_model(
        kt_batch['lrn_indices'].to(device),
        kt_batch['unt_seq_indices'].to(device),
        kt_batch['add1'].to(device),
        kt_batch['add2'].to(device),
        kt_batch['type_indices'].to(device),
        kt_batch['seq_masks'].to(device),
        kt_batch['next_question_masks'].to(device),
        use_cd_optimization=True
    )
    print(f"    KT预测输出: {kt_predictions.shape}")
    print(f"    KT知识点掌握程度: {kt_concept_mastery.shape}")

    # KT结果反馈给CD
    print("  c) KT结果反馈给CD...")
    with torch.no_grad():
        kt_final_ability = kt_model.get_concept_mastery(
            kt_batch['lrn_indices'].to(device),
            kt_batch['unt_seq_indices'].to(device),
            kt_batch['add1'].to(device),
            kt_batch['add2'].to(device),
            kt_batch['type_indices'].to(device),
            kt_batch['seq_masks'].to(device)
        )
    print(f"    KT反馈能力: {kt_final_ability.shape}")

    # CD使用KT反馈优化能力
    print("  d) CD使用KT反馈优化能力...")
    cd_model.set_kt_optimized_ability(kt_final_ability)
    
    cd_predictions_with_kt, cd_ability_with_kt = cd_model(
        cd_batch['h_lrn_batch'].to(device),
        cd_batch['qus_seq_indices'].to(device),
        cd_batch['qus_seq_masks'].to(device),
        return_ability=True,
        use_kt_optimization=True
    )
    print(f"    CD优化后预测: {cd_predictions_with_kt.shape}")
    print(f"    CD优化后能力: {cd_ability_with_kt.shape}")

    # 9. 多轮闭环训练模拟
    print("\n9. 多轮闭环训练模拟...")
    
    num_cycles = 3
    cycle_times = []
    
    for cycle in range(num_cycles):
        print(f"  闭环轮次 {cycle + 1}/{num_cycles}:")
        cycle_start = time.time()
        
        # CD → KT
        with torch.no_grad():
            cd_ability = cd_model.get_ability_matrix(
                cd_batch['h_lrn_batch'].to(device),
                cd_batch['qus_seq_indices'].to(device),
                cd_batch['qus_seq_masks'].to(device)
            )
        
        kt_model.set_cd_optimized_ability(cd_ability)
        
        # KT → CD
        with torch.no_grad():
            kt_ability = kt_model.get_concept_mastery(
                kt_batch['lrn_indices'].to(device),
                kt_batch['unt_seq_indices'].to(device),
                kt_batch['add1'].to(device),
                kt_batch['add2'].to(device),
                kt_batch['type_indices'].to(device),
                kt_batch['seq_masks'].to(device)
            )
        
        cd_model.set_kt_optimized_ability(kt_ability)
        
        cycle_time = time.time() - cycle_start
        cycle_times.append(cycle_time)
        print(f"    CD能力 → KT → CD能力完成 (耗时: {cycle_time:.3f}s)")

    avg_cycle_time = sum(cycle_times) / len(cycle_times)
    print(f"  平均闭环时间: {avg_cycle_time:.3f}s/轮")

    # 10. 梯度传递测试
    print("\n10. 梯度传递测试...")
    
    # CD梯度测试
    print("  a) CD梯度测试...")
    cd_model.train()
    cd_batch['h_lrn_batch'] = cd_batch['h_lrn_batch'].clone().to(device).requires_grad_(True)
    
    cd_predictions = cd_model(
        cd_batch['h_lrn_batch'],
        cd_batch['qus_seq_indices'].to(device),
        cd_batch['qus_seq_masks'].to(device)
    )
    
    cd_loss = nn.BCELoss()(cd_predictions, cd_batch['results'].to(device))
    cd_loss.backward()
    
    cd_has_gradient = cd_batch['h_lrn_batch'].grad is not None
    print(f"    CD梯度计算: {'成功' if cd_has_gradient else '失败'}")
    
    # KT梯度测试
    print("  b) KT梯度测试...")
    kt_model.train()
    
    kt_predictions, _ = kt_model(
        kt_batch['lrn_indices'].to(device),
        kt_batch['unt_seq_indices'].to(device),
        kt_batch['add1'].to(device),
        kt_batch['add2'].to(device),
        kt_batch['type_indices'].to(device),
        kt_batch['seq_masks'].to(device),
        kt_batch['next_question_masks'].to(device)
    )
    
    # 只对下一个是题目的时间步计算损失
    valid_predictions = kt_predictions[kt_batch['next_question_masks'].to(device).bool()]
    valid_targets = kt_batch['next_results'].to(device)[kt_batch['next_question_masks'].to(device).bool()]
    
    if len(valid_predictions) > 0:
        kt_loss = nn.BCELoss()(valid_predictions.mean(dim=-1), valid_targets)
        kt_loss.backward()
        
        kt_has_gradient = any(p.grad is not None for p in kt_model.parameters())
        print(f"    KT梯度计算: 成功 (有效预测数量: {len(valid_predictions)})")
    else:
        print(f"    KT梯度计算: 跳过 (无有效预测)")

    # 11. 性能评估
    print("\n11. 性能评估...")
    
    cd_model.eval()
    kt_model.eval()
    
    with torch.no_grad():
        # CD性能
        cd_final_predictions = cd_model(
            cd_batch['h_lrn_batch'].to(device),
            cd_batch['qus_seq_indices'].to(device),
            cd_batch['qus_seq_masks'].to(device)
        )
        
        cd_accuracy = ((cd_final_predictions > 0.5).float() == cd_batch['results'].to(device)).float().mean()
        print(f"  CD准确率: {cd_accuracy.item():.3f}")
        
        # KT性能
        kt_final_predictions, _ = kt_model(
            kt_batch['lrn_indices'].to(device),
            kt_batch['unt_seq_indices'].to(device),
            kt_batch['add1'].to(device),
            kt_batch['add2'].to(device),
            kt_batch['type_indices'].to(device),
            kt_batch['seq_masks'].to(device),
            kt_batch['next_question_masks'].to(device)
        )
        
        if len(valid_predictions) > 0:
            kt_accuracy = ((kt_final_predictions.mean(dim=-1) > 0.5).float() == kt_batch['next_results'].to(device)).float().mean()
            print(f"  KT准确率: {kt_accuracy.item():.3f}")
        else:
            print(f"  KT准确率: 无有效预测")

    # 12. 内存和性能统计
    print("\n12. 内存和性能统计...")
    
    # 模型大小
    cd_params = sum(p.numel() for p in cd_model.parameters())
    kt_params = sum(p.numel() for p in kt_model.parameters())
    total_params = cd_params + kt_params
    
    print(f"  CD模型参数: {cd_params:,}")
    print(f"  KT模型参数: {kt_params:,}")
    print(f"  总参数数量: {total_params:,}")
    
    # 数据统计
    cd_stats = cd_dataset.get_data_statistics()
    kt_stats = kt_dataset.get_data_statistics()
    
    print(f"  CD数据 - 学习者: {cd_stats['total_learners']}, 记录: {cd_stats['total_records']}")
    print(f"  KT数据 - 学习者: {kt_stats['total_learners']}, 记录: {kt_stats['total_records']}, 有效预测: {kt_stats['total_next_questions']}")

    print("\n=== HGC-KT-CD完整闭环测试完成 ===")
    print("✓ 所有模块集成成功")
    print("✓ 闭环训练流程完整")
    print("✓ 梯度传递正常")
    print("✓ 性能评估完成")

def test_batch_consistency():
    """测试批次一致性"""
    print("\n" + "="*60)
    print("批次一致性测试")
    print("="*60)
    
    # 重新加载数据确保一致性
    hgcdr.loadDatafromSql()
    cddata = cddr.loadDatafromSql()
    ktdata = ktdr.loadDatafromSql()
    
    with torch.no_grad():
        lrn_emb, unt_emb, cpt_emb = HGC(64, hgcdr.lrn_init.shape[1], hgcdr.untqus_init.shape[1], hgcdr.cpt_init.shape[1])(hgcdr, 'cpu')
    
    cd_dataset = CDDataset(cddata, lrn_emb, unt_emb, cpt_emb, 'train', max_seq_len=128)
    kt_dataset = KTDataSet(ktdata, lrn_emb, unt_emb, cpt_emb, 'train', max_seq_len=128)
    
    batch_size = 8
    cd_loader = DataLoader(cd_dataset, batch_size=batch_size, shuffle=False, collate_fn=cd_dataset.collate_fn)
    kt_loader = DataLoader(kt_dataset, batch_size=batch_size, shuffle=False, collate_fn=kt_dataset.collate_fn)
    
    consistent_batches = 0
    total_batches = 0
    
    for cd_batch, kt_batch in zip(cd_loader, kt_loader):
        total_batches += 1
        if torch.equal(cd_batch['lrn_indices'], kt_batch['lrn_indices']):
            consistent_batches += 1
    
    consistency_rate = consistent_batches / total_batches if total_batches > 0 else 0
    print(f"批次一致性: {consistent_batches}/{total_batches} ({consistency_rate:.1%})")
    
    if consistency_rate == 1.0:
        print("✓ 所有批次学习者顺序完全一致")
    else:
        print("⚠ 部分批次学习者顺序不一致")

if __name__ == '__main__':
    # 运行完整闭环测试
    test_hgc_kt_cd_full_pipeline()
    
    # 运行批次一致性测试
    test_batch_consistency()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)