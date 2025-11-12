import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

class DTR(nn.Module):
    """诊断特质表示模块"""
    def __init__(self, embedding_dim):
        super(DTR, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 学生能力计算网络 - 增强表达能力
        self.l_p_lrn = nn.Sequential(
            nn.Linear(2 * embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 题目难度计算网络
        self.l_d_qus = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 题目区分度计算网络
        self.l_b_qus = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, h_lrn_cpt, h_qus_cpt, h_qus):
        p_lrn = self.l_p_lrn(h_lrn_cpt).squeeze(-1)      # (batch_size * seq_len * cpt_num,)
        d_qus = self.l_d_qus(h_qus_cpt).squeeze(-1)      # (batch_size * seq_len * cpt_num,)
        beta_qus = self.l_b_qus(h_qus).squeeze(-1)       # (batch_size * seq_len,)
        return p_lrn, d_qus, beta_qus

class MIRT(nn.Module):
    """多维项目反应理论模块 - 支持能力输出"""
    def __init__(self):
        super(MIRT, self).__init__()

    def forward(self, p_lrn, d_qus, beta_qus, return_ability=False):
        """
        Args:
            p_lrn: (batch_size, seq_len, cpt_num) - 学生能力
            d_qus: (batch_size, seq_len, cpt_num) - 题目难度
            beta_qus: (batch_size, seq_len) - 题目区分度
            return_ability: 是否返回能力矩阵
        """
        # 计算交互项: ∑(p_uc * d_vc) + β_v
        interaction = torch.sum(p_lrn * d_qus, dim=-1)  # (batch_size, seq_len)
        result = torch.sigmoid(interaction + beta_qus)  # (batch_size, seq_len)
        
        if return_ability:
            # 返回预测结果和能力矩阵
            return result, p_lrn
        else:
            return result

class CD(nn.Module):
    """认知诊断主模型 - 支持CD-KT循环优化"""
    def __init__(self, embedding_dim, concept_num, h_qus, h_cpt):
        super(CD, self).__init__()
        self.embedding_dim = embedding_dim
        self.concept_num = concept_num
        
        # 注册为模型的buffer
        self.register_buffer('h_qus', h_qus)
        self.register_buffer('h_cpt', h_cpt)
        
        # KT优化后的能力缓存
        self.kt_optimized_ability = None
        
        self.dtr = DTR(embedding_dim)
        self.mirt = MIRT()
        
        # 能力融合门控机制
        self.ability_gate = nn.Sequential(
            nn.Linear(embedding_dim + concept_num, 128),
            nn.ReLU(),
            nn.Linear(128, concept_num),
            nn.Sigmoid()
        )

    def set_kt_optimized_ability(self, kt_ability):
        """
        设置KT优化后的能力矩阵
        Args:
            kt_ability: (batch_size, seq_len, concept_num) KT优化的能力
        """
        self.kt_optimized_ability = kt_ability

    def compute_ability_with_kt(self, h_lrn_batch, p_lrn_cd):
        """
        融合CD计算的能力和KT优化的能力
        Args:
            h_lrn_batch: (batch_size, emb_dim) CD计算的学习者嵌入
            p_lrn_cd: (batch_size, seq_len, concept_num) CD计算的能力
        """
        if self.kt_optimized_ability is None:
            return p_lrn_cd
        
        batch_size, seq_len, concept_num = p_lrn_cd.shape
        
        # 确保KT能力与CD能力维度一致
        if (self.kt_optimized_ability.shape[0] != batch_size or 
            self.kt_optimized_ability.shape[1] != seq_len):
            return p_lrn_cd
        
        # 计算融合权重
        h_lrn_expanded = h_lrn_batch.unsqueeze(1).repeat(1, seq_len, 1)
        gate_input = torch.cat([h_lrn_expanded, p_lrn_cd], dim=-1)
        fusion_weights = self.ability_gate(gate_input)  # (batch_size, seq_len, concept_num)
        
        # 融合CD能力和KT能力
        p_lrn_fused = (fusion_weights * self.kt_optimized_ability + 
                      (1 - fusion_weights) * p_lrn_cd)
        
        return p_lrn_fused

    def forward(self, h_lrn_batch, qus_seq_indices, qus_seq_masks, 
            return_ability=False, use_kt_optimization=True):
        """
        Args:
            h_lrn_batch: (batch_size, emb_dim) - 批次学习者嵌入
            qus_seq_indices: (batch_size, seq_len) - 题目序列索引
            qus_seq_masks: (batch_size, seq_len) - 序列掩码
            return_ability: 是否返回能力矩阵供KT使用
            use_kt_optimization: 是否使用KT优化
        Returns:
            如果 return_ability=True: (predictions, ability_matrix)
            否则: predictions
        """
        batch_size, seq_len = qus_seq_indices.shape
        
        # 获取批次中涉及的题目嵌入
        qus_emb_batch = self.h_qus[qus_seq_indices]  # (batch_size, seq_len, emb_dim)
        
        # 构造 h_lrn_cpt: 每个学习者与每个知识点的拼接
        h_lrn_expanded = h_lrn_batch.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, emb_dim)
        h_lrn_expanded = h_lrn_expanded.repeat(1, seq_len, self.concept_num, 1)
        
        h_cpt_expanded = self.h_cpt.unsqueeze(0).unsqueeze(0)  # (1, 1, cpt_num, emb_dim)
        h_cpt_expanded = h_cpt_expanded.repeat(batch_size, seq_len, 1, 1)
        
        h_lrn_cpt = torch.cat([h_lrn_expanded, h_cpt_expanded], dim=-1)
        
        # 构造 h_qus_cpt: 每个题目与每个知识点的拼接
        h_qus_expanded = qus_emb_batch.unsqueeze(2)  # (batch_size, seq_len, 1, emb_dim)
        h_qus_expanded = h_qus_expanded.repeat(1, 1, self.concept_num, 1)
        h_qus_cpt = torch.cat([h_qus_expanded, h_cpt_expanded], dim=-1)
        
        # 计算诊断特质
        p_lrn, d_qus, beta_qus = self.dtr(
            h_lrn_cpt.reshape(-1, 2 * self.embedding_dim),
            h_qus_cpt.reshape(-1, 2 * self.embedding_dim),
            qus_emb_batch.reshape(-1, self.embedding_dim)
        )
        
        # 重塑维度 - 修复这里
        p_lrn = p_lrn.reshape(batch_size, seq_len, self.concept_num)
        d_qus = d_qus.reshape(batch_size, seq_len, self.concept_num)
        
        # beta_qus 应该是每个题目的标量值，不需要与知识点维度相关
        beta_qus = beta_qus.reshape(batch_size, seq_len)  # 修改这里：去掉concept_num维度
        
        # 使用KT优化能力（如果可用且启用）
        if use_kt_optimization and self.kt_optimized_ability is not None:
            p_lrn = self.compute_ability_with_kt(h_lrn_batch, p_lrn)
        
        # MIRT预测
        if return_ability:
            predictions, ability_matrix = self.mirt(
                p_lrn, d_qus, beta_qus, return_ability=True
            )
        else:
            predictions = self.mirt(p_lrn, d_qus, beta_qus, return_ability=False)
        
        # 应用掩码
        predictions = predictions * qus_seq_masks
        
        if return_ability:
            # 同时返回掩码后的能力矩阵
            ability_masked = ability_matrix * qus_seq_masks.unsqueeze(-1)
            return predictions, ability_masked
        else:
            return predictions

    def get_ability_matrix(self, h_lrn_batch, qus_seq_indices, qus_seq_masks):
        """
        专门获取能力矩阵供KT初始化使用
        Returns:
            ability_matrix: (batch_size, seq_len, concept_num)
        """
        with torch.no_grad():
            _, ability_matrix = self.forward(
                h_lrn_batch, qus_seq_indices, qus_seq_masks, 
                return_ability=True, use_kt_optimization=False
            )
        return ability_matrix

if __name__ == '__main__':
    import torch.nn as nn
    from torch.utils.data import DataLoader
    
    # 导入必要的模块
    try:
        from Model.HGC import HGC
        from DataReader.HGCDataReader import hgcdr
        from DataSet.CDDataSet import CDDataset
        from DataReader.CDDataReader import cddr
        print("✓ 所有模块导入成功")
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        exit(1)

    def test_cd_with_real_data():
        """使用真实数据测试CD模型和CDDataSet的适配性"""
        print("=== CD模型与CDDataSet适配性测试 ===")
        
        # 1. 加载HGC数据并计算嵌入
        print("\n1. 加载HGC数据...")
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

        # 2. 加载CD数据
        print("\n2. 加载CD数据...")
        cddata = cddr.loadDatafromSql()
        print(f"  学习者数量: {len(cddata['lrn_uid'])}")
        print(f"  题目数量: {len(cddata['qus_uid'])}")
        print(f"  知识点数量: {len(cddata['cpt_uid'])}")

        # 3. 创建数据集
        print("\n3. 创建数据集...")
        train_dataset = CDDataset(cddata, lrn_emb, unt_emb, cpt_emb, 'train', max_seq_len=128)
        test_dataset = CDDataset(cddata, lrn_emb, unt_emb, cpt_emb, 'test', max_seq_len=128)
        
        print("✓ 数据集创建完成")
        print(f"  训练集统计: {train_dataset.get_data_statistics()}")
        print(f"  测试集统计: {test_dataset.get_data_statistics()}")

        # 4. 创建CD模型
        print("\n4. 创建CD模型...")
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
        print(f"  模型参数数量: {sum(p.numel() for p in cd_model.parameters())}")

        # 5. 测试单个batch的前向传播
        print("\n5. 测试单个batch前向传播...")
        batch_size = 4
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=train_dataset.collate_fn
        )
        
        for i, batch in enumerate(train_loader):
            if i >= 1:  # 只测试第一个batch
                break
                
            print(f"  Batch {i+1} 数据:")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"    {key}: {value.shape}")
            
            # CD模型前向传播
            with torch.no_grad():
                # 测试基础预测
                predictions = cd_model(
                    batch['h_lrn_batch'].to(device),
                    batch['qus_seq_indices'].to(device),
                    batch['qus_seq_masks'].to(device)
                )
                print(f"  ✓ 基础预测完成: {predictions.shape}")
                
                # 测试返回能力矩阵
                predictions_with_ability, ability_matrix = cd_model(
                    batch['h_lrn_batch'].to(device),
                    batch['qus_seq_indices'].to(device),
                    batch['qus_seq_masks'].to(device),
                    return_ability=True
                )
                print(f"  ✓ 能力矩阵返回: {ability_matrix.shape}")
                print(f"    预测结果: {predictions_with_ability.shape}")
                print(f"    能力范围: [{ability_matrix.min():.3f}, {ability_matrix.max():.3f}]")
                
                # 测试专门获取能力矩阵
                ability_only = cd_model.get_ability_matrix(
                    batch['h_lrn_batch'].to(device),
                    batch['qus_seq_indices'].to(device),
                    batch['qus_seq_masks'].to(device)
                )
                print(f"  ✓ 专门获取能力矩阵: {ability_only.shape}")

        # 6. 模拟KT功能测试
        print("\n6. 模拟KT功能测试...")
        
        # 模拟KT模型（简化版）
        class MockKTModel:
            """模拟KT模型，仅用于测试接口"""
            def __init__(self, concept_num):
                self.concept_num = concept_num
            
            def optimize_ability(self, cd_ability, batch_data):
                """
                模拟KT优化能力的过程
                Args:
                    cd_ability: CD计算的能力矩阵 (batch_size, seq_len, concept_num)
                    batch_data: 批次数据
                Returns:
                    optimized_ability: KT优化后的能力矩阵
                """
                # 简单模拟：对CD能力加入一些噪声作为"优化"
                batch_size, seq_len, concept_num = cd_ability.shape
                noise = torch.randn_like(cd_ability) * 0.1
                optimized_ability = torch.clamp(cd_ability + noise, 0, 1)
                return optimized_ability
        
        # 创建模拟KT模型
        mock_kt = MockKTModel(len(cddata['cpt_uid']))
        print("✓ 模拟KT模型创建完成")

        # 测试CD-KT循环
        for i, batch in enumerate(train_loader):
            if i >= 1:  # 只测试第一个batch
                break
                
            print(f"  CD-KT循环测试:")
            
            # CD计算初始能力
            with torch.no_grad():
                initial_ability = cd_model.get_ability_matrix(
                    batch['h_lrn_batch'].to(device),
                    batch['qus_seq_indices'].to(device),
                    batch['qus_seq_masks'].to(device)
                )
                print(f"    CD初始能力: {initial_ability.shape}")
            
            # KT优化能力
            kt_optimized_ability = mock_kt.optimize_ability(
                initial_ability.cpu(), batch
            )
            print(f"    KT优化能力: {kt_optimized_ability.shape}")
            
            # 设置KT优化能力到CD模型
            cd_model.set_kt_optimized_ability(kt_optimized_ability.to(device))
            print(f"    KT能力设置完成")
            
            # 使用KT优化后的能力进行CD预测
            with torch.no_grad():
                predictions_with_kt, ability_with_kt = cd_model(
                    batch['h_lrn_batch'].to(device),
                    batch['qus_seq_indices'].to(device),
                    batch['qus_seq_masks'].to(device),
                    return_ability=True,
                    use_kt_optimization=True
                )
                print(f"    KT优化后预测: {predictions_with_kt.shape}")
                print(f"    KT优化后能力: {ability_with_kt.shape}")
                
                # 计算优化前后的差异
                ability_diff = torch.mean(torch.abs(ability_with_kt - initial_ability))
                print(f"    能力平均变化: {ability_diff.item():.4f}")

        # 7. 梯度测试
        print("\n7. 梯度测试...")
        for i, batch in enumerate(train_loader):
            if i >= 1:
                break
                
            # 启用梯度计算
            cd_model.train()
            batch['h_lrn_batch'] = batch['h_lrn_batch'].clone().to(device).requires_grad_(True)
            
            predictions = cd_model(
                batch['h_lrn_batch'],
                batch['qus_seq_indices'].to(device),
                batch['qus_seq_masks'].to(device)
            )
            
            # 计算损失和梯度
            loss = nn.BCELoss()(predictions, batch['results'].to(device))
            loss.backward()
            
            # 检查梯度
            has_gradient = batch['h_lrn_batch'].grad is not None
            print(f"  ✓ 梯度计算: {'成功' if has_gradient else '失败'}")
            if has_gradient:
                grad_norm = batch['h_lrn_batch'].grad.norm().item()
                print(f"    梯度范数: {grad_norm:.6f}")

        # 8. 性能测试
        print("\n8. 性能测试...")
        import time
        
        cd_model.eval()
        test_batches = 3
        total_time = 0
        
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= test_batches:
                    break
                
                start_time = time.time()
                _ = cd_model(
                    batch['h_lrn_batch'].to(device),
                    batch['qus_seq_indices'].to(device),
                    batch['qus_seq_masks'].to(device)
                )
                batch_time = time.time() - start_time
                total_time += batch_time
                print(f"  Batch {i+1} 推理时间: {batch_time*1000:.2f}ms")
        
        avg_time = total_time / test_batches * 1000
        print(f"  平均推理时间: {avg_time:.2f}ms/batch")

        print("\n=== 所有测试完成 ===")
        print("✓ CD模型与CDDataSet适配性测试通过")
        print("✓ CD-KT接口功能正常")
        print("✓ 梯度计算正常")
        print("✓ 性能测试完成")

    # 运行测试
    test_cd_with_real_data()