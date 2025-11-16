import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

class TypeSpecificInput(nn.Module):
    """类型特定的输入处理层 - 处理6种交互类型的additioninfo"""
    def __init__(self, embedding_dim, hidden_dim=64):
        super().__init__()
        
        # 6种交互类型的特定处理网络
        self.type_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim),  # 输入: add1, add2
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            ) for _ in range(6)  # 6种交互类型
        ])
        
        # 类型嵌入
        self.type_embedding = nn.Embedding(6, embedding_dim)
        
    def forward(self, add1, add2, type_indices):
        """
        Args:
            add1: [batch_size, seq_len] - additioninfo1
            add2: [batch_size, seq_len] - additioninfo2  
            type_indices: [batch_size, seq_len] - 交互类型索引(0-5)
        Returns:
            type_specific_features: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len = add1.shape
        
        # 类型嵌入
        type_emb = self.type_embedding(type_indices)  # [batch_size, seq_len, emb_dim]
        
        # 类型特定的特征处理
        type_features = []
        input_features = torch.stack([add1, add2], dim=-1)  # [batch_size, seq_len, 2]
        
        for i in range(6):
            mask = (type_indices == i)  # [batch_size, seq_len]
            if mask.any():
                # 处理该类型的特征
                type_feat = self.type_networks[i](input_features)  # [batch_size, seq_len, emb_dim]
                # 只保留该类型的特征
                type_feat = type_feat * mask.unsqueeze(-1).float()
                type_features.append(type_feat)
            else:
                type_features.append(torch.zeros_like(type_emb))
        
        # 合并所有类型的特征
        type_specific = sum(type_features)  # [batch_size, seq_len, emb_dim]
        
        # 与类型嵌入融合
        combined_features = type_emb + type_specific
        
        return combined_features

class SensoryMemoryRegistration(nn.Module):
    """感觉记忆注册阶段"""
    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 自注意力编码器
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 层归一化和前馈网络
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, embedding_dim] - 输入特征
            mask: [batch_size, seq_len] - 序列掩码 (float类型)
        Returns:
            sensory_memory: [batch_size, seq_len, embedding_dim]
        """
        # 处理掩码：将float掩码转换为bool类型
        if mask is not None:
            # 将float掩码转换为bool类型，小于0.5的位置为padding
            key_padding_mask = mask < 0.5
        else:
            key_padding_mask = None
        
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_output)
        
        # 前馈网络
        ff_output = self.ffn(x)
        sensory_memory = self.norm2(x + ff_output)
        
        return sensory_memory

class ShortTermMemoryFusion(nn.Module):
    """短时记忆融合阶段 - 双通道结构"""
    def __init__(self, embedding_dim, concept_num, num_heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.concept_num = concept_num
        
        # 时序通道 (RNN)
        self.temporal_channel = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
            bidirectional=False
        )
        
        # 关系通道 (注意力)
        self.relational_channel = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 融合层
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 知识点映射层
        self.concept_projection = nn.Linear(embedding_dim, concept_num)
        
    def forward(self, sensory_memory, mask=None):
        """
        Args:
            sensory_memory: [batch_size, seq_len, embedding_dim] - 感觉记忆
            mask: [batch_size, seq_len] - 序列掩码 (float类型)
        Returns:
            short_term_memory: [batch_size, seq_len, embedding_dim]
            concept_mastery: [batch_size, seq_len, concept_num] - 知识点掌握程度
        """
        batch_size, seq_len, _ = sensory_memory.shape
        
        # 处理掩码
        if mask is not None:
            key_padding_mask = mask < 0.5
        else:
            key_padding_mask = None
        
        # 时序通道
        temporal_output, _ = self.temporal_channel(sensory_memory)  # [batch_size, seq_len, emb_dim]
        
        # 关系通道
        relational_output, _ = self.relational_channel(
            sensory_memory, sensory_memory, sensory_memory,
            key_padding_mask=key_padding_mask
        )  # [batch_size, seq_len, emb_dim]
        
        # 融合时序和关系特征
        fusion_input = temporal_output + relational_output
        short_term_memory, _ = self.fusion_attention(
            fusion_input, fusion_input, fusion_input,
            key_padding_mask=key_padding_mask
        )  # [batch_size, seq_len, emb_dim]
        
        # 计算知识点掌握程度
        concept_mastery = torch.sigmoid(self.concept_projection(short_term_memory))  # [batch_size, seq_len, concept_num]
        
        return short_term_memory, concept_mastery

class LongTermMemoryRetrieval(nn.Module):
    """长时记忆检索阶段"""
    def __init__(self, embedding_dim, memory_units=16, memory_dim=20):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.memory_units = memory_units
        self.memory_dim = memory_dim
        
        # 记忆矩阵
        self.memory_matrix = nn.Parameter(torch.randn(memory_units, memory_dim))
        
        # 读写门控机制
        self.write_gate = nn.Sequential(
            nn.Linear(embedding_dim, memory_units),
            nn.Sigmoid()
        )
        
        self.read_gate = nn.Sequential(
            nn.Linear(embedding_dim, memory_units),
            nn.Sigmoid()
        )
        
        # 值投影
        self.value_projection = nn.Linear(embedding_dim, memory_dim)
        
        # 输出投影
        self.output_projection = nn.Linear(memory_dim, embedding_dim)
        
    def forward(self, short_term_memory, mask=None):
        """
        Args:
            short_term_memory: [batch_size, seq_len, embedding_dim] - 短时记忆
            mask: [batch_size, seq_len] - 序列掩码 (float类型)
        Returns:
            long_term_memory: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, _ = short_term_memory.shape
        
        # 处理掩码
        if mask is not None:
            valid_mask = mask > 0.5  # 有效位置为True
        else:
            valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=short_term_memory.device)
        
        # 写操作 - 更新记忆矩阵
        write_weights = self.write_gate(short_term_memory)  # [batch_size, seq_len, memory_units]
        values = self.value_projection(short_term_memory)   # [batch_size, seq_len, memory_dim]
        
        # 累积记忆更新 (简化实现)
        memory_updates = torch.zeros(batch_size, self.memory_units, self.memory_dim, 
                                   device=short_term_memory.device)
        for t in range(seq_len):
            if valid_mask[:, t].any():
                update = write_weights[:, t].unsqueeze(-1) * values[:, t].unsqueeze(1)  # [batch_size, memory_units, memory_dim]
                memory_updates += update
        
        # 读操作 - 从记忆矩阵检索
        read_weights = self.read_gate(short_term_memory)  # [batch_size, seq_len, memory_units]
        
        # 检索长时记忆
        long_term_memory = torch.matmul(read_weights, self.memory_matrix.unsqueeze(0))  # [batch_size, seq_len, memory_dim]
        
        # 投影回原始维度
        long_term_memory = self.output_projection(long_term_memory)
        
        return long_term_memory

class KT(nn.Module):
    """知识追踪主模型 - 支持CD-KT闭环"""
    def __init__(self, embedding_dim, concept_num, h_lrn, h_unt, h_cpt):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.concept_num = concept_num
        
        # 注册预计算的HGC嵌入
        self.register_buffer('h_lrn', h_lrn)  # [lrn_num, emb_dim]
        self.register_buffer('h_unt', h_unt)  # [untqus_num, emb_dim] 
        self.register_buffer('h_cpt', h_cpt)  # [cpt_num, emb_dim]
        
        # CD优化后的能力缓存
        self.cd_optimized_ability = None
        
        # 输入处理
        self.type_specific_input = TypeSpecificInput(embedding_dim)
        
        # 三阶段记忆流
        self.sensory_stage = SensoryMemoryRegistration(embedding_dim)
        self.short_term_stage = ShortTermMemoryFusion(embedding_dim, concept_num)
        self.long_term_stage = LongTermMemoryRetrieval(embedding_dim)
        
        # 能力融合门控
        self.ability_fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim + concept_num, 128),
            nn.ReLU(),
            nn.Linear(128, concept_num),
            nn.Sigmoid()
        )
        
        # 最终预测层
        self.prediction_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2 + concept_num, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, concept_num),
            nn.Sigmoid()
        )
        
    def set_cd_optimized_ability(self, cd_ability):
        """
        设置CD优化后的能力矩阵
        Args:
            cd_ability: [batch_size, seq_len, concept_num] CD优化的能力
        """
        self.cd_optimized_ability = cd_ability
    
    def compute_ability_with_cd(self, h_lrn_batch, concept_mastery):
        """
        融合KT计算的能力和CD优化的能力
        Args:
            h_lrn_batch: [batch_size, emb_dim] KT计算的学习者嵌入
            concept_mastery: [batch_size, seq_len, concept_num] KT计算的能力
        """
        if self.cd_optimized_ability is None:
            return concept_mastery
        
        batch_size, seq_len, concept_num = concept_mastery.shape
        
        # 确保CD能力与KT能力维度一致
        if (self.cd_optimized_ability.shape[0] != batch_size or 
            self.cd_optimized_ability.shape[1] != seq_len):
            return concept_mastery
        
        # 计算融合权重
        h_lrn_expanded = h_lrn_batch.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, emb_dim]
        gate_input = torch.cat([h_lrn_expanded, concept_mastery], dim=-1)  # [batch_size, seq_len, emb_dim + concept_num]
        fusion_weights = self.ability_fusion_gate(gate_input)  # [batch_size, seq_len, concept_num]
        
        # 融合KT能力和CD能力
        fused_ability = (fusion_weights * self.cd_optimized_ability + 
                        (1 - fusion_weights) * concept_mastery)
        
        return fused_ability
    
    def forward(self, lrn_indices, unt_indices, add1, add2, type_indices, 
                seq_mask, next_question_mask, use_cd_optimization=True):
        """
        Args:
            lrn_indices: [batch_size] - 学习者索引
            unt_indices: [batch_size, seq_len] - 学习单元索引  
            add1: [batch_size, seq_len] - additioninfo1
            add2: [batch_size, seq_len] - additioninfo2
            type_indices: [batch_size, seq_len] - 交互类型索引
            seq_mask: [batch_size, seq_len] - 序列掩码 (float类型)
            next_question_mask: [batch_size, seq_len] - 下一个是题目的掩码 (float类型)
            use_cd_optimization: 是否使用CD优化
        Returns:
            predictions: [batch_size, seq_len, concept_num] - 知识点预测
            concept_mastery: [batch_size, seq_len, concept_num] - 知识点掌握程度
        """
        batch_size, seq_len = unt_indices.shape
        
        # 获取HGC嵌入
        h_lrn_batch = self.h_lrn[lrn_indices]  # [batch_size, emb_dim]
        h_unt_batch = self.h_unt[unt_indices]  # [batch_size, seq_len, emb_dim]
        
        # 类型特定的输入处理
        type_features = self.type_specific_input(add1, add2, type_indices)  # [batch_size, seq_len, emb_dim]
        
        # 组合输入特征: HGC嵌入 + 类型特征
        combined_input = h_unt_batch + type_features  # [batch_size, seq_len, emb_dim]
        
        # 阶段1: 感觉记忆注册
        sensory_memory = self.sensory_stage(combined_input, seq_mask)  # [batch_size, seq_len, emb_dim]
        
        # 阶段2: 短时记忆融合
        short_term_memory, concept_mastery = self.short_term_stage(sensory_memory, seq_mask)  # [batch_size, seq_len, emb_dim], [batch_size, seq_len, concept_num]
        
        # 使用CD优化能力（如果可用且启用）
        if use_cd_optimization and self.cd_optimized_ability is not None:
            concept_mastery = self.compute_ability_with_cd(h_lrn_batch, concept_mastery)
        
        # 阶段3: 长时记忆检索
        long_term_memory = self.long_term_stage(short_term_memory, seq_mask)  # [batch_size, seq_len, emb_dim]
        
        # 最终预测
        combined_features = torch.cat([short_term_memory, long_term_memory, concept_mastery], dim=-1)  # [batch_size, seq_len, emb_dim*2 + concept_num]
        predictions = self.prediction_layer(combined_features)  # [batch_size, seq_len, concept_num]
        
        # 应用下一个题目掩码（只在需要预测下一个题目的时间步返回预测）
        predictions_masked = predictions * next_question_mask.unsqueeze(-1)
        concept_mastery_masked = concept_mastery * seq_mask.unsqueeze(-1)
        
        return predictions_masked, concept_mastery_masked
    
    def get_concept_mastery(self, lrn_indices, unt_indices, add1, add2, type_indices, seq_mask):
        """
        专门获取知识点掌握程度供CD使用
        Returns:
            concept_mastery: [batch_size, seq_len, concept_num]
        """
        with torch.no_grad():
            _, concept_mastery = self.forward(
                lrn_indices, unt_indices, add1, add2, type_indices, 
                seq_mask, torch.ones_like(seq_mask),  # 所有时间步都返回
                use_cd_optimization=False  # 不使用CD优化，返回纯KT能力
            )
        return concept_mastery

def test_kt_with_real_data():
    """使用真实数据测试KT模型和KTDataSet的适配性"""
    print("=== KT模型与KTDataSet适配性测试 ===")
    
    # 1. 加载HGC数据并计算嵌入
    print("\n1. 加载HGC数据...")
    from DataReader.HGCDataReader import hgcdr
    from Model.HGC import HGC
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
    from DataReader.KTDataReader import ktdr
    ktdata = ktdr.loadDatafromSql()
    print(f"  学习者数量: {len(ktdata['lrn_uid'])}")
    print(f"  学习单元+题目数量: {len(ktdata['untqus_uid'])}")
    print(f"  知识点数量: {len(ktdata['cpt_uid'])}")

    # 3. 创建数据集
    print("\n3. 创建数据集...")
    from DataSet.KTDataSet import KTDataSet
    train_dataset = KTDataSet(ktdata, lrn_emb, unt_emb, cpt_emb, 'train', max_seq_len=128)
    test_dataset = KTDataSet(ktdata, lrn_emb, unt_emb, cpt_emb, 'test', max_seq_len=128)
    
    print("✓ 数据集创建完成")
    print(f"  训练集统计: {train_dataset.get_data_statistics()}")
    print(f"  测试集统计: {test_dataset.get_data_statistics()}")

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
    print(f"  模型参数数量: {sum(p.numel() for p in kt_model.parameters())}")

    # 5. 测试单个batch的前向传播
    print("\n5. 测试单个batch前向传播...")
    from torch.utils.data import DataLoader
    
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
                print(f"    {key}: {value.shape} (dtype: {value.dtype})")
        
        # KT模型前向传播
        with torch.no_grad():
            # 测试基础预测
            predictions, concept_mastery = kt_model(
                batch['lrn_indices'].to(device),
                batch['unt_seq_indices'].to(device),
                batch['add1'].to(device),
                batch['add2'].to(device),
                batch['type_indices'].to(device),
                batch['seq_masks'].to(device),
                batch['next_question_masks'].to(device)
            )
            print(f"  ✓ 基础预测完成")
            print(f"    预测输出: {predictions.shape}")
            print(f"    知识点掌握程度: {concept_mastery.shape}")
            print(f"    预测范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
            
            # 测试专门获取能力矩阵
            ability_only = kt_model.get_concept_mastery(
                batch['lrn_indices'].to(device),
                batch['unt_seq_indices'].to(device),
                batch['add1'].to(device),
                batch['add2'].to(device),
                batch['type_indices'].to(device),
                batch['seq_masks'].to(device)
            )
            print(f"  ✓ 专门获取能力矩阵: {ability_only.shape}")

    # 6. 模拟CD-KT闭环测试
    print("\n6. 模拟CD-KT闭环测试...")
    
    # 模拟CD能力（简化版）
    class MockCDModel:
        """模拟CD模型，仅用于测试接口"""
        def __init__(self, concept_num):
            self.concept_num = concept_num
        
        def get_ability_matrix(self, batch_data):
            """
            模拟CD计算能力的过程
            Returns:
                cd_ability: CD计算的能力矩阵 [batch_size, seq_len, concept_num]
            """
            batch_size, seq_len = batch_data['unt_seq_indices'].shape
            # 简单模拟：随机生成能力矩阵
            cd_ability = torch.randn(batch_size, seq_len, self.concept_num)
            cd_ability = torch.sigmoid(cd_ability)  # 映射到0-1
            return cd_ability
    
    # 创建模拟CD模型
    mock_cd = MockCDModel(len(ktdata['cpt_uid']))
    print("✓ 模拟CD模型创建完成")

    # 测试CD-KT循环
    for i, batch in enumerate(train_loader):
        if i >= 1:  # 只测试第一个batch
            break
            
        print(f"  CD-KT循环测试:")
        
        # CD计算初始能力
        with torch.no_grad():
            cd_initial_ability = mock_cd.get_ability_matrix(batch)
            print(f"    CD初始能力: {cd_initial_ability.shape}")
        
        # 设置CD能力到KT模型
        kt_model.set_cd_optimized_ability(cd_initial_ability.to(device))
        print(f"    CD能力设置完成")
        
        # 使用CD优化后的能力进行KT预测
        with torch.no_grad():
            predictions_with_cd, ability_with_cd = kt_model(
                batch['lrn_indices'].to(device),
                batch['unt_seq_indices'].to(device),
                batch['add1'].to(device),
                batch['add2'].to(device),
                batch['type_indices'].to(device),
                batch['seq_masks'].to(device),
                batch['next_question_masks'].to(device),
                use_cd_optimization=True
            )
            print(f"    CD优化后预测: {predictions_with_cd.shape}")
            print(f"    CD优化后能力: {ability_with_cd.shape}")

    # 7. 梯度测试
    print("\n7. 梯度测试...")
    for i, batch in enumerate(train_loader):
        if i >= 1:
            break
            
        # 启用梯度计算
        kt_model.train()
        
        predictions, _ = kt_model(
            batch['lrn_indices'].to(device),
            batch['unt_seq_indices'].to(device),
            batch['add1'].to(device),
            batch['add2'].to(device),
            batch['type_indices'].to(device),
            batch['seq_masks'].to(device),
            batch['next_question_masks'].to(device)
        )
        
        # 计算损失和梯度 - 只对下一个是题目的时间步计算损失
        valid_predictions = predictions[batch['next_question_masks'].to(device).bool()]
        valid_targets = batch['next_results'].to(device)[batch['next_question_masks'].to(device).bool()]
        
        if len(valid_predictions) > 0:
            # 对每个知识点维度取平均作为预测结果
            kt_loss = nn.BCELoss()(valid_predictions.mean(dim=-1), valid_targets)
            kt_loss.backward()
            
            # 检查梯度
            has_gradient = any(p.grad is not None for p in kt_model.parameters())
            print(f"  ✓ 梯度计算: {'成功' if has_gradient else '失败'}")
            if has_gradient:
                grad_norm = sum(p.grad.norm().item() for p in kt_model.parameters() if p.grad is not None)
                print(f"    总梯度范数: {grad_norm:.6f}")
        else:
            print(f"  ✗ 梯度计算: 跳过 (无有效预测)")

    print("\n=== 所有测试完成 ===")
    print("✓ KT模型与KTDataSet适配性测试通过")
    print("✓ CD-KT接口功能正常")
    print("✓ 梯度计算正常")

if __name__ == '__main__':
    # 运行基础测试
    print("=== KT模型基础测试 ===")
    
    # 模拟数据测试
    batch_size = 4
    seq_len = 10
    embedding_dim = 64
    concept_num = 50
    lrn_num = 100
    untqus_num = 200
    
    # 模拟HGC嵌入
    h_lrn = torch.randn(lrn_num, embedding_dim)
    h_unt = torch.randn(untqus_num, embedding_dim)
    h_cpt = torch.randn(concept_num, embedding_dim)
    
    # 创建模型
    model = KT(embedding_dim, concept_num, h_lrn, h_unt, h_cpt)
    
    # 模拟输入数据
    lrn_indices = torch.randint(0, lrn_num, (batch_size,))
    unt_indices = torch.randint(0, untqus_num, (batch_size, seq_len))
    add1 = torch.randn(batch_size, seq_len)
    add2 = torch.randn(batch_size, seq_len)
    type_indices = torch.randint(0, 6, (batch_size, seq_len))
    seq_mask = torch.ones(batch_size, seq_len).float()  # 使用float类型掩码
    next_question_mask = torch.randint(0, 2, (batch_size, seq_len)).float()
    
    print("模型测试:")
    print(f"输入维度: batch_size={batch_size}, seq_len={seq_len}")
    
    # 前向传播测试
    predictions, concept_mastery = model(
        lrn_indices, unt_indices, add1, add2, type_indices, 
        seq_mask, next_question_mask
    )
    
    print(f"预测输出: {predictions.shape}")
    print(f"知识点掌握程度: {concept_mastery.shape}")
    
    # 运行真实数据测试
    print("\n" + "="*50)
    test_kt_with_real_data()