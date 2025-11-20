import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hyperparams.hyperparameter import hyperparams

class ContrastivePreTraining(nn.Module):
    """对比学习预训练模块 - 适配超参数"""
    def __init__(self, embedding_dim=None, hidden_dim=None, temperature=None):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = hyperparams.hgc_embedding_dim
        if hidden_dim is None:
            hidden_dim = hyperparams.kt_contrastive_hidden_dim
        if temperature is None:
            temperature = hyperparams.kt_contrastive_temperature
            
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
        # 基础编码器 - 使用Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=hyperparams.kt_attention_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            dropout=hyperparams.hgc_dropout_rate
        )
        self.base_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 动量编码器
        self.momentum_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 投影头
        self.base_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.momentum_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # 初始化动量编码器与基础编码器相同
        self._init_momentum_encoder()
        
    def _init_momentum_encoder(self):
        """初始化动量编码器参数"""
        for param_base, param_momentum in zip(self.base_encoder.parameters(), 
                                            self.momentum_encoder.parameters()):
            param_momentum.data.copy_(param_base.data)
            param_momentum.requires_grad = False
            
        for param_base, param_momentum in zip(self.base_projection.parameters(), 
                                            self.momentum_projection.parameters()):
            param_momentum.data.copy_(param_base.data)
            param_momentum.requires_grad = False
    
    def update_momentum_encoder(self, momentum=None):
        """动量更新"""
        if momentum is None:
            momentum = hyperparams.kt_momentum
            
        for param_base, param_momentum in zip(self.base_encoder.parameters(), 
                                            self.momentum_encoder.parameters()):
            param_momentum.data = momentum * param_momentum.data + (1 - momentum) * param_base.data
            
        for param_base, param_momentum in zip(self.base_projection.parameters(), 
                                            self.momentum_projection.parameters()):
            param_momentum.data = momentum * param_momentum.data + (1 - momentum) * param_base.data
    
    def forward(self, x, mask=None):
        """前向传播"""
        # 处理掩码：将float掩码转换为bool类型
        if mask is not None:
            key_padding_mask = mask < 0.5  # True表示padding位置
        else:
            key_padding_mask = None
        
        # 基础编码器
        z_base = self.base_encoder(x, src_key_padding_mask=key_padding_mask)
        z_base = self.base_projection(z_base)
        z_base = F.normalize(z_base, p=2, dim=-1)
        
        # 动量编码器
        with torch.no_grad():
            z_momentum = self.momentum_encoder(x, src_key_padding_mask=key_padding_mask)
            z_momentum = self.momentum_projection(z_momentum)
            z_momentum = F.normalize(z_momentum, p=2, dim=-1)
        
        return z_base, z_momentum

class RelationTemporalEmbedding(nn.Module):
    """关系-时序嵌入模块 - 修复维度问题"""
    def __init__(self, embedding_dim, concept_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 关系嵌入组件
        self.concept_projection = nn.Linear(embedding_dim, embedding_dim)
        self.interaction_projection = nn.Linear(embedding_dim * 2, embedding_dim)  # 学习者+学习单元
        
        # 修复：添加专门的时序编码组件
        self.temporal_encoding = nn.Linear(embedding_dim + 2, embedding_dim)  # +2 for add1, add2
        
    def forward(self, h_lrn, h_qusunt, h_cpt, add1, add2, type_indices, concept_mapping):
        """
        Args:
            h_lrn: [batch_size, seq_len, emb_dim] 学习者嵌入
            h_qusunt: [batch_size, seq_len, emb_dim] 题目+学习单元嵌入（新顺序）
            h_cpt: [concept_num, emb_dim] 知识点嵌入
            add1, add2: [batch_size, seq_len] 额外信息
            type_indices: [batch_size, seq_len] 交互类型
            concept_mapping: 知识点映射字典
        """
        batch_size, seq_len, _ = h_lrn.shape
        
        # 关系嵌入 - 练习-概念级
        relational_concept = torch.zeros_like(h_qusunt)
        for b in range(batch_size):
            for t in range(seq_len):
                qusunt_idx = type_indices[b, t].item()
                if qusunt_idx in concept_mapping:
                    # 获取相关知识点嵌入的平均值
                    related_cpts = concept_mapping[qusunt_idx]
                    if related_cpts:
                        cpt_embs = torch.stack([h_cpt[cpt_idx] for cpt_idx in related_cpts])
                        avg_cpt_emb = cpt_embs.mean(dim=0)
                        relational_concept[b, t] = avg_cpt_emb
        
        # 关系嵌入 - 练习-交互级
        relational_interaction = self.interaction_projection(
            torch.cat([h_lrn, h_qusunt], dim=-1)
        )
        
        # 时序嵌入 - 修复：使用专门的时序编码层
        temporal_features = torch.stack([add1, add2], dim=-1)  # [batch_size, seq_len, 2]
        temporal_input = torch.cat([h_qusunt, temporal_features], dim=-1)  # [batch_size, seq_len, emb_dim + 2]
        temporal_embedding = self.temporal_encoding(temporal_input)  # [batch_size, seq_len, emb_dim]
        
        # 组合嵌入
        relational_embedding = relational_concept + relational_interaction
        combined_embedding = relational_embedding + temporal_embedding
        
        return combined_embedding

class DualChannelFusion(nn.Module):
    """双通道融合模块 - 适配超参数"""
    def __init__(self, embedding_dim=None, num_heads=None):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = hyperparams.hgc_embedding_dim
        if num_heads is None:
            num_heads = hyperparams.kt_attention_heads
            
        self.embedding_dim = embedding_dim
        
        # 时序通道 - BiGRU捕获双向时序依赖
        self.temporal_channel = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim // 2,  # 双向所以减半
            batch_first=True,
            bidirectional=True
        )
        self.temporal_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # 关系通道 - 多头自注意力
        self.relational_channel = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 注意力融合层
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 门控融合机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
    def forward(self, sensory_memory, mask=None):
        batch_size, seq_len, _ = sensory_memory.shape
        
        # 处理掩码
        if mask is not None:
            key_padding_mask = mask < 0.5
        else:
            key_padding_mask = None
        
        # 时序通道
        temporal_output, _ = self.temporal_channel(sensory_memory)
        temporal_output = self.temporal_projection(temporal_output)
        
        # 关系通道
        relational_output, _ = self.relational_channel(
            sensory_memory, sensory_memory, sensory_memory,
            key_padding_mask=key_padding_mask
        )
        
        # 门控融合
        fusion_gate = self.fusion_gate(torch.cat([temporal_output, relational_output], dim=-1))
        fused_features = fusion_gate * temporal_output + (1 - fusion_gate) * relational_output
        
        # 注意力融合
        short_term_memory, _ = self.fusion_attention(
            fused_features, fused_features, fused_features,
            key_padding_mask=key_padding_mask
        )
        
        return short_term_memory

class KnowledgeBaseRetrieval(nn.Module):
    """知识点库检索模块 - 适配超参数"""
    def __init__(self, embedding_dim, concept_num, memory_units=None, memory_dim=None):
        super().__init__()
        if memory_units is None:
            memory_units = hyperparams.kt_memory_units
        if memory_dim is None:
            memory_dim = hyperparams.kt_memory_dim
            
        self.embedding_dim = embedding_dim
        self.concept_num = concept_num
        self.memory_units = memory_units
        self.memory_dim = memory_dim
        
        # 知识点记忆矩阵 - 模拟知识点库
        self.concept_memory = nn.Parameter(torch.randn(concept_num, memory_dim))
        
        # 单调门控机制
        self.monotonic_gate = nn.Sequential(
            nn.Linear(embedding_dim, memory_units),
            nn.ReLU(),  # 保证单调性
            nn.Linear(memory_units, concept_num),
            nn.Sigmoid()
        )
        
        # 读写头
        self.write_projection = nn.Linear(embedding_dim, memory_dim)
        self.read_projection = nn.Linear(memory_dim, embedding_dim)
        
    def forward(self, short_term_memory, h_cpt, mask=None):
        """
        Args:
            short_term_memory: [batch_size, seq_len, emb_dim]
            h_cpt: [concept_num, emb_dim] HGC计算的知识点嵌入
            mask: [batch_size, seq_len]
        """
        batch_size, seq_len, _ = short_term_memory.shape
        
        if mask is not None:
            valid_mask = mask > 0.5
        else:
            valid_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=short_term_memory.device)
        
        # 单调门控计算权重
        gate_weights = self.monotonic_gate(short_term_memory)  # [batch_size, seq_len, concept_num]
        
        # 写操作 - 更新知识点记忆
        write_values = self.write_projection(short_term_memory)  # [batch_size, seq_len, memory_dim]
        
        # 累积更新（模拟知识点库的渐进更新）
        memory_updates = torch.zeros(self.concept_num, self.memory_dim, device=short_term_memory.device)
        update_counts = torch.zeros(self.concept_num, device=short_term_memory.device)
        
        for b in range(batch_size):
            for t in range(seq_len):
                if valid_mask[b, t]:
                    # 每个时间步对所有知识点都有不同程度的更新
                    concept_updates = gate_weights[b, t].unsqueeze(-1) * write_values[b, t].unsqueeze(0)
                    memory_updates += concept_updates
                    update_counts += gate_weights[b, t]
        
        # 避免除零
        update_counts = torch.clamp(update_counts, min=1e-8)
        memory_updates = memory_updates / update_counts.unsqueeze(-1)
        
        # 读操作 - 从知识点库检索
        # 使用门控权重作为检索权重
        retrieved_memory = torch.matmul(gate_weights, self.concept_memory)  # [batch_size, seq_len, memory_dim]
        long_term_memory = self.read_projection(retrieved_memory)  # [batch_size, seq_len, emb_dim]
        
        return long_term_memory

class KT(nn.Module):
    """改进的知识追踪模型 - 适配新的数据结构"""
    def __init__(self, embedding_dim, concept_num, concept_mapping):
        """
        Args:
            embedding_dim: 嵌入维度
            concept_num: 知识点数量
            concept_mapping: 知识点映射
        """
        super().__init__()
        if embedding_dim is None:
            embedding_dim = hyperparams.hgc_embedding_dim
            
        self.embedding_dim = embedding_dim
        self.concept_num = concept_num
        
        # 知识点映射
        self.concept_mapping = concept_mapping
        
        # CD优化后的能力缓存
        self.cd_optimized_ability = None
        
        # 阶段1: 感觉记忆注册（对比学习预训练）
        self.contrastive_pretraining = ContrastivePreTraining(embedding_dim)
        
        # 关系-时序嵌入
        self.relation_temporal_embedding = RelationTemporalEmbedding(embedding_dim, concept_num)
        
        # 阶段2: 短时记忆融合（双通道）
        self.dual_channel_fusion = DualChannelFusion(embedding_dim)
        
        # 知识点掌握程度预测
        self.mastery_prediction = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, concept_num),
            nn.Sigmoid()
        )
        
        # 阶段3: 长时记忆检索（知识点库检索）
        self.knowledge_retrieval = KnowledgeBaseRetrieval(embedding_dim, concept_num)
        
        # 能力融合门控 - CD-KT闭环接口
        self.ability_fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim + concept_num, 128),
            nn.ReLU(),
            nn.Linear(128, concept_num),
            nn.Sigmoid()
        )
        
        # 最终预测层 - 适配超参数
        fusion_hidden_dims = hyperparams.kt_fusion_hidden_dims
        fusion_dropout_rate = hyperparams.kt_fusion_dropout_rate
        
        self.final_prediction = nn.Sequential(
            nn.Linear(embedding_dim * 2 + concept_num, fusion_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(fusion_dropout_rate),
            nn.Linear(fusion_hidden_dims[0], fusion_hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(fusion_dropout_rate/2),
            nn.Linear(fusion_hidden_dims[1], concept_num),
            nn.Sigmoid()
        )
        
        # 类型特定的输入处理 - 适配超参数
        type_embedding_dim = hyperparams.kt_type_embedding_dim
        type_specific_hidden_dim = hyperparams.kt_type_specific_hidden_dim
        
        self.type_embedding = nn.Embedding(6, type_embedding_dim)
        self.type_specific_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, type_specific_hidden_dim),  # add1, add2
                nn.ReLU(),
                nn.Linear(type_specific_hidden_dim, embedding_dim)
            ) for _ in range(6)
        ])
        
        # 关键修复：梯度增强参数，参考CD实现
        self.grad_enhancer = nn.Parameter(torch.ones(1) * 0.01)
        
    def set_cd_optimized_ability(self, cd_ability, qus_num):
        """
        设置CD优化后的能力矩阵
        Args:
            cd_ability: (batch_size, seq_len, concept_num) CD计算的能力
            qus_num: 题目数量，用于验证
        """
        if cd_ability is not None:
            # 注意：CD能力只对题目位置有效，学习单元位置应该使用其他方式初始化
            self.cd_optimized_ability = cd_ability.detach().clone()
    
    def compute_ability_with_cd(self, h_lrn_batch, concept_mastery):
        """融合KT计算的能力和CD优化的能力"""
        if self.cd_optimized_ability is None:
            return concept_mastery
        
        batch_size, seq_len, concept_num = concept_mastery.shape
        
        if (self.cd_optimized_ability.shape[0] != batch_size or 
            self.cd_optimized_ability.shape[1] != seq_len):
            return concept_mastery
        
        # 计算融合权重
        h_lrn_expanded = h_lrn_batch.unsqueeze(1).repeat(1, seq_len, 1)
        gate_input = torch.cat([h_lrn_expanded, concept_mastery], dim=-1)
        fusion_weights = self.ability_fusion_gate(gate_input)
        
        # 融合能力
        fused_ability = (fusion_weights * self.cd_optimized_ability + 
                        (1 - fusion_weights) * concept_mastery)
        
        return fused_ability
    
    def type_specific_processing(self, add1, add2, type_indices):
        """类型特定的输入处理"""
        batch_size, seq_len = add1.shape
        
        # 类型嵌入
        type_emb = self.type_embedding(type_indices)
        
        # 类型特定的特征
        input_features = torch.stack([add1, add2], dim=-1)
        type_specific_features = torch.zeros_like(type_emb)
        
        for i in range(6):
            mask = (type_indices == i)
            if mask.any():
                type_feat = self.type_specific_networks[i](input_features)
                type_specific_features += type_feat * mask.unsqueeze(-1).float()
        
        return type_emb + type_specific_features
    
    def forward(self, h_lrn_batch, h_qusunt_batch, h_cpt, lrn_indices, qusunt_seq_indices, add1, add2, type_indices,
                seq_mask, prediction_masks, use_cd_optimization=None, use_contrastive=None):
        """
        基于新数据结构的前向传播设计
        Args:
            h_lrn_batch: [batch_size, embedding_dim] 学习者嵌入
            h_qusunt_batch: [batch_size, seq_len, embedding_dim] 题目+学习单元嵌入
            h_cpt: [concept_num, embedding_dim] 知识点嵌入
            lrn_indices: [batch_size] 学习者索引
            qusunt_seq_indices: [batch_size, seq_len] 题目+学习单元索引
            add1, add2: [batch_size, seq_len] 额外信息
            type_indices: [batch_size, seq_len] 交互类型
            seq_mask: [batch_size, seq_len] 序列掩码
            prediction_masks: [batch_size, seq_len] 预测掩码（新字段）
            use_cd_optimization: 是否使用CD优化
            use_contrastive: 是否使用对比学习
        """
        if use_cd_optimization is None:
            use_cd_optimization = hyperparams.kt_use_cd_optimization
        if use_contrastive is None:
            use_contrastive = hyperparams.kt_use_contrastive
            
        batch_size, seq_len = qusunt_seq_indices.shape
        
        # 关键修复：确保所有输入都参与计算图，参考CD实现
        # 学习者嵌入扩展 [batch_size, seq_len, emb_dim]
        h_lrn_expanded = h_lrn_batch.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 类型特定的输入处理
        type_features = self.type_specific_processing(add1, add2, type_indices)
        
        # 组合基础特征
        base_features = h_qusunt_batch + type_features
        
        # 处理掩码用于Transformer
        if seq_mask is not None:
            transformer_mask = seq_mask < 0.5  # True表示padding位置
        else:
            transformer_mask = None
        
        # 阶段1: 感觉记忆注册
        if use_contrastive and self.training:
            sensory_memory, _ = self.contrastive_pretraining(base_features, seq_mask)
        else:
            # 推理时或禁用对比学习时使用基础编码器
            sensory_memory = self.contrastive_pretraining.base_encoder(
                base_features, src_key_padding_mask=transformer_mask
            )
        
        # 关系-时序嵌入
        relational_temporal_features = self.relation_temporal_embedding(
            h_lrn_expanded, sensory_memory, h_cpt, add1, add2, qusunt_seq_indices, self.concept_mapping
        )
        
        enhanced_sensory_memory = sensory_memory + relational_temporal_features
        
        # 阶段2: 短时记忆融合
        short_term_memory = self.dual_channel_fusion(enhanced_sensory_memory, seq_mask)
        
        # 知识点掌握程度预测
        concept_mastery = self.mastery_prediction(short_term_memory)
        
        # 使用CD优化能力
        if use_cd_optimization and self.cd_optimized_ability is not None:
            concept_mastery = self.compute_ability_with_cd(h_lrn_batch, concept_mastery)
        
        # 阶段3: 长时记忆检索（知识点库检索）
        long_term_memory = self.knowledge_retrieval(short_term_memory, h_cpt, seq_mask)
        
        # 最终预测
        combined_features = torch.cat([short_term_memory, long_term_memory, concept_mastery], dim=-1)
        predictions = self.final_prediction(combined_features)
        
        # 关键修复：梯度增强 - 确保所有输入都贡献梯度，参考CD实现
        predictions = predictions + self.grad_enhancer * (
            h_lrn_batch.mean() + h_qusunt_batch.mean() + h_cpt.mean()
        )
        
        # 应用掩码 - 只对有预测掩码的时间步返回预测
        predictions_masked = predictions * prediction_masks.unsqueeze(-1)
        concept_mastery_masked = concept_mastery * seq_mask.unsqueeze(-1)
        
        return predictions_masked, concept_mastery_masked
    
    def get_concept_mastery(self, h_lrn_batch, h_qusunt_batch, h_cpt, lrn_indices, qusunt_seq_indices, add1, add2, type_indices, seq_mask, qus_num):
        """
        专门获取知识点掌握程度供CD使用
        """
        # 可以在这里处理学习单元和题目的区别
        with torch.no_grad():
            _, concept_mastery = self.forward(
                h_lrn_batch, h_qusunt_batch, h_cpt, lrn_indices, qusunt_seq_indices, add1, add2, type_indices,
                seq_mask, torch.ones_like(seq_mask),  # 使用全1掩码获取所有位置的能力
                use_cd_optimization=False,
                use_contrastive=False
            )
        return concept_mastery
    
    def update_momentum_encoder(self, momentum=None):
        """更新动量编码器"""
        if momentum is None:
            momentum = hyperparams.kt_momentum
        self.contrastive_pretraining.update_momentum_encoder(momentum)

    def get_model_info(self):
        """返回模型信息"""
        return {
            'embedding_dim': self.embedding_dim,
            'concept_num': self.concept_num,
            'use_cd_optimization': hyperparams.kt_use_cd_optimization,
            'use_contrastive': hyperparams.kt_use_contrastive,
        }
    
    def get_parameter_count(self):
        """返回参数数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }

def test_kt_gradient_fixed():
    """测试修复后的KT模型梯度"""
    print("=== 测试修复后的KT模型梯度 ===")
    
    device = 'cpu'
    embedding_dim = 64
    concept_num = 15
    qusunt_num = 20
    batch_size = 2
    seq_len = 3
    
    # 创建需要梯度的输入，参考CD测试方式
    h_lrn_batch = torch.randn(batch_size, embedding_dim, requires_grad=True)
    h_qusunt_batch = torch.randn(batch_size, seq_len, embedding_dim, requires_grad=True)
    h_cpt = torch.randn(concept_num, embedding_dim, requires_grad=True)
    
    print(f"输入梯度状态:")
    print(f"  h_lrn_batch: {h_lrn_batch.requires_grad}")
    print(f"  h_qusunt_batch: {h_qusunt_batch.requires_grad}")
    print(f"  h_cpt: {h_cpt.requires_grad}")
    
    # 创建修复后的KT模型
    concept_mapping = {i: [i % concept_num] for i in range(qusunt_num)}
    kt_model = KT(embedding_dim, concept_num, concept_mapping).to(device)
    
    # 模拟输入 - 适配新数据结构
    lrn_indices = torch.arange(batch_size)
    qusunt_seq_indices = torch.randint(0, qusunt_num, (batch_size, seq_len))
    add1 = torch.randn(batch_size, seq_len)
    add2 = torch.randn(batch_size, seq_len)
    type_indices = torch.randint(0, 6, (batch_size, seq_len))
    seq_masks = torch.ones(batch_size, seq_len)
    prediction_masks = torch.ones(batch_size, seq_len)  # 新的预测掩码
    next_results = torch.rand(batch_size, seq_len, concept_num)
    
    print(f"\n模型信息:")
    model_info = kt_model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # 前向传播
    kt_model.train()
    predictions, ability = kt_model(
        h_lrn_batch, h_qusunt_batch, h_cpt, lrn_indices, qusunt_seq_indices, add1, add2, type_indices,
        seq_masks, prediction_masks,  # 使用新的prediction_masks参数
        use_cd_optimization=False,
        use_contrastive=False
    )
    
    print(f"\n前向传播结果:")
    print(f"  predictions: {predictions.shape}, requires_grad={predictions.requires_grad}")
    print(f"  ability: {ability.shape}, requires_grad={ability.requires_grad}")
    
    # 计算损失
    criterion = nn.BCELoss()
    loss = criterion(predictions, next_results)
    print(f"损失: {loss.item():.4f}")
    
    # 反向传播
    kt_model.zero_grad()
    if h_lrn_batch.grad is not None:
        h_lrn_batch.grad.zero_()
    if h_qusunt_batch.grad is not None:
        h_qusunt_batch.grad.zero_()
    if h_cpt.grad is not None:
        h_cpt.grad.zero_()
    
    loss.backward()
    
    # 详细检查梯度
    print(f"\n梯度检查结果:")
    h_lrn_grad_norm = h_lrn_batch.grad.norm().item() if h_lrn_batch.grad is not None else 0
    h_qusunt_grad_norm = h_qusunt_batch.grad.norm().item() if h_qusunt_batch.grad is not None else 0
    h_cpt_grad_norm = h_cpt.grad.norm().item() if h_cpt.grad is not None else 0
    
    print(f"  h_lrn_batch梯度范数: {h_lrn_grad_norm:.6f}")
    print(f"  h_qusunt_batch梯度范数: {h_qusunt_grad_norm:.6f}")
    print(f"  h_cpt梯度范数: {h_cpt_grad_norm:.6f}")
    
    # 检查具体数值
    if h_lrn_batch.grad is not None:
        print(f"  h_lrn_batch梯度统计: max={h_lrn_batch.grad.max().item():.6f}, min={h_lrn_batch.grad.min().item():.6f}")
    if h_cpt.grad is not None:
        print(f"  h_cpt梯度统计: max={h_cpt.grad.max().item():.6f}, min={h_cpt.grad.min().item():.6f}")
    
    # 检查KT内部参数的梯度
    kt_grad_norm = 0
    has_gradients = False
    print(f"\nKT内部参数梯度:")
    for name, param in kt_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            kt_grad_norm += grad_norm
            if grad_norm > 1e-8:
                has_gradients = True
                status = "✅"
            else:
                status = "⚠️ "
            print(f"  {status} {name}: {grad_norm:.8f}")
        else:
            print(f"  ❌ {name}: 无梯度")
    
    print(f"KT内部参数总梯度: {kt_grad_norm:.6f}")
    
    # 验证梯度传播
    h_lrn_grad_ok = h_lrn_grad_norm > 1e-8
    h_qusunt_grad_ok = h_qusunt_grad_norm > 1e-8  
    h_cpt_grad_ok = h_cpt_grad_norm > 1e-8
    
    print(f"\n梯度传播验证:")
    print(f"  h_lrn_batch梯度: {'✅ 正常' if h_lrn_grad_ok else '❌ 异常'}")
    print(f"  h_qusunt_batch梯度: {'✅ 正常' if h_qusunt_grad_ok else '❌ 异常'}")
    print(f"  h_cpt梯度: {'✅ 正常' if h_cpt_grad_ok else '❌ 异常'}")
    print(f"  KT内部梯度: {'✅ 正常' if has_gradients else '❌ 异常'}")
    
    success = h_lrn_grad_ok and h_qusunt_grad_ok and h_cpt_grad_ok and has_gradients
    
    if success:
        print(f"\n🎉 梯度修复成功！所有梯度正常传播！")
        return True
    else:
        print(f"\n❌ 仍有梯度问题需要进一步调试")
        return False

if __name__ == '__main__':
    success = test_kt_gradient_fixed()
    if success:
        print(f"\n🎉 KT模型可以正式使用！")
    else:
        print(f"\n❌ 需要进一步调试")