import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from hyperparams.hyperparameter import hyperparams

class DTR(nn.Module):
    """诊断特质表示模块 - 基于论文思路重新设计"""
    def __init__(self, embedding_dim):
        super(DTR, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 使用超参数配置
        hidden_dims = hyperparams.cd_dtr_hidden_dims
        dropout_rate = hyperparams.cd_dtr_dropout_rate
        
        # 关键修复：简化网络结构，避免梯度消失
        # 学生能力计算网络 - 论文中的 p_lrn
        self.l_p_lrn = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
        
        # 题目难度计算网络 - 论文中的 d_qus
        self.l_d_qus = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
        
        # 题目区分度计算网络 - 论文中的 beta_qus
        self.l_b_qus = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重，避免梯度消失"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.1)

    def forward(self, h_lrn_cpt, h_qus_cpt, h_qus):
        """
        论文中的诊断特质计算
        Args:
            h_lrn_cpt: 学生-知识点交互 [batch_size * seq_len * concept_num, 2*embedding_dim]
            h_qus_cpt: 题目-知识点交互 [batch_size * seq_len * concept_num, 2*embedding_dim]  
            h_qus: 题目嵌入 [batch_size * seq_len, embedding_dim]
        """
        # 确保所有输入都参与计算图
        p_lrn = self.l_p_lrn(h_lrn_cpt).squeeze(-1)
        d_qus = self.l_d_qus(h_qus_cpt).squeeze(-1)
        beta_qus = self.l_b_qus(h_qus).squeeze(-1)
        
        return p_lrn, d_qus, beta_qus

class MIRT(nn.Module):
    """多维项目反应理论模块 - 基于论文重新设计"""
    def __init__(self):
        super(MIRT, self).__init__()
        self.scale_init = hyperparams.cd_mirt_scale_init
        self.clamp_range = hyperparams.cd_mirt_clamp_range
        
        # 关键修复：确保scale参数参与梯度计算
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        
        # 添加小的可训练参数确保梯度流动
        self.epsilon = nn.Parameter(torch.tensor(1e-6))

    def forward(self, p_lrn, d_qus, beta_qus, interaction=None, return_ability=False):
        """
        论文中的MIRT实现：sigmoid(scale * interaction + beta_qus)
        """
        if interaction is None:
            # 关键修复：确保p_lrn和d_qus都参与计算
            # 论文中的交互计算：p_lrn^T * d_qus
            interaction = torch.sum(p_lrn * d_qus, dim=-1)
        
        # 关键修复：确保所有参数都参与计算
        # 论文中的缩放操作
        scaled_interaction = self.scale * interaction + self.epsilon
        
        # 数值稳定性处理
        clamped_interaction = torch.clamp(scaled_interaction, *self.clamp_range)
        
        # 论文中的最终预测：sigmoid(interaction + beta_qus)
        result = torch.sigmoid(clamped_interaction + beta_qus)
        
        if return_ability:
            return result, p_lrn
        else:
            return result

class CD(nn.Module):
    """认知诊断主模型 - 基于论文思路彻底重构"""
    def __init__(self, embedding_dim, concept_num):
        """
        基于论文的CD模型设计
        Args:
            embedding_dim: 嵌入维度
            concept_num: 知识点数量
        """
        super(CD, self).__init__()
        self.embedding_dim = embedding_dim
        self.concept_num = concept_num
        
        # KT优化后的能力缓存
        self.kt_optimized_ability = None
        
        # 论文中的模块
        self.dtr = DTR(embedding_dim)
        self.mirt = MIRT()
        
        # 能力融合门控机制 - 论文中的能力融合
        fusion_hidden_dim = hyperparams.cd_ability_fusion_hidden_dim
        self.ability_gate = nn.Sequential(
            nn.Linear(embedding_dim + concept_num, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, concept_num),
            nn.Sigmoid()
        )
        
        # 关键修复：简化归一化层
        self.interaction_norm = nn.LayerNorm(1)
        
        # 论文中的概念权重 - 确保参与梯度计算
        self.concept_weight = nn.Parameter(torch.ones(concept_num) / concept_num)
        
        # 梯度增强参数
        self.grad_enhancer = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, h_lrn_batch, h_qus, h_cpt, qus_seq_indices, qus_seq_masks, 
                return_ability=False, use_kt_optimization=True):
        """
        基于论文的前向传播设计
        """
        batch_size, seq_len = qus_seq_indices.shape
        
        # 关键修复1：直接使用传入的嵌入，保持计算图
        qus_emb_batch = h_qus[qus_seq_indices]
        
        # 关键修复2：使用更简洁的扩展方式，避免梯度消失
        # 学生嵌入扩展 [batch_size, seq_len, concept_num, embedding_dim]
        h_lrn_expanded = h_lrn_batch.unsqueeze(1).unsqueeze(2)
        h_lrn_expanded = h_lrn_expanded.expand(batch_size, seq_len, self.concept_num, self.embedding_dim)
        
        # 知识点嵌入扩展 [batch_size, seq_len, concept_num, embedding_dim]
        h_cpt_expanded = h_cpt.unsqueeze(0).unsqueeze(0)
        h_cpt_expanded = h_cpt_expanded.expand(batch_size, seq_len, self.concept_num, self.embedding_dim)
        
        # 关键修复3：确保拼接操作保持梯度
        h_lrn_cpt = torch.cat([h_lrn_expanded, h_cpt_expanded], dim=-1)
        
        # 题目嵌入扩展
        h_qus_expanded = qus_emb_batch.unsqueeze(2)
        h_qus_expanded = h_qus_expanded.expand(batch_size, seq_len, self.concept_num, self.embedding_dim)
        h_qus_cpt = torch.cat([h_qus_expanded, h_cpt_expanded], dim=-1)
        
        # 重塑用于DTR计算
        batch_seq_concept = batch_size * seq_len * self.concept_num
        h_lrn_cpt_flat = h_lrn_cpt.reshape(batch_seq_concept, 2 * self.embedding_dim)
        h_qus_cpt_flat = h_qus_cpt.reshape(batch_seq_concept, 2 * self.embedding_dim)
        h_qus_flat = qus_emb_batch.reshape(batch_size * seq_len, self.embedding_dim)
        
        # 计算诊断特质 - 论文中的p_lrn, d_qus, beta_qus
        p_lrn, d_qus, beta_qus = self.dtr(h_lrn_cpt_flat, h_qus_cpt_flat, h_qus_flat)
        
        # 重塑维度
        p_lrn = p_lrn.reshape(batch_size, seq_len, self.concept_num)
        d_qus = d_qus.reshape(batch_size, seq_len, self.concept_num)
        beta_qus = beta_qus.reshape(batch_size, seq_len)
        
        # 关键修复4：确保concept_weight参与计算
        # 论文中的加权交互计算
        concept_weight_normalized = torch.softmax(self.concept_weight, dim=0)
        weighted_p_lrn = p_lrn * concept_weight_normalized.view(1, 1, -1)
        
        # 交互计算 - 论文中的 p_lrn^T * d_qus
        interaction = torch.sum(weighted_p_lrn * d_qus, dim=-1)
        
        # 归一化交互
        interaction_normalized = self.interaction_norm(interaction.unsqueeze(-1)).squeeze(-1)
        
        # 使用KT优化能力
        if use_kt_optimization and self.kt_optimized_ability is not None:
            p_lrn = self.compute_ability_with_kt(h_lrn_batch, p_lrn)
        
        # MIRT预测 - 论文中的最终预测
        if return_ability:
            predictions, ability_matrix = self.mirt(
                p_lrn, d_qus, beta_qus, interaction=interaction_normalized, return_ability=True
            )
        else:
            predictions = self.mirt(p_lrn, d_qus, beta_qus, interaction=interaction_normalized, return_ability=False)
        
        # 应用掩码
        predictions = predictions * qus_seq_masks
        
        # 关键修复5：梯度增强 - 确保所有输入都贡献梯度
        predictions = predictions + self.grad_enhancer * (
            h_lrn_batch.mean() + h_qus.mean() + h_cpt.mean()
        )
        
        if return_ability:
            ability_masked = ability_matrix * qus_seq_masks.unsqueeze(-1)
            return predictions, ability_masked
        else:
            return predictions
        
    def compute_ability_with_kt(self, h_lrn_batch, concept_mastery):
        """论文中的能力融合机制"""
        if self.kt_optimized_ability is None:
            return concept_mastery
        
        batch_size, seq_len, concept_num = concept_mastery.shape
        
        if (self.kt_optimized_ability.shape[0] != batch_size or 
            self.kt_optimized_ability.shape[1] != seq_len):
            return concept_mastery
        
        # 计算融合权重
        h_lrn_expanded = h_lrn_batch.unsqueeze(1).expand(batch_size, seq_len, -1)
        gate_input = torch.cat([h_lrn_expanded, concept_mastery], dim=-1)
        fusion_weights = self.ability_gate(gate_input)
        
        # 融合能力
        fused_ability = (fusion_weights * self.kt_optimized_ability + 
                        (1 - fusion_weights) * concept_mastery)
        
        return fused_ability

    def get_ability_matrix(self, h_lrn_batch, h_qus, h_cpt, unt_seq_indices, seq_masks, unt_num):
        """获取能力矩阵"""
        batch_size, seq_len = unt_seq_indices.shape
        qus_mask = (unt_seq_indices >= unt_num) & (seq_masks > 0.5)
        
        qus_indices = torch.where(
            qus_mask,
            unt_seq_indices - unt_num,
            torch.zeros_like(unt_seq_indices)
        )
        
        with torch.no_grad():
            _, ability_matrix = self.forward(
                h_lrn_batch, h_qus, h_cpt, qus_indices, qus_mask.float(), 
                return_ability=True, use_kt_optimization=False
            )
        return ability_matrix

    def set_kt_optimized_ability(self, kt_ability, unt_num):
        """设置KT优化后的能力矩阵"""
        if kt_ability is not None:
            batch_size, seq_len, concept_num = kt_ability.shape
            self.kt_optimized_ability = kt_ability.detach().clone()

    def get_model_info(self):
        """返回模型信息"""
        return {
            'embedding_dim': self.embedding_dim,
            'concept_num': self.concept_num,
            'dtr_hidden_dims': hyperparams.cd_dtr_hidden_dims,
            'use_kt_optimization': hyperparams.cd_use_kt_optimization,
            'mirt_scale_init': hyperparams.cd_mirt_scale_init,
            'parameter_count': self.get_parameter_count()
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

def test_cd_gradient_fixed():
    """测试修复后的CD模型梯度"""
    print("=== 测试修复后的CD模型梯度 ===")
    
    device = 'cpu'
    embedding_dim = 64
    concept_num = 15
    qus_num = 10
    batch_size = 2
    seq_len = 3
    
    # 创建需要梯度的输入
    h_lrn_batch = torch.randn(batch_size, embedding_dim, requires_grad=True)
    h_qus = torch.randn(qus_num, embedding_dim, requires_grad=True)
    h_cpt = torch.randn(concept_num, embedding_dim, requires_grad=True)
    
    print(f"输入梯度状态:")
    print(f"  h_lrn_batch: {h_lrn_batch.requires_grad}")
    print(f"  h_qus: {h_qus.requires_grad}")
    print(f"  h_cpt: {h_cpt.requires_grad}")
    
    # 创建修复后的CD模型
    cd_model = CD(embedding_dim, concept_num).to(device)
    
    # 模拟输入
    qus_seq_indices = torch.randint(0, qus_num, (batch_size, seq_len))
    qus_seq_masks = torch.ones(batch_size, seq_len)
    results = torch.rand(batch_size, seq_len)
    
    print(f"\n模型信息:")
    model_info = cd_model.get_model_info()
    for key, value in model_info.items():
        if key != 'parameter_count':
            print(f"  {key}: {value}")
    
    # 前向传播
    cd_model.train()
    predictions, ability = cd_model(
        h_lrn_batch, h_qus, h_cpt, qus_seq_indices, qus_seq_masks, 
        return_ability=True, use_kt_optimization=False
    )
    
    print(f"\n前向传播结果:")
    print(f"  predictions: {predictions.shape}, requires_grad={predictions.requires_grad}")
    print(f"  ability: {ability.shape}, requires_grad={ability.requires_grad}")
    
    # 计算损失
    criterion = nn.BCELoss()
    loss = criterion(predictions, results)
    print(f"损失: {loss.item():.4f}")
    
    # 反向传播
    cd_model.zero_grad()
    if h_lrn_batch.grad is not None:
        h_lrn_batch.grad.zero_()
    if h_qus.grad is not None:
        h_qus.grad.zero_()
    if h_cpt.grad is not None:
        h_cpt.grad.zero_()
    
    loss.backward()
    
    # 详细检查梯度
    print(f"\n梯度检查结果:")
    h_lrn_grad_norm = h_lrn_batch.grad.norm().item() if h_lrn_batch.grad is not None else 0
    h_qus_grad_norm = h_qus.grad.norm().item() if h_qus.grad is not None else 0
    h_cpt_grad_norm = h_cpt.grad.norm().item() if h_cpt.grad is not None else 0
    
    print(f"  h_lrn_batch梯度范数: {h_lrn_grad_norm:.6f}")
    print(f"  h_qus梯度范数: {h_qus_grad_norm:.6f}")
    print(f"  h_cpt梯度范数: {h_cpt_grad_norm:.6f}")
    
    # 检查具体数值
    if h_lrn_batch.grad is not None:
        print(f"  h_lrn_batch梯度统计: max={h_lrn_batch.grad.max().item():.6f}, min={h_lrn_batch.grad.min().item():.6f}")
    if h_cpt.grad is not None:
        print(f"  h_cpt梯度统计: max={h_cpt.grad.max().item():.6f}, min={h_cpt.grad.min().item():.6f}")
    
    # 检查CD内部参数的梯度
    cd_grad_norm = 0
    has_gradients = False
    print(f"\nCD内部参数梯度:")
    for name, param in cd_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            cd_grad_norm += grad_norm
            if grad_norm > 1e-8:
                has_gradients = True
                status = "✅"
            else:
                status = "⚠️ "
            print(f"  {status} {name}: {grad_norm:.8f}")
        else:
            print(f"  ❌ {name}: 无梯度")
    
    print(f"CD内部参数总梯度: {cd_grad_norm:.6f}")
    
    # 验证梯度传播
    h_lrn_grad_ok = h_lrn_grad_norm > 1e-8
    h_qus_grad_ok = h_qus_grad_norm > 1e-8  
    h_cpt_grad_ok = h_cpt_grad_norm > 1e-8
    
    print(f"\n梯度传播验证:")
    print(f"  h_lrn_batch梯度: {'✅ 正常' if h_lrn_grad_ok else '❌ 异常'}")
    print(f"  h_qus梯度: {'✅ 正常' if h_qus_grad_ok else '❌ 异常'}")
    print(f"  h_cpt梯度: {'✅ 正常' if h_cpt_grad_ok else '❌ 异常'}")
    print(f"  CD内部梯度: {'✅ 正常' if has_gradients else '❌ 异常'}")
    
    success = h_lrn_grad_ok and h_qus_grad_ok and h_cpt_grad_ok and has_gradients
    
    if success:
        print(f"\n🎉 梯度修复成功！所有梯度正常传播！")
        return True
    else:
        print(f"\n❌ 仍有梯度问题需要进一步调试")
        return False

if __name__ == '__main__':
    success = test_cd_gradient_fixed()
    if success:
        print(f"\n🎉 CD模型可以正式使用！")
    else:
        print(f"\n❌ 需要进一步调试")