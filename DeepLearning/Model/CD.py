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
    """认知诊断主模型 - 修复版本"""
    def __init__(self, embedding_dim, concept_num, h_qus, h_cpt):
        super(CD, self).__init__()
        self.embedding_dim = embedding_dim
        self.concept_num = concept_num
        
        # 注册为parameter
        self.h_qus = nn.Parameter(h_qus, requires_grad=True)
        self.h_cpt = nn.Parameter(h_cpt, requires_grad=True)
        
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
        
        # 关键修复：添加归一化层
        self.interaction_norm = nn.LayerNorm(1)  # 用于归一化interaction
        self.concept_weight = nn.Parameter(torch.ones(concept_num) / concept_num)  # 知识点权重

    def forward(self, h_lrn_batch, qus_seq_indices, qus_seq_masks, 
                return_ability=False, use_kt_optimization=True):
        """
        修复版本：解决interaction过大问题
        """
        batch_size, seq_len = qus_seq_indices.shape
        
        # 获取题目嵌入
        qus_emb_batch = self.h_qus[qus_seq_indices]
        
        # 构造h_lrn_cpt和h_qus_cpt
        h_lrn_expanded = h_lrn_batch.unsqueeze(1).unsqueeze(2)
        h_lrn_expanded = h_lrn_expanded.repeat(1, seq_len, self.concept_num, 1)
        
        h_cpt_expanded = self.h_cpt.unsqueeze(0).unsqueeze(0)
        h_cpt_expanded = h_cpt_expanded.repeat(batch_size, seq_len, 1, 1)
        
        h_lrn_cpt = torch.cat([h_lrn_expanded, h_cpt_expanded], dim=-1)
        
        h_qus_expanded = qus_emb_batch.unsqueeze(2)
        h_qus_expanded = h_qus_expanded.repeat(1, 1, self.concept_num, 1)
        h_qus_cpt = torch.cat([h_qus_expanded, h_cpt_expanded], dim=-1)
        
        # 计算诊断特质
        p_lrn, d_qus, beta_qus = self.dtr(
            h_lrn_cpt.reshape(-1, 2 * self.embedding_dim),
            h_qus_cpt.reshape(-1, 2 * self.embedding_dim),
            qus_emb_batch.reshape(-1, self.embedding_dim)
        )
        
        # 重塑维度
        p_lrn = p_lrn.reshape(batch_size, seq_len, self.concept_num)
        d_qus = d_qus.reshape(batch_size, seq_len, self.concept_num)
        beta_qus = beta_qus.reshape(batch_size, seq_len)
        
        # 关键修复：使用加权的interaction，避免数值过大
        weighted_interaction = torch.sum(
            p_lrn * d_qus * self.concept_weight.unsqueeze(0).unsqueeze(0), 
            dim=-1
        )
        
        # 归一化interaction
        interaction_normalized = self.interaction_norm(weighted_interaction.unsqueeze(-1)).squeeze(-1)
        
        # 使用KT优化能力
        if use_kt_optimization and self.kt_optimized_ability is not None:
            p_lrn = self.compute_ability_with_kt(h_lrn_batch, p_lrn)
        
        # MIRT预测
        if return_ability:
            # 关键修复：使用归一化的interaction
            predictions, ability_matrix = self.mirt(
                p_lrn, d_qus, beta_qus, interaction=interaction_normalized, return_ability=True
            )
        else:
            predictions = self.mirt(p_lrn, d_qus, beta_qus, interaction=interaction_normalized, return_ability=False)
        
        # 应用掩码
        predictions = predictions * qus_seq_masks
        
        if return_ability:
            ability_masked = ability_matrix * qus_seq_masks.unsqueeze(-1)
            return predictions, ability_masked
        else:
            return predictions
        
    def get_ability_matrix(self, h_lrn_batch, unt_seq_indices, seq_masks, unt_num):
        """
        获取能力矩阵 - 修复版本
        Args:
            h_lrn_batch: (batch_size, emb_dim) 学习者嵌入
            unt_seq_indices: (batch_size, seq_len) 完整的学习单元+题目索引
            seq_masks: (batch_size, seq_len) 序列掩码
            unt_num: 学习单元数量，用于索引转换
        """
        # 过滤出题目索引（索引 >= unt_num 的才是题目）
        batch_size, seq_len = unt_seq_indices.shape
        qus_mask = (unt_seq_indices >= unt_num) & (seq_masks > 0.5)
        
        # 转换为CD题目索引
        qus_indices = torch.where(
            qus_mask,
            unt_seq_indices - unt_num,  # 转换为0~qus_num-1
            torch.zeros_like(unt_seq_indices)  # 非题目位置设为0（会被掩码过滤）
        )
        
        with torch.no_grad():
            _, ability_matrix = self.forward(
                h_lrn_batch, qus_indices, qus_mask.float(), 
                return_ability=True, use_kt_optimization=False
            )
        return ability_matrix

    def set_kt_optimized_ability(self, kt_ability, unt_num):
        """
        设置KT优化后的能力矩阵 - 修复版本
        Args:
            kt_ability: (batch_size, seq_len, concept_num) KT优化的能力
            unt_num: 学习单元数量，用于验证维度
        """
        if kt_ability is not None:
            # 验证维度
            batch_size, seq_len, concept_num = kt_ability.shape
            self.kt_optimized_ability = kt_ability.detach().clone()

class MIRT(nn.Module):
    """修复的多维项目反应理论模块"""
    def __init__(self):
        super(MIRT, self).__init__()
        # 添加缩放参数
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, p_lrn, d_qus, beta_qus, interaction=None, return_ability=False):
        """
        Args:
            interaction: 可选的预计算interaction（修复版本）
        """
        if interaction is None:
            # 原始计算方式
            interaction = torch.sum(p_lrn * d_qus, dim=-1)
        
        # 关键修复：使用缩放和clamp避免数值问题
        scaled_interaction = self.scale * interaction
        clamped_interaction = torch.clamp(scaled_interaction, -20, 20)  # 防止sigmoid饱和
        
        result = torch.sigmoid(clamped_interaction + beta_qus)
        
        if return_ability:
            return result, p_lrn
        else:
            return result

if __name__ == '__main__':
    def test_cd_fixed_version():
        """测试修复后的CD模型"""
        print("=== CD模型修复版本测试 ===")
        
        # 基础设置
        from DataReader.HGCDataReader import hgcdr
        hgcdr.loadDatafromSql()
        device = 'cpu'
        
        from Model.HGC import HGC
        from DataReader.CDDataReader import cddr
        from DataSet.CDDataSet import CDDataset
        from torch.utils.data import DataLoader
        
        lrn_input_dim = hgcdr.lrn_init.shape[1]
        unt_input_dim = hgcdr.untqus_init.shape[1]
        cpt_input_dim = hgcdr.cpt_init.shape[1]
        
        model_hgc = HGC(embedding_dim=64, lrn_input_dim=lrn_input_dim,
                       unt_input_dim=unt_input_dim, cpt_input_dim=cpt_input_dim).to(device)
        
        cddata = cddr.loadDatafromSql()
        
        # 计算HGC嵌入
        with torch.no_grad():
            lrn_emb, unt_emb, cpt_emb = model_hgc(hgcdr, device)
        
        # 创建数据集
        train_dataset = CDDataset(cddata, lrn_emb, unt_emb, cpt_emb, 'train', max_seq_len=128)
        batch_size = 4
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                collate_fn=train_dataset.collate_fn)
        batch = next(iter(train_loader))
        
        # 创建修复后的CD模型
        unt_num = unt_emb.shape[0] - len(cddata['qus_uid'])
        h_qus = unt_emb[unt_num:]
        
        cd_model_fixed = CD(embedding_dim=64, concept_num=len(cddata['cpt_uid']),
                           h_qus=h_qus, h_cpt=cpt_emb).to(device)
        
        print("\n1. 修复版本诊断")
        h_lrn_batch = batch['h_lrn_batch'].to(device)
        qus_seq_indices = batch['qus_seq_indices'].to(device)
        qus_seq_masks = batch['qus_seq_masks'].to(device)
        
        # 前向传播
        predictions, ability = cd_model_fixed(
            h_lrn_batch, qus_seq_indices, qus_seq_masks, 
            return_ability=True, use_kt_optimization=False
        )
        
        print(f"  预测值范围: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"  预测值均值: {predictions.mean():.3f}")
        
        # 检查有效位置
        valid_mask = qus_seq_masks > 0.5
        valid_predictions = predictions[valid_mask]
        valid_results = batch['results'].to(device)[valid_mask]
        
        print(f"  有效预测数量: {len(valid_predictions)}")
        if len(valid_predictions) > 0:
            print(f"  有效预测范围: [{valid_predictions.min():.3f}, {valid_predictions.max():.3f}]")
            
            # 损失计算
            criterion = nn.BCELoss()
            loss = criterion(valid_predictions, valid_results)
            print(f"  损失值: {loss.item():.6f}")
            
            # 梯度测试
            cd_model_fixed.zero_grad()
            loss.backward()
            
            # 检查梯度
            has_grad = False
            for name, param in cd_model_fixed.named_parameters():
                if param.grad is not None and param.grad.norm().item() > 1e-10:
                    has_grad = True
                    print(f"  ✓ {name} 有梯度: {param.grad.norm().item():.8f}")
                    break
            
            if has_grad:
                print("  ✓ 梯度计算正常")
                
                # 参数更新测试
                optimizer = torch.optim.Adam(cd_model_fixed.parameters(), lr=0.001)
                optimizer.step()
                print("  ✓ 参数更新成功")
            else:
                print("  ✗ 仍然没有梯度")
        else:
            print("  ✗ 没有有效预测位置")
        
        print("\n=== 修复测试完成 ===")
    
    # 运行修复测试
    test_cd_fixed_version()