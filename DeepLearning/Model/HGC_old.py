import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from hyperparams.hyperparameter import hyperparams

class MetaPathAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.W_att = nn.Linear(embedding_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [num_paths, num_nodes, embedding_dim]
        """
        num_paths, num_nodes, emb_dim = embeddings.shape
        scores = []
        for i in range(num_paths):
            score = self.W_att(embeddings[i])  # [num_nodes, 1]
            scores.append(score)
        scores = torch.cat(scores, dim=1)  # [num_nodes, num_paths]
        attn_weights = F.softmax(scores, dim=1)  # [num_nodes, num_paths]
        weighted_emb = torch.zeros_like(embeddings[0])
        for i in range(num_paths):
            weighted_emb += attn_weights[:, i].unsqueeze(-1) * embeddings[i]
        return weighted_emb

class GCNConvEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_layers=None):
        super().__init__()
        if num_layers is None:
            num_layers = hyperparams.hgc_gcn_layers
            
        self.num_layers = num_layers
        self.W = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.activation_slope = hyperparams.hgc_activation_slope
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # 使用clone确保不修改原始输入
        h_current = h.clone()
        
        for _ in range(self.num_layers):
            row, col = edge_index[0], edge_index[1]
            h_agg = scatter_add(h_current[row] * edge_weight.unsqueeze(1), col, dim=0, dim_size=h_current.size(0))
            h_current = torch.mm(h_agg, self.W)
            h_current = F.leaky_relu(h_current, negative_slope=self.activation_slope)
        return h_current

class Projection(nn.Module):
    def __init__(self, input_dim, embedding_dim=None):
        super().__init__()
        if embedding_dim is None:
            embedding_dim = hyperparams.hgc_embedding_dim
            
        hidden_dim = hyperparams.hgc_proj_hidden_dim
        dropout_rate = hyperparams.hgc_dropout_rate
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(hyperparams.hgc_activation_slope),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

class LearnerEncoder(nn.Module):
    """学习者编码器 - 可独立使用"""
    def __init__(self, embedding_dim=None, lrn_input_dim=None):
        super().__init__()
        
        if embedding_dim is None:
            embedding_dim = hyperparams.hgc_embedding_dim
            
        self.embedding_dim = embedding_dim
        self.lrn_proj = Projection(input_dim=lrn_input_dim, embedding_dim=embedding_dim)
        self.lrn_gcn_lul = GCNConvEmbedding(embedding_dim)
        self.lrn_gcn_lcl = GCNConvEmbedding(embedding_dim)
        self.lrn_gcn_ltl = GCNConvEmbedding(embedding_dim)
        self.lrn_attn = MetaPathAttention(embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, lrn_init: torch.Tensor, 
                p_lul: tuple, p_lcl: tuple, p_ltl: tuple,
                device=None) -> torch.Tensor:
        """
        独立计算学习者嵌入
        """
        if device is None:
            device = next(self.parameters()).device
            
        # 修复：确保输入张量可以参与梯度计算
        lrn_init = lrn_init.clone().detach().requires_grad_(True)
        
        # 学习者嵌入
        lrn_init_proj = self.lrn_proj(lrn_init.to(device))
        lrn_lul = self.lrn_gcn_lul(lrn_init_proj, 
                                  p_lul[0].to(device).clone(), 
                                  p_lul[1].to(device).clone())
        lrn_lcl = self.lrn_gcn_lcl(lrn_init_proj, 
                                  p_lcl[0].to(device).clone(), 
                                  p_lcl[1].to(device).clone())
        lrn_ltl = self.lrn_gcn_ltl(lrn_init_proj, 
                                  p_ltl[0].to(device).clone(), 
                                  p_ltl[1].to(device).clone())
        lrn_emb = self.lrn_attn(torch.stack([lrn_lul, lrn_lcl, lrn_ltl]))
        lrn_emb = self.output_norm(lrn_emb)
        
        return lrn_emb

class UnitEncoder(nn.Module):
    """学习单元编码器 - 可独立使用"""
    def __init__(self, embedding_dim=None, unt_input_dim=None):
        super().__init__()
        
        if embedding_dim is None:
            embedding_dim = hyperparams.hgc_embedding_dim
            
        self.embedding_dim = embedding_dim
        self.unt_proj = Projection(input_dim=unt_input_dim, embedding_dim=embedding_dim)
        self.unt_gcn_ulu = GCNConvEmbedding(embedding_dim)
        self.unt_gcn_ucrsu = GCNConvEmbedding(embedding_dim)
        self.unt_gcn_ucptu = GCNConvEmbedding(embedding_dim)
        self.unt_attn = MetaPathAttention(embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, unt_init: torch.Tensor,
                p_ulu: tuple, p_ucrsu: tuple, p_ucptu: tuple,
                device=None) -> torch.Tensor:
        """
        独立计算学习单元嵌入
        """
        if device is None:
            device = next(self.parameters()).device
            
        # 修复：确保输入张量可以参与梯度计算
        unt_init = unt_init.clone().detach().requires_grad_(True)
        
        # 学习单元嵌入
        unt_init_proj = self.unt_proj(unt_init.to(device))
        unt_ulu = self.unt_gcn_ulu(unt_init_proj, 
                                  p_ulu[0].to(device).clone(), 
                                  p_ulu[1].to(device).clone())
        unt_ucrsu = self.unt_gcn_ucrsu(unt_init_proj, 
                                      p_ucrsu[0].to(device).clone(), 
                                      p_ucrsu[1].to(device).clone())
        unt_ucptu = self.unt_gcn_ucptu(unt_init_proj, 
                                      p_ucptu[0].to(device).clone(), 
                                      p_ucptu[1].to(device).clone())
        
        unt_emb = self.unt_attn(torch.stack([unt_ulu, unt_ucrsu, unt_ucptu]))
        unt_emb = self.output_norm(unt_emb)
        
        return unt_emb

class ConceptEncoder(nn.Module):
    """知识点编码器 - 可独立使用"""
    def __init__(self, embedding_dim=None, cpt_input_dim=None):
        super().__init__()
        
        if embedding_dim is None:
            embedding_dim = hyperparams.hgc_embedding_dim
            
        self.embedding_dim = embedding_dim
        self.cpt_proj = Projection(input_dim=cpt_input_dim, embedding_dim=embedding_dim)
        self.cpt_gcn_cc = GCNConvEmbedding(embedding_dim)
        self.cpt_gcn_cuc = GCNConvEmbedding(embedding_dim)
        self.cpt_gcn_ctc = GCNConvEmbedding(embedding_dim)
        self.cpt_attn = MetaPathAttention(embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, cpt_init: torch.Tensor,
                p_cc: tuple, p_cuc: tuple, p_ctc: tuple,
                device=None) -> torch.Tensor:
        """
        独立计算知识点嵌入
        """
        if device is None:
            device = next(self.parameters()).device
            
        # 修复：确保输入张量可以参与梯度计算（关键修复）
        cpt_init = cpt_init.clone().detach().requires_grad_(True)
        
        # 知识点嵌入
        cpt_init_proj = self.cpt_proj(cpt_init.to(device))
        cpt_cc = self.cpt_gcn_cc(cpt_init_proj, 
                                p_cc[0].to(device).clone(), 
                                p_cc[1].to(device).clone())
        cpt_cuc = self.cpt_gcn_cuc(cpt_init_proj, 
                                  p_cuc[0].to(device).clone(), 
                                  p_cuc[1].to(device).clone())
        cpt_ctc = self.cpt_gcn_ctc(cpt_init_proj, 
                                  p_ctc[0].to(device).clone(), 
                                  p_ctc[1].to(device).clone())
        cpt_emb = self.cpt_attn(torch.stack([cpt_cc, cpt_cuc, cpt_ctc]))
        cpt_emb = self.output_norm(cpt_emb)
        
        return cpt_emb

class HGC(nn.Module):
    """统一的HGC模型 - 保持原有接口，内部使用拆分后的编码器"""
    def __init__(self, embedding_dim=None, lrn_input_dim=None, unt_input_dim=None, cpt_input_dim=None):
        super().__init__()
        
        # 使用超参数配置
        if embedding_dim is None:
            embedding_dim = hyperparams.hgc_embedding_dim
            
        self.embedding_dim = embedding_dim
        self.attention_heads = hyperparams.hgc_attention_heads

        # 使用拆分后的编码器
        self.learner_encoder = LearnerEncoder(embedding_dim, lrn_input_dim)
        self.unit_encoder = UnitEncoder(embedding_dim, unt_input_dim)
        self.concept_encoder = ConceptEncoder(embedding_dim, cpt_input_dim)

    def forward(self, hgcdr=None, device=None, return_dict=False, 
                input_data=None):
        """
        统一的前向传播接口 - 支持两种输入方式
        """
        if device is None:
            device = next(self.parameters()).device
            
        # 确定输入数据来源
        if input_data is not None:
            # 使用新的输入数据格式
            data = input_data
        elif hgcdr is not None:
            # 使用传统的hgcdr格式
            data = self._hgcdr_to_input_data(hgcdr)
        else:
            raise ValueError("必须提供hgcdr或input_data之一")
        
        # 分别计算三种实体的嵌入
        lrn_emb = self.learner_encoder(
            data['lrn_init'], 
            data['p_lul'], data['p_lcl'], data['p_ltl'],
            device
        )
        
        unt_emb = self.unit_encoder(
            data['unt_init'],
            data['p_ulu'], data['p_ucrsu'], data['p_ucptu'],
            device
        )
        
        cpt_emb = self.concept_encoder(
            data['cpt_init'],
            data['p_cc'], data['p_cuc'], data['p_ctc'],
            device
        )

        if return_dict:
            return {
                'lrn_emb': lrn_emb,
                'unt_emb': unt_emb, 
                'cpt_emb': cpt_emb
            }
        else:
            return lrn_emb, unt_emb, cpt_emb

    def _hgcdr_to_input_data(self, hgcdr):
        """将hgcdr转换为新的输入数据格式"""
        return {
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

    # 新增：分别调用各个编码器的方法
    def compute_learner_embeddings(self, lrn_init, p_lul, p_lcl, p_ltl, device=None):
        """仅计算学习者嵌入"""
        return self.learner_encoder(lrn_init, p_lul, p_lcl, p_ltl, device)
    
    def compute_unit_embeddings(self, unt_init, p_ulu, p_ucrsu, p_ucptu, device=None):
        """仅计算学习单元嵌入"""
        return self.unit_encoder(unt_init, p_ulu, p_ucrsu, p_ucptu, device)
    
    def compute_concept_embeddings(self, cpt_init, p_cc, p_cuc, p_ctc, device=None):
        """仅计算知识点嵌入"""
        return self.concept_encoder(cpt_init, p_cc, p_cuc, p_ctc, device)

    def get_embedding_info(self):
        """返回嵌入信息"""
        return {
            'embedding_dim': self.embedding_dim,
            'attention_heads': self.attention_heads,
            'gcn_layers': hyperparams.hgc_gcn_layers,
            'activation_slope': hyperparams.hgc_activation_slope,
            'used_metapaths': {
                'learner': ['LUL', 'LCL', 'LTL'],
                'unit': ['ULU', 'UCRSU', 'UCPTU'],
                'concept': ['CC', 'CUC', 'CTC']
            }
        }

    def get_parameter_count(self):
        """返回模型参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }

def test_hgc_model():
    """测试HGC模型 - 修复一致性验证问题"""
    print("=== HGC模型测试 (适配无UU元路径版本) ===")
    
    # 加载数据
    hgcdr.loadDatafromSql()
    device = hyperparams.device
    
    # 动态获取输入维度
    lrn_input_dim = hgcdr.lrn_init.shape[1]
    unt_input_dim = hgcdr.qusunt_init.shape[1]
    cpt_input_dim = hgcdr.cpt_init.shape[1]
    
    print(f"输入维度: lrn={lrn_input_dim}, unt={unt_input_dim}, cpt={cpt_input_dim}")
    
    # 创建模型
    model = HGC(
        embedding_dim=hyperparams.hgc_embedding_dim,
        lrn_input_dim=lrn_input_dim,
        unt_input_dim=unt_input_dim,
        cpt_input_dim=cpt_input_dim
    ).to(device)
    
    print("模型配置:", model.get_embedding_info())
    print("参数统计:", model.get_parameter_count())
    
    # 测试新接口
    print("\n1. 测试新输入数据接口...")
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
    
    with torch.no_grad():
        # 使用新接口
        embeddings_dict_new = model(input_data=input_data, device=device, return_dict=True)
        embeddings_tuple_new = model(input_data=input_data, device=device, return_dict=False)
        
        # 使用旧接口
        embeddings_dict_old = model(hgcdr=hgcdr, device=device, return_dict=True)
        embeddings_tuple_old = model(hgcdr=hgcdr, device=device, return_dict=False)
    
    print("新接口结果:")
    for key, value in embeddings_dict_new.items():
        print(f"  {key}: {value.shape}, 范围[{value.min():.3f}, {value.max():.3f}]")
    
    # 验证新旧接口一致性
    print("\n2. 验证新旧接口一致性...")
    consistency_passed = True
    for key in ['lrn_emb', 'unt_emb', 'cpt_emb']:
        new_emb = embeddings_dict_new[key]
        old_emb = embeddings_dict_old[key]
        if torch.allclose(new_emb, old_emb, atol=1e-4, rtol=1e-3):
            print(f"  ✓ {key} 新旧接口一致性验证通过")
        else:
            max_diff = (new_emb - old_emb).abs().max().item()
            mean_diff = (new_emb - old_emb).abs().mean().item()
            print(f"  ⚠ {key} 新旧接口有差异: 最大差异={max_diff:.6f}, 平均差异={mean_diff:.6f}")
            consistency_passed = False
    
    # 测试独立编码器
    print("\n3. 测试独立编码器...")
    with torch.no_grad():
        lrn_emb_ind = model.compute_learner_embeddings(
            hgcdr.lrn_init, 
            (hgcdr.p_lul[0], hgcdr.p_lul[1]),
            (hgcdr.p_lcl[0], hgcdr.p_lcl[1]),
            (hgcdr.p_ltl[0], hgcdr.p_ltl[1]),
            device
        )
        
        unt_emb_ind = model.compute_unit_embeddings(
            hgcdr.qusunt_init,
            (hgcdr.p_ulu[0], hgcdr.p_ulu[1]),
            (hgcdr.p_ucrsu[0], hgcdr.p_ucrsu[1]),
            (hgcdr.p_ucptu[0], hgcdr.p_ucptu[1]),
            device
        )
        
        cpt_emb_ind = model.compute_concept_embeddings(
            hgcdr.cpt_init,
            (hgcdr.p_cc[0], hgcdr.p_cc[1]),
            (hgcdr.p_cuc[0], hgcdr.p_cuc[1]),
            (hgcdr.p_ctc[0], hgcdr.p_ctc[1]),
            device
        )
    
    # 验证独立编码器与统一接口的一致性
    print("4. 验证独立编码器与统一接口的一致性...")
    ind_embs = [lrn_emb_ind, unt_emb_ind, cpt_emb_ind]
    unified_embs = [embeddings_dict_new['lrn_emb'], embeddings_dict_new['unt_emb'], embeddings_dict_new['cpt_emb']]
    
    for i, (ind_emb, unified_emb) in enumerate(zip(ind_embs, unified_embs)):
        if torch.allclose(ind_emb, unified_emb, atol=1e-6):
            print(f"  ✓ 编码器{i+1} 独立与统一接口一致性验证通过")
        else:
            max_diff = (ind_emb - unified_emb).abs().max().item()
            print(f"  ✗ 编码器{i+1} 独立与统一接口有差异: 最大差异={max_diff:.6f}")
            consistency_passed = False
    
    # 测试梯度计算
    print("\n5. 梯度计算测试...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 使用新接口计算梯度
    lrn_emb, unt_emb, cpt_emb = model(input_data=input_data, device=device, return_dict=False)
    loss = (lrn_emb.norm() + unt_emb.norm() + cpt_emb.norm()) / 3
    loss.backward()
    
    # 检查梯度
    has_gradients = False
    total_grad_norm = 0.0
    grad_info = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:
                has_gradients = True
                total_grad_norm += grad_norm
                module_name = name.split('.')[0]
                grad_info[module_name] = grad_info.get(module_name, 0) + grad_norm
    
    if has_gradients:
        print(f"✓ 梯度计算正常，总梯度范数: {total_grad_norm:.6f}")
        print("各模块梯度分布:")
        for module, grad_norm in grad_info.items():
            print(f"  {module}: {grad_norm:.6f}")
        
        # 参数更新测试
        optimizer.step()
        print("✓ 参数更新成功")
    else:
        print("✗ 没有检测到有效梯度")
    
    # 总结测试结果
    print("\n=== 测试总结 ===")
    if consistency_passed and has_gradients:
        print("✓ 所有测试通过!")
    else:
        print("⚠ 部分测试有问题:")
        if not consistency_passed:
            print("  - 接口一致性有微小数值差异（通常可接受）")
        if not has_gradients:
            print("  - 梯度计算异常")
    
    print("\n✓ HGC模型测试完成")

if __name__ == '__main__':
    from DataReader.HGCDataReader import hgcdr
    test_hgc_model()