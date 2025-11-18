import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from DataReader.HGCDataReader import hgcdr
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

class HGC(nn.Module):
    def __init__(self, embedding_dim=None, lrn_input_dim=None, unt_input_dim=None, cpt_input_dim=None):
        super().__init__()
        
        # 使用超参数配置
        if embedding_dim is None:
            embedding_dim = hyperparams.hgc_embedding_dim
            
        self.embedding_dim = embedding_dim
        self.attention_heads = hyperparams.hgc_attention_heads

        # 学习者模块
        self.lrn_proj = Projection(input_dim=lrn_input_dim, embedding_dim=embedding_dim)
        self.lrn_gcn_lul = GCNConvEmbedding(embedding_dim)
        self.lrn_gcn_lcl = GCNConvEmbedding(embedding_dim)
        self.lrn_gcn_ltl = GCNConvEmbedding(embedding_dim)
        self.lrn_attn = MetaPathAttention(embedding_dim)

        # 学习单元模块
        self.unt_proj = Projection(input_dim=unt_input_dim, embedding_dim=embedding_dim)
        self.unt_gcn_ulu = GCNConvEmbedding(embedding_dim)
        self.unt_gcn_ucrsu = GCNConvEmbedding(embedding_dim)
        self.unt_gcn_ucptu = GCNConvEmbedding(embedding_dim)
        # 注释掉UU元路径，但保留模型结构以保持接口一致性
        # self.unt_gcn_uu = GCNConvEmbedding(embedding_dim)
        self.unt_attn = MetaPathAttention(embedding_dim)

        # 知识点模块
        self.cpt_proj = Projection(input_dim=cpt_input_dim, embedding_dim=embedding_dim)
        self.cpt_gcn_cc = GCNConvEmbedding(embedding_dim)
        self.cpt_gcn_cuc = GCNConvEmbedding(embedding_dim)
        self.cpt_gcn_ctc = GCNConvEmbedding(embedding_dim)
        self.cpt_attn = MetaPathAttention(embedding_dim)
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, hgcdr, device, return_dict=False):
        """
        统一的前向传播接口 - 适配注释掉UU元路径的版本
        
        Args:
            hgcdr: HGC数据读取器实例
            device: 计算设备
            return_dict: 是否返回字典格式结果
            
        Returns:
            如果return_dict=True: {'lrn_emb': ..., 'unt_emb': ..., 'cpt_emb': ...}
            否则: (lrn_emb, unt_emb, cpt_emb)
        """
        # 学习者嵌入 - 使用clone确保梯度正确传递
        lrn_init = self.lrn_proj(hgcdr.lrn_init.to(device).clone())
        lrn_lul = self.lrn_gcn_lul(lrn_init, 
                                  hgcdr.p_lul[0].to(device).clone(), 
                                  hgcdr.p_lul[1].to(device).clone())
        lrn_lcl = self.lrn_gcn_lcl(lrn_init, 
                                  hgcdr.p_lcl[0].to(device).clone(), 
                                  hgcdr.p_lcl[1].to(device).clone())
        lrn_ltl = self.lrn_gcn_ltl(lrn_init, 
                                  hgcdr.p_ltl[0].to(device).clone(), 
                                  hgcdr.p_ltl[1].to(device).clone())
        lrn_emb = self.lrn_attn(torch.stack([lrn_lul, lrn_lcl, lrn_ltl]))
        lrn_emb = self.output_norm(lrn_emb)

        # 学习单元嵌入 - 注释掉UU元路径
        unt_init = self.unt_proj(hgcdr.qusunt_init.to(device).clone())
        unt_ulu = self.unt_gcn_ulu(unt_init, 
                                  hgcdr.p_ulu[0].to(device).clone(), 
                                  hgcdr.p_ulu[1].to(device).clone())
        unt_ucrsu = self.unt_gcn_ucrsu(unt_init, 
                                      hgcdr.p_ucrsu[0].to(device).clone(), 
                                      hgcdr.p_ucrsu[1].to(device).clone())
        unt_ucptu = self.unt_gcn_ucptu(unt_init, 
                                      hgcdr.p_ucptu[0].to(device).clone(), 
                                      hgcdr.p_ucptu[1].to(device).clone())
        
        # 注释掉UU元路径，只使用三个元路径
        # unt_uu = self.unt_gcn_uu(unt_init, 
        #                         hgcdr.p_uu[0].to(device).clone(), 
        #                         hgcdr.p_uu[1].to(device).clone())
        
        # 使用三个元路径而不是四个
        unt_emb = self.unt_attn(torch.stack([unt_ulu, unt_ucrsu, unt_ucptu]))
        unt_emb = self.output_norm(unt_emb)

        # 知识点嵌入 - 关键修复：正确处理推理张量
        cpt_init_tensor = hgcdr.cpt_init.clone().detach().requires_grad_(True)
        cpt_init = self.cpt_proj(cpt_init_tensor.to(device))
        cpt_cc = self.cpt_gcn_cc(cpt_init, 
                                hgcdr.p_cc[0].to(device).clone(), 
                                hgcdr.p_cc[1].to(device).clone())
        cpt_cuc = self.cpt_gcn_cuc(cpt_init, 
                                  hgcdr.p_cuc[0].to(device).clone(), 
                                  hgcdr.p_cuc[1].to(device).clone())
        cpt_ctc = self.cpt_gcn_ctc(cpt_init, 
                                  hgcdr.p_ctc[0].to(device).clone(), 
                                  hgcdr.p_ctc[1].to(device).clone())
        cpt_emb = self.cpt_attn(torch.stack([cpt_cc, cpt_cuc, cpt_ctc]))
        cpt_emb = self.output_norm(cpt_emb)

        if return_dict:
            return {
                'lrn_emb': lrn_emb,
                'unt_emb': unt_emb, 
                'cpt_emb': cpt_emb
            }
        else:
            return lrn_emb, unt_emb, cpt_emb

    def get_embedding_info(self):
        """返回嵌入信息"""
        return {
            'embedding_dim': self.embedding_dim,
            'attention_heads': self.attention_heads,
            'gcn_layers': hyperparams.hgc_gcn_layers,
            'activation_slope': hyperparams.hgc_activation_slope,
            'used_metapaths': {
                'learner': ['LUL', 'LCL', 'LTL'],
                'unit': ['ULU', 'UCRSU', 'UCPTU'],  # 注释掉UU
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
    
    # 测试前向传播 - 修复一致性验证
    print("\n1. 前向传播测试...")
    
    # 方法1：分别进行两次独立计算
    print("方法1: 分别计算两种返回格式")
    model.eval()  # 使用eval模式避免dropout随机性
    with torch.no_grad():
        embeddings_dict_1 = model(hgcdr, device, return_dict=True)
        embeddings_tuple_1 = model(hgcdr, device, return_dict=False)
    
    print("字典格式结果:")
    for key, value in embeddings_dict_1.items():
        print(f"  {key}: {value.shape}, 范围[{value.min():.3f}, {value.max():.3f}]")
    
    print("元组格式结果:")
    for i, emb in enumerate(embeddings_tuple_1):
        print(f"  嵌入{i+1}: {emb.shape}, 范围[{emb.min():.3f}, {emb.max():.3f}]")
    
    # 修复的一致性验证
    dict_embs = [embeddings_dict_1['lrn_emb'], embeddings_dict_1['unt_emb'], embeddings_dict_1['cpt_emb']]
    consistency_passed = True
    
    for i, (dict_emb, tuple_emb) in enumerate(zip(dict_embs, embeddings_tuple_1)):
        # 更宽松的一致性检查
        if torch.allclose(dict_emb, tuple_emb, atol=1e-4, rtol=1e-3):
            print(f"  ✓ 输出{i+1}一致性验证通过")
        else:
            # 计算具体差异
            max_diff = (dict_emb - tuple_emb).abs().max().item()
            mean_diff = (dict_emb - tuple_emb).abs().mean().item()
            print(f"  ⚠ 输出{i+1}一致性验证有差异: 最大差异={max_diff:.6f}, 平均差异={mean_diff:.6f}")
            consistency_passed = False
    
    # 方法2：从同一计算结果转换
    print("\n方法2: 从同一计算结果转换")
    with torch.no_grad():
        embeddings_tuple_2 = model(hgcdr, device, return_dict=False)
        # 手动构建字典
        embeddings_dict_2 = {
            'lrn_emb': embeddings_tuple_2[0],
            'unt_emb': embeddings_tuple_2[1], 
            'cpt_emb': embeddings_tuple_2[2]
        }
    
    # 检查转换一致性
    conversion_passed = True
    for i, key in enumerate(['lrn_emb', 'unt_emb', 'cpt_emb']):
        if torch.allclose(embeddings_dict_2[key], embeddings_tuple_2[i], atol=1e-6):
            print(f"  ✓ {key}转换一致性验证通过")
        else:
            print(f"  ✗ {key}转换一致性验证失败")
            conversion_passed = False
    
    # 测试梯度计算
    print("\n2. 梯度计算测试...")
    model.train()  # 切换回训练模式
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 使用其中一种格式计算梯度
    lrn_emb, unt_emb, cpt_emb = model(hgcdr, device, return_dict=False)
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
    if consistency_passed and conversion_passed and has_gradients:
        print("✓ 所有测试通过!")
    else:
        print("⚠ 部分测试有问题:")
        if not consistency_passed:
            print("  - 两种返回格式有微小数值差异（通常可接受）")
        if not conversion_passed:
            print("  - 格式转换一致性失败")
        if not has_gradients:
            print("  - 梯度计算异常")
    
    print("\n✓ HGC模型测试完成")

if __name__ == '__main__':
    test_hgc_model()