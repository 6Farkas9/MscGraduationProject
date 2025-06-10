import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from typing import Dict, Tuple

class RRGRU(nn.Module):
    def __init__(self, input_size=32, hidden_size=32, num_layers=1):
        """
        动态学习需求建模的GRU网络
        参数:
        - input_size: 输入特征维度（与静态嵌入维度一致默认32）
        - hidden_size: GRU隐藏层维度（通常设为输入维度的一半）
        - num_layers: GRU层数（默认1层）
        """
        super(RRGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # 输入形状为(batch, seq_len, input_size)
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        输入:
        - x: 用户知识水平序列 (batch_size, seq_len, input_size)
           seq_len为用户历史场景交互数，每个场景的嵌入表达由HGC获得
        
        输出:
        - h_last: 最终时刻的隐藏状态 (batch_size, hidden_size)
        """
        # 初始化隐藏状态（论文未明确初始化方式，默认全零）
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size, device=x.device)
        
        # GRU前向传播
        # out: (batch_size, seq_len, hidden_size)
        # hn: (num_layers, batch_size, hidden_size)
        out, _ = self.gru(x, h0)
        
        # 取最后一个时间步的输出（论文公式18）
        h_last = out[:, -1, :]
        return h_last

class RR(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim):
        super(RR, self).__init__()
        self.gru = RRGRU(embedding_dim, embedding_dim, 1)
        self.project = nn.Linear(embedding_dim, hidden_dim)

        self.w = nn.Parameter(torch.Tensor(embedding_dim * 2, embedding_dim))

        self.lrn_lmd = nn.Parameter(torch.Tensor(embedding_dim * 2, hidden_dim))
        self.cpt_lmd = nn.Parameter(torch.Tensor(hidden_dim, embedding_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.lrn_lmd)
        nn.init.xavier_uniform_(self.cpt_lmd)  # Xavier初始化

    def forward(self, 
                lrn_static : torch.Tensor, 
                scn_dynamic : torch.Tensor,
                scn_seq_index : torch.Tensor,
                scn_seq_mask : torch.Tensor,
                cpt_static : torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # lrn_static : 学习者静态嵌入
        # scn_dynamic : 计算出的场景的动态嵌入
        # scn_seq_index : 当前batch的学习者和场景的互动序列
        # scn_seq_mask : 指名scn_seq_index中的有效位
        # -> 返回的应该是计算出的r_uk

        # print(lrn_static.shape)
        # print(scn_dynamic.shape)
        # print(scn_seq_index.shape)
        # print(scn_seq_mask.shape)
        # print(cpt_static.shape)

        # 只能假定知识点是相对稳定，不会随时变化的，那么将知识点的潜在嵌入直接作为参数
        # 学习者的潜在嵌入通过在HGC中类似的投影手段来获得

        scn_record = scn_dynamic[scn_seq_index]
        scn_record = scn_record * scn_seq_mask.unsqueeze(-1)

        lrn_dynamic = self.gru(scn_record)

        lrn = torch.cat((lrn_static, lrn_dynamic), dim=1)
        
        # 潜在因子计算
        # (lrn_num * (emb*2)) --> (lrn_num * hidden_dim) : (lrn_num * (emb*2)) * ((emb*2) * hidden_dim)
        # (emb * cpt_num) --> (hidden_dim * cpt_num) : (hidden_dim * emb)

        # 计算主体
        # (lrn_num * (emb*2)) - ((emb*2) * emb) - (emb * cpt_num) --> (lrn_num * cpt_num)

        h_lrn = lrn @ self.lrn_lmd
        h_cpt = self.cpt_lmd @ cpt_static.t()

        h = h_lrn @ h_cpt

        temp_ = lrn @ self.w @ cpt_static.t()

        r = h + temp_
        # 虽然，在计算上，推荐的结果和cpt数量或者其他数量无关，但是，不知道效果会怎么样

        return r, h_lrn, h_cpt