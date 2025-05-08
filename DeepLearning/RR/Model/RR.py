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
    def __init__(self, input_size=32, hidden_size=16, num_layers=1):
        """
        动态学习需求建模的GRU网络
        参数:
        - input_size: 输入特征维度（与静态嵌入维度一致默认128）
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
        
    def forward(self, x : torch.tensor) -> torch.tensor:
        """
        输入:
        - x: 用户知识水平序列 (batch_size, seq_len, input_size)
           seq_len为用户历史场景交互数，每个场景的嵌入表达由HGC获得
        
        输出:
        - h_last: 最终时刻的隐藏状态 (batch_size, hidden_size)
        """
        # 初始化隐藏状态（论文未明确初始化方式，默认全零）
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        
        # GRU前向传播
        # out: (batch_size, seq_len, hidden_size)
        # hn: (num_layers, batch_size, hidden_size)
        out, _ = self.gru(x, h0)
        
        # 取最后一个时间步的输出（论文公式18）
        h_last = out[:, -1, :]
        return h_last
    
class RR(nn.Module):
    
    def __init__(self, embedding_dim, device):
        super(RR, self).__init__()
        self.device = device
        self.gru = RRGRU(embedding_dim, embedding_dim // 2, 1)

    def forward(self, 
                lrn_static : torch.tensor, scn_dynamic : torch.tensor
                
                ) -> tuple[torch.tensor, torch.tensor, torch.tensor]:

        # lrn_emb, scn_emb, cpt_emb = self.hgc(inits, p_martixes)

        return 0