import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

KCGE_path = deeplearning_root + '\\KCGE'
sys.path.append(KCGE_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from DataSet.KCGEDataReader import KCGEDataReader

# class ECGEConv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(ECGEConv, self).__init__(aggr='add')  # 使用加法聚合
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         # 为每种关系定义一个可学习的权重矩阵 W
#         self.weights = nn.Parameter(torch.Tensor(4, in_channels, out_channels))
#         self.bias = nn.Parameter(torch.Tensor(out_channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weights)
#         nn.init.constant_(self.bias, 0)

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
#         # 计算节点的度并进行归一化
#         row = edge_index[0]
#         col = edge_index[1]
#         # print(type(edge_index), type(row), type(col))
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         # print(f"节点度：{deg.min()}, {deg.max()}")
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         res = self.propagate(edge_index=edge_index, size=(x.size(0), x.size(0)), x=x, edge_type=edge_type, norm=norm, edge_weight=edge_weight)

#         res = F.leaky_relu(res, negative_slope=0.01)
#         res = F.dropout(res, p=0.2, training=self.training)
#         return res

#     def message(self, x_j, edge_type, norm, edge_weight):
#         w = self.weights[edge_type]
#         x_j = torch.matmul(x_j.unsqueeze(1), w).squeeze(1)
#         x_j *= edge_weight.view(-1,1)
#         return norm.view(-1, 1) * x_j

#     def update(self, aggr_out):
#         # 应用偏置并返回最终的节点嵌入
#         return aggr_out + self.bias

class ECGEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ECGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 为每种关系定义一个可学习的权重矩阵 W
        self.weights = nn.Parameter(torch.Tensor(4, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # 计算节点的度并进行归一化
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        deg.scatter_add_(0, col, torch.ones_like(col, dtype=x.dtype))
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight

        # 消息传递实现
        # 1. 为每条边选择对应的权重矩阵
        w = self.weights[edge_type]  # [E, in_channels, out_channels]
        # 2. 应用权重变换
        x_j = x[col]  # [E, in_channels]
        x_j = torch.bmm(x_j.unsqueeze(1), w).squeeze(1)  # [E, out_channels]
        # 3. 应用归一化
        x_j = norm.view(-1, 1) * x_j
        
        # 聚合消息 (替代MessagePassing的'aggr'功能)
        out = torch.zeros(x.size(0), self.out_channels, dtype=x.dtype, device=x.device)
        out.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.out_channels), x_j)
        
        # 应用偏置和非线性激活
        out = out + self.bias
        out = F.leaky_relu(out, negative_slope=0.01)
        out = F.dropout(out, p=0.2, training=self.training)
        
        return out

class KCGE(nn.Module):
    def __init__(self, embedding_dim):
        super(KCGE, self).__init__()
        # 三层卷积
        self.conv1 = ECGEConv(embedding_dim, embedding_dim)  # 第1层
        self.conv2 = ECGEConv(embedding_dim, embedding_dim)  # 第2层
        self.conv3 = ECGEConv(embedding_dim, embedding_dim)  # 第3层

    def forward(self, x : torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:

        # print(x.shape)
        # print(edge_index.shape)
        # print(edge_type.shape)
        # print(edge_attr.shape)

        z_1 = self.conv1(x, edge_index, edge_type, edge_attr)
        z_2 = self.conv2(z_1, edge_index, edge_type, edge_attr)
        z_3 = self.conv3(z_2, edge_index, edge_type, edge_attr)
        
        z = (x + z_1 + z_2 + z_3) / 4
        # 进行简化，令z_shape和z_star相同

        return z
    
if __name__ == '__main__':
    kcgedatareader = KCGEDataReader('are_3fee9e47d0f3428382f4afbcb1004117')

    _, _, _, _, edge_index, edge_attr, edge_type = kcgedatareader.load_data_from_db()

    model_kcge = KCGE(32, 'cpu')

    z = model_kcge(edge_index, edge_type, edge_attr)

    print(z)