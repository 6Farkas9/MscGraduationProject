import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

KCGE_path = deeplearning_root + '\\KCGE'
sys.path.append(KCGE_path)

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class ECGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations):
        super(ECGEConv, self).__init__(aggr='add')  # 使用加法聚合
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        # 为每种关系定义一个可学习的权重矩阵 W
        self.weights = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.constant_(self.bias, 0)

    # def forward(self, x, edge_index, edge_type, edge_weight=None):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # 计算节点的度并进行归一化
        row = edge_index[0]
        col = edge_index[1]
        # print(type(edge_index), type(row), type(col))
        deg = degree(col, x.size(0), dtype=x.dtype)
        # print(f"节点度：{deg.min()}, {deg.max()}")
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 1  # 将度为零的节点设置为1  # 将度为零的节点的归一化因子设为0
        # print(f"归一化因子：{deg_inv_sqrt.min()}, {deg_inv_sqrt.max()}")
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 调用 propagate 方法进行消息传递
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_type=edge_type, norm=norm, edge_weight=edge_weight)

    def message(self, x_j, edge_type, norm, edge_weight):
        # 根据边的类型选择不同的权重矩阵
        w = self.weights[edge_type]
        # 计算节点嵌入，乘以归一化因子和边的权重
        x_j = torch.matmul(x_j.unsqueeze(1), w).squeeze(1)
        x_j *= edge_weight.view(-1,1)
        # print(f"edge_weight 的最小值: {edge_weight.min()}, 最大值: {edge_weight.max()}")
        # print(f"norm 的最小值: {norm.min()}, 最大值: {norm.max()}")
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # 应用偏置并返回最终的节点嵌入
        return aggr_out + self.bias

class KCGE(nn.Module):
    def __init__(self, embedding_dim, num_relations, lamda):
        super(KCGE, self).__init__()
        self.lamda = lamda

        self.conv1 = ECGEConv(embedding_dim, embedding_dim, num_relations)  # 第1层
        self.conv2 = ECGEConv(embedding_dim, embedding_dim, num_relations)  # 第2层

    # def forward(self, data):
    # def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        # print(type(data))
        # print(f"x 输入的最小值: {x.min()}, 最大值: {x.max()}")
        # print(f"conv1 之前的 x 的最小值: {x.min()}, 最大值: {x.max()}")
        z_1 = F.leaky_relu(self.conv1(x, edge_index, edge_type, edge_weight), negative_slope=0.01)  # 第1层
        # print(f"conv1 之后的 z_1 的最小值: {z_1.min()}, 最大值: {z_1.max()}")
        # print(f"conv1 输出的最小值: {z_1.min()}, 最大值: {z_1.max()}")
        z_1 = F.dropout(z_1, training=self.training)
        z_2 = self.conv2(z_1, edge_index, edge_type, edge_weight)  # 第2层
        # print(f"conv2 输出的最小值: {z_2.min()}, 最大值: {z_2.max()}")
        
        z_star = (x + z_1 + z_2) / 4

        temp = [x, z_1, z_2]

        z_sharp = temp[self.lamda]
        for i in range(self.lamda + 1, 3):
            z_sharp += temp[i]
        z_sharp = z_sharp / (4 - self.lamda)

        # print(f"conv1 权重的最小值: {self.conv1.weights.min()}, 最大值: {self.conv1.weights.max()}")
        # print(f"conv1 偏置的最小值: {self.conv1.bias.min()}, 最大值: {self.conv1.bias.max()}")
        # print(f"conv2 权重的最小值: {self.conv2.weights.min()}, 最大值: {self.conv2.weights.max()}")
        # print(f"conv2 偏置的最小值: {self.conv2.bias.min()}, 最大值: {self.conv2.bias.max()}")

        return z_star, z_sharp