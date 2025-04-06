import torch
import torch.nn as nn
import torch.optim as optim
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
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        # 计算节点的度并进行归一化
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 调用 propagate 方法进行消息传递
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_type=edge_type, norm=norm, edge_weight=edge_weight)

    def message(self, x_j, edge_type, norm, edge_weight):
        # 根据边的类型选择不同的权重矩阵
        w = self.weights[edge_type]
        # 计算节点嵌入，乘以归一化因子和边的权重
        x_j = torch.matmul(x_j.unsqueeze(1), w).squeeze(1)
        x_j *= edge_weight.view(-1,1)
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

    def forward(self, data):
        x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        z_1 = F.relu(self.conv1(x, edge_index, edge_type, edge_weight))  # 第1层
        z_1 = F.dropout(z_1, training=self.training)
        z_2 = self.conv2(z_1, edge_index, edge_type, edge_weight)  # 第2层
        
        z_star = (x + z_1 + z_2) / 4

        temp = [x, z_1, z_2]

        z_sharp = temp[self.lamda]
        for i in range(self.lamda + 1, 3):
            z_sharp += temp[i]
        z_sharp = z_sharp / (4 - self.lamda)

        return z_star, z_sharp