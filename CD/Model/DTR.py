import torch
import torch.nn as nn

class Model_PDBeta(nn.Module):
    def __init__(self,embedding_dim):
        super(Model_PDBeta,self).__init__()
        self.embedding_dim = embedding_dim
        self.linear_pu = nn.Linear(2 * self.embedding_dim, 1)
        self.linear_dv = nn.Linear(2 * self.embedding_dim, 1)
        self.linear_bv = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        out_pu = torch.nn.functional.leaky_relu(self.linear_pu(x[0]), negative_slope=0.01)

        # print(f"linear_pu 输入的最小值: {x[0].min()}, 最大值: {x[0].max()}")
        # print(f"linear_pu 输出的最小值: {out_pu.min()}, 最大值: {out_pu.max()}")

        out_dv = torch.nn.functional.leaky_relu(self.linear_dv(x[1]), negative_slope=0.01)

        # print(f"linear_dv 输入的最小值: {x[1].min()}, 最大值: {x[1].max()}")
        # print(f"linear_dv 输出的最小值: {out_dv.min()}, 最大值: {out_dv.max()}")

        out_bv = torch.nn.functional.leaky_relu(self.linear_bv(x[2]), negative_slope=0.01)

        # print(f"linear_bv 输入的最小值: {x[2].min()}, 最大值: {x[2].max()}")
        # print(f"linear_bv 输出的最小值: {out_bv.min()}, 最大值: {out_bv.max()}")

        # out_dv = self.relu(out_linear)
        # out_linear = self.relu(self.linear(x.clone()))

        return out_pu, out_dv, out_bv
