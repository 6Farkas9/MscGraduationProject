import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
import torch.nn as nn

class DTR(nn.Module):
    def __init__(self, embedding_dim, devive):
        super(DTR,self).__init__()
        self.embedding_dim = embedding_dim
        self.device = devive
        self.l_p_lrn = nn.Linear(2 * self.embedding_dim, 1).to(self.device)
        self.l_d_scn = nn.Linear(2 * self.embedding_dim, 1).to(self.device)
        self.l_b_scn = nn.Linear(self.embedding_dim, 1).to(self.device)

    # def forward(self, x):
    def forward(self, 
                h_lrn_cpt :torch.Tensor, 
                h_scn_cpt :torch.Tensor,
                h_scn :torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_lrn = torch.nn.functional.leaky_relu(self.l_p_lrn(h_lrn_cpt), negative_slope=0.01).squeeze(-1)

        d_scn = torch.nn.functional.leaky_relu(self.l_d_scn(h_scn_cpt), negative_slope=0.01).squeeze(-1)

        beta_scn = torch.nn.functional.leaky_relu(self.l_b_scn(h_scn), negative_slope=0.01).squeeze(-1)

        return p_lrn, d_scn, beta_scn
    
class MIRT(nn.Module):
    def __init__(self):
        super(MIRT,self).__init__()

    # def forward(self, p_u, d_v, beta_v):
    def forward(self, p_lrn: torch.Tensor, d_scn: torch.Tensor, beta_scn: torch.Tensor) -> torch.Tensor:
        # 限制使用的时候batch也必须要有，等于1
        # print(p_lrn.shape, d_scn.shape, beta_scn.shape)
        result = torch.sigmoid(torch.einsum('bic,bc->bi', d_scn, p_lrn) + beta_scn)
        return result

class CD(nn.Module):
    def __init__(self, embedding_dim, device):
        super(CD, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.dtr = DTR(self.embedding_dim, self.device).to(self.device)
        self.mirt = MIRT().to(self.device)

    def forward(self, 
                scn_seq_index : torch.Tensor, 
                scn_seq_mask : torch.Tensor, 
                lrn_static : torch.Tensor,
                scn_emb : torch.Tensor, 
                cpt_emb : torch.Tensor) -> torch.Tensor:
        # 先计算h矩阵
        # 然后拼接h矩阵
        # 然后输入到dtr中获得p，d，β向量
        # 最后输入到mirt中获得预测值

        # print(scn_seq_index.shape, scn_seq_mask.shape)

        h_lrn = (scn_emb[scn_seq_index] * scn_seq_mask.unsqueeze(-1)).sum(dim=1) / scn_seq_mask.sum(dim=1, keepdim=True)
        h_lrn.add_(lrn_static).div_(2.0)
        # print('h_lrn:', h_lrn.shape)

        h_scn = scn_emb
        h_cpt = cpt_emb

        num_lrn = h_lrn.size(0)
        num_scn = h_scn.size(0)
        num_cpt = h_cpt.size(0)

        h_lrn_expended = h_lrn.unsqueeze(1).expand(-1, num_cpt, -1)
        h_cpt_expended = h_cpt.unsqueeze(0).expand(num_lrn, -1, -1)
        h_lrn_cpt = torch.cat((h_lrn_expended, h_cpt_expended), dim=-1).view(-1, 2 * self.embedding_dim)

        h_scn_expended = h_scn.unsqueeze(1).expand(-1, num_cpt, -1)
        h_cpt_expended = h_cpt.unsqueeze(0).expand(num_scn, -1, -1)
        h_scn_cpt = torch.cat((h_scn_expended, h_cpt_expended), dim=-1).view(-1, 2 * self.embedding_dim)

        p_lrn, d_scn, beta_scn = self.dtr(h_lrn_cpt, h_scn_cpt, h_scn)
        
        p_lrn = p_lrn.view(num_lrn, num_cpt)
        d_scn = d_scn.view(num_scn, num_cpt)

        # p_lrn : lrn_num * cpt_num
        # d_scn : scn_num * cpt_num
        # beta_scn : scn_num
        # print(p_lrn.shape, d_scn.shape, beta_scn.shape)
        
        # 是不是应该根据lrn_id和scn_id来获取对应的向量？yes,yes,yes
        
        # print(scn_seq_index.max(), scn_seq_index.min())

        d_scn = d_scn[scn_seq_index]
        beta_scn = beta_scn[scn_seq_index]

        # print(d_scn.shape, beta_scn.shape)

        result = self.mirt(p_lrn, d_scn, beta_scn) * scn_seq_mask

        return result
