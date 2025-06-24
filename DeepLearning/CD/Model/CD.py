import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

CDDataReader_path = deeplearning_root + '\\CD'
sys.path.append(CDDataReader_path)

import torch
import torch.nn as nn

from Dataset.CDDataReader import CDDataReader

class DTR(nn.Module):
    def __init__(self, embedding_dim):
        super(DTR,self).__init__()
        self.embedding_dim = embedding_dim
        self.l_p_lrn = nn.Linear(2 * self.embedding_dim, 1)
        self.l_d_unt = nn.Linear(2 * self.embedding_dim, 1)
        self.l_b_unt = nn.Linear(self.embedding_dim, 1)

    # def forward(self, x):
    def forward(self, 
                h_lrn_cpt :torch.Tensor, 
                h_unt_cpt :torch.Tensor,
                h_unt :torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        p_lrn = torch.nn.functional.leaky_relu(self.l_p_lrn(h_lrn_cpt), negative_slope=0.01).squeeze(-1)

        d_unt = torch.nn.functional.leaky_relu(self.l_d_unt(h_unt_cpt), negative_slope=0.01).squeeze(-1)

        beta_unt = torch.nn.functional.leaky_relu(self.l_b_unt(h_unt), negative_slope=0.01).squeeze(-1)

        return p_lrn, d_unt, beta_unt
    
class MIRT(nn.Module):
    def __init__(self):
        super(MIRT,self).__init__()

    # def forward(self, p_u, d_v, beta_v):
    def forward(self, p_lrn: torch.Tensor, d_unt: torch.Tensor, beta_unt: torch.Tensor) -> torch.Tensor:
        # 限制使用的时候batch也必须要有，等于1
        # print(p_lrn.shape, d_unt.shape, beta_unt.shape)
        # result = torch.sigmoid(torch.einsum('bic,bc->bi', d_unt, p_lrn) + beta_unt)

        einsum_out = torch.einsum('bic,bc->bi', d_unt, p_lrn)
        # print("einsum 输出范围:", einsum_out.min(), einsum_out.max())
        sigmoid_input = einsum_out + beta_unt
        # print("sigmoid 输入范围:", sigmoid_input.min(), sigmoid_input.max())
        result = torch.sigmoid(sigmoid_input)
        return result

class CD(nn.Module):
    def __init__(self, embedding_dim):
        super(CD, self).__init__()
        self.embedding_dim = embedding_dim

        self.dtr = DTR(self.embedding_dim)
        self.mirt = MIRT()

    def forward(self, 
                unt_seq_index : torch.Tensor,
                unt_seq_mask : torch.Tensor,
                h_lrn : torch.Tensor,
                h_unt : torch.Tensor,
                h_cpt : torch.Tensor) -> torch.Tensor:
        # 模型外要根据kcge的输出从中提取
        # 模型外部计算h，使用unt_seq_index进行计算

        # print(unt_seq_index.shape)
        # print(unt_seq_mask.shape)
        # print(h_lrn.shape)
        # print(h_unt.shape)
        # print(h_cpt.shape)

        lrn_num = h_lrn.size(0)
        unt_num = h_unt.size(0)
        cpt_num = h_cpt.size(0)

        # 1. 计算 h_lrn_cpt
        h_lrn_expanded = h_lrn.repeat_interleave(cpt_num, dim=0)  # (lrn_num * cpt_num, emb_dim)
        h_cpt_expanded = h_cpt.repeat(lrn_num, 1)                 # (lrn_num * cpt_num, emb_dim)
        h_lrn_cpt = torch.cat([h_lrn_expanded, h_cpt_expanded], dim=1)  # (lrn_num * cpt_num, emb_dim * 2)

        # 2. 计算 h_unt_cpt
        h_unt_expanded = h_unt.repeat_interleave(cpt_num, dim=0)  # (unt_num * cpt_num, emb_dim)
        h_cpt_expanded_unt = h_cpt.repeat(unt_num, 1)             # (unt_num * cpt_num, emb_dim)
        h_unt_cpt = torch.cat([h_unt_expanded, h_cpt_expanded_unt], dim=1)  # (unt_num * cpt_num, emb_dim * 2)

        p_lrn, d_unt, beta_unt = self.dtr(h_lrn_cpt, h_unt_cpt, h_unt)

        # p_lrn (lrn_num * cpt_num)
        p_lrn = p_lrn.reshape(lrn_num, cpt_num)
        # d_unt (unt_num * cpt_num)
        d_unt = d_unt.reshape(unt_num, cpt_num)
        # beta_unt (unt_num * 1)

        d_unt = d_unt[unt_seq_index]
        beta_unt = beta_unt[unt_seq_index]

        # # print(d_unt.shape, beta_unt.shape)

        result = self.mirt(p_lrn, d_unt, beta_unt) * unt_seq_mask

        # print("MIRT 输出范围:", result.min(), result.max())
        # assert not torch.isnan(result).any(), "MIRT 输出包含 NaN!"
        return result

if __name__ == '__main__':
    cddr = CDDataReader()
    cddr.set_are_uid('are_3fee9e47d0f3428382f4afbcb1004117')
    
    train_data, master_data, cpt_uids, lrn_uids, unt_uids, edge_index, edge_attr, edge_type = cddr.load_Data_from_db()

