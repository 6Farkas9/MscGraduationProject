import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from DataReader.HGCDataReader import hgcdr

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
    def __init__(self, embedding_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        for _ in range(3):  # 3层共享参数GCN
            row, col = edge_index[0], edge_index[1]
            h_agg = scatter_add(h[row] * edge_weight.unsqueeze(1), col, dim=0, dim_size=h.size(0))
            h = torch.mm(h_agg, self.W)
            h = F.relu(h)
        return h

class Projection(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, embedding_dim)
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
    def __init__(self, embedding_dim=64, lrn_input_dim=None, unt_input_dim=None, cpt_input_dim=None):
        super().__init__()
        self.embedding_dim = embedding_dim

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
        self.unt_gcn_uu = GCNConvEmbedding(embedding_dim)
        self.unt_attn = MetaPathAttention(embedding_dim)

        # 知识点模块
        self.cpt_proj = Projection(input_dim=cpt_input_dim, embedding_dim=embedding_dim)
        self.cpt_gcn_cc = GCNConvEmbedding(embedding_dim)
        self.cpt_gcn_cuc = GCNConvEmbedding(embedding_dim)
        self.cpt_gcn_ctc = GCNConvEmbedding(embedding_dim)
        self.cpt_attn = MetaPathAttention(embedding_dim)

    def forward(self, hgcdr, device):
        # 学习者嵌入
        lrn_init = self.lrn_proj(hgcdr.lrn_init.to(device))
        lrn_lul = self.lrn_gcn_lul(lrn_init, 
                                  hgcdr.p_lul[0].to(device), 
                                  hgcdr.p_lul[1].to(device))
        lrn_lcl = self.lrn_gcn_lcl(lrn_init, 
                                  hgcdr.p_lcl[0].to(device), 
                                  hgcdr.p_lcl[1].to(device))
        lrn_ltl = self.lrn_gcn_ltl(lrn_init, 
                                  hgcdr.p_ltl[0].to(device), 
                                  hgcdr.p_ltl[1].to(device))
        lrn_emb = self.lrn_attn(torch.stack([lrn_lul, lrn_lcl, lrn_ltl]))

        # 学习单元嵌入
        unt_init = self.unt_proj(hgcdr.untqus_init.to(device))
        unt_ulu = self.unt_gcn_ulu(unt_init, 
                                  hgcdr.p_ulu[0].to(device), 
                                  hgcdr.p_ulu[1].to(device))
        unt_ucrsu = self.unt_gcn_ucrsu(unt_init, 
                                      hgcdr.p_ucrsu[0].to(device), 
                                      hgcdr.p_ucrsu[1].to(device))
        unt_ucptu = self.unt_gcn_ucptu(unt_init, 
                                      hgcdr.p_ucptu[0].to(device), 
                                      hgcdr.p_ucptu[1].to(device))
        unt_uu = self.unt_gcn_uu(unt_init, 
                                hgcdr.p_uu[0].to(device), 
                                hgcdr.p_uu[1].to(device))
        unt_emb = self.unt_attn(torch.stack([unt_ulu, unt_ucrsu, unt_ucptu, unt_uu]))

        # 知识点嵌入 - 关键修复：将推理张量转换为可训练张量
        cpt_init_tensor = hgcdr.cpt_init.clone().detach().requires_grad_(True)
        cpt_init = self.cpt_proj(cpt_init_tensor.to(device))
        cpt_cc = self.cpt_gcn_cc(cpt_init, 
                                hgcdr.p_cc[0].to(device), 
                                hgcdr.p_cc[1].to(device))
        cpt_cuc = self.cpt_gcn_cuc(cpt_init, 
                                  hgcdr.p_cuc[0].to(device), 
                                  hgcdr.p_cuc[1].to(device))
        cpt_ctc = self.cpt_gcn_ctc(cpt_init, 
                                  hgcdr.p_ctc[0].to(device), 
                                  hgcdr.p_ctc[1].to(device))
        cpt_emb = self.cpt_attn(torch.stack([cpt_cc, cpt_cuc, cpt_ctc]))

        return lrn_emb, unt_emb, cpt_emb

if __name__ == '__main__':
    hgcdr.loadDatafromSql()
    device = 'cpu'
    
    # 动态获取输入维度
    lrn_input_dim = hgcdr.lrn_init.shape[1]
    unt_input_dim = hgcdr.untqus_init.shape[1]
    cpt_input_dim = hgcdr.cpt_init.shape[1]
    
    model = HGC(
        embedding_dim=64,
        lrn_input_dim=lrn_input_dim,
        unt_input_dim=unt_input_dim,
        cpt_input_dim=cpt_input_dim
    ).to(device)

    lrn_emb, unt_emb, cpt_emb = model(hgcdr, device)

    print("Learner Embedding:", lrn_emb.shape)
    print("Unit Embedding:", unt_emb.shape)
    print("Concept Embedding:", cpt_emb.shape)