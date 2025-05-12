import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

HGC_path = deeplearning_root + '\\HGC'
sys.path.append(HGC_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from Dataset.HGCDataReader import HGCDataReader

class MetaPathAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(MetaPathAttention, self).__init__()
        self.W_att = nn.Linear(embedding_dim, 1)  # 可训练参数（原文中的 W_att^mp）
        self.activation = nn.ReLU()

    def forward(self, embeddings : torch.Tensor) -> torch.Tensor:
        """
        embeddings: 形状为 [num_paths, x, embedding_dim]
        - num_paths: 元路径的数量
        - x: 节点数（如场景数量）
        - embedding_dim: 每个节点的嵌入维度
        """
        # embeddings 是一个形状为 [num_paths, x, embedding_dim] 的张量
        num_paths, _, _ = embeddings.shape
        
        # 计算每个元路径的注意力分数
        scores = []
        for i in range(num_paths):
            score = self.W_att(embeddings[i])  # 计算每个元路径的注意力分数，形状为 [x, 1]
            scores.append(score)
        
        # 拼接所有元路径的分数，形状为 [x, num_paths]
        scores = torch.cat(scores, dim=1)

        # 计算注意力权重，进行 softmax 归一化
        attention_weights = F.softmax(scores, dim=1)  # softmax，形状为 [x, num_paths]

        # 加权求和，得到最终的融合嵌入
        weighted_embeddings = torch.zeros_like(embeddings[0])  # 初始化融合的结果
        for i in range(num_paths):
            weighted_embeddings += attention_weights[:, i].unsqueeze(-1) * embeddings[i]

        return weighted_embeddings

class GCNConvEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super(GCNConvEmbedding, self).__init__()
        self.W = nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)  # Xavier初始化

    def forward(self, 
                h : torch.Tensor,
                edge_index : torch.Tensor,
                edge_attr : torch.Tensor
                ) -> torch.Tensor:
        
        for _ in range(3):  # 三层共享参数的卷积
            # 1. 计算 P*h (通过边索引和边权重隐式构造传播矩阵P)
            row, col = edge_index[0], edge_index[1]
            # 加权聚合 (edge_attr作为P的非零元素值)

            h_agg = scatter_add(h[row] * edge_attr.unsqueeze(1), col, dim=0, dim_size=h.size(0))

            # 2. 计算 h*P*W (等价于 (P*h)*W)
            h = torch.mm(h_agg, self.W)

            # 3. ReLU激活
            h = F.relu(h)
        
        return h
    
class Projection(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # 输入维度为场景数量（动态），输出固定维度
        self.proj = nn.Sequential(
            nn.Linear(1, 16),    # 处理单个归一化交互值
            nn.ReLU(),
            nn.Linear(16, embedding_dim)
        )
        
    def forward(self, normalized_matrix: torch.Tensor) -> torch.Tensor:
        """
        输入: normalized_matrix = D⁻¹A [num_learners, num_scenes]
        输出: [num_learners, hidden_dim]
        """
        num_learners = normalized_matrix.size(0)
        embeddings = []
        
        for i in range(num_learners):
            # 提取非零归一化交互值 [num_interacted_scenes, 1]
            non_zero = normalized_matrix[i].nonzero().squeeze(-1).float()
            if len(non_zero) == 0:
                emb = torch.zeros(1, self.proj[-1].out_features, dtype=torch.float)
            else:
                # 对每个归一化值独立编码 [num_interacted_scenes, hidden_dim]
                embs = self.proj(non_zero.unsqueeze(-1))
                # 均值聚合 [hidden_dim]
                emb = embs.mean(dim=0, keepdim=True)
            embeddings.append(emb)
            
        return torch.cat(embeddings, dim=0)

class HGC_LRN(nn.Module):
    def __init__(self, embedding_dim, device):
        super(HGC_LRN, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        self.proj_lrn = Projection(embedding_dim).to(device)

        self.GCN_lsl = GCNConvEmbedding(embedding_dim).to(device)

    def forward(self,
                init : torch.Tensor,
                p_lsl_edge_index : torch.Tensor, p_lsl_edge_attr : torch.Tensor
                ) -> torch.Tensor:    

        embeddings_lrn = self.proj_lrn(init)

        out_lsl = self.GCN_lsl(embeddings_lrn,
                               p_lsl_edge_index,
                               p_lsl_edge_attr)

        return out_lsl
    
class HGC_SCN(nn.Module):
    def __init__(self, embedding_dim, device):
        super(HGC_SCN, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        self.proj_scn = Projection(embedding_dim).to(device)

        self.GCN_scs = GCNConvEmbedding(embedding_dim).to(device)
        self.GCN_sls = GCNConvEmbedding(embedding_dim).to(device)

        self.attention_scn = MetaPathAttention(embedding_dim).to(device)

    def forward(self, 
                init : torch.Tensor,
                p_scs_edge_index : torch.Tensor, p_scs_edge_attr : torch.Tensor,
                p_sls_edge_index : torch.Tensor, p_sls_edge_attr : torch.Tensor
                ) -> torch.Tensor:

        embeddings_scn = self.proj_scn(init)

        out_scs = self.GCN_scs(embeddings_scn.clone(),
                                p_scs_edge_index.clone(),
                                p_scs_edge_attr.clone())
        
        out_sls = self.GCN_sls(embeddings_scn.clone(),
                                p_sls_edge_index.clone(),
                                p_sls_edge_attr.clone())
        
        combined_scn = torch.stack([out_scs, out_sls], dim=0)

        fin_out_scn = self.attention_scn(combined_scn)

        return fin_out_scn

class HGC_CPT(nn.Module):
    def __init__(self, embedding_dim, device):
        super(HGC_CPT, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        self.proj_cpt = Projection(embedding_dim).to(device)

        self.GCN_cc = GCNConvEmbedding(embedding_dim).to(device)
        self.GCN_cac = GCNConvEmbedding(embedding_dim).to(device)
        self.GCN_csc = GCNConvEmbedding(embedding_dim).to(device)

        self.attention_cpt = MetaPathAttention(embedding_dim).to(device)

    def forward(self, 
                init : torch.Tensor, 
                p_cc_edge_index : torch.Tensor, p_cc_edge_attr : torch.Tensor,
                p_cac_edge_index : torch.Tensor, p_cac_edge_attr : torch.Tensor,
                p_csc_edge_index : torch.Tensor, p_csc_edge_attr : torch.Tensor
                ) -> torch.Tensor:

        embeddings_cpt = self.proj_cpt(init)

        out_cc  = self.GCN_cc(embeddings_cpt.clone(),
                                p_cc_edge_index.clone(),
                                p_cc_edge_attr.clone())
        out_cac = self.GCN_cac(embeddings_cpt.clone(),
                                p_cac_edge_index.clone(),
                                p_cac_edge_attr.clone())
        out_csc = self.GCN_csc(embeddings_cpt.clone(),
                                p_csc_edge_index.clone(),
                                p_csc_edge_attr.clone())
        
        combined_cpt = torch.stack([out_cc, out_cac, out_csc], dim=0)

        fin_out_cpt = self.attention_cpt(combined_cpt)

        return fin_out_cpt

class HGC_ALL(nn.Module):
    def __init__(self, embedding_dim, device):
        super(HGC_ALL, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        self.proj_lrn = Projection(embedding_dim).to(device)
        self.proj_scn = Projection(embedding_dim).to(device)
        self.proj_cpt = Projection(embedding_dim).to(device)

        self.GCN_lsl = GCNConvEmbedding(embedding_dim).to(device)

        self.GCN_cc = GCNConvEmbedding(embedding_dim).to(device)
        self.GCN_cac = GCNConvEmbedding(embedding_dim).to(device)
        self.GCN_csc = GCNConvEmbedding(embedding_dim).to(device)

        self.GCN_scs = GCNConvEmbedding(embedding_dim).to(device)
        self.GCN_sls = GCNConvEmbedding(embedding_dim).to(device)

        self.attention_cpt = MetaPathAttention(embedding_dim).to(device)
        self.attention_scn = MetaPathAttention(embedding_dim).to(device)

    def forward(self, 
                lrn_init : torch.Tensor, 
                scn_init : torch.Tensor, 
                cpt_init : torch.Tensor, 
                p_lsl_edge_index : torch.Tensor, p_lsl_edge_attr : torch.Tensor,
                p_scs_edge_index : torch.Tensor, p_scs_edge_attr : torch.Tensor,
                p_sls_edge_index : torch.Tensor, p_sls_edge_attr : torch.Tensor,
                p_cc_edge_index : torch.Tensor, p_cc_edge_attr : torch.Tensor,
                p_cac_edge_index : torch.Tensor, p_cac_edge_attr : torch.Tensor,
                p_csc_edge_index : torch.Tensor, p_csc_edge_attr : torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        embeddings_lrn = self.proj_lrn(lrn_init)
        embeddings_scn = self.proj_scn(scn_init)
        embeddings_cpt = self.proj_cpt(cpt_init)

        out_lsl = self.GCN_lsl(embeddings_lrn.to(self.device),
                               p_lsl_edge_index.to(self.device),
                               p_lsl_edge_attr.to(self.device))
        
        out_scs = self.GCN_scs(embeddings_scn.clone().to(self.device),
                                p_scs_edge_index.to(self.device),
                                p_scs_edge_attr.to(self.device))
        out_sls = self.GCN_sls(embeddings_scn.to(self.device),
                                p_sls_edge_index.to(self.device),
                                p_sls_edge_attr.to(self.device))
        
        out_cc  = self.GCN_cc(embeddings_cpt.clone().to(self.device),
                                p_cc_edge_index.to(self.device),
                                p_cc_edge_attr.to(self.device))
        out_cac = self.GCN_cac(embeddings_cpt.clone().to(self.device),
                                p_cac_edge_index.to(self.device),
                                p_cac_edge_attr.to(self.device))
        out_csc = self.GCN_csc(embeddings_cpt.to(self.device),
                                p_csc_edge_index.to(self.device),
                                p_csc_edge_attr.to(self.device))
        
        combined_cpt = torch.stack([out_cc, out_cac, out_csc], dim=0)
        combined_scn = torch.stack([out_scs, out_sls], dim=0)

        fin_out_cpt = self.attention_cpt(combined_cpt)
        fin_out_scn = self.attention_scn(combined_scn)

        return out_lsl, fin_out_scn, fin_out_cpt

if __name__ == '__main__':
    hgcdr = HGCDataReader()
    uids, inits, p_matrixes = hgcdr.load_data_from_db()
    model = HGC_ALL(32, 'cpu')

    lrn_, scn_, cpt_ = model(inits[0], inits[1], inits[2],
                              p_matrixes[0][0], p_matrixes[0][1],
                              p_matrixes[1][0], p_matrixes[1][1],
                              p_matrixes[2][0], p_matrixes[2][1],
                              p_matrixes[3][0], p_matrixes[3][1],
                              p_matrixes[4][0], p_matrixes[4][1],
                              p_matrixes[5][0], p_matrixes[5][1],
                            )

    print(lrn_.sum(), scn_.sum(), cpt_.sum())

