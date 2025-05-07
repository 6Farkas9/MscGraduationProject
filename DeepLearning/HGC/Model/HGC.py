import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.data import Data
from typing import Dict, Tuple

import sys
sys.path.append('..')
# sys.path.append('../..')
from HGC.Dataset.HGCDataReader import HGCDataReader

class MetaPathAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(MetaPathAttention, self).__init__()
        self.W_att = nn.Linear(embedding_dim, 1)  # 可训练参数（原文中的 W_att^mp）
        self.activation = nn.ReLU()

    def forward(self, embeddings : torch.tensor) -> torch.tensor:
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

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNConvEmbedding, self).__init__()
        # 第一层卷积层，输入维度为in_channels，输出维度为hidden_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # 第二层卷积层，输入维度为hidden_channels，输出维度为hidden_channels
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # 第三层卷积层，输入维度为hidden_channels，输出维度为out_channels
        self.conv3 = GCNConv(hidden_channels, out_channels)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, data : Data) -> torch.tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # 第一层卷积 + ReLU
        x_out = self.conv1(x, edge_index, edge_attr)
        x_out = self.relu(x_out)
        # 第二层卷积 + ReLU
        x_out = self.conv2(x_out, edge_index, edge_attr)
        x_out = self.relu(x_out)
        # 第三层卷积 + ReLU
        x_out = self.conv3(x_out, edge_index, edge_attr)
        x_out = self.relu(x_out)
        return x_out
    
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

        self.GCN_lsl = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)

    def forward(self,
                init : torch.tensor,
                p_matrix : torch.tensor
                ) -> torch.tensor:
        # (self.learners_init, self.scenes_init, self.concepts_init)
        # (self.p_lsl, self.p_cc, self.p_cac, self.p_csc, self.p_scs, self.p_sls)
        

        embeddings_lrn = self.proj_lrn(init)

        p_lsl = p_matrix
        p_lsl.x = embeddings_lrn.clone()
        # p_lsl = p_lsl.to(self.device)

        out_lsl = self.GCN_lsl(p_lsl)

        return out_lsl
    
class HGC_SCN(nn.Module):
    def __init__(self, embedding_dim, device):
        super(HGC_SCN, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        self.proj_scn = Projection(embedding_dim).to(device)

        self.GCN_scs = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)
        self.GCN_sls = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)

        self.attention_scn = MetaPathAttention(embedding_dim).to(device)

    def forward(self, 
                init : torch.tensor, 
                p_matrix : tuple[torch.tensor, torch.tensor]
                ) -> torch.tensor:
        # (self.learners_init, self.scenes_init, self.concepts_init)
        # (self.p_lsl, self.p_cc, self.p_cac, self.p_csc, self.p_scs, self.p_sls)
        
        p_scs = p_matrix[0]
        p_sls = p_matrix[1]

        embeddings_scn = self.proj_scn(init)

        p_scs.x = embeddings_scn.clone()
        p_sls.x = embeddings_scn.clone()

        # p_scs = p_scs.to(self.device)
        # p_sls = p_sls.to(self.device)

        out_scs = self.GCN_scs(p_scs)
        out_sls = self.GCN_sls(p_sls)
        
        combined_scn = torch.stack([out_scs, out_sls], dim=0)

        fin_out_scn = self.attention_scn(combined_scn)

        return fin_out_scn

class HGC_CPT(nn.Module):
    def __init__(self, embedding_dim, device):
        super(HGC_CPT, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        self.proj_cpt = Projection(embedding_dim).to(device)

        self.GCN_cc = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)
        self.GCN_cac = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)
        self.GCN_csc = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)

        self.attention_cpt = MetaPathAttention(embedding_dim).to(device)

    def forward(self, 
                init : torch.tensor, 
                p_matrix : tuple[torch.tensor, torch.tensor, torch.tensor]
                ) -> torch.tensor:
        # (self.learners_init, self.scenes_init, self.concepts_init)
        # (self.p_lsl, self.p_cc, self.p_cac, self.p_csc, self.p_scs, self.p_sls)
        
        p_cc = p_matrix[0]
        p_cac = p_matrix[1]
        p_csc = p_matrix[2]

        embeddings_cpt = self.proj_cpt(init)

        p_cc.x = embeddings_cpt.clone()
        p_cac.x = embeddings_cpt.clone()
        p_csc.x = embeddings_cpt.clone()

        # p_cc = p_cc.to(self.device)
        # p_cac = p_cac.to(self.device)
        # p_csc = p_csc.to(self.device)

        out_cc  = self.GCN_cc(p_cc)
        out_cac = self.GCN_cac(p_cac)
        out_csc = self.GCN_csc(p_csc)
        
        combined_cpt = torch.stack([out_cc, out_cac, out_csc], dim=0)

        fin_out_cpt = self.attention_cpt(combined_cpt)

        return fin_out_cpt

class HGC(nn.Module):
    def __init__(self, embedding_dim, device):
        super(HGC, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim

        self.proj_lrn = Projection(embedding_dim).to(device)
        self.proj_scn = Projection(embedding_dim).to(device)
        self.proj_cpt = Projection(embedding_dim).to(device)

        self.GCN_lsl = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)

        self.GCN_cc = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)
        self.GCN_cac = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)
        self.GCN_csc = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)

        self.GCN_scs = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)
        self.GCN_sls = GCNConvEmbedding(embedding_dim, embedding_dim, embedding_dim).to(device)

        self.attention_cpt = MetaPathAttention(embedding_dim).to(device)
        self.attention_scn = MetaPathAttention(embedding_dim).to(device)

    def forward(self, 
                inits : tuple[torch.tensor, torch.tensor, torch.tensor], 
                p_matrix : tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]
                ) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        # (self.learners_init, self.scenes_init, self.concepts_init)
        # (self.p_lsl, self.p_cc, self.p_cac, self.p_csc, self.p_scs, self.p_sls)
        
        p_lsl = p_matrix[0]
        p_cc = p_matrix[1]
        p_cac = p_matrix[2]
        p_csc = p_matrix[3]
        p_scs = p_matrix[4]
        p_sls = p_matrix[5]

        embeddings_lrn = self.proj_lrn(inits[0])
        embeddings_scn = self.proj_scn(inits[1])
        embeddings_cpt = self.proj_cpt(inits[2])


        p_lsl.x = embeddings_lrn.clone().to(self.device)
        p_scs.x = embeddings_scn.clone().to(self.device)
        p_sls.x = embeddings_scn.clone().to(self.device)
        p_cc.x = embeddings_cpt.clone().to(self.device)
        p_cac.x = embeddings_cpt.clone().to(self.device)
        p_csc.x = embeddings_cpt.clone().to(self.device)

        out_lsl = self.GCN_lsl(p_lsl)
        out_scs = self.GCN_scs(p_scs)
        out_sls = self.GCN_sls(p_sls)
        out_cc  = self.GCN_cc(p_cc)
        out_cac = self.GCN_cac(p_cac)
        out_csc = self.GCN_csc(p_csc)
        
        combined_cpt = torch.stack([out_cc, out_cac, out_csc], dim=0)
        combined_scn = torch.stack([out_scs, out_sls], dim=0)

        fin_out_cpt = self.attention_cpt(combined_cpt)
        fin_out_scn = self.attention_scn(combined_scn)

        return out_lsl, fin_out_scn, fin_out_cpt

if __name__ == '__main__':
    HGCDataReader = HGCDataReader()
    uids, inits, p_matrixes = HGCDataReader.load_data_from_db()
    model = HGC(32, 'cpu')
    lrn_, scn_, cpt_ = model(inits, p_matrixes)

    print(lrn_.sum(), scn_.sum(), cpt_.sum())

