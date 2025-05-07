import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
import pandas as pd
from tqdm import tqdm
        
def get_H_Data(stu_exer, exer, topic, z_sharp, z_star, device):
    # h_u = []
    # h_v = []
    # h_c = []

    # for stu_row in stu_exer:
    #     h_u.append(sum(z_sharp[stu_row]) / len(stu_row))
    # h_u = torch.stack(h_u).to(device)
    # h_v = z_star[list(exer.keys())]
    # h_c = z_star[list(topic.keys())]

    # return h_u, h_v, h_c

    # 使用torch操作替代Python循环
    # 预转换所有索引为张量
    stu_indices = [torch.tensor(stu_row, device=device) for stu_row in stu_exer]
    
    # 使用torch.mean保持梯度
    h_u = torch.stack([
        torch.index_select(z_sharp, 0, indices).mean(dim=0)
        for indices in stu_indices
    ])
    
    # 使用index_select保持梯度
    h_v = z_star.index_select(0, torch.tensor(list(exer.keys()), device=device))
    h_c = z_star.index_select(0, torch.tensor(list(topic.keys()), device=device))
    
    return h_u, h_v, h_c

def get_PDBeta_Data(h_u, h_v, h_c, model, embedding_dim):
    # p_u = []
    # d_v = []
    # beta_v = []

    num_stu = h_u.size(0)
    num_topic = h_c.size(0)
    num_exer = h_v.size(0)

    h_u_expended = h_u.unsqueeze(1).expand(-1, num_topic, -1)
    h_c_expended = h_c.unsqueeze(0).expand(num_stu, -1, -1)
    combine_uc = torch.cat((h_u_expended,h_c_expended),dim=-1).view(-1, 2 * embedding_dim)
    h_v_expended = h_v.unsqueeze(1).expand(-1, num_topic, -1)
    h_c_expended = h_c.unsqueeze(0).expand(num_exer, -1, -1)
    combine_dc = torch.cat((h_v_expended,h_c_expended),dim=-1).view(-1, 2 * embedding_dim)

    p_u, d_v, beta_v = model([combine_uc, combine_dc, h_v])

    p_u_res = p_u.squeeze().view(num_stu, num_topic)
    d_v_res = d_v.squeeze().view(num_exer, num_topic)
    beta_v_res = beta_v.squeeze()

    return p_u_res, d_v_res, beta_v_res
