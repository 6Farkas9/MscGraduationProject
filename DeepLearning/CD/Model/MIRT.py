import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
import torch.nn as nn

class MIRT(nn.Module):
    def __init__(self):
        super(MIRT,self).__init__()

    # def forward(self, p_u, d_v, beta_v):
    def forward(self, p_u: torch.Tensor, d_v: torch.Tensor, beta_v: torch.Tensor) -> torch.Tensor:
        # print('p_u:{},d_v:{},beta_v:{}'.format(p_u.shape,d_v.shape,beta_v.shape))
        if p_u.dim() == 1:
            p_u = p_u.unsqueeze(0)
        if d_v.dim() == 1:
            d_v = d_v.unsqueeze(0)
        if beta_v.dim() == 1:
            beta_v = beta_v.unsqueeze(0)
        
        output = torch.sigmoid(torch.einsum('ij,ij->i', p_u, d_v) + beta_v)
        return output