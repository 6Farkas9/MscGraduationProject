import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import numpy as np
from torch.utils.data import Dataset

# class DADDataset(Dataset):
#     def __init__(self, p_u, d_v, beta_v, exer_ids, log_data):
#         super(DADDataset,self).__init__()
#         self.p_u = p_u
#         self.d_v = d_v
#         self.beta_v = beta_v
#         self.exer_ids = exer_ids
#         self.log_data = log_data
    
#     def __len__(self):
#         return len(self.log_data)

#     def __getitem__(self, index):
#         stu_id = self.log_data[index][0].item()
#         exer_id = self.log_data[index][1].item()
#         correct = self.log_data[index][2].item()
#         exer_id = self.exer_ids[exer_id]
#         p_u_stu = self.p_u[stu_id]
#         d_v_exer = self.d_v[exer_id]
#         beta_v_exer = self.beta_v[exer_id]
        
#         return p_u_stu,d_v_exer,beta_v_exer,correct

class DADDataset(Dataset):
    def __init__(self, exer_ids, log_data):
        super(DADDataset,self).__init__()
        self.exer_ids = exer_ids
        self.log_data = log_data
    
    def __len__(self):
        return len(self.log_data)

    def __getitem__(self, index):
        stu_id = self.log_data[index][0].item()
        exer_id = self.log_data[index][1].item()
        correct = self.log_data[index][2].item()
        exer_id = self.exer_ids[exer_id]
        
        return stu_id,exer_id,correct

