import torch
import pandas as pd

from tqdm import tqdm

class DADDataReader():
    def __init__(self, path, device):
        self.path = path
        self.device = device

    def load_Data(self):
        reader_file = pd.read_csv(self.path, header=0)

        num_stu = 0
        stu_all = {} # {stu_origin_id:newid,....}
        stu_exer = []
        len_row, _ = reader_file.shape

        res_data = []

        for i in range(len_row):
            stu_name = reader_file.iloc[i,0]
            exer_id = reader_file.iloc[i,1]
            correct = reader_file.iloc[i,2]

            if stu_name not in stu_all.keys():
                stu_all[stu_name] = num_stu
                stu_exer.append([])
                num_stu += 1
            
            stu_id = stu_all[stu_name]
            if exer_id not in stu_exer[stu_id]:
                stu_exer[stu_id].append(exer_id)
            res_data.append([stu_id,exer_id,correct])

        res_data = torch.tensor(res_data, dtype=torch.long).to(self.device)

        return res_data, stu_exer
