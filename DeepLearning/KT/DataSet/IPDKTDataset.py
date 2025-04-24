import numpy as np
from torch.utils.data import Dataset

class IPDKTDataset(Dataset):
    def __init__(self,data, cpt_num, data_len):
        super(IPDKTDataset,self).__init__()
        self.data = data
        self.cpt_num = cpt_num
        self.data_len = data_len # 最多考虑同一学生的多少次做题记录

        self.lrn_ids = []
        for lrn_id in data.index:
            temp_data = self.data.loc[lrn_id]
            if len(temp_data['correct']) < 2:
                continue
            self.lrn_ids.append(lrn_id)

    def __len__(self):
        return len(self.lrn_ids)
    
    def __getitem__(self, index):
        lrn_id = self.lrn_ids[index]
        cpt_ids, cor = self.data.loc[lrn_id]
        current_len = len(cor)
        cpt = []

        cor_expended = np.zeros(self.data_len, dtype=int)

        if current_len <= self.data_len:
            cor_expended[-current_len : ] = cor
            for i in range(current_len):
                cpt.insert(0, cpt_ids[current_len - i - 1])
            for i in range(self.data_len - current_len):
                cpt.insert(0, [0] * self.cpt_num)
        else:
            cor_expended = cor[-self.data_len : ]
            for i in range(self.data_len):
                cpt.insert(0, cpt_ids[current_len - i - 1])
        
        onehot = self.onehot(cpt.copy(), cor_expended.copy())

        cor_expended = np.array(cor_expended[1:], dtype=float)
        cpt = cpt[1:]
        cpt_matrix = np.zeros([self.data_len - 1, self.cpt_num], dtype=float)

        for i in range(self.data_len - 1):
            for j in cpt[i]:
                cpt_matrix[i][j] = 1

        return onehot, cpt_matrix, cor_expended

    def onehot(self, cpt, cor):
        res = np.zeros(shape = [self.data_len - 1, 2 * self.cpt_num])
        for i in range(self.data_len - 1):
            if sum(cpt[i]) == 0:
                continue
            if cor[i] > 0:
                for item in cpt[i]:
                    res[i][item] = 1
            else:
                for item in cpt[i]:
                    res[i][item + self.cpt_num] = 1
        return res