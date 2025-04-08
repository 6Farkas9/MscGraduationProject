import numpy as np
from torch.utils.data import Dataset

class IPDKTDataset(Dataset):
    def __init__(self,data,kc_num,data_len):
        super(IPDKTDataset,self).__init__()
        self.data = data
        self.kc_num = kc_num
        self.data_len = data_len # 同一学生做题的最大次数
        self.stu_id = []
        for stuid in data.index:
            temp_data = self.data.loc[stuid]
            if len(temp_data['que_id']) < 2:
                continue
            self.stu_id.append(stuid)

    def __len__(self):
        return len(self.stu_id)
    
    def __getitem__(self, index):
        stuid = self.stu_id[index]
        que_,kc_,cor_,cor_rate_ = self.data.loc[stuid]
        current_len = len(que_)
        kc = []
        cor_rate = []
        que = np.zeros(self.data_len,dtype=int)
        cor = np.zeros(self.data_len,dtype=int)

        if current_len <= self.data_len:
            que[-current_len:] = que_
            cor[-current_len:] = cor_
            for i in range(current_len):
                kc.insert(0,kc_[current_len - i - 1])
                cor_rate.insert(0,np.array(cor_rate_[current_len - i - 1]))
            for i in range(self.data_len - current_len):
                kc.insert(0,[0] * self.kc_num)
                # cor_rate.insert(0,[0.0] * self.kc_num)
                cor_rate.insert(0,np.zeros(self.kc_num, dtype=float))
        else:
            que[:] = que_[-self.data_len:]
            cor[:] = cor_[-self.data_len:]
            for i in range(self.data_len):
                kc.insert(0,kc_[current_len - i - 1])
                cor_rate.insert(0,np.array(cor_rate_[current_len - i - 1]))
        
        onehot = self.onehot(kc.copy(),cor.copy())
        que = que[1:]
        cor = cor[1:]
        cor_rate = cor_rate[1:]

        kc = kc[1:]
        kc_res = np.zeros([self.data_len-1,self.kc_num],dtype=float)

        for i in range(self.data_len - 1):
            for j in kc[i]:
                kc_res[i][j] = 1

        return onehot,que,kc_res,cor,np.array(cor_rate)

    def onehot(self,kc,cor):
        res = np.zeros(shape = [self.data_len - 1,2 * self.kc_num])
        for i in range(self.data_len - 1):
            if cor[i] > 0:
                for item in kc[i]:
                    res[i][item] = 1
            else:
                for item in kc[i]:
                    res[i][item + self.kc_num] = 1
        return res