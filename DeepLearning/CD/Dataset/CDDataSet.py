import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
import numpy as np
from torch.utils.data import Dataset

from tqdm import tqdm
from torch.utils.data import DataLoader
from CD.Dataset.CDDataReader import CDDataReader

# HGCdatareader计算出lrn/unt/cpt的初始嵌入init
# 经过hgc计算获得lrn/unt/cpt的静态嵌入，作为 z
# 使用z计算出h
# h获得拼接后的矩阵h_expended
# 通过dtr计算出p/d/β
# 通过mirt计算出概率

class CDDataset(Dataset):
    def __init__(self, data, lrn_uids, cpt_uids, unt_uids, max_step):
        super(CDDataset,self).__init__()
        self.data = data
        self.max_step = max_step

        self.lrn_uids = lrn_uids
        self.cpt_uids = cpt_uids
        self.unt_uids = unt_uids

        self.lrn_id2uid = {lrn_uids[lrn_uid] : lrn_uid for lrn_uid in lrn_uids}

        self.unt_seq_indices = torch.zeros(len(self.lrn_uids), self.max_step, dtype=torch.long)
        self.unt_seq_masks = torch.zeros(len(self.lrn_uids), self.max_step, dtype=torch.float32)

        self.results = torch.zeros(len(self.lrn_uids), self.max_step, dtype=torch.float32)

        for lrn_uid in self.data:
            unt_seq = self.data[lrn_uid][0]
            result = self.data[lrn_uid][1]

            row = self.lrn_uids[lrn_uid]
            seq_indices = [self.unt_uids[unt_uid] for unt_uid in unt_seq]
            valid_len = min(len(seq_indices), self.max_step)

            start_idx = self.max_step - valid_len
            self.unt_seq_indices[row][start_idx:] = torch.tensor(seq_indices[-valid_len:], dtype=torch.long)
            self.results[row][start_idx:] = torch.tensor(result[-valid_len:], dtype=torch.float32)

            self.unt_seq_masks[row][start_idx:] = 1.0

            # 将无结果位置置为无效
            without_result_mask = (self.results[row] == -1)
            self.results[row][without_result_mask] = 0
            self.unt_seq_masks[row][without_result_mask] = 0.0
    
    def __len__(self):
        return len(self.lrn_uids)

    def __getitem__(self, idx):
        # 对于单个学习者：
        # 1. 该学习者的初始嵌入 - 该学习者的idx
        # 2. 该学习者的相关的unt的初始嵌入 - 场景的idx
        # 3. 该学习者的的真实的对比数据 - 有问题，好像还是需要
        # 首先要明确的是：dataset给出的数据首先是输入到HGC中的，所以HGC需要的数据一个也不能少
        # 然后是以学习者为键值返回数据的话，
        #     学习者要取子集，场景也要取子集
        #     这样就要求每个要返回一个场景数的tensor

        # lrn_uid = self.idx2lrn_uid[idx]

        unt_seq_index = self.unt_seq_indices[idx]
        unt_seq_mask = self.unt_seq_masks[idx]

        result = self.results[idx]

        return {
            'learner_uid' : self.lrn_id2uid[idx],
            'unt_seq_index' : unt_seq_index,
            'unt_seq_mask' : unt_seq_mask,
            'result' : result
        }
    
    def collate_fn(self, batch):
        learner_idx = torch.stack([item['learner_uid'] for item in batch])
        unt_seq_index = torch.stack([item['unt_seq_index'] for item in batch])
        unt_seq_mask = torch.stack([item['unt_seq_mask'] for item in batch])
        result = torch.stack([item['result'] for item in batch])

        # sub_p_lsl = subgraph(learner_idx, edge_index=self.p_lsl.edge_index, edge_attr=self.p_lsl.edge_attr, num_nodes=self.p_lsl.x.size(0))

        return {
            'learner_uid' : learner_idx,
            'unt_seq_index' : unt_seq_index,
            'unt_seq_mask' : unt_seq_mask,
            'result' : result
        }
    
if __name__ == '__main__':
    cddr = CDDataReader()

    cddr.set_are_uid('are_3fee9e47d0f3428382f4afbcb1004117')
    
    train_data, master_data, lrn_uids, cpt_uids, unt_uids, cpt_idx, unt_idx, edge_index, edge_attr, edge_type = cddr.load_Data_from_db()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    cdds = CDDataset(train_data, lrn_uids, cpt_uids, unt_uids, 128)
    cddl = DataLoader(cdds, batch_size=32, shuffle=True, num_workers=3, **dataloader_kwargs)

    for item in tqdm(cddl):
        # print(item['unt_seq_index'].shape)
        lrn_idx = item['learner_idx']
        unt_seq_index = item['unt_seq_index']
        unt_seq_mask = item['unt_seq_mask']
        result = item['result']

        print(lrn_idx.shape, unt_seq_index.shape, unt_seq_mask.shape, result.shape)


