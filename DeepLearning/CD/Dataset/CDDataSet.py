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

# HGCdatareader计算出lrn/scn/cpt的初始嵌入init
# 经过hgc计算获得lrn/scn/cpt的静态嵌入，作为 z
# 使用z计算出h
# h获得拼接后的矩阵h_expended
# 通过dtr计算出p/d/β
# 通过mirt计算出概率

class CDDataset(Dataset):
    def __init__(self, data, uids, lrn_init, max_step):
        super(CDDataset,self).__init__()
        self.data = data
        self.max_step = max_step

        self.lrn_uids = uids[0]
        self.scn_uids = uids[1]
        self.cpt_uids = uids[2]

        # datareader提供的：train_data, uids, inits, p_matrixes, lrn_scn_mat
        # HGC需要的：
        # 如果在dataset内提供scn的子集，那么随dataset返回的应该也包含每个的子图？（不定长）
        # 所以只能在外面获取对应的scn集合，不能在内部返回，内部处理索引和掩码

        self.idx2lrn_uid = {idx : lrn_uid for lrn_uid, idx in uids[0].items()}

        self.lrn_init = lrn_init  # shape: [num_users, embed_dim]

        self.scn_seq_indices = torch.zeros(len(self.lrn_uids), self.max_step, dtype=torch.long)
        self.scn_seq_masks = torch.zeros(len(self.lrn_uids), self.max_step, dtype=torch.float32)

        self.results = torch.zeros(len(self.lrn_uids), self.max_step, dtype=torch.float32)
        for lrn_uid in self.data:
            scn_seq = self.data[lrn_uid][0]
            result = self.data[lrn_uid][1]

            row = self.lrn_uids[lrn_uid]
            seq_indices = [self.scn_uids[scn_uid] for scn_uid in scn_seq]
            valid_len = min(len(seq_indices), self.max_step)

            start_idx = self.max_step - valid_len
            self.scn_seq_indices[row][start_idx:] = torch.tensor(seq_indices[-valid_len:], dtype=torch.long)
            self.results[row][start_idx:] = torch.tensor(result[-valid_len:], dtype=torch.float32)

            self.scn_seq_masks[row][start_idx:] = 1.0

    
    def __len__(self):
        return len(self.lrn_uids)

    def __getitem__(self, idx):
        # 对于单个学习者：
        # 1. 该学习者的初始嵌入 - 该学习者的idx
        # 2. 该学习者的相关的scn的初始嵌入 - 场景的idx
        # 3. 该学习者的的真实的对比数据 - 有问题，好像还是需要
        # 首先要明确的是：dataset给出的数据首先是输入到HGC中的，所以HGC需要的数据一个也不能少
        # 然后是以学习者为键值返回数据的话，
        #     学习者要取子集，场景也要取子集
        #     这样就要求每个要返回一个场景数的tensor

        # lrn_uid = self.idx2lrn_uid[idx]

        learner_init = self.lrn_init[idx]

        scn_seq_index = self.scn_seq_indices[idx]
        scn_seq_mask = self.scn_seq_masks[idx]

        result = self.results[idx]

        return {
            'learner_idx' : idx,
            'learner_init' : learner_init,
            'scn_seq_index' : scn_seq_index,
            'scn_seq_mask' : scn_seq_mask,
            'result' : result
        }
    
    def collate_fn(self, batch):
        learner_idx = torch.stack([item['learner_idx'] for item in batch])
        learner_init = torch.stack([item['learner_init'] for item in batch])
        scn_seq_index = torch.stack([item['scn_seq_index'] for item in batch])
        scn_seq_mask = torch.stack([item['scn_seq_mask'] for item in batch])
        result = torch.stack([item['result'] for item in batch])

        # sub_p_lsl = subgraph(learner_idx, edge_index=self.p_lsl.edge_index, edge_attr=self.p_lsl.edge_attr, num_nodes=self.p_lsl.x.size(0))

        return {
            'learner_idx' : learner_idx,
            'learner_init' : learner_init,
            'scn_seq_index' : scn_seq_index,
            'scn_seq_mask' : scn_seq_mask,
            'result' : result
        }
    
if __name__ == '__main__':
    cddr = CDDataReader()
    
    td, md, uids, inits, p_matrixes = cddr.load_Data_from_db()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    cdds = CDDataset(md, uids, inits[0], 128)
    cddl = DataLoader(cdds, batch_size=32, shuffle=True, num_workers=3, **dataloader_kwargs)

    for item in tqdm(cddl):
        # print(item['scn_seq_index'].shape)
        lrn_idx = item['learner_idx']
        learner_init = item['learner_init']
        scn_seq_index = item['scn_seq_index']
        scn_seq_mask = item['scn_seq_mask']
        result = item['result']

        print(result.shape)

        # 这里要调用HGC获取静态嵌入也就是z，其中learner是子图调用，scn和cpt都是全图调用
        # 实际使用的时候可以子图调用，这里是因为使用dataset无法返回

        # 获得三个tensor ： lrn， scn， cpt，这个和batch_size无关
        # 直接输入到cd中
        # cd先拼接scn和cpt
        # 然后根据

