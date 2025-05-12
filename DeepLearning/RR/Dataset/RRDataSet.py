import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class RRDataSet(Dataset):

    def __init__(self, data, uids, lrn_init, max_step):
        super(RRDataSet, self).__init__()
        self.data = data

        self.lrn_uids = uids[0]
        self.scn_uids = uids[1]
        self.cpt_uids = uids[2]

        self.idx2lrn_uid = {idx : lrn_uid for lrn_uid, idx in uids[0].items()}

        self.lrn_init = lrn_init  # shape: [num_users, embed_dim]

        self.scn_seq_indices = torch.zeros(len(self.lrn_uids), max_step, dtype=torch.long)
        self.scn_seq_masks = torch.zeros(len(self.lrn_uids), max_step, dtype=torch.float32)
        for lrn_uid in self.data:
            scn_seq = self.data[lrn_uid][0]
            row = self.lrn_uids[lrn_uid]
            seq_indices = [self.scn_uids[scn_uid] for scn_uid in scn_seq]
            valid_len = min(len(seq_indices), max_step)

            start_idx = max_step - valid_len
            self.scn_seq_indices[row][start_idx:] = torch.tensor(seq_indices[-valid_len:], dtype=torch.long)

            self.scn_seq_masks[row][start_idx:] = 1.0

    def __len__(self):
        return len(self.lrn_uids)
    
    def __getitem__(self, idx):

        # 对于单个学习者，返回的是
        # 1. 对应学习者的行的初始嵌入（固定大小）
        # 2. 场景总体的初始嵌入（固定大小）
        # 3. 知识点总体的初始嵌入（固定大小）
        # 4. data中对应键值的列表（非固定大小）？
        # 直接计算出4岁对应的tensor：大小为max_step，每个位置存储对应scn的行号，mask为是否有效的掩码

        lrn_uid = self.idx2lrn_uid[idx]

        learner_init = self.lrn_init[idx]
        # scene_init = self.scn_init
        # concept_init = self.cpt_init

        scn_seq_index = self.scn_seq_indices[idx]
        scn_seq_mask = self.scn_seq_masks[idx]

        r_data = self.data[lrn_uid][1]

        # return learner_init, scene_init, concept_init, scn_seq_index, scn_seq_mask
        return {
            'learner_idx' : idx,
            'learner_init' : learner_init,
            'scn_seq_index' : scn_seq_index,
            'scn_seq_mask' : scn_seq_mask,
            'r_uk_data' : r_data
        }
    
    def collate_fn(self, batch):
        learner_idx = torch.stack([item['learner_idx'] for item in batch])
        learner_init = torch.stack([item['learner_init'] for item in batch])
        scn_seq_index = torch.stack([item['scn_seq_index'] for item in batch])
        scn_seq_mask = torch.stack([item['scn_seq_mask'] for item in batch])
        r_uk_data = torch.stack([item['r_uk_data'] for item in batch])

        # sub_p_lsl = subgraph(learner_idx, edge_index=self.p_lsl.edge_index, edge_attr=self.p_lsl.edge_attr, num_nodes=self.p_lsl.x.size(0))

        return {
            'learner_idx' : learner_idx,
            'learner_init' : learner_init,
            'scn_seq_index' : scn_seq_index,
            'scn_seq_mask' : scn_seq_mask,
            'r_uk_data' : r_uk_data
        }
