import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import subgraph
from torch_geometric.data import Data

from RR.Dataset.RRDataReader import RRDataReader
from RR.Dataset.RRDataSet import RRDataSet
from HGC.Model.HGC import HGC_LRN, HGC_SCN, HGC_CPT
from RR.Model.RR import RR

parser = argparse.ArgumentParser(description='RR')
parser.add_argument('--batch_size',type=int,default=32,help='number of batch size to train (defauly 32 )')
parser.add_argument('--epochs',type=int,default=32,help='number of epochs to train (defauly 32 )')
parser.add_argument('--lr',type=float,default=0.01,help='number of learning rate')
parser.add_argument('--embedding_dim',type=int,default=32,help='the number of the embedding_dim')
parser.add_argument('--hidden_dim',type=int,default=30,help='the number of the hidden_dim')
parser.add_argument('--max_step',type=int,default=128,help='the number of max step')
parser.add_argument('--sample_num',type=int,default=100,help='the number of max sample')

if __name__ == '__main__':
    parsers = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    print(device)

    train_data, master_data, uids, inits, p_matrixes, dynamic_scn_mat = RRDataReader(parsers.sample_num).load_data_from_db()

    # (self.learners_init, self.scenes_init, self.concepts_init), \
    # (self.p_lsl, self.p_scs, self.p_sls, self.p_cc, self.p_cac, self.p_csc,)
    lrn_uids, scn_uids, cpt_uids = uids
    learners_init, scenes_init, concepts_init = inits

    (p_lsl_edge_index, p_lsl_edge_attr), \
    (p_scs_edge_index, p_scs_edge_attr), \
    (p_sls_edge_index, p_sls_edge_attr), \
    (p_cc_edge_index, p_cc_edge_attr), \
    (p_cac_edge_index, p_cac_edge_attr), \
    (p_csc_edge_index, p_csc_edge_attr) = p_matrixes

    train_dataset = RRDataSet(train_data, uids, learners_init, parsers.max_step)
    train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

    # model_hgc = HGC(parsers.embedding_dim, device).to(device)
    
    model_hgc_lrn = HGC_LRN(parsers.embedding_dim, device).to(device)
    model_hgc_scn = HGC_SCN(parsers.embedding_dim, device).to(device)
    model_hgc_cpt = HGC_CPT(parsers.embedding_dim, device).to(device)

    model_rr = RR(parsers.embedding_dim, parsers.hidden_dim, device).to(device)

    print(model_hgc_lrn)
    print(model_hgc_scn)
    print(model_hgc_cpt)
    print(RR)

    tbar = tqdm(train_dataloader)
    for item in tbar:
        model_hgc_lrn.train()
        model_hgc_scn.train()
        model_hgc_cpt.train()
        model_rr.train()
        learner_idx = item['learner_idx']
        learner_init = item['learner_init']
        scn_seq_index = item['scn_seq_index']
        scn_seq_mask = item['scn_seq_mask']
        r_uk_data = item['r_uk_data']

        p_lsl = Data(x = learners_init, edge_index = p_lsl_edge_index, edge_attr = p_lsl_edge_attr)
        sub_p_lsl = p_lsl.subgraph(learner_idx)

        lrn_emb = model_hgc_lrn(sub_p_lsl.x.to(device), 
                                sub_p_lsl.edge_index.to(device), sub_p_lsl.edge_attr.to(device)
                                )
        scn_emb = model_hgc_scn(scenes_init.to(device), 
                                p_scs_edge_index.to(device), p_scs_edge_attr.to(device),
                                p_sls_edge_index.to(device), p_sls_edge_attr.to(device)
                                )
        cpt_emb = model_hgc_cpt(concepts_init.to(device), 
                                p_cc_edge_index.to(device), p_cc_edge_attr.to(device),
                                p_cac_edge_index.to(device), p_cac_edge_attr.to(device), 
                                p_csc_edge_index.to(device), p_csc_edge_attr.to(device)
                                )

        scn_dynamic_emb = torch.sparse.mm(dynamic_scn_mat.to(device), cpt_emb)

        # lrn_static : torch.tensor, 
        # scn_dynamic : torch.tensor,
        # scn_seq_index : torch.tensor,
        # scn_seq_mask : torch.tensor,
        # cpt_static : torch.tensor
        r, h_lrn, h_cpt =  model_rr(lrn_emb, 
                                    scn_dynamic_emb, 
                                    scn_seq_index.to(device),
                                    scn_seq_mask.to(device),
                                    cpt_emb)
        
        print(r.shape, h_lrn.shape, h_cpt.shape)




    
