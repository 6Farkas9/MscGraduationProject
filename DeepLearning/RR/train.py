import os
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

from Dataset.RRDataReader import RRDataReader
from Dataset.RRDataSet import RRDataSet
from HGC.Model.HGC import HGC

from Model.RR import RR

parser = argparse.ArgumentParser(description='RR')
parser.add_argument('--batch_size',type=int,default=32,help='number of batch size to train (defauly 32 )')
parser.add_argument('--epochs',type=int,default=32,help='number of epochs to train (defauly 32 )')
parser.add_argument('--lr',type=float,default=0.01,help='number of learning rate')
parser.add_argument('--embedding_dim',type=int,default=32,help='the number of the embedding_dim')
parser.add_argument('--max_step',type=int,default=128,help='the number of max step')
parser.add_argument('--sample_num',type=int,default=100,help='the number of max sample')

if __name__ == '__main__':
    parsers = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    print(device)

    train_data, master_data, uids, inits, p_matrixes, dynamic_indices, dynamic_scatter_idx = RRDataReader(parsers.sample_num).load_data_from_db()

    lrn_uids, scn_uids, cpt_uids = uids
    learners_init, scenes_init, concepts_init = inits
    p_lsl, p_cc, p_cac, p_csc, p_scs, p_sls = p_matrixes

    train_dataset = RRDataSet(train_data, uids, learners_init, parsers.max_step)
    train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

    model_hgc = HGC(parsers.embedding_dim, device).to(device)
    model_rr = RR(parsers.embedding_dim, device).to(device)

    print(model_hgc, model_rr)

    tbar = tqdm(train_dataloader)
    for item in tbar:
        model_hgc.train()
        model_rr.train()
        learner_idx = item['learner_idx'].to(device)
        learner_init = item['learner_init'].to(device)
        scn_seq_index = item['scn_seq_index'].to(device)
        scn_seq_mask = item['scn_seq_mask'].to(device)

        print(p_lsl.edge_index.shape)
        print(p_lsl.edge_attr.shape)

        p_lsl.x = learners_init
        # subgraph_edge_index, subgraph_edge_attr = subgraph(learner_idx.to('cpu'), edge_index=p_lsl.edge_index, edge_attr=p_lsl.edge_attr, num_nodes=p_lsl.x.size(0))

        sub_p_lsl = p_lsl.subgraph(learner_idx.to('cpu'))
        # print(learner_idx.shape)
        # print(subgraph_edge_index.shape)
        # print(subgraph_edge_attr.shape)
        
        # sub_p_lsl = Data(x = learner_init, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr)

        (lrn_emb, scn_emb, cpt_emb) = model_hgc(
            (learner_init, scenes_init.to(device) ,concepts_init.to(device)), 
            (sub_p_lsl.to(device), p_cc.to(device), p_cac.to(device), p_csc.to(device), p_scs.to(device), p_sls.to(device)))
        print(lrn_emb.shape, scn_emb.shape, cpt_emb.shape)
        break



    
