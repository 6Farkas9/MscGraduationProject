import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import os
import torch
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from operator import itemgetter

from RR.Dataset.RRDataReader import RRDataReader
from RR.Dataset.RRDataSet import RRDataSet
from HGC.Model.HGC import HGC_LRN, HGC_SCN, HGC_CPT
from RR.Model.RR import RR

parser = argparse.ArgumentParser(description='RR')
parser.add_argument('--batch_size',type=int,default=32,help='number of batch size to train (defauly 32 )')
parser.add_argument('--epochs',type=int,default=2,help='number of epochs to train (defauly 32 )')
parser.add_argument('--lr',type=float,default=0.01,help='number of learning rate')
parser.add_argument('--embedding_dim',type=int,default=32,help='the number of the embedding_dim')
parser.add_argument('--hidden_dim',type=int,default=30,help='the number of the hidden_dim')
parser.add_argument('--max_step',type=int,default=200,help='the number of max step')
parser.add_argument('--sample_num',type=int,default=100,help='the number of max sample')
parser.add_argument('--lambda_reg',type=float,default=1e-3,help='the number of lambda_reg')

def RRloss(r_pred : torch.tensor, r : torch.tensor, h_lrn : torch.tensor, h_cpt : torch.tensor, lambda_reg : float):
    mse_loss = torch.mean((r_pred - r) ** 2)
    regularization_loss = torch.norm(h_lrn, p=2) + torch.norm(h_cpt, p=2)
    loss = mse_loss + lambda_reg * regularization_loss
    return loss

def save_final_hgc_data(uids, inits, p_matrixes, dynamic_unt_mat, datareader : RRDataReader):
    device = 'cpu'

    lrn_uids, unt_uids, cpt_uids = uids
    learners_init, units_init, concepts_init = inits

    (p_lsl_edge_index, p_lsl_edge_attr), \
    (p_scs_edge_index, p_scs_edge_attr), \
    (p_sls_edge_index, p_sls_edge_attr), \
    (p_cc_edge_index, p_cc_edge_attr), \
    (p_cac_edge_index, p_cac_edge_attr), \
    (p_csc_edge_index, p_csc_edge_attr) = p_matrixes

    HGC_pt_path = os.path.join(deeplearning_root, 'HGC', 'PT')
    HGC_LRN_use_path = os.path.join(HGC_pt_path, 'HGC_LRN_use.pt')
    HGC_SCN_use_path = os.path.join(HGC_pt_path, 'HGC_SCN_use.pt')
    HGC_CPT_use_path = os.path.join(HGC_pt_path, 'HGC_CPT_ues.pt')

    model_hgc_lrn = torch.jit.load(HGC_LRN_use_path)
    model_hgc_unt = torch.jit.load(HGC_SCN_use_path)
    model_hgc_cpt = torch.jit.load(HGC_CPT_use_path)

    model_hgc_lrn = model_hgc_lrn.to(device)
    model_hgc_unt = model_hgc_unt.to(device)
    model_hgc_cpt = model_hgc_cpt.to(device)

    model_hgc_lrn.eval()
    model_hgc_unt.eval()
    model_hgc_cpt.eval()

    with torch.no_grad():

        lrn_emb = model_hgc_lrn(learners_init, 
                                p_lsl_edge_index, p_lsl_edge_attr
                                )
        unt_emb = model_hgc_unt(units_init, 
                                p_scs_edge_index, p_scs_edge_attr,
                                p_sls_edge_index, p_sls_edge_attr
                                )
        cpt_emb = model_hgc_cpt(concepts_init, 
                                p_cc_edge_index, p_cc_edge_attr,
                                p_cac_edge_index, p_cac_edge_attr, 
                                p_csc_edge_index, p_csc_edge_attr
                                )

        unt_dynamic_emb = torch.sparse.mm(dynamic_unt_mat, cpt_emb)

    lrn_uids_list = [lrn_uid for lrn_uid, _ in sorted(lrn_uids.items(), key=itemgetter(1))]
    unt_uids_list = [unt_uid for unt_uid, _ in sorted(unt_uids.items(), key=itemgetter(1))]
    cpt_uids_list = [cpt_uid for cpt_uid, _ in sorted(cpt_uids.items(), key=itemgetter(1))]

    # 保存lrn_emb，unt_emb，cpt_emb
    datareader.save_final_hgc_data(
        lrn_uids_list, unt_uids_list, cpt_uids_list, 
        lrn_emb, unt_dynamic_emb, cpt_emb,
    )

    return cpt_uids_list, (lrn_emb, unt_dynamic_emb, cpt_emb)

def save_final_rr_data(
        lrn_uid, unt_uids, 
        lrn_emb, unt_dynamic_emb, cpt_emb,
        cpt_uids_list, 
        datareader : RRDataReader
    ):
    # 其实，就是最后一次的master的特化版本
    # 根据输入的数据获取各种图
    # 然后获取所有学生的近一个月的所有交互数据
    # 加载模型
    # 得出四个结果
    # 学习者、知识点、场景的嵌入式表达直接保存
    # 推荐得分按学习者分类保存
    device = 'cpu'

    RR_pt_path = os.path.join(deeplearning_root, 'RR', 'PT')
    RR_use_path = os.path.join(RR_pt_path, 'RR_use.pt')
    RR_pt_lrn_dir_path = os.path.join(RR_pt_path, 'lrn_use')
    RR_pt_lrn_use_path = os.path.join(RR_pt_lrn_dir_path, lrn_uid + '_use.pt')

    RR_pt_final_use = ''
    if os.path.exists(RR_pt_lrn_use_path):
        RR_pt_final_use = RR_pt_lrn_use_path
    else:
        RR_pt_final_use = RR_use_path


    model_rr = torch.jit.load(RR_pt_final_use)

    model_rr = model_rr.to(device)

    model_rr.eval()

    with torch.no_grad():

        unt_index, unt_mask = datareader.get_final_lrn_unt_index(lrn_uid, unt_uids)

        r_pred, h_lrn, h_cpt =  model_rr(
            lrn_emb, 
            unt_dynamic_emb, 
            unt_index,
            unt_mask,
            cpt_emb)

    # 保存lrn_emb，unt_emb，cpt_emb
    datareader.save_final_rr_data(lrn_uid, cpt_uids_list, r_pred)

def train_basic():
    parsers = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    print(device)
    rrdatareader = RRDataReader(parsers.sample_num)
    train_data, master_data, uids, inits, p_matrixes, dynamic_unt_mat = rrdatareader.load_data_from_db()

    # (self.learners_init, self.units_init, self.concepts_init), \
    # (self.p_lsl, self.p_scs, self.p_sls, self.p_cc, self.p_cac, self.p_csc,)
    lrn_uids, unt_uids, cpt_uids = uids
    learners_init, units_init, concepts_init = inits

    (p_lsl_edge_index, p_lsl_edge_attr), \
    (p_scs_edge_index, p_scs_edge_attr), \
    (p_sls_edge_index, p_sls_edge_attr), \
    (p_cc_edge_index, p_cc_edge_attr), \
    (p_cac_edge_index, p_cac_edge_attr), \
    (p_csc_edge_index, p_csc_edge_attr) = p_matrixes

    # model_hgc = HGC(parsers.embedding_dim, device).to(device)
    
    model_hgc_lrn = HGC_LRN(parsers.embedding_dim).to(device)
    model_hgc_unt = HGC_SCN(parsers.embedding_dim).to(device)
    model_hgc_cpt = HGC_CPT(parsers.embedding_dim).to(device)

    model_rr = RR(parsers.embedding_dim, parsers.hidden_dim).to(device)

    print(model_hgc_lrn)
    print(model_hgc_unt)
    print(model_hgc_cpt)
    print(model_rr)

    optimizer = torch.optim.Adam([
            {'params':model_hgc_lrn.parameters()},
            {'params':model_hgc_unt.parameters()},
            {'params':model_hgc_cpt.parameters()},
            {'params':model_rr.parameters()}
        ], lr=parsers.lr)
    
    HGC_pt_path = os.path.join(deeplearning_root, 'HGC', 'PT')
    HGC_LRN_train_path = os.path.join(HGC_pt_path, 'HGC_LRN_train.pt')
    HGC_SCN_train_path = os.path.join(HGC_pt_path, 'HGC_SCN_train.pt')
    HGC_CPT_train_path = os.path.join(HGC_pt_path, 'HGC_CPT_train.pt')
    HGC_LRN_use_path = os.path.join(HGC_pt_path, 'HGC_LRN_use.pt')
    HGC_SCN_use_path = os.path.join(HGC_pt_path, 'HGC_SCN_use.pt')
    HGC_CPT_use_path = os.path.join(HGC_pt_path, 'HGC_CPT_ues.pt')

    RR_pt_path = os.path.join(deeplearning_root, 'RR', 'PT')
    RR_train_path = os.path.join(RR_pt_path, 'RR_train.pt')
    RR_use_path = os.path.join(RR_pt_path, 'RR_use.pt')

    RR_temp_path = os.path.join(RR_pt_path, 'RR_temp.pt')

    continue_train = False
    epoch_start = 0
    
    if os.path.exists(RR_temp_path):
        print('继续训练')
        continue_train = True
        checkpoint = torch.load(RR_temp_path, map_location= device)
        model_hgc_lrn.load_state_dict(checkpoint['model_hgc_lrn'])
        model_hgc_unt.load_state_dict(checkpoint['model_hgc_unt'])
        model_hgc_cpt.load_state_dict(checkpoint['model_hgc_cpt'])
        model_rr.load_state_dict(checkpoint['model_rr'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch'] + 1

    update_train = False
    loss_last = None
    if not continue_train:
        if os.path.exists(HGC_LRN_train_path):
            print('HGC_LRN增量训练')
            # update_train = True
            checkpoint = torch.load(HGC_LRN_train_path, map_location=device)
            model_hgc_lrn.load_state_dict(checkpoint['model_hgc_lrn'])
        else:
            print('HGC_LRN初始训练')

        if os.path.exists(HGC_SCN_train_path):
            print('HGC_SCN增量训练')
            # update_train = True
            checkpoint = torch.load(HGC_SCN_train_path, map_location=device)
            model_hgc_unt.load_state_dict(checkpoint['model_hgc_unt'])
        else:
            print('HGC_SCN初始训练')

        if os.path.exists(HGC_CPT_train_path):
            print('HGC_CPT增量训练')
            # update_train = True
            checkpoint = torch.load(HGC_CPT_train_path, map_location=device)
            model_hgc_cpt.load_state_dict(checkpoint['model_hgc_cpt'])
        else:
            print('HGC_LRN初始训练')

        if os.path.exists(RR_train_path):
            print('RR增量训练')
            update_train = True
            checkpoint = torch.load(RR_train_path, map_location=device)
            model_rr.load_state_dict(checkpoint['model_rr'])
            loss_last = checkpoint['loss']
        else:
            print('RR初始训练')

    epoch_tqdm = tqdm(range(epoch_start, parsers.epochs))

    # loss_all = []
    # for s_loss in loss_master:
        # loss_all.append(s_loss.detach().cpu().numpy())

    for epoch in epoch_tqdm:
        epoch_tqdm.set_description('epoch - train - {}'.format(epoch))

        train_dataset = RRDataSet(train_data, uids, learners_init, parsers.max_step)
        train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

        batch_tqdm = tqdm(train_dataloader)
        batch_tqdm.set_description('batch start')
        loss_train = []

        model_hgc_lrn.train()
        model_hgc_unt.train()
        model_hgc_cpt.train()
        model_rr.train()

        for item in batch_tqdm:
            learner_idx = item['learner_idx']
            learner_init = item['learner_init']
            unt_seq_index = item['unt_seq_index']
            unt_seq_mask = item['unt_seq_mask']
            r_uk_data = item['r_uk_data']

            p_lsl = Data(x = learners_init, edge_index = p_lsl_edge_index, edge_attr = p_lsl_edge_attr)
            sub_p_lsl = p_lsl.subgraph(learner_idx)

            lrn_emb = model_hgc_lrn(sub_p_lsl.x.to(device), 
                                    sub_p_lsl.edge_index.to(device), sub_p_lsl.edge_attr.to(device)
                                    )
            unt_emb = model_hgc_unt(units_init.to(device), 
                                    p_scs_edge_index.to(device), p_scs_edge_attr.to(device),
                                    p_sls_edge_index.to(device), p_sls_edge_attr.to(device)
                                    )
            cpt_emb = model_hgc_cpt(concepts_init.to(device), 
                                    p_cc_edge_index.to(device), p_cc_edge_attr.to(device),
                                    p_cac_edge_index.to(device), p_cac_edge_attr.to(device), 
                                    p_csc_edge_index.to(device), p_csc_edge_attr.to(device)
                                    )

            unt_dynamic_emb = torch.sparse.mm(dynamic_unt_mat.to(device), cpt_emb)

            r_pred, h_lrn, h_cpt =  model_rr(lrn_emb, 
                                        unt_dynamic_emb, 
                                        unt_seq_index.to(device),
                                        unt_seq_mask.to(device),
                                        cpt_emb)

            loss = RRloss(r_pred, r_uk_data.to(device), h_lrn, h_cpt, parsers.lambda_reg)
            batch_tqdm.set_description('loss:{:.4f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train.append(loss.detach().cpu().numpy())

        epoch_tqdm.set_description('epoch - {} train_loss - {:.2f}'.format(epoch, np.average(loss_train)))

        del train_dataloader

        epoch_tqdm.set_description('epoch - master - {}'.format(epoch))

        master_dataset = RRDataSet(master_data, uids, learners_init, parsers.max_step)
        master_dataloader = DataLoader(master_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

        batch_tqdm = tqdm(master_dataloader)
        batch_tqdm.set_description('batch start')
        loss_master = []

        model_hgc_lrn.eval()
        model_hgc_unt.eval()
        model_hgc_cpt.eval()
        model_rr.eval()

        for item in batch_tqdm:
            learner_idx = item['learner_idx']
            learner_init = item['learner_init']
            unt_seq_index = item['unt_seq_index']
            unt_seq_mask = item['unt_seq_mask']
            r_uk_data = item['r_uk_data']

            p_lsl = Data(x = learners_init, edge_index = p_lsl_edge_index, edge_attr = p_lsl_edge_attr)
            sub_p_lsl = p_lsl.subgraph(learner_idx)

            with torch.no_grad():

                lrn_emb = model_hgc_lrn(sub_p_lsl.x.to(device), 
                                        sub_p_lsl.edge_index.to(device), sub_p_lsl.edge_attr.to(device)
                                        )
                unt_emb = model_hgc_unt(units_init.to(device), 
                                        p_scs_edge_index.to(device), p_scs_edge_attr.to(device),
                                        p_sls_edge_index.to(device), p_sls_edge_attr.to(device)
                                        )
                cpt_emb = model_hgc_cpt(concepts_init.to(device), 
                                        p_cc_edge_index.to(device), p_cc_edge_attr.to(device),
                                        p_cac_edge_index.to(device), p_cac_edge_attr.to(device), 
                                        p_csc_edge_index.to(device), p_csc_edge_attr.to(device)
                                        )

                unt_dynamic_emb = torch.sparse.mm(dynamic_unt_mat.to(device), cpt_emb)

                r_pred, h_lrn, h_cpt =  model_rr(lrn_emb, 
                                            unt_dynamic_emb, 
                                            unt_seq_index.to(device),
                                            unt_seq_mask.to(device),
                                            cpt_emb)

            loss = RRloss(r_pred, r_uk_data.to(device), h_lrn, h_cpt, parsers.lambda_reg)
            batch_tqdm.set_description('loss:{:.4f}'.format(loss))
            loss_master.append(loss.detach().cpu().numpy())
            # loss_all.append(loss.detach().cpu().numpy())

        epoch_tqdm.set_description('epoch - {} master_loss - {:.2f}'.format(epoch, np.average(loss_master)))

        del master_dataloader

        if (epoch + 1) % 8 == 0:
            torch.save({
                'model_hgc_lrn': model_hgc_lrn.state_dict(),
                'model_hgc_unt': model_hgc_unt.state_dict(),
                'model_hgc_cpt': model_hgc_cpt.state_dict(),
                'model_rr': model_rr.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, RR_temp_path)

    if os.path.exists(RR_temp_path):
        os.remove(RR_temp_path)

    if not update_train or loss < loss_last:
        torch.save({
            'model_hgc_lrn': model_hgc_lrn.state_dict(),
        }, HGC_LRN_train_path)
        torch.save({
            'model_hgc_unt': model_hgc_unt.state_dict(),
        }, HGC_SCN_train_path)
        torch.save({
            'model_hgc_cpt': model_hgc_cpt.state_dict(),
        }, HGC_CPT_train_path)
        torch.save({
            'model_rr': model_rr.state_dict(),
            'loss' : loss
        }, RR_train_path)
    
        # torch.save(model.state_dict(), IPDKT_pt_use_path)
        model_hgc_lrn = model_hgc_lrn.to('cpu')
        model_hgc_unt = model_hgc_unt.to('cpu')
        model_hgc_cpt = model_hgc_cpt.to('cpu')
        model_rr = model_rr.to('cpu')

        torch.cuda.empty_cache()

        with torch.no_grad():  # 禁用梯度计算
            with torch.jit.optimized_execution(False):  # 禁止优化时隐式转移到GPU
                scripted_model_hgc_lrn = torch.jit.script(model_hgc_lrn)
                scripted_model_hgc_unt = torch.jit.script(model_hgc_unt)
                scripted_model_hgc_cpt = torch.jit.script(model_hgc_cpt)
                scripted_model_rr = torch.jit.script(model_rr)
        
        scripted_model_hgc_lrn = torch.jit.optimize_for_inference(scripted_model_hgc_lrn)
        scripted_model_hgc_lrn.save(HGC_LRN_use_path)

        scripted_model_hgc_unt = torch.jit.optimize_for_inference(scripted_model_hgc_unt)
        scripted_model_hgc_unt.save(HGC_SCN_use_path)

        scripted_model_hgc_cpt = torch.jit.optimize_for_inference(scripted_model_hgc_cpt)
        scripted_model_hgc_cpt.save(HGC_CPT_use_path)

        scripted_model_rr = torch.jit.optimize_for_inference(scripted_model_rr)
        scripted_model_rr.save(RR_use_path)

        cpt_uids_list, embs = save_final_hgc_data(uids, inits, p_matrixes, dynamic_unt_mat, rrdatareader)

    train_single_lrn(
        train_data, master_data, uids, inits, p_matrixes, dynamic_unt_mat, optimizer,
        cpt_uids_list,
        embs[0], embs[1], embs[2],
        rrdatareader
    )

def train_single_lrn(
    train_data, master_data, uids, inits, p_matrixes, dynamic_unt_mat, optimizer,
    cpt_uids_list,
    lrn_emb_final, unt_emb_final, cpt_emb_final,
    rrdatareader : RRDataReader):
    # 加载设备
    parsers = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
    
    # 加载各种数据
    lrn_uids, unt_uids, cpt_uids = uids
    learners_init, units_init, concepts_init = inits

    (p_lsl_edge_index, p_lsl_edge_attr), \
    (p_scs_edge_index, p_scs_edge_attr), \
    (p_sls_edge_index, p_sls_edge_attr), \
    (p_cc_edge_index, p_cc_edge_attr), \
    (p_cac_edge_index, p_cac_edge_attr), \
    (p_csc_edge_index, p_csc_edge_attr) = p_matrixes

    # 声明路径
    HGC_pt_path = os.path.join(deeplearning_root, 'HGC', 'PT')
    HGC_LRN_train_path = os.path.join(HGC_pt_path, 'HGC_LRN_train.pt')
    HGC_SCN_train_path = os.path.join(HGC_pt_path, 'HGC_SCN_train.pt')
    HGC_CPT_train_path = os.path.join(HGC_pt_path, 'HGC_CPT_train.pt')

    RR_pt_path = os.path.join(deeplearning_root, 'RR', 'PT')
    RR_train_path = os.path.join(RR_pt_path, 'RR_train.pt')
    RR_use_path = os.path.join(RR_pt_path, 'RR_use.pt')

    RR_pt_lrn_dir_path = os.path.join(RR_pt_path, 'lrn_use')

    if not os.path.exists(RR_pt_lrn_dir_path):
        os.makedirs(RR_pt_lrn_dir_path)

    # 声明模型
    model_hgc_lrn = HGC_LRN(parsers.embedding_dim).to(device)
    model_hgc_unt = HGC_SCN(parsers.embedding_dim).to(device)
    model_hgc_cpt = HGC_CPT(parsers.embedding_dim).to(device)

    model_rr = RR(parsers.embedding_dim, parsers.hidden_dim).to(device)

    lrn_tqdm = tqdm(lrn_uids)
    for lrn_uid in lrn_tqdm:
        lrn_tqdm.set_description('{}'.format(lrn_uid))

        # 加载模型
        checkpoint = torch.load(HGC_LRN_train_path, map_location=device)
        model_hgc_lrn.load_state_dict(checkpoint['model_hgc_lrn'])
        checkpoint = torch.load(HGC_SCN_train_path, map_location=device)
        model_hgc_unt.load_state_dict(checkpoint['model_hgc_unt'])
        checkpoint = torch.load(HGC_CPT_train_path, map_location=device)
        model_hgc_cpt.load_state_dict(checkpoint['model_hgc_cpt'])
        checkpoint = torch.load(RR_train_path, map_location=device)
        model_rr.load_state_dict(checkpoint['model_rr'])

        model_rr = model_rr.to(device)

        epoch_tqdm = tqdm(range(parsers.epochs))
        for epoch in epoch_tqdm:
            epoch_tqdm.set_description('epoch - train - {}'.format(epoch))

            train_dataset = RRDataSet(
                {lrn_uid : train_data[lrn_uid]}, 
                ({lrn_uid : 0}, unt_uids, cpt_uids), 
                learners_init, 
                len(train_data[lrn_uid][0])
            )
            train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

            batch_tqdm = tqdm(train_dataloader)
            batch_tqdm.set_description('batch start')
            loss_train = []

            model_hgc_lrn.train()
            model_hgc_unt.train()
            model_hgc_cpt.train()
            model_rr.train()

            for item in batch_tqdm:
                learner_idx = item['learner_idx']
                learner_init = item['learner_init']
                unt_seq_index = item['unt_seq_index']
                unt_seq_mask = item['unt_seq_mask']
                r_uk_data = item['r_uk_data']

                p_lsl = Data(x = learners_init, edge_index = p_lsl_edge_index, edge_attr = p_lsl_edge_attr)
                sub_p_lsl = p_lsl.subgraph(learner_idx)

                lrn_emb = model_hgc_lrn(sub_p_lsl.x.to(device), 
                                        sub_p_lsl.edge_index.to(device), sub_p_lsl.edge_attr.to(device)
                                        )
                unt_emb = model_hgc_unt(units_init.to(device), 
                                        p_scs_edge_index.to(device), p_scs_edge_attr.to(device),
                                        p_sls_edge_index.to(device), p_sls_edge_attr.to(device)
                                        )
                cpt_emb = model_hgc_cpt(concepts_init.to(device), 
                                        p_cc_edge_index.to(device), p_cc_edge_attr.to(device),
                                        p_cac_edge_index.to(device), p_cac_edge_attr.to(device), 
                                        p_csc_edge_index.to(device), p_csc_edge_attr.to(device)
                                        )

                unt_dynamic_emb = torch.sparse.mm(dynamic_unt_mat.to(device), cpt_emb)

                r_pred, h_lrn, h_cpt =  model_rr(lrn_emb, 
                                            unt_dynamic_emb, 
                                            unt_seq_index.to(device),
                                            unt_seq_mask.to(device),
                                            cpt_emb)

                loss = RRloss(r_pred, r_uk_data.to(device), h_lrn, h_cpt, parsers.lambda_reg)
                batch_tqdm.set_description('loss:{:.4f}'.format(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_train.append(loss.detach().cpu().numpy())

            epoch_tqdm.set_description('epoch - {} train_loss - {:.2f}'.format(epoch, np.average(loss_train)))

            del train_dataloader

            epoch_tqdm.set_description('epoch - master - {}'.format(epoch))

            master_dataset = RRDataSet(
                {lrn_uid : master_data[lrn_uid]}, 
                ({lrn_uid : 0}, unt_uids, cpt_uids), 
                learners_init, 
                len(master_data[lrn_uid][0])
            )
            master_dataloader = DataLoader(master_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

            batch_tqdm = tqdm(master_dataloader)
            batch_tqdm.set_description('batch start')
            loss_master = []

            model_hgc_lrn.eval()
            model_hgc_unt.eval()
            model_hgc_cpt.eval()
            model_rr.eval()

            for item in batch_tqdm:
                learner_idx = item['learner_idx']
                learner_init = item['learner_init']
                unt_seq_index = item['unt_seq_index']
                unt_seq_mask = item['unt_seq_mask']
                r_uk_data = item['r_uk_data']

                p_lsl = Data(x = learners_init, edge_index = p_lsl_edge_index, edge_attr = p_lsl_edge_attr)
                sub_p_lsl = p_lsl.subgraph(learner_idx)

                with torch.no_grad():

                    lrn_emb = model_hgc_lrn(sub_p_lsl.x.to(device), 
                                            sub_p_lsl.edge_index.to(device), sub_p_lsl.edge_attr.to(device)
                                            )
                    unt_emb = model_hgc_unt(units_init.to(device), 
                                            p_scs_edge_index.to(device), p_scs_edge_attr.to(device),
                                            p_sls_edge_index.to(device), p_sls_edge_attr.to(device)
                                            )
                    cpt_emb = model_hgc_cpt(concepts_init.to(device), 
                                            p_cc_edge_index.to(device), p_cc_edge_attr.to(device),
                                            p_cac_edge_index.to(device), p_cac_edge_attr.to(device), 
                                            p_csc_edge_index.to(device), p_csc_edge_attr.to(device)
                                            )

                    unt_dynamic_emb = torch.sparse.mm(dynamic_unt_mat.to(device), cpt_emb)

                    r_pred, h_lrn, h_cpt =  model_rr(lrn_emb, 
                                                unt_dynamic_emb, 
                                                unt_seq_index.to(device),
                                                unt_seq_mask.to(device),
                                                cpt_emb)

                loss = RRloss(r_pred, r_uk_data.to(device), h_lrn, h_cpt, parsers.lambda_reg)
                batch_tqdm.set_description('loss:{:.4f}'.format(loss))
                loss_master.append(loss.detach().cpu().numpy())
                # loss_all.append(loss.detach().cpu().numpy())

            epoch_tqdm.set_description('epoch - {} master_loss - {:.2f}'.format(epoch, np.average(loss_master)))

            del master_dataloader

        RR_pt_lrn_use_path = os.path.join(RR_pt_lrn_dir_path, lrn_uid + '_use.pt')

        model_rr = model_rr.to('cpu')
        torch.cuda.empty_cache()

        with torch.no_grad():  # 禁用梯度计算
            with torch.jit.optimized_execution(False):  # 禁止优化时隐式转移到GPU
                scripted_model_rr = torch.jit.script(model_rr)

        scripted_model_rr = torch.jit.optimize_for_inference(scripted_model_rr)
        scripted_model_rr.save(RR_pt_lrn_use_path)

        # 在这里保存mongo数据
        # 保存 1.学习者嵌入式表达2.场景嵌入式表达3.知识点嵌入式表达
        # 针对特定的学习者保存推荐结果？是的
        # row_i = A[i:i+1, :]

        save_final_rr_data(
            lrn_uid, unt_uids, 
            lrn_emb_final[lrn_uids[lrn_uid] : lrn_uids[lrn_uid] + 1, :].to('cpu'), 
            unt_emb_final.to('cpu'), cpt_emb_final.to('cpu'),
            cpt_uids_list, 
            rrdatareader
        )

if __name__ == '__main__':
    train_basic()

