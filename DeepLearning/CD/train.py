import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import os
import torch
import argparse
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from operator import itemgetter

from CD.Dataset.CDDataReader import CDDataReader
from CD.Dataset.CDDataSet import CDDataset
from CD.Model.CD import CD
from HGC.Model.HGC import HGC_LRN, HGC_SCN, HGC_CPT

def protect_norm(models):
    return 
    for model in models:
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1e4, neginf=-1e4)

def save_final_data(uids, inits, p_matrixes, datareader : CDDataReader):

    lrn_uids, scn_uids, cpt_uids = uids
    learners_init, scenes_init, concepts_init = inits

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

    CD_pt_path = os.path.join(deeplearning_root, 'CD', 'PT')
    CD_use_path = os.path.join(CD_pt_path, 'CD_use.pt')

    model_hgc_lrn = torch.jit.load(HGC_LRN_use_path)
    model_hgc_scn = torch.jit.load(HGC_SCN_use_path)
    model_hgc_cpt = torch.jit.load(HGC_CPT_use_path)
    model_cd = torch.jit.load(CD_use_path)

    model_hgc_lrn.eval()
    model_hgc_scn.eval()
    model_hgc_cpt.eval()
    model_cd.eval()

    with torch.no_grad():
        lrn_emb = model_hgc_lrn(
            learners_init.to(device), 
            p_lsl_edge_index.to(device), p_lsl_edge_attr.to(device)
        )
        scn_emb = model_hgc_scn(
            scenes_init.to(device), 
            p_scs_edge_index.to(device), p_scs_edge_attr.to(device),
            p_sls_edge_index.to(device), p_sls_edge_attr.to(device)
        )
        cpt_emb = model_hgc_cpt(
            concepts_init.to(device), 
            p_cc_edge_index.to(device), p_cc_edge_attr.to(device),
            p_cac_edge_index.to(device), p_cac_edge_attr.to(device), 
            p_csc_edge_index.to(device), p_csc_edge_attr.to(device)
        )
        
        # 这里已经获得了相当于z的矩阵
        # 然后输入到cd中

        scn_index, scn_mask, ordered_cpt_uids = datareader.get_final_lrn_scn_index(lrn_uids, scn_uids)

        r_pred = model_cd(
            scn_index.to(device), 
            scn_mask.to(device), 
            lrn_emb,
            scn_emb, 
            cpt_emb
        )

    lrn_uids_list = [lrn_uid for lrn_uid, _ in sorted(lrn_uids.items(), key=itemgetter(1))]
    scn_uids_list = [scn_uid for scn_uid, _ in sorted(scn_uids.items(), key=itemgetter(1))]
    cpt_uids_list = [cpt_uid for cpt_uid, _ in sorted(cpt_uids.items(), key=itemgetter(1))]
    

    # 保存lrn_emb，scn_emb，cpt_emb
    datareader.save_final_data(lrn_uids_list, scn_uids_list, cpt_uids_list, 
                               lrn_emb, scn_emb, cpt_emb,
                               r_pred, ordered_cpt_uids)



parser = argparse.ArgumentParser(description='CD')
parser.add_argument('--batch_size',type=int,default=32,help='number of batch size to train (defauly 32 )')
parser.add_argument('--epochs',type=int,default=2,help='number of epochs to train (defauly 32 )')
parser.add_argument('--lr',type=float,default=0.001,help='number of learning rate')
parser.add_argument('--embedding_dim',type=int,default=32,help='number of embedding dim')
parser.add_argument('--lamda_kcge',type=int,default=1,help='lamda used in kCGE')
parser.add_argument('--num_workers',type=int,default=3,help='num of workers')
parser.add_argument('--max_step',type=int,default=128,help='num of max_step')

if __name__ == '__main__':
    parsers = parser.parse_args()
    train_file_path = ''
    master_file_path = ''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    cddatareader = CDDataReader()
    train_data, master_data, uids, inits, p_matrixes = cddatareader.load_Data_from_db()

    lrn_uids, scn_uids, cpt_uids = uids
    learners_init, scenes_init, concepts_init = inits

    (p_lsl_edge_index, p_lsl_edge_attr), \
    (p_scs_edge_index, p_scs_edge_attr), \
    (p_sls_edge_index, p_sls_edge_attr), \
    (p_cc_edge_index, p_cc_edge_attr), \
    (p_cac_edge_index, p_cac_edge_attr), \
    (p_csc_edge_index, p_csc_edge_attr) = p_matrixes

    model_hgc_lrn = HGC_LRN(parsers.embedding_dim, device).to(device)
    model_hgc_scn = HGC_SCN(parsers.embedding_dim, device).to(device)
    model_hgc_cpt = HGC_CPT(parsers.embedding_dim, device).to(device)
    model_cd = CD(parsers.embedding_dim, device).to(device)

    optimizer = torch.optim.Adam([{'params':model_hgc_lrn.parameters()},
                                {'params':model_hgc_scn.parameters()},
                                {'params':model_hgc_cpt.parameters()},
                                {'params':model_cd.parameters()}], lr= parsers.lr)

    criterion = nn.BCELoss().to(device)

    HGC_pt_path = os.path.join(deeplearning_root, 'HGC', 'PT')
    HGC_LRN_train_path = os.path.join(HGC_pt_path, 'HGC_LRN_train.pt')
    HGC_SCN_train_path = os.path.join(HGC_pt_path, 'HGC_SCN_train.pt')
    HGC_CPT_train_path = os.path.join(HGC_pt_path, 'HGC_CPT_train.pt')
    HGC_LRN_use_path = os.path.join(HGC_pt_path, 'HGC_LRN_use.pt')
    HGC_SCN_use_path = os.path.join(HGC_pt_path, 'HGC_SCN_use.pt')
    HGC_CPT_use_path = os.path.join(HGC_pt_path, 'HGC_CPT_ues.pt')

    CD_pt_path = os.path.join(deeplearning_root, 'CD', 'PT')
    CD_train_path = os.path.join(CD_pt_path, 'CD_train.pt')
    CD_use_path = os.path.join(CD_pt_path, 'CD_use.pt')
    CD_temp_path = os.path.join(CD_pt_path, 'CD_temp.pt')

    epoch_start = 0
    continue_train = False
    
    if os.path.exists(CD_temp_path):
        print('继续训练')
        continue_train = True
        check_point = torch.load(CD_temp_path, map_location=device)
        model_hgc_lrn.load_state_dict(check_point['model_state_dict_dtr'])
        model_hgc_scn.load_state_dict(check_point['model_state_dict_mirt'])
        model_hgc_cpt.load_state_dict(check_point['model_state_dict_kcge'])
        model_cd.load_state_dict(check_point['model_state_dict_kcge'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        epoch_start = check_point['epoch'] + 1

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
            model_hgc_scn.load_state_dict(checkpoint['model_hgc_scn'])
        else:
            print('HGC_SCN初始训练')

        if os.path.exists(HGC_CPT_train_path):
            print('HGC_CPT增量训练')
            # update_train = True
            checkpoint = torch.load(HGC_CPT_train_path, map_location=device)
            model_hgc_cpt.load_state_dict(checkpoint['model_hgc_cpt'])
        else:
            print('HGC_LRN初始训练')

        if os.path.exists(CD_train_path):
            print('CD增量训练')
            update_train = True
            checkpoint = torch.load(CD_train_path, map_location=device)
            model_cd.load_state_dict(checkpoint['model_cd'])
            loss_last = checkpoint['loss']
        else:
            print('CD初始训练')

    epoch_tqdm = tqdm(range(epoch_start, parsers.epochs))

    for epoch in epoch_tqdm:
        epoch_tqdm.set_description('epoch {} - train'.format(epoch))

        model_hgc_lrn.train()
        model_hgc_scn.train()
        model_hgc_cpt.train()
        model_cd.train()

        train_dataset = CDDataset(train_data, uids, learners_init, parsers.max_step)
        train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

        batch_tqdm = tqdm(train_dataloader)
        batch_tqdm.set_description('train batch:')

        num_correct = 0
        num_total = 0
        loss_train = []

        for item in batch_tqdm:
            learner_idx = item['learner_idx']
            learner_init = item['learner_init']
            scn_seq_index = item['scn_seq_index']
            scn_seq_mask = item['scn_seq_mask']
            result = item['result']

            p_lsl = Data(x = learners_init, edge_index = p_lsl_edge_index, edge_attr = p_lsl_edge_attr)
            sub_p_lsl = p_lsl.subgraph(learner_idx)

            lrn_emb = model_hgc_lrn(
                sub_p_lsl.x.to(device), 
                sub_p_lsl.edge_index.to(device), sub_p_lsl.edge_attr.to(device)
            )
            scn_emb = model_hgc_scn(
                scenes_init.to(device), 
                p_scs_edge_index.to(device), p_scs_edge_attr.to(device),
                p_sls_edge_index.to(device), p_sls_edge_attr.to(device)
            )
            cpt_emb = model_hgc_cpt(
                concepts_init.to(device), 
                p_cc_edge_index.to(device), p_cc_edge_attr.to(device),
                p_cac_edge_index.to(device), p_cac_edge_attr.to(device), 
                p_csc_edge_index.to(device), p_csc_edge_attr.to(device)
            )
            
            # 这里已经获得了相当于z的矩阵
            # 然后输入到cd中
            # print('lrn_emb', lrn_emb.shape)

            result_pred = model_cd(
                scn_seq_index.to(device), 
                scn_seq_mask.to(device), 
                lrn_emb,
                scn_emb, 
                cpt_emb
            )

            result = result.flatten().to(device)
            result_pred = result_pred.flatten()

            # print(result.max(), result.min())
            # print(result_pred.max(), result_pred.min())

            loss = criterion(result, result_pred)

            optimizer.zero_grad()
            loss.backward()
            protect_norm([model_hgc_lrn, model_hgc_scn, model_hgc_cpt, model_cd])
            optimizer.step()
            protect_norm([model_hgc_lrn, model_hgc_scn, model_hgc_cpt, model_cd])

            num_correct += ((result_pred >= 0.5).long() == result).sum().item()
            num_total += len(result)
            loss_train.append(loss.detach().cpu().numpy())
            batch_tqdm.set_description('loss:{:.4f}'.format(loss))
        
        acc = num_correct / num_total
        loss = np.average(loss_train)
        epoch_tqdm.set_description('epoch {} - train - loss:{:.4f} - acc:{:.4f}'.format(epoch, loss, acc))

        del train_dataloader

        epoch_tqdm.set_description('epoch {} - master'.format(epoch))

        model_hgc_lrn.eval()
        model_hgc_scn.eval()
        model_hgc_cpt.eval()
        model_cd.eval()

        master_dataset = CDDataset(master_data, uids, learners_init, parsers.max_step)
        master_dataloader = DataLoader(master_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

        batch_tqdm = tqdm(master_dataloader)
        batch_tqdm.set_description('master batch:')

        num_correct = 0
        num_total = 0
        loss_master = []

        for item in batch_tqdm:
            learner_idx = item['learner_idx']
            learner_init = item['learner_init']
            scn_seq_index = item['scn_seq_index']
            scn_seq_mask = item['scn_seq_mask']
            result = item['result']

            p_lsl = Data(x = learners_init, edge_index = p_lsl_edge_index, edge_attr = p_lsl_edge_attr)
            sub_p_lsl = p_lsl.subgraph(learner_idx)

            with torch.no_grad():

                lrn_emb = model_hgc_lrn(
                    sub_p_lsl.x.to(device), 
                    sub_p_lsl.edge_index.to(device), sub_p_lsl.edge_attr.to(device)
                )
                scn_emb = model_hgc_scn(
                    scenes_init.to(device), 
                    p_scs_edge_index.to(device), p_scs_edge_attr.to(device),
                    p_sls_edge_index.to(device), p_sls_edge_attr.to(device)
                )
                cpt_emb = model_hgc_cpt(
                    concepts_init.to(device), 
                    p_cc_edge_index.to(device), p_cc_edge_attr.to(device),
                    p_cac_edge_index.to(device), p_cac_edge_attr.to(device), 
                    p_csc_edge_index.to(device), p_csc_edge_attr.to(device)
                )
                
                # 这里已经获得了相当于z的矩阵
                # 然后输入到cd中

                result_pred = model_cd(
                    scn_seq_index.to(device), 
                    scn_seq_mask.to(device), 
                    lrn_emb,
                    scn_emb, 
                    cpt_emb
                )

            result = result.flatten().to(device)
            result_pred = result_pred.flatten()

            loss = criterion(result, result_pred)

            num_correct += ((result_pred >= 0.5).long() == result).sum().item()
            num_total += len(result)
            loss_master.append(loss.detach().cpu().numpy())
            batch_tqdm.set_description('loss:{:.4f}'.format(loss))
        
        acc = num_correct / num_total
        loss = np.average(loss_master)
        epoch_tqdm.set_description('epoch {} - master - loss:{:.4f} - acc:{:.4f}'.format(epoch, loss, acc))

        del master_dataloader

        if (epoch + 1) % 8 == 0:
            torch.save({
                'model_hgc_lrn': model_hgc_lrn.state_dict(),
                'model_hgc_scn': model_hgc_scn.state_dict(),
                'model_hgc_cpt': model_hgc_cpt.state_dict(),
                'model_cd': model_cd.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, CD_temp_path)

    if os.path.exists(CD_temp_path):
        os.remove(CD_temp_path)

    if not update_train or loss < loss_last:
        torch.save({
            'model_hgc_lrn': model_hgc_lrn.state_dict(),
        }, HGC_LRN_train_path)
        torch.save({
            'model_hgc_scn': model_hgc_scn.state_dict(),
        }, HGC_SCN_train_path)
        torch.save({
            'model_hgc_cpt': model_hgc_cpt.state_dict(),
        }, HGC_CPT_train_path)
        torch.save({
            'model_cd': model_cd.state_dict(),
            'loss' : loss
        }, CD_train_path)
    
        # torch.save(model.state_dict(), IPDKT_pt_use_path)

        scripted_model = torch.jit.script(model_hgc_lrn)
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        scripted_model.save(HGC_LRN_use_path)
        scripted_model = torch.jit.script(model_hgc_scn)
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        scripted_model.save(HGC_SCN_use_path)
        scripted_model = torch.jit.script(model_hgc_cpt)
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        scripted_model.save(HGC_CPT_use_path)
        scripted_model = torch.jit.script(model_cd)
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        scripted_model.save(CD_use_path)

    save_final_data(uids, inits, p_matrixes, cddatareader)