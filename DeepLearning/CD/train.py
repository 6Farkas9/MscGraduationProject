import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import os
import torch
import argparse
import sys
import json
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
from KCGE.Model.KCGE import KCGE

parser = argparse.ArgumentParser(description='CD')
parser.add_argument('--batch_size',type=int,default=32,help='number of batch size to train (defauly 32 )')
parser.add_argument('--epochs',type=int,default=2,help='number of epochs to train (defauly 32 )')
parser.add_argument('--lr',type=float,default=1e-5,help='number of learning rate')
parser.add_argument('--embedding_dim',type=int,default=32,help='number of embedding dim')
parser.add_argument('--num_workers',type=int,default=3,help='num of workers')
parser.add_argument('--max_step',type=int,default=256,help='num of max_step')

def save_final_data(x, datareader : CDDataReader):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CD_pt_path = os.path.join(deeplearning_root, 'CD', 'PT')
    CD_use_path = os.path.join(CD_pt_path, are_uid + '_use.pt')

    model_cd = torch.jit.load(CD_use_path)
    model_cd = model_cd.to('cpu')

    model_cd.eval()

    scn_index, scn_mask, scn_index_special, scn_mask_special, scn_idx ,cpt_idx = datareader.get_final_Data()

    # 根据返回的scn_idx和cpt_idx从x中获取h_scn和h_cpt
    h_scn = x[scn_idx]
    h_cpt = x[cpt_idx]

    # 根据近期的交互记录结合h_scn获取h_lrn

    # 1. 提取所有可能需要的行 (lrn_num, max_step, embedding_dim)
    selected = h_scn[scn_index]  # 形状 (lrn_num, max_step, embedding_dim)
    # 2. 计算加权和（利用广播机制）
    weighted_sum = (selected * scn_mask.unsqueeze(-1)).sum(dim=1)  # (lrn_num, embedding_dim)
    # 3. 计算有效计数（每行有多少个 1）
    valid_counts = scn_mask.sum(dim=1, keepdim=True)  # (lrn_num, 1)
    # 4. 直接归一化
    h_lrn = weighted_sum / valid_counts  # (lrn_num, embedding_dim)

    # 然后输入到model中的交互记录是和特殊scn的交互记录
    r_pred = model_cd(scn_index_special, scn_mask_special, h_lrn, h_scn, h_cpt)
    
    # print(r_pred.shape)

    cddatareader.save_final_data(r_pred, h_scn, h_cpt)

def train_single_are(cddatareader, parsers, are_uid):
    train_data, master_data, lrn_uids, cpt_uids, scn_uids, cpt_idx, scn_idx, edge_index, edge_attr, edge_type = cddatareader.load_Data_from_db()

    # 这里获得两个set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
    print(f'current device:{device}')

    model_kcge = KCGE(parsers.embedding_dim).to(device)
    model_cd = CD(parsers.embedding_dim).to(device)
    
    optimizer = torch.optim.Adam([{'params':model_kcge.parameters()},
                                {'params':model_cd.parameters()}], lr= parsers.lr)

    criterion = nn.BCEWithLogitsLoss().to(device)

    KCGE_pt_path = os.path.join(deeplearning_root, 'KCGE', 'PT')
    KCGE_train_path = os.path.join(KCGE_pt_path, 'KCGE_train.pt')
    KCGE_use_path = os.path.join(KCGE_pt_path, 'KCGE_use.pt')

    CD_pt_path = os.path.join(deeplearning_root, 'CD', 'PT')
    CD_train_path = os.path.join(CD_pt_path, are_uid + '_train.pt')
    CD_use_path = os.path.join(CD_pt_path, are_uid + '_use.pt')
    CD_temp_path = os.path.join(CD_pt_path, 'CD_temp.pt')

    continue_train = False
    epoch_start = 0
    if os.path.exists(CD_temp_path):
        print('继续训练')
        continue_train = True
        check_point = torch.load(CD_temp_path, map_location=device)
        model_kcge.load_state_dict(check_point['model_state_dict_kcge'])
        model_cd.load_state_dict(check_point['model_state_dict_cd'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        epoch_start = check_point['epoch'] + 1

    update_train = False
    loss_last = None
    if not continue_train:
        if os.path.exists(KCGE_train_path):
            print('KCGE增量训练')
            # update_train = True
            checkpoint = torch.load(KCGE_train_path, map_location=device)
            model_kcge.load_state_dict(checkpoint['model_kcge'])
        else:
            print('KCGE初始训练')

        if os.path.exists(CD_train_path):
            print('CD增量训练')
            update_train = True
            checkpoint = torch.load(CD_train_path, map_location=device)
            model_cd.load_state_dict(checkpoint['model_cd'])
            loss_last = checkpoint['loss']
        else:
            print('CD初始训练')

    # x初始化为全1tensor
    # 比较特殊，这个x既是输入又是优化参数，所以这个x是不是也应该注册到优化器中？
    # 不是，x在训练过程中就会更新？并不会更新
    # 可以让x每次最后更新为z，dropout是否会有影响？应该没有
    # x在模型外初始化，每次进行更新，在final保存的时候保存最终的矩阵
    # 这个x的初始化很特殊，需要从数据库中读取，交给datareader完成
    x = torch.ones((edge_index.size(1), parsers.embedding_dim), dtype=torch.float32, device=device)
    # 这个x一直用到最后

    epoch_tqdm = tqdm(range(epoch_start, parsers.epochs))

    for epoch in epoch_tqdm:
        epoch_tqdm.set_description('epoch {} - train'.format(epoch))

        model_kcge.train()
        model_cd.train()

        train_dataset = CDDataset(train_data, lrn_uids, cpt_uids, scn_uids, parsers.max_step)
        train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

        batch_tqdm = tqdm(train_dataloader)

        num_correct = 0
        num_total = 0
        loss_train = []

        for item in batch_tqdm:
            # 'learner_idx' : learner_idx,
            # 'scn_seq_index' : scn_seq_index,
            # 'scn_seq_mask' : scn_seq_mask,
            # 'result' : result
            lrn_uids_in = item['learner_uid']
            scn_seq_idx = item['scn_seq_index']
            scn_seq_mask = item['scn_seq_mask']
            result = item['result']

            z = model_kcge(x, edge_index.to(device), edge_type.to(device), edge_attr.to(device))
            x = z.detach().clone()

            h_scn = z[scn_idx]
            h_cpt = z[cpt_idx]

            # 1. 提取所有可能需要的行 (lrn_num, max_step, embedding_dim)
            selected = h_scn[scn_seq_idx]  # 形状 (lrn_num, max_step, embedding_dim)
            # 2. 计算加权和（利用广播机制）
            weighted_sum = (selected * scn_seq_mask.unsqueeze(-1).to(device)).sum(dim=1)  # (lrn_num, embedding_dim)
            # 3. 计算有效计数（每行有多少个 1）
            valid_counts = scn_seq_mask.sum(dim=1, keepdim=True)  # (lrn_num, 1)
            # 4. 直接归一化
            h_lrn = weighted_sum / valid_counts.to(device)  # (lrn_num, embedding_dim)

            r_pred = model_cd(scn_seq_idx.to(device), scn_seq_mask.to(device), h_lrn, h_scn, h_cpt)

            result = result.flatten().to(device)
            r_pred = r_pred.flatten()

            loss = criterion(result, r_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct += ((r_pred >= 0.5).long() == result).sum().item()
            num_total += len(result)
            loss_train.append(loss.detach().cpu().numpy())
            batch_tqdm.set_description('loss:{:.4f}'.format(loss))
        
        acc = num_correct / num_total
        loss = np.average(loss_train)
        epoch_tqdm.set_description('epoch {} - train - loss:{:.4f} - acc:{:.4f}'.format(epoch, loss, acc))

        del train_dataloader

        epoch_tqdm.set_description('epoch {} - master'.format(epoch))

        model_kcge.eval()
        model_cd.eval()

        master_dataset = CDDataset(master_data, lrn_uids, cpt_uids, scn_uids, parsers.max_step)
        master_dataloader = DataLoader(master_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

        batch_tqdm = tqdm(master_dataloader)

        num_correct = 0
        num_total = 0
        loss_master = []

        for item in batch_tqdm:
            # 'learner_idx' : learner_idx,
            # 'scn_seq_index' : scn_seq_index,
            # 'scn_seq_mask' : scn_seq_mask,
            # 'result' : result
            lrn_uids_in = item['learner_uid']
            scn_seq_idx = item['scn_seq_index']
            scn_seq_mask = item['scn_seq_mask']
            result = item['result']

            with torch.no_grad():

                z = model_kcge(x, edge_index.to(device), edge_type.to(device), edge_attr.to(device))
                x = z.detach().clone()

                # 1. 提取所有可能需要的行 (lrn_num, max_step, embedding_dim)
                selected = h_scn[scn_seq_idx]  # 形状 (lrn_num, max_step, embedding_dim)
                # 2. 计算加权和（利用广播机制）
                weighted_sum = (selected * scn_seq_mask.unsqueeze(-1).to(device)).sum(dim=1)  # (lrn_num, embedding_dim)
                # 3. 计算有效计数（每行有多少个 1）
                valid_counts = scn_seq_mask.sum(dim=1, keepdim=True)  # (lrn_num, 1)
                # 4. 直接归一化
                h_lrn = weighted_sum / valid_counts.to(device)  # (lrn_num, embedding_dim)

                r_pred = model_cd(scn_seq_idx.to(device), scn_seq_mask.to(device), h_lrn, h_scn, h_cpt)

            result = result.flatten().to(device)
            r_pred = r_pred.flatten()

            loss = criterion(result, r_pred)

            num_correct += ((r_pred >= 0.5).long() == result).sum().item()
            num_total += len(result)
            loss_master.append(loss.detach().cpu().numpy())
            batch_tqdm.set_description('loss:{:.4f}'.format(loss))
        
        acc = num_correct / num_total
        loss = np.average(loss_master)
        epoch_tqdm.set_description('epoch {} - master - loss:{:.4f} - acc:{:.4f}'.format(epoch, loss, acc))

        del master_dataloader

        if (epoch + 1) % 8 == 0:
            torch.save({
                'model_kcge': model_kcge.state_dict(),
                'model_cd': model_cd.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, CD_temp_path)

        # 这里要暂时保存一下x，先不管

    if os.path.exists(CD_temp_path):
        os.remove(CD_temp_path)

    if not update_train or loss < loss_last:
        torch.save({
            'model_kcge': model_kcge.state_dict(),
        }, KCGE_train_path)
        torch.save({
            'model_cd': model_cd.state_dict(),
            'loss' : loss
        }, CD_train_path)

        model_kcge = model_kcge.to('cpu')
        model_cd = model_cd.to('cpu')
        torch.cuda.empty_cache()

        with torch.no_grad():  # 禁用梯度计算
            with torch.jit.optimized_execution(False):  # 禁止优化时隐式转移到GPU
                scripted_model_kcge = torch.jit.script(model_kcge)
                scripted_model_cd = torch.jit.script(model_cd)

        scripted_model_kcge = torch.jit.optimize_for_inference(scripted_model_kcge)
        scripted_model_kcge.save(KCGE_use_path)
        scripted_model_cd = torch.jit.optimize_for_inference(scripted_model_cd)
        scripted_model_cd.save(CD_use_path)

    save_final_data(x.to('cpu'), cddatareader)


if __name__ == '__main__':
    parsers = parser.parse_args()

    CD_pt_path = os.path.join('PT')
    CD_are_schedule_path = os.path.join(CD_pt_path, 'CD_schedule.json')

    cddatareader = CDDataReader()

    are_uids = []
    are_uids_dict = {}
    if os.path.exists(CD_are_schedule_path):
        with open(CD_are_schedule_path, 'r') as f:
            are_uids_dict = json.load(f)
            for are_uid in are_uids_dict:
                if are_uids_dict[are_uid] == 0:
                    are_uids.append(are_uid)
                elif are_uids_dict[are_uid] == 1:
                    are_uids.insert(0, are_uid)
                else:
                    continue
    else:
        are_uids = cddatareader.load_area_uids()
        are_uids_dict = {are_uid : 0 for are_uid in are_uids}
    
    for are_uid in are_uids:
        are_uids_dict[are_uid] = 1

        with open(CD_are_schedule_path, 'w') as f:
            json.dump(are_uids_dict, f)

        cddatareader.set_are_uid(are_uid)
        train_single_are(cddatareader, parsers, are_uid)

        # 这里根据训练出的参数，保存所有学生在该领域的知识点的KT预测结果
        # save_final_predict(are_uid, datareader)

        are_uids_dict[are_uid] = 2

        with open(CD_are_schedule_path, 'w') as f:
            json.dump(are_uids_dict, f)
    
    if os.path.exists(CD_are_schedule_path):
        os.remove(CD_are_schedule_path)
    