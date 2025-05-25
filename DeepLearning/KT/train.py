import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import os
import torch
import argparse
import json
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from KT.DataSet.IPDKTDataReader import IPDKTDataReader
from KT.DataSet.IPDKTDataset import IPDKTDataset
from KT.Model.IPDKT import IPDKT

def train_epoch(model, train_iterator, optim, criterion, device="cpu"):
    model.train()

    num_correct = 0
    num_sum = 0
    loss_total = []

    tbar = tqdm(train_iterator)

    for item in tbar:
        x = item[0].to(device).float()

        # print(x.shape)
        # exit()

        cpt = item[1].to(device).long()
        cor = item[2].to(device).float()

        optim.zero_grad()
        output = model(x)

        # print(output.shape)
        # exit()

        cor_expanded = cor.unsqueeze(-1).expand_as(output)
        mask = cpt.bool()

        output = torch.masked_select(output, mask).reshape(-1)
        cor_expanded = torch.masked_select(cor_expanded, mask).reshape(-1)

        loss = criterion(output, cor_expanded)

        num_correct += ((output >= 0.5).long() == cor_expanded).sum().item()
        num_sum += len(cor_expanded)

        loss.backward()
        optim.step()
        loss_total.append(loss.item())

        tbar.set_description('loss:{:.4f}'.format(loss))

    acc = num_correct / num_sum
    loss = np.average(loss_total)
    return loss, acc

def master_epoch(model, train_iterator, criterion, device="cpu"):
    model.eval()

    num_correct = 0
    num_sum = 0
    loss_total = []

    tbar = tqdm(train_iterator)

    for item in tbar:
        x = item[0].to(device).float()
        cpt = item[1].to(device).long()
        cor = item[2].to(device).float()

        with torch.no_grad():
            output = model(x)

        cor_expanded = cor.unsqueeze(-1).expand_as(output)
        mask = cpt.bool()

        output = torch.masked_select(output, mask).reshape(-1)
        cor_expanded = torch.masked_select(cor_expanded, mask).reshape(-1)

        loss = criterion(output, cor_expanded)

        num_correct += ((output >= 0.5).long() == cor_expanded).sum().item()
        num_sum += len(cor_expanded)

        loss_total.append(loss.item())

        tbar.set_description('loss:{:.4f}'.format(loss))

    acc = num_correct / num_sum
    loss = np.average(loss_total)
    return loss, acc

# 解析传入的参数
parser = argparse.ArgumentParser(description='IPDKT')
parser.add_argument('--batch_size',type=int,default=32,help='number of batch size to train (defauly 32 )')
parser.add_argument('--epochs',type=int,default=2,help='number of epochs to train (defauly 32 )')
parser.add_argument('--lr',type=float,default=0.01,help='number of learning rate')
parser.add_argument('--hidden_size',type=int,default=256,help='the number of the hidden-size')
parser.add_argument('--max_step',type=int,default=128,help='the number of max step')
parser.add_argument('--num_layers',type=int,default=2,help='the number of layers')

# parser.add_argument('--are_uid',type=str,default='are_3fee9e47d0f3428382f4afbcb1004117',help='the uid of area')

def train_single_are(datareader, parsers):
    # train_data, master_data, cpt_uids = IPDKTDataReader(are_uid).load_data_from_db()
    train_data, master_data, cpt_uids = datareader.load_data_from_db()
    train_data_frame = pd.DataFrame(train_data, columns=['lrn_id','cpt_ids','correct']).set_index('lrn_id')
    master_data_frame = pd.DataFrame(master_data, columns=['lrn_id','cpt_ids','correct']).set_index('lrn_id')

    cpt_num = len(cpt_uids)

    model = 'IPDKT'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}   
    print(f'current device:{device}')

    model = IPDKT(
        input_size= 2 * cpt_num,
        hidden_size= parsers.hidden_size,
        num_layer= parsers.num_layers,
        output_size= cpt_num
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= parsers.lr)

    IPDKT_pt_path = os.path.join('PT')
    IPDKT_pt_train_path = os.path.join(IPDKT_pt_path, are_uid + '_train.pt')
    # print(IPDKT_pt_path)
    IPDKT_pt_train_path = os.path.normpath(IPDKT_pt_train_path)
    isaddupdate = True
    IPDKT_pt_use_path = os.path.join(IPDKT_pt_path, are_uid + '_use.pt')
    # print(IPDKT_pt_use_path)
    IPDKT_pt_use_path = os.path.normpath(IPDKT_pt_use_path)
    IPDKT_pt_temp_path = os.path.join(IPDKT_pt_path, 'IPDKT_train_temp.pt')
    IPDKT_pt_temp_path = os.path.normpath(IPDKT_pt_temp_path)

    if os.path.exists(IPDKT_pt_train_path):
        print(are_uid + '增量训练')
        check_point = torch.load(IPDKT_pt_train_path, map_location=device)
        model.load_state_dict(check_point['model_state_dict'])
        lastloss = check_point['loss']
    else:
        print('初始训练')
        isaddupdate = False

    print('model:')
    print(model)
    
    criterion = nn.BCELoss().to(device)

    loss_all = []

    epoch_start = 0
    if os.path.exists(IPDKT_pt_temp_path):
        check_point = torch.load(IPDKT_pt_temp_path, map_location=device)
        model.load_state_dict(check_point['model_state_dict'])
        optimizer.load_state_dict(check_point['optimizer_state_dict']) 
        epoch_start = check_point['epoch'] + 1
        loss_all = check_point['loss all']

    epoch_tqdm = tqdm(range(epoch_start, parsers.epochs))
    for epoch in epoch_tqdm:
        epoch_tqdm.set_description('epoch - {}'.format(epoch))

        train_dataset = IPDKTDataset(train_data_frame, cpt_num, parsers.max_step)
        train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

        loss, acc = train_epoch(model, train_dataloader, optimizer, criterion, device)
        epoch_tqdm.set_description('epoch - {} train_loss - {:.2f} acc - {:.2f}'.format(epoch, loss, acc))
        del train_dataset, train_dataloader

        master_dataset = IPDKTDataset(master_data_frame, cpt_num, parsers.max_step)
        master_dataloader = DataLoader(master_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

        loss, acc = master_epoch(model, master_dataloader, criterion, device)
        epoch_tqdm.set_description('epoch - {} master_loss - {:.2f} acc - {:.2f}'.format(epoch, loss, acc))
        # print('epoch - {} master_loss - {:.2f} acc - {:.2f}'.format(epoch, loss, acc))
        loss_all.append(loss)
        del master_dataset,master_dataloader

        if (epoch + 1) % 8 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss all': loss_all,
                'epoch': epoch
            }, IPDKT_pt_temp_path)
    
    if os.path.exists(IPDKT_pt_temp_path):
        os.remove(IPDKT_pt_temp_path)

    loss = np.average(loss_all)
    if not isaddupdate or loss < lastloss :
        torch.save({
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, IPDKT_pt_train_path)
    
        # torch.save(model.state_dict(), IPDKT_pt_use_path)

        scripted_model = torch.jit.script(model)
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        scripted_model.save(IPDKT_pt_use_path)

    datareader.make_cpt_trained()

    save_final_predict(are_uid, datareader, device)


def save_final_predict(are_uid, datareader: IPDKTDataReader, device):
    # 获取当前领域的所有学生的各自的数据 - 因为是最终预测，不需要数据大小对齐，所以获取所有数据就行
    final_data, cpt_id2uid = datareader.load_final_data(device)
    
    # 然后加载模型，
    IPDKT_pt_path = os.path.join('PT')
    IPDKT_pt_use_path = os.path.join(IPDKT_pt_path, are_uid + '_use.pt')

    model_ipdkt = torch.jit.load(IPDKT_pt_use_path)

    # 预测出结果
    pre_result = {}
    for lrn_uid in final_data:
        result = model_ipdkt(final_data[lrn_uid])[-1]
        # print(pre_result[lrn_uid])
        pre_result[lrn_uid] = {cpt_uid : result[id].item() for id, cpt_uid in cpt_id2uid.items()}

    # 保存到MongoDB中
    datareader.save_final_data(pre_result)


if __name__ == '__main__': 
    parsers = parser.parse_args()
    IPDKT_pt_path = os.path.join('PT')
    IPDKT_are_schedule_path = os.path.join(IPDKT_pt_path, 'IPDKT_schedule.json')
    datareader = IPDKTDataReader()
    # KT比较特殊，不能用一个temp完全解决中途崩溃的问题，需要记录哪些area已经处理过了
    # 是挺复杂的逻辑，但是这里不需要考虑那么复杂
    # 每个are_uid有三个值，waiting 0/processing 1/done 2
    # 先获取processing的are_uid,然后判断是否有temp.pt
    # 如果有temp.pt则说明是在训练过程中崩溃的？ -- 不能这样，设置的是8个一保存，可能没到8个就崩溃了
    # 所以：排除所有事done的area，从processing开始，训练processing和waiting的
    # 是否重复训练无所谓其实
    are_uids = []
    are_uids_dict = {}
    if os.path.exists(IPDKT_are_schedule_path):
        with open(IPDKT_are_schedule_path, 'r') as f:
            are_uids_dict = json.load(f)
            for are_uid in are_uids_dict:
                if are_uids_dict[are_uid] == 0:
                    are_uids.append(are_uid)
                elif are_uids_dict[are_uid] == 1:
                    are_uids.insert(0, are_uid)
                else:
                    continue
    else:
        are_uids = datareader.load_area_uids()
        are_uids_dict = {are_uid : 0 for are_uid in are_uids}

    for are_uid in are_uids:
        are_uids_dict[are_uid] = 1

        with open(IPDKT_are_schedule_path, 'w') as f:
            json.dump(are_uids_dict, f)

        datareader.set_are_uid(are_uid)
        train_single_are(datareader, parsers)

        # 这里根据训练出的参数，保存所有学生在该领域的知识点的KT预测结果
        # save_final_predict(are_uid, datareader)

        are_uids_dict[are_uid] = 2

        with open(IPDKT_are_schedule_path, 'w') as f:
            json.dump(are_uids_dict, f)
    
    if os.path.exists(IPDKT_are_schedule_path):
        os.remove(IPDKT_are_schedule_path)