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
        cpt = item[1].to(device).long()
        cor = item[2].to(device).float()

        optim.zero_grad()
        output = model(x)

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
parser.add_argument('--epochs',type=int,default=32,help='number of epochs to train (defauly 32 )')
parser.add_argument('--lr',type=float,default=0.01,help='number of learning rate')
parser.add_argument('--hidden_size',type=int,default=256,help='the number of the hidden-size')
parser.add_argument('--max_step',type=int,default=64,help='the number of max step')
parser.add_argument('--num_layers',type=int,default=2,help='the number of layers')

parser.add_argument('--are_uid',type=str,default='are_3fee9e47d0f3428382f4afbcb1004117',help='the uid of area')

# parser.add_argument('--data_dir', type=str, default='../Data',help='the data directory, default as ../Data/KT')
# parser.add_argument('--train_file',type=str,default='train_simple.txt',help='name of train_file')
# parser.add_argument('--master_file',type=str,default='master_simple.txt',help='name of master_file')

# 根据领域来决定的数据
# parser.add_argument('--n_kc',type=int,default=150,help='name of master_file')

# 从数据库中根据指定的领域获取数据来训练
# 领域 - 知识点数量
# 当前时间 - 过去30天内的数据
#     80%用来训练
#     20%用来检测

if __name__ == '__main__': 
    parsers = parser.parse_args()

    # 这里用来获取数据
    train_data, master_data, cpt_num = IPDKTDataReader(parsers.are_uid).load_data_from_db()
    train_data_frame = pd.DataFrame(train_data, columns=['lrn_id','cpt_ids','correct']).set_index('lrn_id')
    master_data_frame = pd.DataFrame(master_data, columns=['lrn_id','cpt_ids','correct']).set_index('lrn_id')

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
    IPDKT_pt_train_path = os.path.join(IPDKT_pt_path, 'IPDKT_train.pt')
    IPDKT_pt_train_path = os.path.normpath(IPDKT_pt_train_path)
    isaddupdate = True
    IPDKT_pt_use_path = os.path.join(IPDKT_pt_path, 'IPDKT_use.pt')
    IPDKT_pt_use_path = os.path.normpath(IPDKT_pt_use_path)
    IPDKT_pt_temp_path = os.path.join(IPDKT_pt_path, 'IPDKT_train_temp.pt')
    IPDKT_pt_temp_path = os.path.normpath(IPDKT_pt_temp_path)

    if os.path.exists(IPDKT_pt_train_path):
        print('增量训练')
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
