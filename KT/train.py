import os
import torch
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from DataSet.DataReader import DataReader
from DataSet.IPDKTDataset import IPDKTDataset
from Model.IPDKT import IPDKT

def train_epoch(model, train_iterator, optim, lamda_kt, criterion_kt, lamda_ik,criterion_ik, device="cpu"):
    model.train()

    num_correct_kt = 0
    num_sum_kt = 0
    loss_total = []

    tbar = tqdm(train_iterator)

    for item in tbar:
        x = item[0].to(device).float()
        que = item[1].to(device).long()
        kc = item[2].to(device).long()
        cor = item[3].to(device).float()
        cor_rate = item[4].to(device).float()

        optim.zero_grad()
        output_kt,output_ik = model(x)

        que_expanded = que.unsqueeze(-1).expand_as(output_kt)
        cor_expanded = cor.unsqueeze(-1).expand_as(output_kt)
        mask_kt = kc.bool() & que_expanded.bool()

        output_kt = torch.masked_select(output_kt, mask_kt).reshape(-1)
        cor_expanded = torch.masked_select(cor_expanded, mask_kt).reshape(-1)

        loss_kt = criterion_kt(output_kt, cor_expanded)

        num_correct_kt += ((output_kt >= 0.5).long() == cor_expanded).sum().item()
        num_sum_kt += len(cor_expanded)

        output_ik = output_ik.reshape(-1)
        cor_rate = cor_rate.reshape(-1)

        loss_ik = criterion_ik(output_ik, cor_rate)
        loss = lamda_kt * loss_kt + lamda_ik * loss_ik
        loss.backward()
        optim.step()
        loss_total.append(loss.item())

        tbar.set_description('loss:{:.4f}'.format(loss))

    acc = num_correct_kt / num_sum_kt
    loss = np.average(loss_total)
    return loss,acc

def master_epoch(model, train_iterator, lamda_kt, criterion_kt, lamda_ik,criterion_ik, device="cpu"):
    model.eval()

    num_correct_kt = 0
    num_sum_kt = 0
    loss_total = []

    tbar = tqdm(train_iterator)

    for item in tbar:
        x = item[0].to(device).float()
        que = item[1].to(device).long()
        kc = item[2].to(device).long()
        cor = item[3].to(device).float()
        cor_rate = item[4].to(device).float()

        with torch.no_grad():
            output_kt,output_ik = model(x)

        que_expanded = que.unsqueeze(-1).expand_as(output_kt)
        cor_expanded = cor.unsqueeze(-1).expand_as(output_kt)
        mask_kt = kc.bool() & que_expanded.bool()

        output_kt = torch.masked_select(output_kt, mask_kt).reshape(-1)
        cor_expanded = torch.masked_select(cor_expanded, mask_kt).reshape(-1)

        loss_kt = criterion_kt(output_kt, cor_expanded)

        num_correct_kt += ((output_kt >= 0.5).long() == cor_expanded).sum().item()
        num_sum_kt += len(cor_expanded)

        output_ik = output_ik.reshape(-1)
        cor_rate = cor_rate.reshape(-1)

        loss_ik = criterion_ik(output_ik, cor_rate)
        loss = lamda_kt * loss_kt + lamda_ik * loss_ik
        loss_total.append(loss.item())

        tbar.set_description('loss:{:.4f}'.format(loss))

    acc = num_correct_kt / num_sum_kt
    loss = np.average(loss_total)
    return loss,acc


# 解析传入的参数
parser = argparse.ArgumentParser(description='IPDKT')
parser.add_argument('--batch_size',type=int,default=32,help='number of batch size to train (defauly 32 )')
parser.add_argument('--epochs',type=int,default=32,help='number of epochs to train (defauly 35 )')
parser.add_argument('--lr',type=float,default=0.01,help='number of learning rate')
parser.add_argument('--hidden_size',type=int,default=200,help='the number of the hidden-size')
parser.add_argument('--max_step',type=int,default=100,help='the number of max step')
parser.add_argument('--num_layers',type=int,default=2,help='the number of layers')
parser.add_argument('--lamda_kt',type=float,default=0.5,help='the number of lamda KT')
parser.add_argument('--lamda_ik',type=float,default=0.5,help='the number of lamda IK')

parser.add_argument('--data_dir', type=str, default='../Data/KT',help='the data directory, default as ../Data/KT')
parser.add_argument('--train_file',type=str,default='train.csv',help='name of train_file')
parser.add_argument('--master_file',type=str,default='master.csv',help='name of master_file')

parser.add_argument('--n_kc',type=int,default=110,help='name of master_file')

if __name__ == '__main__': 
    parsers = parser.parse_args()
    train_file_path = ''
    master_file_path = ''
    
    if(parsers.data_dir != '../Data/KT'):
        train_file_path = os.path.join(parsers.data_dir, parsers.train_file)
        master_file_path = os.path.join(parsers.data_dir, parsers.master_file)
    else:
        data_dir_path = os.path.join('..', 'Data', 'KT')
        train_file_path = os.path.join(data_dir_path, parsers.train_file)
        master_file_path = os.path.join(data_dir_path, parsers.master_file)
    
    train_file_path = os.path.normpath(train_file_path)
    master_file_path = os.path.normpath(master_file_path)

    if not os.path.exists(train_file_path) or not os.path.exists(master_file_path):
        print(train_file_path, ' , ' , master_file_path)
        print('wrong file path')
        os._exit(0)

    model = 'IPDKT'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}   
    print(f'current device:{device}')

    model = IPDKT(
        input_size= 2 * parsers.n_kc,
        hidden_size= parsers.hidden_size,
        num_layer= parsers.num_layers,
        output_size= parsers.n_kc
    )
    model.to(device)
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
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        lastloss = check_point['loss']
    else:
        print('初始训练')
        isaddupdate = False

    print('model:')
    print(model)
    
    criterion_kt = nn.BCELoss()
    criterion_ik = nn.MSELoss()
    
    criterion_kt.to(device)
    criterion_ik.to(device)

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
        train_data,train_len,train_kc_num = DataReader(train_file_path).load_data()
        train_data = pd.DataFrame(train_data,columns=['stu_id','que_id','kc_id','correct','cor_rate']).set_index('stu_id')
        train_dataset = IPDKTDataset(train_data,train_kc_num,parsers.max_step)
        train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)
        loss,acc = train_epoch(model, train_dataloader, optimizer, parsers.lamda_kt,criterion_kt, parsers.lamda_ik, criterion_ik, device)
        print('epoch - {} train_loss - {:.2f} acc - {:.2f}'.format(epoch,loss,acc))
        del train_data,train_len,train_kc_num,train_dataset,train_dataloader

        master_data,master_len,master_kc_num = DataReader(master_file_path).load_data()
        master_data = pd.DataFrame(master_data,columns=['stu_id','que_id','kc_id','correct','cor_rate']).set_index('stu_id')
        master_dataset = IPDKTDataset(master_data,master_kc_num,parsers.max_step)
        master_dataloader = DataLoader(master_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)
        loss,acc = master_epoch(model, master_dataloader, parsers.lamda_kt,criterion_kt, parsers.lamda_ik, criterion_ik, device)
        print('epoch - {} master_loss - {:.2f} acc - {:.2f}'.format(epoch,loss,acc))
        loss_all.append(loss)
        del master_data,master_len,master_kc_num,master_dataset,master_dataloader

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
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, IPDKT_pt_train_path)
    
    torch.save(model.state_dict(), IPDKT_pt_use_path)