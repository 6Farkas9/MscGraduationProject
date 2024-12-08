import torch
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

from Dataset.KCGEDataReader import KCGEDataReader
from Model.KCGE import KCGE
from Model.DTR import Model_PDBeta
from Model.MIRT import MIRT
from Dataset.DADDateReader import DADDataReader
from Dataset.DataOperator import get_H_Data,get_PDBeta_Data
from Dataset.DADDataSet import DADDataset

parser = argparse.ArgumentParser(description='CD')
parser.add_argument('--batch_size',type=int,default=32,help='number of batch size to train (defauly 32 )')
parser.add_argument('--epochs',type=int,default=32,help='number of epochs to train (defauly 32 )')
parser.add_argument('--lr',type=float,default=0.01,help='number of learning rate')
parser.add_argument('--embedding_dim',type=int,default=16,help='number of embedding dim')
parser.add_argument('--path',type=str,default='./Data/JunYi/',help='path of data file')
parser.add_argument('--lamda_kcge',type=int,default=1,help='lamda used in kCGE')
parser.add_argument('--num_workers',type=int,default=3,help='num of workers')

if __name__ == '__main__':

    torch.autograd.set_detect_anomaly(True)

    relation_file_name = 'relations.csv'
    train_file_name = 'test.csv'
    test_file_name = 'test.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    parsers = parser.parse_args()
    path_relation_file = parsers.path + relation_file_name
    path_train_file = parsers.path + train_file_name
    path_test_file = parsers.path + test_file_name

    reader_relation = KCGEDataReader(path_relation_file, device, parsers.embedding_dim)
    data_relation, exer_all, topic_all = reader_relation.load_Data()
    train_data_loader = DADDataReader(path_train_file, device)
    train_log_data, train_stu_exer = train_data_loader.load_Data()
    test_data_loader = DADDataReader(path_test_file, device)
    test_log_data, test_stu_exer = test_data_loader.load_Data()

    model_kcge = KCGE(parsers.embedding_dim, 5, parsers.lamda_kcge).to(device)
    model_dtr = Model_PDBeta(parsers.embedding_dim).to(device)
    model_mirt = MIRT().to(device)

    optimizer = torch.optim.Adam([{'params':model_kcge.parameters()},
                                  {'params':model_dtr.parameters()},
                                  {'params':model_mirt.parameters()},], lr= parsers.lr)

    criterion = nn.BCELoss().to(device)

    for epoch in range(parsers.epochs):
        print('train epoch:{}'.format(epoch))
        model_kcge.train()
        model_dtr.train()
        model_mirt.train()
        num_correct = 0
        num_total = 0
        loss_total = []
        dad_dataset = DADDataset(exer_all, train_log_data)
        dad_dataloader = DataLoader(dad_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=parsers.num_workers, **dataloader_kwargs)
        batch_tqdm = tqdm(dad_dataloader)
        batch_tqdm.set_description('train batch:')
        for item in batch_tqdm:
            stu_id = item[0]
            exer_id = item[1]
            correct = item[2].to(device).float()

            z_star, z_sharp = model_kcge(data_relation)
            h_u, h_v, h_c = get_H_Data(train_stu_exer, exer_all, topic_all, z_sharp, z_star, device)
            p_u, d_v, beta_v = get_PDBeta_Data(h_u, h_v, h_c, model_dtr, parsers.embedding_dim)
            
            p_u_temp = p_u[stu_id]
            d_v_temp = d_v[exer_id]
            beta_v_temp = beta_v[exer_id]

            output = model_mirt(p_u_temp, d_v_temp, beta_v_temp).squeeze()
            loss = criterion(output,correct)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_correct += ((output >= 0.5).long() == correct).sum().item()
            num_total += len(output)
            loss_total.append(loss.item())
            batch_tqdm.set_description('loss:{:.4f}'.format(loss))
        acc = num_correct / num_total
        loss = np.average(loss_total)
        print('epoch {} - loss:{:.4f} - acc:{:.4f}'.format(epoch,loss,acc))
        
        print()

        print('test epoch:{}'.format(epoch))
        model_kcge.eval()
        model_dtr.eval()
        model_mirt.eval()
        num_correct = 0
        num_total = 0
        loss_total = []
        dad_dataset = DADDataset(exer_all, train_log_data)
        dad_dataloader = DataLoader(dad_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=parsers.num_workers, **dataloader_kwargs)
        batch_tqdm = tqdm(dad_dataloader)
        batch_tqdm.set_description('test batch:')
        for item in batch_tqdm:
            stu_id = item[0]
            exer_id = item[1]
            correct = item[2].to(device).float()
            with torch.no_grad():
                z_star, z_sharp = model_kcge(data_relation)
            h_u, h_v, h_c = get_H_Data(test_stu_exer, exer_all, topic_all, z_sharp, z_star, device)
            with torch.no_grad():
                p_u, d_v, beta_v = get_PDBeta_Data(h_u, h_v, h_c, model_dtr, parsers.embedding_dim)
            p_u_temp = p_u[stu_id]
            d_v_temp = d_v[exer_id]
            beta_v_temp = beta_v[exer_id]
            with torch.no_grad():
                output = model_mirt(p_u_temp, d_v_temp, beta_v_temp).squeeze()
            loss = criterion(output,correct)
            num_correct += ((output >= 0.5).long() == correct).sum().item()
            num_total += len(output)
            loss_total.append(loss.item())
            batch_tqdm.set_description('loss:{:.4f}'.format(loss))
        acc = num_correct / num_total
        loss = np.average(loss_total)
        print('epoch {} - loss:{:.4f} - acc:{:.4f}'.format(epoch,loss,acc))
