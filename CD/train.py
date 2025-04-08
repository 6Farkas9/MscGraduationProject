import os
import torch
import argparse
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Model.DTR import Model_PDBeta
from Model.MIRT import MIRT
from Dataset.DADDateReader import DADDataReader
from Dataset.DataOperator import get_H_Data,get_PDBeta_Data
from Dataset.DADDataSet import DADDataset
sys.path.append('..')
from KCGE.DataSet.KCGEDataReader import KCGEDataReader
from KCGE.Model.KCGE import KCGE

# def save_model_state(model_dtr, model_mirt, model_kcge, optimizer_cd, optimizer_kcge, loss, save_cd_path, save_kcge_path):
#     torch.save({
#         'model_state_dict_dtr': model_dtr.state_dict(),
#         'model_state_dict_mirt': model_mirt.state_dict(),
#         'optimizer_state_dict_cd': optimizer_cd.state_dict(),  # 仅保存dtr和mirt的优化器状态
#         'loss': loss
#     }, save_cd_path)

#     torch.save({
#         'model_state_dict_kcge': model_kcge.state_dict(),
#         'optimizer_state_dict_kcge': optimizer_kcge.state_dict()  # 仅保存kcge的优化器状态
#     }, save_kcge_path)

# def load_model_state(model_dtr, model_mirt, model_kcge, optimizer_cd, optimizer_kcge, save_cd_path, save_kcge_path, device):
#     # 加载dtr和mirt的模型参数及优化器状态
#     checkpoint_AB = torch.load(save_cd_path)
#     model_dtr.load_state_dict(checkpoint_AB['model_state_dict_dtr'], map_location=device)
#     model_mirt.load_state_dict(checkpoint_AB['model_state_dict_mirt'], map_location=device)
#     optimizer_cd.load_state_dict(checkpoint_AB['optimizer_state_dict_cd'], map_location=device)  # 加载dtr和mirt的优化器状态

#     # 加载C的模型参数及优化器状态
#     checkpoint_C = torch.load(save_kcge_path)
#     model_kcge.load_state_dict(checkpoint_C['model_state_dict_kcge'], map_location=device)
#     optimizer_kcge.load_state_dict(checkpoint_C['optimizer_state_dict_kcge'], map_location=device)  # 加载kcge的优化器状态

parser = argparse.ArgumentParser(description='CD')
parser.add_argument('--batch_size',type=int,default=32,help='number of batch size to train (defauly 32 )')
parser.add_argument('--epochs',type=int,default=32,help='number of epochs to train (defauly 32 )')
parser.add_argument('--lr',type=float,default=0.01,help='number of learning rate')
parser.add_argument('--embedding_dim',type=int,default=16,help='number of embedding dim')
parser.add_argument('--lamda_kcge',type=int,default=1,help='lamda used in kCGE')
parser.add_argument('--num_workers',type=int,default=3,help='num of workers')

parser.add_argument('--data_dir',type=str,default='../Data/CD/',help='path of data file')
parser.add_argument('--train_file',type=str,default='train.csv',help='name of train_file')
parser.add_argument('--master_file',type=str,default='master.csv',help='name of master_file')

if __name__ == '__main__':
    parsers = parser.parse_args()
    train_file_path = ''
    master_file_path = ''

    if(parsers.data_dir != '../Data/CD/'):
        train_file_path = os.path.join(parsers.data_dir, parsers.train_file)
        master_file_path = os.path.join(parsers.data_dir, parsers.master_file)
    else:
        data_dir_path = os.path.join('..', 'Data', 'CD')
        train_file_path = os.path.join(data_dir_path, parsers.train_file)
        master_file_path = os.path.join(data_dir_path, parsers.master_file)
    
    train_file_path = os.path.normpath(train_file_path)
    master_file_path = os.path.normpath(master_file_path)

    if not os.path.exists(train_file_path) or not os.path.exists(master_file_path):
        print(train_file_path, ' , ' , master_file_path)
        print('wrong file path')
        os._exit(0)
    
    # 这里引用的知识图谱需要更改
    kcge_data_path = os.path.join('..','Data','KG')
    relation_file_path = os.path.join(kcge_data_path, 'relations.csv')
    relation_file_path = os.path.normpath(relation_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    reader_relation = KCGEDataReader(relation_file_path, device, parsers.embedding_dim)
    data_relation, exer_all, topic_all = reader_relation.load_Data()

    train_data_loader = DADDataReader(train_file_path, device)
    train_log_data, train_stu_exer = train_data_loader.load_Data()

    master_data_loader = DADDataReader(master_file_path, device)
    master_log_data, test_stu_exer = master_data_loader.load_Data()

    model_kcge = KCGE(parsers.embedding_dim, 5, parsers.lamda_kcge).to(device)
    model_dtr = Model_PDBeta(parsers.embedding_dim).to(device)
    model_mirt = MIRT().to(device)

    optimizer = torch.optim.Adam([{'params':model_dtr.parameters()},
                                {'params':model_mirt.parameters()},
                                {'params':model_kcge.parameters()}], lr= parsers.lr)

    criterion = nn.BCELoss().to(device)

    pt_path = os.path.join('PT')
    cd_pt_train_path = os.path.join(pt_path, 'cd_train.pt')
    cd_pt_train_path = os.path.normpath(cd_pt_train_path)
    cd_pt_temp_path = os.path.join(pt_path, 'cd_temp.pt')
    cd_pt_temp_path = os.path.normpath(cd_pt_temp_path)
    cd_pt_use_path = os.path.join(pt_path, 'cd_use.pt')
    cd_pt_use_path = os.path.normpath(cd_pt_use_path)

    kcge_pt_path = os.path.join('..', 'KCGE', 'PT')
    kcge_pt_path = os.path.normpath(kcge_pt_path)
    kcge_pt_use_path = os.path.join(kcge_pt_path, 'kcge_use.pt')
    kcge_pt_use_path = os.path.normpath(kcge_pt_use_path)

    cd_add_update = True
    kcge_add_update = True

    # 有问题：优化器参数的分开组合
    if os.path.exists(cd_pt_train_path):
        print('CD增量训练')
        # 加载dtr和mirt的模型参数及优化器状态
        checkpoint_cd = torch.load(cd_pt_train_path,  map_location=device)
        model_dtr.load_state_dict(checkpoint_cd['model_state_dict_dtr'])
        model_mirt.load_state_dict(checkpoint_cd['model_state_dict_mirt'])
        lastloss = checkpoint_cd['loss']
    else:
        print('CD初始训练')
        cd_add_update = False

    if os.path.exists(kcge_pt_use_path):
        print('KCGE增量训练')
        checkpoint_kcge = torch.load(kcge_pt_use_path, map_location=device)
        model_kcge.load_state_dict(checkpoint_kcge['model_state_dict_kcge'])
    else:
        print('KCGE初始训练')
        kcge_add_update = False

    epoch_start = 0
    loss_all = []
    if os.path.exists(cd_pt_temp_path):
        check_point = torch.load(cd_pt_temp_path, map_location=device)
        model_dtr.load_state_dict(check_point['model_state_dict_dtr'])
        model_mirt.load_state_dict(check_point['model_state_dict_mirt'])
        model_kcge.load_state_dict(check_point['model_state_dict_kcge'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        epoch_start = check_point['epoch'] + 1
        loss_all = check_point['loss all']

    for epoch in range(epoch_start, parsers.epochs):
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
            loss = criterion(output, correct)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("KCGE grad:", [p.grad for p in model_kcge.parameters() if p.grad is not None])

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
        print('epoch {} - loss:{:.4f} - acc:{:.4f}'.format(epoch, loss, acc))

        loss_all.append(loss)
        if (epoch + 1) % 8 == 0:
            torch.save({
                'model_state_dict_dtr': model_dtr.state_dict(),
                'model_state_dict_mirt': model_mirt.state_dict(),
                'model_state_dict_kcge': model_kcge.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss all': loss_all
            }, cd_pt_temp_path)

    if os.path.exists(cd_pt_temp_path):
        os.remove(cd_pt_temp_path)
    
    loss = np.average(loss_all)
    if not cd_add_update or not kcge_add_update or loss < lastloss:
        torch.save({
            'model_state_dict_dtr': model_dtr.state_dict(),
            'model_state_dict_mirt': model_mirt.state_dict(),
            'loss': loss
        }, cd_pt_train_path)

        torch.save({
            'model_state_dict_kcge': model_kcge.state_dict(),
        }, kcge_pt_use_path)

    torch.save({
        'model_state_dict_dtr': model_dtr.state_dict(),
        'model_state_dict_mirt': model_mirt.state_dict()
    }, cd_pt_use_path)

    torch.save({
        'model_state_dict_kcge': model_kcge.state_dict()
    }, kcge_pt_use_path)
