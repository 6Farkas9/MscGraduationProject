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
from KCGE.Model.KCGE import KCGE

def protect_norm(models):
    return
    for model in models:
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1e4, neginf=-1e4)

def check_nan(model, step_name):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN/Inf in {step_name}: {name}")
            raise ValueError(f"参数 {name} 在 {step_name} 出现NaN/Inf")

def check_grad_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            stats[name] = {
                "norm": grad.norm(2).item(),
                "max": grad.max().item(),
                "min": grad.min().item(),
                "mean": grad.mean().item()
            }
    print(stats)

def check_data(inputs):
    assert torch.isfinite(inputs).all(), "数据包含 NaN/Inf!"
    print(f"输入范围: [{inputs.min():.4f}, {inputs.max():.4f}]")

parser = argparse.ArgumentParser(description='CD')
parser.add_argument('--batch_size',type=int,default=32,help='number of batch size to train (defauly 32 )')
parser.add_argument('--epochs',type=int,default=2,help='number of epochs to train (defauly 32 )')
parser.add_argument('--lr',type=float,default=1e-5,help='number of learning rate')
parser.add_argument('--embedding_dim',type=int,default=32,help='number of embedding dim')
parser.add_argument('--num_workers',type=int,default=3,help='num of workers')
parser.add_argument('--max_step',type=int,default=128,help='num of max_step')

def save_final_data(datareader : CDDataReader):
    return 0

def train_single_are(cddatareader, parsers, are_uid):
    train_data, master_data, lrn_uids, cpt_uids, scn_uids, cpt_idx, scn_idx, edge_index, edge_attr, edge_type = cddatareader.load_Data_from_db()

    # 这里获得两个set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}

    model_kcge = KCGE(parsers.embedding_dim, device).to(device)
    model_cd = CD(parsers.embedding_dim, device).to(device)
    
    optimizer = torch.optim.Adam([{'params':model_kcge.parameters()},
                                {'params':model_cd.parameters()}], lr= parsers.lr)

    criterion = nn.BCEWithLogitsLoss().to(device)

    KCGE_pt_path = os.path.join(deeplearning_root, 'KCGE', 'PT')
    KCGE_train_path = os.path.join(KCGE_pt_path, 'KCGE_train.pt')
    KCGE_use_path = os.path.join(KCGE_pt_path, 'KCGE_use.pt')
    KCGE_temp_path = os.path.join(KCGE_pt_path, 'KCGE_temp.pt')

    CD_pt_path = os.path.join(deeplearning_root, 'CD', 'PT')
    CD_train_path = os.path.join(CD_pt_path, 'CD_train.pt')
    CD_use_path = os.path.join(CD_pt_path, are_uid + '_use.pt')
    CD_temp_path = os.path.join(CD_pt_path, 'CD_temp.pt')

    # x初始化为全1tensor
    # 比较特殊，这个x既是输入又是优化参数，所以这个x是不是也应该注册到优化器中？
    # 不是，x在训练过程中就会更新？并不会更新
    # 可以让x每次最后更新为z，dropout是否会有影响？应该没有
    # x在模型外初始化，每次进行更新，在final保存的时候保存最终的矩阵
    x = torch.ones((edge_index.size(1), parsers.embedding_dim), dtype=torch.float32, device=device)

    # for ep
    # batch...

    model_kcge.train()
    model_cd.train()

    train_dataset = CDDataset(train_data, lrn_uids, cpt_uids, scn_uids, parsers.max_step)
    train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

    batch_tqdm = tqdm(train_dataloader)

    for item in batch_tqdm:
        # 'learner_idx' : learner_idx,
        # 'scn_seq_index' : scn_seq_index,
        # 'scn_seq_mask' : scn_seq_mask,
        # 'result' : result

        lrn_idx = item['learner_idx']
        scn_seq_idx = item['scn_seq_index']
        scn_seq_mask = item['scn_seq_mask']
        result = item['result']

        z = model_kcge(x, edge_index.to(device), edge_type.to(device), edge_attr.to(device))
        x = z.clone()

        h_scn = z[scn_idx]
        h_cpt = z[cpt_idx]

        r_pred = model_cd(scn_seq_idx.to(device), scn_seq_mask.to(device), h_scn, h_cpt)

        print(result.shape, r_pred.shape)

        result = result.flatten().to(device)
        r_pred = r_pred.flatten()

        # print(result.max(), result.min())
        # print(result_pred.max(), result_pred.min())

        loss = criterion(result, r_pred)

        print(f"损失值: {loss.item()}")

        check_nan(model_kcge, "损失计算")
        check_nan(model_cd, "损失计算")

        check_grad_stats(model_kcge)
        check_grad_stats(model_cd)

        optimizer.zero_grad()

        check_grad_stats(model_kcge)
        check_grad_stats(model_cd)

        loss.backward()

        check_nan(model_kcge, "反向传播")
        check_nan(model_cd, "反向传播")

        check_grad_stats(model_kcge)
        check_grad_stats(model_cd)

        # protect_norm([model_hgc_lrn, model_hgc_scn, model_hgc_cpt, model_cd])
        
        optimizer.step()

        return






if __name__ == '__main__':
    parsers = parser.parse_args()

    cddatareader = CDDataReader()

    are_uids = cddatareader.load_area_uids()

    for are_uid in are_uids:
        cddatareader.set_are_uid(are_uid)
        train_single_are(cddatareader, parsers, are_uid)
    
    # if os.path.exists(CD_temp_path):
    #     print('继续训练')
    #     continue_train = True
    #     check_point = torch.load(CD_temp_path, map_location=device)
        
    #     model_cd.load_state_dict(check_point['model_state_dict_kcge'])
    #     optimizer.load_state_dict(check_point['optimizer_state_dict'])
    #     epoch_start = check_point['epoch'] + 1

    # update_train = False
    # loss_last = None
    # if not continue_train:
    #     if os.path.exists(HGC_LRN_train_path):
    #         print('HGC_LRN增量训练')
    #         # update_train = True
    #         checkpoint = torch.load(HGC_LRN_train_path, map_location=device)
    #         model_hgc_lrn.load_state_dict(checkpoint['model_hgc_lrn'])
    #     else:
    #         print('HGC_LRN初始训练')

    #     if os.path.exists(HGC_SCN_train_path):
    #         print('HGC_SCN增量训练')
    #         # update_train = True
    #         checkpoint = torch.load(HGC_SCN_train_path, map_location=device)
    #         model_hgc_scn.load_state_dict(checkpoint['model_hgc_scn'])
    #     else:
    #         print('HGC_SCN初始训练')

    #     if os.path.exists(HGC_CPT_train_path):
    #         print('HGC_CPT增量训练')
    #         # update_train = True
    #         checkpoint = torch.load(HGC_CPT_train_path, map_location=device)
    #         model_hgc_cpt.load_state_dict(checkpoint['model_hgc_cpt'])
    #     else:
    #         print('HGC_LRN初始训练')

    #     if os.path.exists(CD_train_path):
    #         print('CD增量训练')
    #         update_train = True
    #         checkpoint = torch.load(CD_train_path, map_location=device)
    #         model_cd.load_state_dict(checkpoint['model_cd'])
    #         loss_last = checkpoint['loss']
    #     else:
    #         print('CD初始训练')

    # epoch_tqdm = tqdm(range(epoch_start, parsers.epochs))

    # for epoch in epoch_tqdm:
    #     epoch_tqdm.set_description('epoch {} - train'.format(epoch))

    #     model_hgc_lrn.eval()
    #     model_hgc_scn.eval()
    #     model_hgc_cpt.eval()
    #     model_cd.train()

    #     train_dataset = CDDataset(train_data, uids, learners_init, parsers.max_step)
    #     train_dataloader = DataLoader(train_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

    #     batch_tqdm = tqdm(train_dataloader)
    #     batch_tqdm.set_description('train batch:')

    #     num_correct = 0
    #     num_total = 0
    #     loss_train = []

    #     for item in batch_tqdm:
    #         learner_idx = item['learner_idx']
    #         learner_init = item['learner_init']
    #         scn_seq_index = item['scn_seq_index']
    #         scn_seq_mask = item['scn_seq_mask']
    #         result = item['result']

    #         p_lsl = Data(x = learners_init, edge_index = p_lsl_edge_index, edge_attr = p_lsl_edge_attr)
    #         sub_p_lsl = p_lsl.subgraph(learner_idx)

    #         with torch.no_grad():

    #             lrn_emb = model_hgc_lrn(
    #                 sub_p_lsl.x.to(device), 
    #                 sub_p_lsl.edge_index.to(device), sub_p_lsl.edge_attr.to(device)
    #             )
    #             scn_emb = model_hgc_scn(
    #                 scenes_init.to(device), 
    #                 p_scs_edge_index.to(device), p_scs_edge_attr.to(device),
    #                 p_sls_edge_index.to(device), p_sls_edge_attr.to(device)
    #             )
    #             cpt_emb = model_hgc_cpt(
    #                 concepts_init.to(device), 
    #                 p_cc_edge_index.to(device), p_cc_edge_attr.to(device),
    #                 p_cac_edge_index.to(device), p_cac_edge_attr.to(device), 
    #                 p_csc_edge_index.to(device), p_csc_edge_attr.to(device)
    #             )
            
    #         # 这里已经获得了相当于z的矩阵
    #         # 然后输入到cd中
    #         # print('lrn_emb', lrn_emb.shape)

    #         result_pred = model_cd(
    #             scn_seq_index.to(device), 
    #             scn_seq_mask.to(device), 
    #             lrn_emb,
    #             scn_emb, 
    #             cpt_emb
    #         )

    #         check_data(result_pred)
    #         check_data(result)

    #         check_nan(model_cd, "前向传播")

    #         # print('in train', result_pred.max(), result_pred.min())

    #         result = result.flatten().to(device)
    #         result_pred = result_pred.flatten()

    #         # print(result.max(), result.min())
    #         # print(result_pred.max(), result_pred.min())

    #         loss = criterion(result, result_pred)

    #         print(f"损失值: {loss.item()}")

    #         check_nan(model_cd, "损失计算")

    #         optimizer.zero_grad()

    #         loss.backward()

    #         check_nan(model_cd, "反向传播")

    #         torch.nn.utils.clip_grad_norm_(model_hgc_lrn.parameters(), max_norm=1.0)
    #         torch.nn.utils.clip_grad_norm_(model_hgc_scn.parameters(), max_norm=1.0)
    #         torch.nn.utils.clip_grad_norm_(model_hgc_cpt.parameters(), max_norm=1.0)
    #         torch.nn.utils.clip_grad_norm_(model_cd.parameters(), max_norm=1.0)

    #         check_grad_stats(model_cd)

    #         # protect_norm([model_hgc_lrn, model_hgc_scn, model_hgc_cpt, model_cd])
            
    #         optimizer.step()

    #         check_nan(model_cd, "参数更新")

    #         # protect_norm([model_hgc_lrn, model_hgc_scn, model_hgc_cpt, model_cd])

    #         num_correct += ((result_pred >= 0.5).long() == result).sum().item()
    #         num_total += len(result)
    #         loss_train.append(loss.detach().cpu().numpy())
    #         batch_tqdm.set_description('loss:{:.4f}'.format(loss))
        
    #     acc = num_correct / num_total
    #     loss = np.average(loss_train)
    #     epoch_tqdm.set_description('epoch {} - train - loss:{:.4f} - acc:{:.4f}'.format(epoch, loss, acc))

    #     del train_dataloader

    #     epoch_tqdm.set_description('epoch {} - master'.format(epoch))

    #     model_hgc_lrn.eval()
    #     model_hgc_scn.eval()
    #     model_hgc_cpt.eval()
    #     model_cd.eval()

    #     master_dataset = CDDataset(master_data, uids, learners_init, parsers.max_step)
    #     master_dataloader = DataLoader(master_dataset, batch_size=parsers.batch_size, shuffle=True, num_workers=3, **dataloader_kwargs)

    #     batch_tqdm = tqdm(master_dataloader)
    #     batch_tqdm.set_description('master batch:')

    #     num_correct = 0
    #     num_total = 0
    #     loss_master = []

    #     for item in batch_tqdm:
    #         learner_idx = item['learner_idx']
    #         learner_init = item['learner_init']
    #         scn_seq_index = item['scn_seq_index']
    #         scn_seq_mask = item['scn_seq_mask']
    #         result = item['result']

    #         p_lsl = Data(x = learners_init, edge_index = p_lsl_edge_index, edge_attr = p_lsl_edge_attr)
    #         sub_p_lsl = p_lsl.subgraph(learner_idx)

    #         with torch.no_grad():

    #             lrn_emb = model_hgc_lrn(
    #                 sub_p_lsl.x.to(device), 
    #                 sub_p_lsl.edge_index.to(device), sub_p_lsl.edge_attr.to(device)
    #             )
    #             scn_emb = model_hgc_scn(
    #                 scenes_init.to(device), 
    #                 p_scs_edge_index.to(device), p_scs_edge_attr.to(device),
    #                 p_sls_edge_index.to(device), p_sls_edge_attr.to(device)
    #             )
    #             cpt_emb = model_hgc_cpt(
    #                 concepts_init.to(device), 
    #                 p_cc_edge_index.to(device), p_cc_edge_attr.to(device),
    #                 p_cac_edge_index.to(device), p_cac_edge_attr.to(device), 
    #                 p_csc_edge_index.to(device), p_csc_edge_attr.to(device)
    #             )
                
    #             # 这里已经获得了相当于z的矩阵
    #             # 然后输入到cd中

    #             result_pred = model_cd(
    #                 scn_seq_index.to(device), 
    #                 scn_seq_mask.to(device), 
    #                 lrn_emb,
    #                 scn_emb, 
    #                 cpt_emb
    #             )

    #             print('in master', result_pred.max(), result_pred.min())

    #         result = result.flatten().to(device)
    #         result_pred = result_pred.flatten()

    #         loss = criterion(result, result_pred)

    #         num_correct += ((result_pred >= 0.5).long() == result).sum().item()
    #         num_total += len(result)
    #         loss_master.append(loss.detach().cpu().numpy())
    #         batch_tqdm.set_description('loss:{:.4f}'.format(loss))
        
    #     acc = num_correct / num_total
    #     loss = np.average(loss_master)
    #     epoch_tqdm.set_description('epoch {} - master - loss:{:.4f} - acc:{:.4f}'.format(epoch, loss, acc))

    #     del master_dataloader

    #     if (epoch + 1) % 8 == 0:
    #         torch.save({
    #             'model_hgc_lrn': model_hgc_lrn.state_dict(),
    #             'model_hgc_scn': model_hgc_scn.state_dict(),
    #             'model_hgc_cpt': model_hgc_cpt.state_dict(),
    #             'model_cd': model_cd.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'epoch': epoch
    #         }, CD_temp_path)

    # if os.path.exists(CD_temp_path):
    #     os.remove(CD_temp_path)

    # if not update_train or loss < loss_last:
    #     torch.save({
    #         'model_hgc_lrn': model_hgc_lrn.state_dict(),
    #     }, HGC_LRN_train_path)
    #     torch.save({
    #         'model_hgc_scn': model_hgc_scn.state_dict(),
    #     }, HGC_SCN_train_path)
    #     torch.save({
    #         'model_hgc_cpt': model_hgc_cpt.state_dict(),
    #     }, HGC_CPT_train_path)
    #     torch.save({
    #         'model_cd': model_cd.state_dict(),
    #         'loss' : loss
    #     }, CD_train_path)
    
    #     # torch.save(model.state_dict(), IPDKT_pt_use_path)

    #     scripted_model = torch.jit.script(model_hgc_lrn)
    #     scripted_model = torch.jit.optimize_for_inference(scripted_model)
    #     scripted_model.save(HGC_LRN_use_path)
    #     scripted_model = torch.jit.script(model_hgc_scn)
    #     scripted_model = torch.jit.optimize_for_inference(scripted_model)
    #     scripted_model.save(HGC_SCN_use_path)
    #     scripted_model = torch.jit.script(model_hgc_cpt)
    #     scripted_model = torch.jit.optimize_for_inference(scripted_model)
    #     scripted_model.save(HGC_CPT_use_path)
    #     scripted_model = torch.jit.script(model_cd)
    #     scripted_model = torch.jit.optimize_for_inference(scripted_model)
    #     scripted_model.save(CD_use_path)

    # save_final_data(uids, inits, p_matrixes, cddatareader)