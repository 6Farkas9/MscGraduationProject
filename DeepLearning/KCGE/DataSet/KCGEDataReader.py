import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import torch
from datetime import datetime, timedelta
from torch_geometric.data import Data

from Data.MySQLOperator import mysqldb

class KCGEDataReader():

    def __init__(self, are_uid):
        self.are_uid = are_uid

    # 知识点 - 领域 : 双向边，边类型1，边权重1
    def get_cpt_uid_of_are(self):
        cpt_uids_list = mysqldb.get_cpt_uid_of_are(self.are_uid)
        self.cpt_uids = {cpt_uid : idx + 1 for idx, cpt_uid in enumerate(cpt_uids_list)}
        edge = [(self.cpt_uids[cpt_uid], 0) for cpt_uid in cpt_uids_list]
        edge = edge + [(0, cpt_id) for cpt_id, _ in edge]
        edge_index = torch.tensor(edge, dtype=torch.long).t()
        edge_attr = torch.ones(edge_index.size(1), dtype=torch.float32)
        edge_type = torch.full((edge_index.size(1),), fill_value = 1, dtype=torch.long)
        return edge_index, edge_attr, edge_type

    # 知识点 - 知识点 ： 单向边，边类型2，边权重1
    def get_cpt_cpt_of_are(self):
        cpt_cpt_tuple = mysqldb.get_cpt_cpt_of_are(self.are_uid)
        edge = [(self.cpt_uids[item[0]], self.cpt_uids[item[1]]) for item in cpt_cpt_tuple]
        edge_index = torch.tensor(edge, dtype=torch.long).t()
        edge_attr = torch.ones(edge_index.size(1), dtype=torch.float32)
        edge_type = torch.full((edge_index.size(1),), fill_value = 2, dtype=torch.long)
        return edge_index, edge_attr, edge_type

    # 知识点 - 场景 ： 双向边， 边类型3， 边权重difficulty
    def get_scn_cpt_uid_of_are(self):
        scn_cpt_diff_tuple = mysqldb.get_scn_cpt_uid_of_are(self.are_uid)

        unique_scn_uids = {item[0] for item in scn_cpt_diff_tuple}

        cpt_num = len(self.cpt_uids)
        self.scn_uids = {scn_uid : idx + cpt_num + 1 for idx, scn_uid in enumerate(unique_scn_uids)}

        self.scn_cpt = {}
        for scn_uid, cpt_uid, _ in scn_cpt_diff_tuple:
            if scn_uid not in self.scn_cpt:
                self.scn_cpt[scn_uid] = []
            self.scn_cpt[scn_uid].append(cpt_uid)

        edge = [(self.scn_uids[scn_uid], self.cpt_uids[cpt_uid]) for scn_uid, cpt_uid, _ in scn_cpt_diff_tuple]
        edge = edge + [(cpt_id, scn_id) for scn_id, cpt_id in edge]

        attr = [difficuty for _, _, difficuty in scn_cpt_diff_tuple]
        attr = attr + attr

        edge_index = torch.tensor(edge, dtype=torch.long).t()
        edge_attr = torch.tensor(attr, dtype=torch.float32)
        edge_type = torch.full((edge_index.size(1),), fill_value = 3, dtype=torch.long)

        return edge_index, edge_attr, edge_type
    
    # 自连接边：单向就行，边权重1，边类型0
    def add_self_link(self):
        node_num = len(self.cpt_uids) + len(self.scn_uids) + 1
        edge = [(i, i) for i in range(node_num)]

        edge_index = torch.tensor(edge, dtype=torch.long).t()
        edge_attr = torch.ones(edge_index.size(1), dtype=torch.float32)
        edge_type = torch.full((edge_index.size(1),), fill_value = 0, dtype=torch.long)

        return edge_index, edge_attr, edge_type 

    def load_data_from_db(self):
        # 异构网络中有实体：学习者、场景、知识点、领域
        # 除开学习者有关系：
        #   1.场景 - 知识点 -- 难度 -- graph_involve
        #   2.知识点 - 知识点 -- 无权重 -- graph_precondition
        #   3.知识点 - 领域 -- 权重 -- graph_belong
        # 其中知识点 - 领域的关系也是是需要训练的
        # 也就是有三种关系 - 构建三个参数矩阵w
        # 加上节点自环边，一共四种关系，构建四个参数矩阵w

        # KCGE - 计算出z矩阵(h_scn, h_cpt同时得到) - 使用cddatareader获得的30天内的交互数据 - 得到h_lrn
        # 在cd内部进行dtr和mirt操作
        # 仔细想想，cd也应该是按领域去搞

        # 那就是：
        # 1.根据输入的are_uid获取
        #     1.1.graph_belong -- 也就获得了所有该领域内的知识点
        #     1.2.graph_precondition
        #     1.3.graph_involve -- 只找涉及这些知识点的场景


        # 2.根据上述数据构建图data
        # 因为在使用的时候不会在工程中调用KCGE
        #     中途添加scn则使用scn涉及的知识点的加权均值暂时替代
        #     中途添加cpt则使用同领域内的cpt的均值替代
        # 不过以防万一还是不使用pyg的data去存储，将图拆分为x, index, attr三部分


        # 3.然后直接将数据返回就行了，因为是该领域内的全局计算

        # 由于图中的实体都是葫芦搅茄子放在一起，所以一定要有从uid到特征索引的索引
        # 让are_uid作为0
        # cpt_uid 从1 - cpt_num
        # scn_uid 从cpt_num + 1 - 最后

        edge_index, edge_attr, edge_type = self.get_cpt_uid_of_are()

        temp_edge_index, temp_edge_attr, temp_edge_type = self.get_cpt_cpt_of_are()
        edge_index = torch.cat((edge_index, temp_edge_index), dim = 1)
        edge_attr = torch.cat((edge_attr, temp_edge_attr), dim = 0)
        edge_type = torch.cat((edge_type, temp_edge_type), dim = 0)

        temp_edge_index, temp_edge_attr, temp_edge_type = self.get_scn_cpt_uid_of_are()
        edge_index = torch.cat((edge_index, temp_edge_index), dim = 1)
        edge_attr = torch.cat((edge_attr, temp_edge_attr), dim = 0)
        edge_type = torch.cat((edge_type, temp_edge_type), dim = 0)

        temp_edge_index, temp_edge_attr, temp_edge_type = self.add_self_link()
        edge_index = torch.cat((edge_index, temp_edge_index), dim = 1)
        edge_attr = torch.cat((edge_attr, temp_edge_attr), dim = 0)
        edge_type = torch.cat((edge_type, temp_edge_type), dim = 0)

        return edge_index, edge_attr, edge_type
    
if __name__ == '__main__':
    kcgedatareader =  KCGEDataReader('are_3fee9e47d0f3428382f4afbcb1004117')
    a,b,c = kcgedatareader.get_cpt_uid_of_are()
    d,e,f = kcgedatareader.get_cpt_cpt_of_are()
    g,h,j = kcgedatareader.get_scn_cpt_uid_of_are()

    print(a.shape, b.shape, c.shape)
    print(d.shape, e.shape, f.shape)
    print(g.shape, h.shape, j.shape)

    edge_index, edge_attr, edge_type = kcgedatareader.load_data_from_db()

    print(edge_index.shape, edge_attr.shape, edge_type.shape)
