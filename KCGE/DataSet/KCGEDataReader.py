import torch
import pandas as pd
from torch_geometric.data import Data

class KCGEDataReader():
    def __init__(self, path, device, embedding_dim):
        self.path = path
        self.device = device
        self.embedding_dim = embedding_dim

    def load_Data(self):
        reader_file = pd.read_csv(self.path, header=0)
        # print(f"文件前几行:\n{reader_file.head()}")
        start_points = []
        end_points = []
        edge_type = []
        edge_weight = []
        len_row, _ = reader_file.shape
        entity_all = {}

        exer_all = []
        topic_all = []

        for i in range(len_row):
            e_1 = reader_file.iloc[i,0]
            e_2 = reader_file.iloc[i,1]
            e_type = reader_file.iloc[i,2]
            e_weight = reader_file.iloc[i,3]

            if e_1 not in entity_all.keys():
                entity_all[e_1] = None
                if e_type != 4:
                    exer_all.append(e_1)
                else:
                    topic_all.append(e_1)
            if e_2 not in entity_all.keys():
                entity_all[e_2] = None
                if e_type == 0 or e_type == 1 or e_type == 2:
                    exer_all.append(e_2)
                elif e_type == 3:
                    topic_all.append(e_2)

            start_points.extend([e_1,e_2])
            end_points.extend([e_2,e_1])
            edge_type.extend([e_type,e_type])
            edge_weight.extend([e_weight,e_weight])
        
        for node in entity_all.keys():
            start_points.append(node)
            end_points.append(node)
            edge_type.append(0)  # 假设自环边的类型是0
            edge_weight.append(1)  # 自环边的权重设为1
        
        # print(f"总实体数量: {len(entity_all.keys())}")
        # print(f"exer_all 列表的长度: {len(exer_all)}")
        # print(f"topic_all 列表的长度: {len(topic_all)}")
        # print(f"边的数量: {len(start_points)}")
        # print(f"边的示例: {start_points[:5]}, {end_points[:5]}, {edge_type[:5]}, {edge_weight[:5]}")

        x = torch.ones(len(entity_all.keys()), self.embedding_dim, dtype=torch.float).to(self.device)
        # x = torch.randn(len(entity_all.keys()), self.embedding_dim, dtype=torch.float).to(self.device)

        # print(f"x 的最小值: {x.min()}, 最大值: {x.max()}")

        edge_index = torch.tensor([start_points, end_points], dtype=torch.long).to(self.device)

        edge_type = torch.tensor(edge_type, dtype=torch.long).to(self.device)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float).to(self.device)
        # edge_weight = torch.ones(len(start_points), dtype=torch.float).to(self.device)  # 将所有边权重初始化为1
        # edge_weight[edge_weight == 0] = 1
        data = Data(x = x, edge_index = edge_index, edge_type = edge_type, edge_weight = edge_weight)

        exer_dict = {}
        topic_dict = {}

        # print(f"data.x 的形状: {data.x.shape}")
        # print(f"data.edge_index 的形状: {data.edge_index.shape}")
        # print(f"data.edge_type 的形状: {data.edge_type.shape}")
        # print(f"data.edge_weight 的形状: {data.edge_weight.shape}")

        for i in range(len(exer_all)):
            exer_dict[exer_all[i]] = i
        for i in range(len(topic_all)):
            topic_dict[topic_all[i]] = i

        return data, exer_dict, topic_dict
