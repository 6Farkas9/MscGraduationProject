import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict

from DataReader.BasicDataReader import basicdr
from Data.CDRepository import cdrepo

class CDDataReader():

    def __init__(self):
        self.lrn_uid = basicdr.lrn_uid
        self.lrn_num = basicdr.lrn_num

        self.untqus_uid = basicdr.untqus_uid
        self.untqus_num = basicdr.untqus_num

        self.cpt_uid = basicdr.cpt_uid
        self.cpt_num = basicdr.cpt_num

        self.qus_uid = basicdr.qus_uid
        self.qus_num = basicdr.qus_num

        self.getLrnQusData()
    
    def getLrnQusData(self):
        self.lrn_qus_uids = cdrepo.getLrnQus()

    def getTrainTestData(self, train_ratio=0.8):
        """划分训练集和测试集 - 按每个学习者划分，保持时间顺序"""
        interactions = self.lrn_qus_uids
        
        # 首先按学习者分组
        learner_interactions = defaultdict(list)
        for lrn_uid, qus_uid, result in interactions:
            if lrn_uid not in self.lrn_uid or qus_uid not in self.qus_uid:
                continue
            learner_interactions[lrn_uid].append((qus_uid, result))
        
        # 按每个学习者划分训练测试集
        train_data = defaultdict(lambda: [[], []])
        test_data = defaultdict(lambda: [[], []])
        
        for lrn_uid, interactions_list in learner_interactions.items():
            if len(interactions_list) < 2:  # 至少需要2条记录才能划分
                # 如果只有一条记录，放入训练集
                if interactions_list:
                    qus_uid, result = interactions_list[0]
                    train_data[lrn_uid][0].append(qus_uid)
                    train_data[lrn_uid][1].append(result)
                continue
                
            # 计算划分点
            split_index = int(len(interactions_list) * train_ratio)
            
            # 确保至少有一条训练数据和一条测试数据
            if split_index == 0:
                split_index = 1
            
            # 前 train_ratio 作为训练集
            for i in range(split_index):
                qus_uid, result = interactions_list[i]
                train_data[lrn_uid][0].append(qus_uid)
                train_data[lrn_uid][1].append(result)
            
            # 后 (1-train_ratio) 作为测试集
            for i in range(split_index, len(interactions_list)):
                qus_uid, result = interactions_list[i]
                test_data[lrn_uid][0].append(qus_uid)
                test_data[lrn_uid][1].append(result)
        
        # 统计信息
        # train_count = sum(len(data[0]) for data in train_data.values())
        # test_count = sum(len(data[0]) for data in test_data.values())
        # print(f"训练集: {train_count} 条记录, 测试集: {test_count} 条记录")
        
        # return train_data, test_data
        self.train_data = train_data
        self.test_data = test_data
    
    def loadDatafromSql(self):

        self.getTrainTestData()

        return {
            # 交互数据
            'train_data': self.train_data,
            'test_data': self.test_data,
            
            # ID映射
            'lrn_uid': self.lrn_uid,
            'qus_uid': self.qus_uid,  # 使用qus替换scn
            'cpt_uid': self.cpt_uid,
        }
    
cddr = CDDataReader()
    
if __name__ == '__main__':
    # train_data, test_data = cddr.getTrainTestData()
    
    # print(train_data)
    # print(test_data)
    cddata = cddr.loadDatafromSql()
    print(cddata)

