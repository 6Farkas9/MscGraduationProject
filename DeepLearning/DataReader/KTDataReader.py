import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from Data.KTRepository import ktrepo
from DataReader.BasicDataReader import basicdr

class KTDataReader():
    """
    知识追踪数据读取器 - 准备训练所需的静态数据
    """

    def __init__(self):
        # 基础ID映射
        self.lrn_uid = basicdr.lrn_uid
        self.lrn_num = basicdr.lrn_num
        
        self.untqus_uid = basicdr.untqus_uid
        self.untqus_num = basicdr.untqus_num
        
        self.cpt_uid = basicdr.cpt_uid
        self.cpt_num = basicdr.cpt_num
        
        # 从Repository获取数据
        self.getInteractionData()
        self.getUnitTypeData()
        self.getConceptMappingData()
        
        # 划分训练测试集
        self.getTrainTestData()
    
    def getInteractionData(self):
        """获取交互数据并处理"""
        raw_interactions = ktrepo.getLrnInteractions()
        
        # 处理交互数据：添加交互类型和结果标记
        self.interactions = []
        for lrn_uid, unt_uid, add1, add2, create_time in raw_interactions:
            if lrn_uid not in self.lrn_uid or unt_uid not in self.untqus_uid:
                continue
                
            # 判断是否为题目交互（有结果）
            is_question = 1 if unt_uid in basicdr.qus_uid else 0
            # 对于题目交互，add2表示正确性；对于其他交互，add1/add2作为特征
            result = add2 if is_question else -1  # -1表示无结果
            
            self.interactions.append((lrn_uid, unt_uid, add1, add2, create_time, is_question, result))
    
    def getUnitTypeData(self):
        """获取学习单元类型数据"""
        raw_unit_types = ktrepo.getUntTypes()
        self.unit_types = {}
        
        for unt_uid, unit_type in raw_unit_types.items():
            if unt_uid in self.untqus_uid:
                # 将类型字符串映射为数字
                type_mapping = {
                    'video': 0, 'vr': 1, 'ar': 2, 
                    'interact': 3, 'cooperate': 4, 'question': 5
                }
                self.unit_types[unt_uid] = type_mapping.get(unit_type, 5)
    
    def getConceptMappingData(self):
        """获取知识点映射数据"""
        # 题目-知识点映射
        qus_cpts = ktrepo.getQusCpts()
        self.question_concepts = {}
        
        for qus_uid, cpt_list in qus_cpts.items():
            if qus_uid in basicdr.qus_uid:
                # 只保留有效的知识点ID
                valid_cpts = [cpt for cpt in cpt_list if cpt in self.cpt_uid]
                if valid_cpts:
                    self.question_concepts[qus_uid] = valid_cpts
        
        # 学习单元-知识点映射
        unt_cpts = ktrepo.getUntCpts()
        self.unit_concepts = {}
        
        for unt_uid, cpt_list in unt_cpts.items():
            if unt_uid in self.untqus_uid:
                valid_cpts = [cpt for cpt in cpt_list if cpt in self.cpt_uid]
                if valid_cpts:
                    self.unit_concepts[unt_uid] = valid_cpts
    
    def getTrainTestData(self, train_ratio=0.8):
        """划分训练集和测试集 - 按每个学习者划分，保持时间顺序"""
        # 按学习者分组
        learner_interactions = defaultdict(list)
        for interaction in self.interactions:
            lrn_uid, unt_uid, add1, add2, create_time, is_question, result = interaction
            learner_interactions[lrn_uid].append((unt_uid, add1, add2, create_time, is_question, result))
        
        # 按每个学习者划分训练测试集
        train_data = defaultdict(lambda: [[], [], [], [], [], []])  # [unt_uids, add1s, add2s, is_questions, results, next_is_questions]
        test_data = defaultdict(lambda: [[], [], [], [], [], []])
        
        for lrn_uid, interactions_list in learner_interactions.items():
            if len(interactions_list) < 2:  # 至少需要2条记录才能划分
                continue  # 跳过只有一条记录的学习者
                
            # 计算划分点
            split_index = int(len(interactions_list) * train_ratio)
            if split_index == 0:
                split_index = 1
            
            # 处理训练集数据（需要下一个时间步信息）
            for i in range(split_index):
                unt_uid, add1, add2, create_time, is_question, result = interactions_list[i]
                
                # 标记下一个时间步是否是题目交互
                next_is_question = interactions_list[i+1][4] if i < split_index - 1 else 0
                next_result = interactions_list[i+1][5] if i < split_index - 1 else -1
                
                train_data[lrn_uid][0].append(unt_uid)
                train_data[lrn_uid][1].append(add1)
                train_data[lrn_uid][2].append(add2)
                train_data[lrn_uid][3].append(is_question)
                train_data[lrn_uid][4].append(result)
                train_data[lrn_uid][5].append(next_is_question)  # 下一个时间步掩码
            
            # 处理测试集数据
            for i in range(split_index, len(interactions_list)):
                unt_uid, add1, add2, create_time, is_question, result = interactions_list[i]
                
                # 标记下一个时间步是否是题目交互
                next_is_question = interactions_list[i+1][4] if i < len(interactions_list) - 1 else 0
                next_result = interactions_list[i+1][5] if i < len(interactions_list) - 1 else -1
                
                test_data[lrn_uid][0].append(unt_uid)
                test_data[lrn_uid][1].append(add1)
                test_data[lrn_uid][2].append(add2)
                test_data[lrn_uid][3].append(is_question)
                test_data[lrn_uid][4].append(result)
                test_data[lrn_uid][5].append(next_is_question)  # 下一个时间步掩码
        
        self.train_data = train_data
        self.test_data = test_data
    
    def loadDatafromSql(self):
        """加载数据并返回静态数据字典"""
        return {
            # 交互数据
            'train_data': self.train_data,
            'test_data': self.test_data,
            
            # 类型和映射数据
            'unit_types': self.unit_types,
            'question_concepts': self.question_concepts,
            'unit_concepts': self.unit_concepts,
            
            # ID映射
            'lrn_uid': self.lrn_uid,
            'untqus_uid': self.untqus_uid,
            'cpt_uid': self.cpt_uid,
        }
    
ktdr = KTDataReader()

if __name__ == '__main__':
    # ktdr = KTDataReader()
    
    print("测试KTDataReader功能:")
    print("=" * 50)
    
    # 加载数据
    kt_data = ktdr.loadDatafromSql()
    
    print("数据统计信息:")
    print(f"学习者数量: {len(kt_data['lrn_uid'])}")
    print(f"学习单元+题目数量: {len(kt_data['untqus_uid'])}")
    print(f"知识点数量: {len(kt_data['cpt_uid'])}")
    print(f"学习单元类型数量: {len(kt_data['unit_types'])}")
    print(f"题目-知识点映射数量: {len(kt_data['question_concepts'])}")
    print(f"学习单元-知识点映射数量: {len(kt_data['unit_concepts'])}")
    
    # 训练测试集统计
    train_count = sum(len(data[0]) for data in kt_data['train_data'].values())
    test_count = sum(len(data[0]) for data in kt_data['test_data'].values())
    print(f"训练集交互记录: {train_count}")
    print(f"测试集交互记录: {test_count}")
    
    # 检查掩码数据（下一个时间步是题目的数量）
    train_next_question_count = 0
    test_next_question_count = 0
    
    for lrn_data in kt_data['train_data'].values():
        train_next_question_count += sum(lrn_data[5])  # next_is_questions列表
    
    for lrn_data in kt_data['test_data'].values():
        test_next_question_count += sum(lrn_data[5])  # next_is_questions列表
    
    print(f"训练集下一个是题目的数量: {train_next_question_count}")
    print(f"测试集下一个是题目的数量: {test_next_question_count}")
    
    print("=" * 50)
    print("KTDataReader测试完成!")

