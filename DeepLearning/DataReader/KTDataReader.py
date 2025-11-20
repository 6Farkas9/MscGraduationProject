import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from Data.KTRepository import ktrepo
from DataReader.BasicDataReader import basicdr

class KTDataReader():
    """
    知识追踪数据读取器 - 重构掩码逻辑
    掩码说明：
    - prediction_mask: 标记当前步骤的预测是否有意义（即下一步骤是否是题目）
    - 序列: A B C D E F (D、E是题目)
    - 有效预测: 
        C预测D的结果 (prediction_mask=1)
        D预测E的结果 (prediction_mask=1)
    """

    def __init__(self):
        # 基础ID映射
        self.lrn_uid = basicdr.lrn_uid
        self.lrn_num = basicdr.lrn_num
        
        self.qusunt_uid = basicdr.qusunt_uid
        self.qusunt_num = basicdr.qusunt_num
        
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
            if lrn_uid not in self.lrn_uid or unt_uid not in self.qusunt_uid:
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
            if unt_uid in self.qusunt_uid:
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
            if unt_uid in self.qusunt_uid:
                valid_cpts = [cpt for cpt in cpt_list if cpt in self.cpt_uid]
                if valid_cpts:
                    self.unit_concepts[unt_uid] = valid_cpts
    
    def getTrainTestData(self, train_ratio=0.8):
        """
        划分训练集和测试集 - 重构掩码逻辑
        
        数据结构: [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]
        
        掩码逻辑:
        - prediction_mask: 当前步骤的预测是否有意义（下一步骤是否是题目）
        - 序列: A B C D E F (D、E是题目)
        - 有效预测位置: 
            C (预测D) -> prediction_mask=1
            D (预测E) -> prediction_mask=1
        """
        # 按学习者分组
        learner_interactions = defaultdict(list)
        for interaction in self.interactions:
            lrn_uid, unt_uid, add1, add2, create_time, is_question, result = interaction
            learner_interactions[lrn_uid].append((unt_uid, add1, add2, create_time, is_question, result))
        
        # 按每个学习者划分训练测试集
        # 扩展为7个元素的列表，添加prediction_mask和next_results
        train_data = defaultdict(lambda: [[], [], [], [], [], [], []])  # [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]
        test_data = defaultdict(lambda: [[], [], [], [], [], [], []])
        
        for lrn_uid, interactions_list in learner_interactions.items():
            if len(interactions_list) < 2:  # 至少需要2条记录才能划分
                continue  # 跳过只有一条记录的学习者
                
            # 计算划分点
            split_index = int(len(interactions_list) * train_ratio)
            if split_index == 0:
                split_index = 1
            
            # 处理训练集数据
            for i in range(split_index):
                unt_uid, add1, add2, create_time, is_question, result = interactions_list[i]
                
                # 计算预测掩码和下一个结果
                if i < split_index - 1:  # 不是最后一个元素
                    next_unt_uid, next_add1, next_add2, next_create_time, next_is_question, next_result = interactions_list[i+1]
                    # prediction_mask: 下一步骤是否是题目
                    prediction_mask = 1 if next_is_question == 1 else 0
                    # next_result: 下一步骤的结果（如果是题目）
                    next_result_value = next_result if next_is_question == 1 else 0
                else:
                    # 序列末尾，没有下一步骤
                    prediction_mask = 0
                    next_result_value = 0
                
                train_data[lrn_uid][0].append(unt_uid)
                train_data[lrn_uid][1].append(add1)
                train_data[lrn_uid][2].append(add2)
                train_data[lrn_uid][3].append(is_question)
                train_data[lrn_uid][4].append(result)
                train_data[lrn_uid][5].append(prediction_mask)  # 预测掩码
                train_data[lrn_uid][6].append(next_result_value)  # 下一个结果
            
            # 处理测试集数据
            for i in range(split_index, len(interactions_list)):
                unt_uid, add1, add2, create_time, is_question, result = interactions_list[i]
                
                # 计算预测掩码和下一个结果
                if i < len(interactions_list) - 1:  # 不是最后一个元素
                    next_unt_uid, next_add1, next_add2, next_create_time, next_is_question, next_result = interactions_list[i+1]
                    # prediction_mask: 下一步骤是否是题目
                    prediction_mask = 1 if next_is_question == 1 else 0
                    # next_result: 下一步骤的结果（如果是题目）
                    next_result_value = next_result if next_is_question == 1 else 0
                else:
                    # 序列末尾，没有下一步骤
                    prediction_mask = 0
                    next_result_value = 0
                
                test_data[lrn_uid][0].append(unt_uid)
                test_data[lrn_uid][1].append(add1)
                test_data[lrn_uid][2].append(add2)
                test_data[lrn_uid][3].append(is_question)
                test_data[lrn_uid][4].append(result)
                test_data[lrn_uid][5].append(prediction_mask)  # 预测掩码
                test_data[lrn_uid][6].append(next_result_value)  # 下一个结果
        
        self.train_data = train_data
        self.test_data = test_data
        
        # 打印掩码统计信息
        # self._print_mask_statistics(train_data, test_data)
    
    def _print_mask_statistics(self, train_data, test_data):
        """打印掩码统计信息"""
        train_total_positions = 0
        train_valid_masks = 0
        test_total_positions = 0
        test_valid_masks = 0
        
        for lrn_data in train_data.values():
            train_total_positions += len(lrn_data[5])  # prediction_masks
            train_valid_masks += sum(lrn_data[5])
        
        for lrn_data in test_data.values():
            test_total_positions += len(lrn_data[5])  # prediction_masks
            test_valid_masks += sum(lrn_data[5])
        
        print(f"训练集掩码统计: 有效预测位置 {train_valid_masks}/{train_total_positions} ({train_valid_masks/train_total_positions*100:.1f}%)")
        print(f"测试集掩码统计: 有效预测位置 {test_valid_masks}/{test_total_positions} ({test_valid_masks/test_total_positions*100:.1f}%)")
    
    def getCompleteData(self):
        """
        获取完整的KT数据 - 重构掩码逻辑
        返回格式与train_data相同
        """
        # 按学习者分组，保留所有数据
        complete_data = defaultdict(lambda: [[], [], [], [], [], [], []])  # [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]
        
        # 按学习者分组所有交互
        learner_interactions = defaultdict(list)
        for interaction in self.interactions:
            lrn_uid, unt_uid, add1, add2, create_time, is_question, result = interaction
            learner_interactions[lrn_uid].append((unt_uid, add1, add2, create_time, is_question, result))
        
        # 为每个学习者构建完整序列
        for lrn_uid, interactions_list in learner_interactions.items():
            for i in range(len(interactions_list)):
                unt_uid, add1, add2, create_time, is_question, result = interactions_list[i]
                
                # 计算预测掩码和下一个结果
                if i < len(interactions_list) - 1:  # 不是最后一个元素
                    next_unt_uid, next_add1, next_add2, next_create_time, next_is_question, next_result = interactions_list[i+1]
                    # prediction_mask: 下一步骤是否是题目
                    prediction_mask = 1 if next_is_question == 1 else 0
                    # next_result: 下一步骤的结果（如果是题目）
                    next_result_value = next_result if next_is_question == 1 else 0
                else:
                    # 序列末尾，没有下一步骤
                    prediction_mask = 0
                    next_result_value = 0
                
                complete_data[lrn_uid][0].append(unt_uid)
                complete_data[lrn_uid][1].append(add1)
                complete_data[lrn_uid][2].append(add2)
                complete_data[lrn_uid][3].append(is_question)
                complete_data[lrn_uid][4].append(result)
                complete_data[lrn_uid][5].append(prediction_mask)  # 预测掩码
                complete_data[lrn_uid][6].append(next_result_value)  # 下一个结果
        
        # 统计信息
        total_records = sum(len(data[0]) for data in complete_data.values())
        total_learners = len(complete_data)
        total_valid_masks = sum(sum(data[5]) for data in complete_data.values())
        
        print(f"完整KT数据统计: {total_learners}个学习者, {total_records}条记录, {total_valid_masks}个有效预测位置")
        print(f"完整数据掩码覆盖率: {total_valid_masks}/{total_records} ({total_valid_masks/total_records*100:.1f}%)")
        
        return complete_data
    
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
            'qusunt_uid': self.qusunt_uid,
            'cpt_uid': self.cpt_uid,
        }
    
ktdr = KTDataReader()

if __name__ == '__main__':
    # 测试重构后的KTDataReader
    print("测试重构后的KTDataReader:")
    print("=" * 50)
    
    # 加载数据
    kt_data = ktdr.loadDatafromSql()
    
    print("数据统计信息:")
    print(f"学习者数量: {len(kt_data['lrn_uid'])}")
    print(f"学习单元+题目数量: {len(kt_data['qusunt_uid'])}")
    print(f"知识点数量: {len(kt_data['cpt_uid'])}")
    
    # 训练测试集统计
    train_count = sum(len(data[0]) for data in kt_data['train_data'].values())
    test_count = sum(len(data[0]) for data in kt_data['test_data'].values())
    print(f"训练集交互记录: {train_count}")
    print(f"测试集交互记录: {test_count}")
    
    # 检查样例数据
    if kt_data['train_data']:
        sample_lrn = list(kt_data['train_data'].keys())[0]
        sample_data = kt_data['train_data'][sample_lrn]
        print(f"\n样例学习者 {sample_lrn} 的数据结构:")
        print(f"  学习单元序列长度: {len(sample_data[0])}")
        print(f"  预测掩码序列长度: {len(sample_data[5])}")
        print(f"  下一个结果序列长度: {len(sample_data[6])}")
        
        # 显示前几个位置的掩码和结果对应关系
        print(f"\n前5个位置的掩码和结果:")
        for i in range(min(5, len(sample_data[0]))):
            print(f"  位置{i}: unt={sample_data[0][i]}, is_q={sample_data[3][i]}, "
                  f"result={sample_data[4][i]}, mask={sample_data[5][i]}, next_result={sample_data[6][i]}")
    
    # 测试完整数据
    print(f"\n测试完整数据:")
    complete_data = ktdr.getCompleteData()
    
    print("=" * 50)
    print("KTDataReader测试完成!")