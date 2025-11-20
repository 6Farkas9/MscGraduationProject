import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataReader.CDDataReader import cddr
from DataReader.KTDataReader import ktdr
from collections import defaultdict

class CD_KTDataMerger:
    """
    CD和KT数据合并器 - 适配KTDataReader的新数据结构
    KT数据结构: [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]
    """
    
    def __init__(self):
        self.cddr = cddr
        self.ktdr = ktdr
        self._train_test_merged = False
        self._complete_data_merged = False
    
    def merge_and_filter_train_test(self):
        """合并并筛选CD和KT的训练测试数据"""
        if self._train_test_merged:
            print("训练测试数据已经合并过，跳过重复合并")
            return
        
        print("开始合并CD和KT的训练测试数据（在各自筛选后）...")
        
        # 先让DataReader完成各自的初始筛选
        cd_data = self.cddr.loadDatafromSql()
        kt_data = self.ktdr.loadDatafromSql()
        
        # 获取各自筛选后的学习者
        cd_train_learners = set(self.cddr.train_data.keys())
        cd_test_learners = set(self.cddr.test_data.keys())
        cd_learners = cd_train_learners.union(cd_test_learners)
        
        kt_train_learners = set(self.ktdr.train_data.keys())
        kt_test_learners = set(self.ktdr.test_data.keys())
        kt_learners = kt_train_learners.union(kt_test_learners)
        
        common_learners = cd_learners.intersection(kt_learners)
        
        print(f"CD筛选后学习者数量: {len(cd_learners)}")
        print(f"KT筛选后学习者数量: {len(kt_learners)}")
        print(f"共同学习者数量: {len(common_learners)}")
        
        if not common_learners:
            print("警告: 没有共同的学习者!")
            return
        
        # 按照CD的lrn_uid索引顺序排序共同学习者
        common_learners_sorted = sorted(
            common_learners, 
            key=lambda uid: self.cddr.lrn_uid[uid]
        )
        
        # 更新CD训练测试数据（二次筛选）
        self._update_cd_train_test_data(common_learners_sorted)
        
        # 更新KT训练测试数据（二次筛选）
        self._update_kt_train_test_data(common_learners_sorted)
        
        self._train_test_merged = True
        print("✓ CD和KT训练测试数据合并完成")
    
    def merge_and_filter_complete_data(self):
        """合并并筛选CD和KT的完整数据"""
        if self._complete_data_merged:
            print("完整数据已经合并过，跳过重复合并")
            return
        
        print("开始合并CD和KT的完整数据...")
        
        # 获取完整数据
        cd_complete_data = self.cddr.getCompleteData()
        kt_complete_data = self.ktdr.getCompleteData()
        
        # 获取各自完整数据的学习者
        cd_complete_learners = set(cd_complete_data.keys())
        kt_complete_learners = set(kt_complete_data.keys())
        common_learners = cd_complete_learners.intersection(kt_complete_learners)
        
        print(f"CD完整数据学习者数量: {len(cd_complete_learners)}")
        print(f"KT完整数据学习者数量: {len(kt_complete_learners)}")
        print(f"共同学习者数量: {len(common_learners)}")
        
        if not common_learners:
            print("警告: 没有共同的学习者!")
            return
        
        # 按照CD的lrn_uid索引顺序排序共同学习者
        common_learners_sorted = sorted(
            common_learners, 
            key=lambda uid: self.cddr.lrn_uid[uid]
        )
        
        # 更新CD完整数据（二次筛选）
        self._update_cd_complete_data(common_learners_sorted, cd_complete_data)
        
        # 更新KT完整数据（二次筛选）
        self._update_kt_complete_data(common_learners_sorted, kt_complete_data)
        
        self._complete_data_merged = True
        print("✓ CD和KT完整数据合并完成")
    
    def _update_cd_train_test_data(self, common_learners):
        """更新CD训练测试数据，只保留共同学习者"""
        # 更新训练数据
        new_train_data = defaultdict(lambda: [[], []])
        for lrn_uid in common_learners:
            if lrn_uid in self.cddr.train_data:
                new_train_data[lrn_uid] = self.cddr.train_data[lrn_uid]
        
        # 更新测试数据
        new_test_data = defaultdict(lambda: [[], []])
        for lrn_uid in common_learners:
            if lrn_uid in self.cddr.test_data:
                new_test_data[lrn_uid] = self.cddr.test_data[lrn_uid]
        
        # 更新单例对象的数据
        self.cddr.train_data = new_train_data
        self.cddr.test_data = new_test_data
        
        print(f"✓ CD训练测试数据二次筛选完成: {len(new_train_data)} 个训练学习者, {len(new_test_data)} 个测试学习者")
    
    def _update_kt_train_test_data(self, common_learners):
        """更新KT训练测试数据，只保留共同学习者 - 适配新数据结构"""
        # 更新训练数据 - 使用7元素的defaultdict
        new_train_data = defaultdict(lambda: [[], [], [], [], [], [], []])
        for lrn_uid in common_learners:
            if lrn_uid in self.ktdr.train_data:
                # 复制所有7个数据列表
                for i in range(7):
                    new_train_data[lrn_uid][i].extend(self.ktdr.train_data[lrn_uid][i])
        
        # 更新测试数据 - 使用7元素的defaultdict
        new_test_data = defaultdict(lambda: [[], [], [], [], [], [], []])
        for lrn_uid in common_learners:
            if lrn_uid in self.ktdr.test_data:
                # 复制所有7个数据列表
                for i in range(7):
                    new_test_data[lrn_uid][i].extend(self.ktdr.test_data[lrn_uid][i])
        
        # 更新单例对象的数据
        self.ktdr.train_data = new_train_data
        self.ktdr.test_data = new_test_data
        
        # 统计掩码信息
        train_valid_masks = sum(sum(data[5]) for data in new_train_data.values())
        train_total_positions = sum(len(data[5]) for data in new_train_data.values())
        test_valid_masks = sum(sum(data[5]) for data in new_test_data.values())
        test_total_positions = sum(len(data[5]) for data in new_test_data.values())
        
        print(f"✓ KT训练测试数据二次筛选完成:")
        print(f"  训练数据: {len(new_train_data)}个学习者, {train_valid_masks}/{train_total_positions}有效预测位置")
        print(f"  测试数据: {len(new_test_data)}个学习者, {test_valid_masks}/{test_total_positions}有效预测位置")
    
    def _update_cd_complete_data(self, common_learners, original_complete_data):
        """更新CD完整数据，只保留共同学习者"""
        new_complete_data = defaultdict(lambda: [[], []])
        for lrn_uid in common_learners:
            if lrn_uid in original_complete_data:
                new_complete_data[lrn_uid] = original_complete_data[lrn_uid]
        
        # 保存到单例对象
        self.cddr.complete_data = new_complete_data
        
        print(f"✓ CD完整数据二次筛选完成: {len(new_complete_data)} 个学习者")
    
    def _update_kt_complete_data(self, common_learners, original_complete_data):
        """更新KT完整数据，只保留共同学习者 - 适配新数据结构"""
        new_complete_data = defaultdict(lambda: [[], [], [], [], [], [], []])
        for lrn_uid in common_learners:
            if lrn_uid in original_complete_data:
                # 复制所有7个数据列表
                for i in range(7):
                    new_complete_data[lrn_uid][i].extend(original_complete_data[lrn_uid][i])
        
        # 保存到单例对象
        self.ktdr.complete_data = new_complete_data
        
        # 统计掩码信息
        valid_masks = sum(sum(data[5]) for data in new_complete_data.values())
        total_positions = sum(len(data[5]) for data in new_complete_data.values())
        
        print(f"✓ KT完整数据二次筛选完成: {len(new_complete_data)} 个学习者")
        print(f"  有效预测位置: {valid_masks}/{total_positions} ({valid_masks/total_positions*100:.1f}%)")
    
    def get_merged_statistics(self):
        """获取合并后的统计信息"""
        stats = {}
        
        if self._train_test_merged:
            cd_train_count = len(self.cddr.train_data)
            cd_test_count = len(self.cddr.test_data)
            kt_train_count = len(self.ktdr.train_data)
            kt_test_count = len(self.ktdr.test_data)
            
            # CD数据统计
            cd_train_records = sum(len(data[0]) for data in self.cddr.train_data.values())
            cd_test_records = sum(len(data[0]) for data in self.cddr.test_data.values())
            
            # KT数据统计 - 适配新数据结构
            kt_train_records = sum(len(data[0]) for data in self.ktdr.train_data.values())
            kt_test_records = sum(len(data[0]) for data in self.ktdr.test_data.values())
            kt_train_valid_masks = sum(sum(data[5]) for data in self.ktdr.train_data.values())
            kt_test_valid_masks = sum(sum(data[5]) for data in self.ktdr.test_data.values())
            
            stats['train_test'] = {
                'common_learners_count': cd_train_count + cd_test_count,
                'cd_train_learners': cd_train_count,
                'cd_test_learners': cd_test_count,
                'kt_train_learners': kt_train_count,
                'kt_test_learners': kt_test_count,
                'cd_train_records': cd_train_records,
                'cd_test_records': cd_test_records,
                'kt_train_records': kt_train_records,
                'kt_test_records': kt_test_records,
                'kt_train_valid_predictions': kt_train_valid_masks,
                'kt_test_valid_predictions': kt_test_valid_masks,
                'kt_train_prediction_coverage': f"{kt_train_valid_masks/kt_train_records*100:.1f}%",
                'kt_test_prediction_coverage': f"{kt_test_valid_masks/kt_test_records*100:.1f}%",
            }
        
        if self._complete_data_merged:
            cd_complete_count = len(getattr(self.cddr, 'complete_data', {}))
            kt_complete_count = len(getattr(self.ktdr, 'complete_data', {}))
            
            # CD完整数据统计
            cd_complete_records = sum(len(data[0]) for data in getattr(self.cddr, 'complete_data', {}).values())
            
            # KT完整数据统计 - 适配新数据结构
            kt_complete_records = sum(len(data[0]) for data in getattr(self.ktdr, 'complete_data', {}).values())
            kt_complete_valid_masks = sum(sum(data[5]) for data in getattr(self.ktdr, 'complete_data', {}).values())
            
            stats['complete'] = {
                'common_learners_count': cd_complete_count,
                'cd_complete_learners': cd_complete_count,
                'kt_complete_learners': kt_complete_count,
                'cd_complete_records': cd_complete_records,
                'kt_complete_records': kt_complete_records,
                'kt_complete_valid_predictions': kt_complete_valid_masks,
                'kt_complete_prediction_coverage': f"{kt_complete_valid_masks/kt_complete_records*100:.1f}%",
            }
        
        if not stats:
            print("请先调用 merge_and_filter_train_test() 或 merge_and_filter_complete_data() 方法")
            return None
        
        return stats
    
    def verify_consistency(self, data_type='train_test'):
        """验证CD和KT数据的一致性"""
        if data_type == 'train_test' and not self._train_test_merged:
            print("请先调用 merge_and_filter_train_test() 方法")
            return False
        elif data_type == 'complete' and not self._complete_data_merged:
            print("请先调用 merge_and_filter_complete_data() 方法")
            return False
        
        if data_type == 'train_test':
            # 检查训练测试数据学习者集合是否一致
            cd_train_learners = set(self.cddr.train_data.keys())
            cd_test_learners = set(self.cddr.test_data.keys())
            cd_learners = cd_train_learners.union(cd_test_learners)
            
            kt_train_learners = set(self.ktdr.train_data.keys())
            kt_test_learners = set(self.ktdr.test_data.keys())
            kt_learners = kt_train_learners.union(kt_test_learners)
            
            if cd_learners != kt_learners:
                print(f"✗ 训练测试数据学习者集合不一致: CD={len(cd_learners)}, KT={len(kt_learners)}")
                print(f"  CD独有: {cd_learners - kt_learners}")
                print(f"  KT独有: {kt_learners - cd_learners}")
                return False
            
            # 检查索引顺序是否一致
            for lrn_uid in cd_learners:
                if self.cddr.lrn_uid[lrn_uid] != self.ktdr.lrn_uid[lrn_uid]:
                    print(f"✗ 学习者 {lrn_uid} 的索引不一致: CD={self.cddr.lrn_uid[lrn_uid]}, KT={self.ktdr.lrn_uid[lrn_uid]}")
                    return False
            
            print("✓ CD和KT训练测试数据一致性验证通过")
            
        elif data_type == 'complete':
            # 检查完整数据学习者集合是否一致
            cd_complete_data = getattr(self.cddr, 'complete_data', {})
            kt_complete_data = getattr(self.ktdr, 'complete_data', {})
            
            cd_learners = set(cd_complete_data.keys())
            kt_learners = set(kt_complete_data.keys())
            
            if cd_learners != kt_learners:
                print(f"✗ 完整数据学习者集合不一致: CD={len(cd_learners)}, KT={len(kt_learners)}")
                print(f"  CD独有: {cd_learners - kt_learners}")
                print(f"  KT独有: {kt_learners - cd_learners}")
                return False
            
            # 检查索引顺序是否一致
            for lrn_uid in cd_learners:
                if self.cddr.lrn_uid[lrn_uid] != self.ktdr.lrn_uid[lrn_uid]:
                    print(f"✗ 学习者 {lrn_uid} 的索引不一致: CD={self.cddr.lrn_uid[lrn_uid]}, KT={self.ktdr.lrn_uid[lrn_uid]}")
                    return False
            
            print("✓ CD和KT完整数据一致性验证通过")
        
        return True
    
    def get_merged_data(self, data_type='train_test'):
        """获取合并后的数据"""
        if data_type == 'train_test' and self._train_test_merged:
            return {
                'cd_train_data': self.cddr.train_data,
                'cd_test_data': self.cddr.test_data,
                'kt_train_data': self.ktdr.train_data,
                'kt_test_data': self.ktdr.test_data
            }
        elif data_type == 'complete' and self._complete_data_merged:
            return {
                'cd_complete_data': getattr(self.cddr, 'complete_data', {}),
                'kt_complete_data': getattr(self.ktdr, 'complete_data', {})
            }
        else:
            print(f"请先调用对应的合并方法: {data_type}")
            return None

# 创建全局单例
cd_kt_merger = CD_KTDataMerger()

def test_merger():
    """测试数据合并器"""
    print("=== CD_KT数据合并器测试 (适配新数据结构) ===")
    
    # 测试训练测试数据合并
    print("\n1. 训练测试数据合并测试:")
    cd_kt_merger.merge_and_filter_train_test()
    
    # 获取统计信息
    stats = cd_kt_merger.get_merged_statistics()
    if stats and 'train_test' in stats:
        print("\n训练测试数据合并后统计信息:")
        for key, value in stats['train_test'].items():
            print(f"  {key}: {value}")
    
    # 验证一致性
    print("\n训练测试数据一致性验证:")
    cd_kt_merger.verify_consistency('train_test')
    
    # 测试完整数据合并
    print("\n2. 完整数据合并测试:")
    cd_kt_merger.merge_and_filter_complete_data()
    
    # 获取统计信息
    stats = cd_kt_merger.get_merged_statistics()
    if stats and 'complete' in stats:
        print("\n完整数据合并后统计信息:")
        for key, value in stats['complete'].items():
            print(f"  {key}: {value}")
    
    # 验证一致性
    print("\n完整数据一致性验证:")
    cd_kt_merger.verify_consistency('complete')
    
    # 测试获取合并数据
    print("\n3. 获取合并数据测试:")
    train_test_data = cd_kt_merger.get_merged_data('train_test')
    complete_data = cd_kt_merger.get_merged_data('complete')
    
    if train_test_data:
        print(f"训练测试数据获取成功:")
        print(f"  CD训练学习者: {len(train_test_data['cd_train_data'])}")
        print(f"  CD测试学习者: {len(train_test_data['cd_test_data'])}")
        print(f"  KT训练学习者: {len(train_test_data['kt_train_data'])}")
        print(f"  KT测试学习者: {len(train_test_data['kt_test_data'])}")
        
        # 检查KT数据结构
        if train_test_data['kt_train_data']:
            sample_lrn = list(train_test_data['kt_train_data'].keys())[0]
            sample_data = train_test_data['kt_train_data'][sample_lrn]
            print(f"  KT数据结构验证: {len(sample_data)}个数据列表")
            for i, data_list in enumerate(sample_data):
                print(f"    列表{i}: {len(data_list)}个元素")
    
    if complete_data:
        print(f"完整数据获取成功:")
        print(f"  CD完整学习者: {len(complete_data['cd_complete_data'])}")
        print(f"  KT完整学习者: {len(complete_data['kt_complete_data'])}")
    
    print("\n=== 合并器测试完成 ===")

if __name__ == '__main__':
    test_merger()