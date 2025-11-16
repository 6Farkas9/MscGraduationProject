import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from Data.MySQLOperator import mysqlop

class KTRepository():
    """
    知识追踪数据仓库 - 从数据库提取KT训练所需数据
    处理6种交互类型：video/vr/ar/interact/cooperate/question
    """

    def getLrnInteractions(self):
        """
        获取所有学习者的交互序列，按时间排序
        返回: [(lrn_uid, unt_uid, additioninfo1, additioninfo2, create_time)]
        """
        interactions = mysqlop.get_all_records(
            "Interaction",
            "lrn_uid, unt_uid, additioninfo1, additioninfo2, create_time",
            order_by="lrn_uid, create_time"
        )
        return interactions

    def getUntTypes(self):
        """
        获取学习单元类型映射
        返回: {unt_uid: unit_type}
        """
        units = mysqlop.get_all_records("Units", "uid, type")
        return {uid: unit_type for uid, unit_type in units}

    def getQusCpts(self):
        """
        获取题目-知识点关联关系
        返回: {qus_uid: [cpt_uid1, cpt_uid2, ...]}
        """
        qus_cpt_records = mysqlop.get_all_records("Question_Concept", "qus_uid, cpt_uid")
        
        qus_cpts = defaultdict(list)
        for qus_uid, cpt_uid in qus_cpt_records:
            qus_cpts[qus_uid].append(cpt_uid)
        
        return dict(qus_cpts)

    def getUntCpts(self):
        """
        获取学习单元-知识点关联关系
        返回: {unt_uid: [cpt_uid1, cpt_uid2, ...]}
        """
        unt_cpt_records = mysqlop.get_all_records("Unit_Concept", "unt_uid, cpt_uid")
        
        unt_cpts = defaultdict(list)
        for unt_uid, cpt_uid in unt_cpt_records:
            unt_cpts[unt_uid].append(cpt_uid)
        
        return dict(unt_cpts)
    
ktrepo = KTRepository()

if __name__ == '__main__':
    # ktrepo = KTRepository()
    
    print("测试KTRepository功能:")
    print("=" * 50)
    
    # 测试获取交互数据
    interactions = ktrepo.getLrnInteractions()
    print(f"1. 交互数据数量: {len(interactions)}")
    if interactions:
        print(f"   示例数据: {interactions[0]}")
    
    # 测试获取单元类型
    unt_types = ktrepo.getUntTypes()
    print(f"2. 学习单元类型数量: {len(unt_types)}")
    if unt_types:
        sample_uid = list(unt_types.keys())[0]
        print(f"   示例: {sample_uid} -> {unt_types[sample_uid]}")
    
    # 测试获取题目-知识点关系
    qus_cpts = ktrepo.getQusCpts()
    print(f"3. 题目-知识点关系数量: {len(qus_cpts)}")
    if qus_cpts:
        sample_qus = list(qus_cpts.keys())[0]
        print(f"   示例题目 {sample_qus} 关联知识点: {qus_cpts[sample_qus]}")
    
    # 测试获取学习单元-知识点关系
    unt_cpts = ktrepo.getUntCpts()
    print(f"4. 学习单元-知识点关系数量: {len(unt_cpts)}")
    if unt_cpts:
        sample_unt = list(unt_cpts.keys())[0]
        print(f"   示例单元 {sample_unt} 关联知识点: {unt_cpts[sample_unt]}")
    
    print("=" * 50)
    print("KTRepository测试完成!")

