# BackEnd/app/test/test_hgc_cd_kt.py
"""
HGC-CD-KT推理引擎综合测试
测试三个推理器的协同工作效果

流程说明：
1. HGC计算学习者嵌入（仅用于新学习者）
2. CD计算认知诊断结果（可能使用KT能力融合）
3. KT计算知识追踪结果（可能使用CD优化能力）

测试路径：
1. 已有单一学习者路径：直接使用CD和KT（不需要HGC）
2. 已有多个学习者路径：批量使用CD和KT
3. 新单一学习者路径：HGC -> CD -> KT
4. 新多个学习者路径：HGC -> CD -> KT（批量）
"""

import os
import sys
import logging
import torch
from typing import List, Dict, Any
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入三个推理引擎
from app.engine.hgc_engine import HGCEngine
from app.engine.cd_engine import CDEngine
from app.engine.kt_engine import KTEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 测试用的学习者UID（真实数据）
TEST_LEARNER_UIDS = [
    "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
    "lrn_004a9c3f5bf246faab3d390ce716e658"
]

class HGC_CD_KT_IntegratedTester:
    """HGC-CD-KT集成测试器"""
    
    def __init__(self, device: str = 'cpu'):
        """
        初始化集成测试器
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.hgc_engine = None
        self.cd_engine = None
        self.kt_engine = None
        
        logger.info(f"集成测试器初始化，设备: {device}")
    
    def initialize_engines(self) -> bool:
        """
        初始化所有推理引擎
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("开始初始化所有推理引擎...")
            
            # 初始化HGC引擎
            logger.info("初始化HGC引擎...")
            self.hgc_engine = HGCEngine(device=self.device)
            if not self.hgc_engine.initialize():
                logger.error("HGC引擎初始化失败")
                return False
            logger.info("✅ HGC引擎初始化成功")
            
            # 初始化CD引擎
            logger.info("初始化CD引擎...")
            self.cd_engine = CDEngine(device=self.device)
            if not self.cd_engine.initialize():
                logger.error("CD引擎初始化失败")
                return False
            logger.info("✅ CD引擎初始化成功")
            
            # 初始化KT引擎
            logger.info("初始化KT引擎...")
            self.kt_engine = KTEngine(device=self.device)
            if not self.kt_engine.initialize():
                logger.error("KT引擎初始化失败")
                return False
            logger.info("✅ KT引擎初始化成功")
            
            logger.info("所有推理引擎初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"初始化推理引擎失败: {e}")
            return False
    
    def test_path1_existing_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """
        测试路径1：已有单一学习者
        流程：CD -> KT（不需要HGC）
        
        Args:
            learner_uid: 学习者UID
            
        Returns:
            Dict: 测试结果
        """
        logger.info(f"=== 测试路径1：已有单一学习者 [{learner_uid}] ===")
        
        try:
            # 步骤1：CD计算知识点掌握程度
            logger.info(f"步骤1：CD计算学习者 {learner_uid} 的知识点掌握程度...")
            cd_result = self.cd_engine.compute_single_learner_concept_mastery(learner_uid)
            
            if not cd_result:
                logger.error(f"CD计算失败: 学习者 {learner_uid} 没有交互记录或计算失败")
                return {
                    'success': False,
                    'error': f"CD计算失败: 学习者 {learner_uid} 没有交互记录或计算失败",
                    'path': 'path1_existing_single'
                }
            
            logger.info(f"✅ CD计算完成: 知识点数={cd_result['concept_count']}")
            
            # 步骤2：KT计算知识追踪结果
            logger.info(f"步骤2：KT计算学习者 {learner_uid} 的知识追踪结果...")
            kt_result = self.kt_engine.compute_single_learner_concept_mastery(learner_uid)
            
            if not kt_result:
                logger.error(f"KT计算失败: 学习者 {learner_uid}")
                return {
                    'success': False,
                    'error': f"KT计算失败: 学习者 {learner_uid}",
                    'path': 'path1_existing_single',
                    'cd_result': cd_result
                }
            
            logger.info(f"✅ KT计算完成: 知识点数={kt_result.get('concept_count', 'unknown')}")
            
            # 分析结果
            concept_mastery = kt_result.get('concept_mastery', {})
            values = list(concept_mastery.values()) if concept_mastery else []
            
            result = {
                'success': True,
                'path': 'path1_existing_single',
                'learner_uid': learner_uid,
                'cd_result_summary': {
                    'concept_count': cd_result['concept_count'],
                    'vector_length': len(cd_result.get('concept_mastery_vector', []))
                },
                'kt_result_summary': {
                    'concept_count': len(concept_mastery),
                    'non_zero_count': sum(1 for x in values if abs(x) > 0.001) if values else 0,
                    'min_value': min(values) if values else 0,
                    'max_value': max(values) if values else 0,
                    'avg_value': sum(values)/len(values) if values else 0
                }
            }
            
            logger.info(f"路径1测试完成: 学习者 {learner_uid}")
            logger.info(f"  CD结果: {cd_result['concept_count']}个知识点")
            logger.info(f"  KT结果: {result['kt_result_summary']['non_zero_count']}个非零值")
            
            return result
            
        except Exception as e:
            logger.error(f"路径1测试失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': 'path1_existing_single'
            }
    
    def test_path2_existing_multiple_learners(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        测试路径2：已有多个学习者
        流程：CD -> KT（批量，不需要HGC）
        
        Args:
            learner_uids: 学习者UID列表
            
        Returns:
            Dict: 测试结果
        """
        logger.info(f"=== 测试路径2：已有多个学习者 [{len(learner_uids)}个] ===")
        
        try:
            # 步骤1：批量CD计算
            logger.info(f"步骤1：批量CD计算 {len(learner_uids)} 个学习者的知识点掌握程度...")
            cd_results = self.cd_engine.compute_multiple_learners_concept_mastery(learner_uids)
            
            if not cd_results['success']:
                logger.error(f"批量CD计算失败: {cd_results.get('error', '未知错误')}")
                return {
                    'success': False,
                    'error': f"批量CD计算失败: {cd_results.get('error', '未知错误')}",
                    'path': 'path2_existing_multiple'
                }
            
            logger.info(f"✅ 批量CD计算完成: 成功 {cd_results['success_count']}/{cd_results['total_count']}")
            
            # 步骤2：批量KT计算
            logger.info(f"步骤2：批量KT计算 {len(learner_uids)} 个学习者的知识追踪结果...")
            kt_results = self.kt_engine.compute_multiple_learners_concept_mastery(learner_uids)
            
            if not kt_results['success']:
                logger.error(f"批量KT计算失败: {kt_results.get('error', '未知错误')}")
                return {
                    'success': False,
                    'error': f"批量KT计算失败: {kt_results.get('error', '未知错误')}",
                    'path': 'path2_existing_multiple',
                    'cd_results': cd_results
                }
            
            logger.info(f"✅ 批量KT计算完成: 成功 {kt_results['success_count']}/{kt_results['total_count']}")
            
            # 分析结果
            kt_summaries = {}
            for result in kt_results['results']:
                learner_id = result['learner_id']
                concept_mastery = result.get('concept_mastery', {})
                values = list(concept_mastery.values()) if concept_mastery else []
                
                kt_summaries[learner_id] = {
                    'concept_count': len(concept_mastery),
                    'non_zero_count': sum(1 for x in values if abs(x) > 0.001) if values else 0,
                    'has_result': 'concept_mastery' in result
                }
            
            result = {
                'success': True,
                'path': 'path2_existing_multiple',
                'learner_count': len(learner_uids),
                'cd_summary': {
                    'total_count': cd_results['total_count'],
                    'valid_count': cd_results['valid_count'],
                    'success_count': cd_results['success_count']
                },
                'kt_summary': {
                    'total_count': kt_results['total_count'],
                    'valid_count': kt_results['valid_count'],
                    'success_count': kt_results['success_count']
                },
                'kt_individual_summaries': kt_summaries
            }
            
            logger.info(f"路径2测试完成:")
            logger.info(f"  CD: {cd_results['success_count']}/{cd_results['total_count']} 成功")
            logger.info(f"  KT: {kt_results['success_count']}/{kt_results['total_count']} 成功")
            
            return result
            
        except Exception as e:
            logger.error(f"路径2测试失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': 'path2_existing_multiple'
            }
    
    def test_path3_new_single_learner(self, learner_uid: str) -> Dict[str, Any]:
        """
        测试路径3：新单一学习者
        流程：HGC -> CD -> KT
        
        Args:
            learner_uid: 学习者UID
            
        Returns:
            Dict: 测试结果
        """
        logger.info(f"=== 测试路径3：新单一学习者 [{learner_uid}] ===")
        
        try:
            # 步骤1：HGC计算学习者嵌入
            logger.info(f"步骤1：HGC计算新学习者 {learner_uid} 的嵌入表达...")
            hgc_result = self.hgc_engine.compute_single_learner_embedding(learner_uid)
            
            if not hgc_result:
                logger.error(f"HGC计算失败: 学习者 {learner_uid}")
                return {
                    'success': False,
                    'error': f"HGC计算失败: 学习者 {learner_uid}",
                    'path': 'path3_new_single'
                }
            
            logger.info(f"✅ HGC计算完成: 嵌入维度={hgc_result['embedding_dim']}")
            
            # 步骤2：CD使用HGC嵌入计算知识点掌握程度
            logger.info(f"步骤2：CD使用HGC嵌入计算学习者 {learner_uid} 的知识点掌握程度...")
            
            # 将嵌入转换为Tensor
            embedding_tensor = torch.tensor(
                hgc_result['embedding'], 
                dtype=torch.float32, 
                device=self.device
            )
            
            # 使用模式2：使用提供的嵌入计算（新学习者模式）
            cd_result = self.cd_engine.compute_concept_mastery_with_embeddings(
                learner_embeddings=[embedding_tensor],
                learner_uids=[learner_uid]
            )
            
            if not cd_result['success']:
                logger.error(f"CD计算失败: {cd_result.get('error', '未知错误')}")
                return {
                    'success': False,
                    'error': f"CD计算失败: {cd_result.get('error', '未知错误')}",
                    'path': 'path3_new_single',
                    'hgc_result': hgc_result
                }
            
            logger.info(f"✅ CD计算完成: 成功 {cd_result['success_count']}/{cd_result['total_count']}")
            
            # 步骤3：KT计算知识追踪结果
            logger.info(f"步骤3：KT计算学习者 {learner_uid} 的知识追踪结果...")
            kt_result = self.kt_engine.compute_concept_mastery_with_embeddings(
                learner_embeddings=[embedding_tensor],
                learner_uids=[learner_uid]
            )
            
            if not kt_result['success']:
                logger.error(f"KT计算失败: {kt_result.get('error', '未知错误')}")
                return {
                    'success': False,
                    'error': f"KT计算失败: {kt_result.get('error', '未知错误')}",
                    'path': 'path3_new_single',
                    'hgc_result': hgc_result,
                    'cd_result': cd_result
                }
            
            logger.info(f"✅ KT计算完成: 成功 {kt_result['success_count']}/{kt_result['total_count']}")
            
            # 分析结果
            kt_results_list = kt_result['results']
            kt_individual_result = None
            if kt_results_list:
                kt_individual_result = kt_results_list[0]
            
            concept_mastery = kt_individual_result.get('concept_mastery', {}) if kt_individual_result else {}
            values = list(concept_mastery.values()) if concept_mastery else []
            
            result = {
                'success': True,
                'path': 'path3_new_single',
                'learner_uid': learner_uid,
                'hgc_result_summary': {
                    'embedding_dim': hgc_result['embedding_dim']
                },
                'cd_result_summary': {
                    'total_count': cd_result['total_count'],
                    'valid_count': cd_result['valid_count'],
                    'success_count': cd_result['success_count']
                },
                'kt_result_summary': {
                    'total_count': kt_result['total_count'],
                    'valid_count': kt_result['valid_count'],
                    'success_count': kt_result['success_count'],
                    'concept_count': len(concept_mastery),
                    'non_zero_count': sum(1 for x in values if abs(x) > 0.001) if values else 0
                },
                'is_new_learner': True
            }
            
            logger.info(f"路径3测试完成: 新学习者 {learner_uid}")
            logger.info(f"  HGC: 嵌入维度={hgc_result['embedding_dim']}")
            logger.info(f"  CD: {cd_result['success_count']}成功")
            logger.info(f"  KT: {result['kt_result_summary']['non_zero_count']}个非零值")
            
            return result
            
        except Exception as e:
            logger.error(f"路径3测试失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': 'path3_new_single'
            }
    
    def test_path4_new_multiple_learners(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        测试路径4：新多个学习者
        流程：HGC -> CD -> KT（批量）
        
        Args:
            learner_uids: 学习者UID列表
            
        Returns:
            Dict: 测试结果
        """
        logger.info(f"=== 测试路径4：新多个学习者 [{len(learner_uids)}个] ===")
        
        try:
            # 步骤1：批量HGC计算学习者嵌入
            logger.info(f"步骤1：批量HGC计算 {len(learner_uids)} 个新学习者的嵌入表达...")
            
            hgc_results = {}
            learner_embeddings = []
            valid_learner_uids = []
            
            for learner_uid in learner_uids:
                hgc_result = self.hgc_engine.compute_single_learner_embedding(learner_uid)
                
                if hgc_result:
                    hgc_results[learner_uid] = hgc_result
                    embedding_tensor = torch.tensor(
                        hgc_result['embedding'], 
                        dtype=torch.float32, 
                        device=self.device
                    )
                    learner_embeddings.append(embedding_tensor)
                    valid_learner_uids.append(learner_uid)
                    logger.info(f"  {learner_uid}: 嵌入维度={hgc_result['embedding_dim']}")
                else:
                    logger.warning(f"  {learner_uid}: HGC计算失败")
            
            if not valid_learner_uids:
                logger.error("所有学习者的HGC计算都失败")
                return {
                    'success': False,
                    'error': "所有学习者的HGC计算都失败",
                    'path': 'path4_new_multiple'
                }
            
            logger.info(f"✅ 批量HGC计算完成: 成功 {len(valid_learner_uids)}/{len(learner_uids)}")
            
            # 步骤2：批量CD使用HGC嵌入计算知识点掌握程度
            logger.info(f"步骤2：批量CD使用HGC嵌入计算 {len(valid_learner_uids)} 个学习者的知识点掌握程度...")
            
            cd_result = self.cd_engine.compute_concept_mastery_with_embeddings(
                learner_embeddings=learner_embeddings,
                learner_uids=valid_learner_uids
            )
            
            if not cd_result['success']:
                logger.error(f"批量CD计算失败: {cd_result.get('error', '未知错误')}")
                return {
                    'success': False,
                    'error': f"批量CD计算失败: {cd_result.get('error', '未知错误')}",
                    'path': 'path4_new_multiple',
                    'hgc_results_summary': {
                        'total': len(learner_uids),
                        'success': len(valid_learner_uids)
                    }
                }
            
            logger.info(f"✅ 批量CD计算完成: 成功 {cd_result['success_count']}/{cd_result['total_count']}")
            
            # 步骤3：批量KT计算知识追踪结果
            logger.info(f"步骤3：批量KT计算 {len(valid_learner_uids)} 个学习者的知识追踪结果...")
            
            kt_result = self.kt_engine.compute_concept_mastery_with_embeddings(
                learner_embeddings=learner_embeddings,
                learner_uids=valid_learner_uids
            )
            
            if not kt_result['success']:
                logger.error(f"批量KT计算失败: {kt_result.get('error', '未知错误')}")
                return {
                    'success': False,
                    'error': f"批量KT计算失败: {kt_result.get('error', '未知错误')}",
                    'path': 'path4_new_multiple',
                    'hgc_results_summary': {
                        'total': len(learner_uids),
                        'success': len(valid_learner_uids)
                    },
                    'cd_result': cd_result
                }
            
            logger.info(f"✅ 批量KT计算完成: 成功 {kt_result['success_count']}/{kt_result['total_count']}")
            
            # 分析结果
            kt_summaries = {}
            for result in kt_result['results']:
                learner_id = result['learner_id']
                concept_mastery = result.get('concept_mastery', {})
                values = list(concept_mastery.values()) if concept_mastery else []
                
                kt_summaries[learner_id] = {
                    'concept_count': len(concept_mastery),
                    'non_zero_count': sum(1 for x in values if abs(x) > 0.001) if values else 0,
                    'has_result': 'concept_mastery' in result
                }
            
            result = {
                'success': True,
                'path': 'path4_new_multiple',
                'total_learner_count': len(learner_uids),
                'valid_learner_count': len(valid_learner_uids),
                'hgc_summary': {
                    'total': len(learner_uids),
                    'success': len(valid_learner_uids)
                },
                'cd_summary': {
                    'total_count': cd_result['total_count'],
                    'valid_count': cd_result['valid_count'],
                    'success_count': cd_result['success_count']
                },
                'kt_summary': {
                    'total_count': kt_result['total_count'],
                    'valid_count': kt_result['valid_count'],
                    'success_count': kt_result['success_count']
                },
                'kt_individual_summaries': kt_summaries,
                'is_new_learners': True
            }
            
            logger.info(f"路径4测试完成:")
            logger.info(f"  HGC: {len(valid_learner_uids)}/{len(learner_uids)} 成功")
            logger.info(f"  CD: {cd_result['success_count']}/{cd_result['total_count']} 成功")
            logger.info(f"  KT: {kt_result['success_count']}/{kt_result['total_count']} 成功")
            
            return result
            
        except Exception as e:
            logger.error(f"路径4测试失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': 'path4_new_multiple'
            }
    
    def run_all_tests(self, learner_uids: List[str] = None) -> Dict[str, Any]:
        """
        运行所有测试路径
        
        Args:
            learner_uids: 测试用的学习者UID列表，如果为None则使用默认测试UID
            
        Returns:
            Dict: 所有测试结果
        """
        if learner_uids is None:
            learner_uids = TEST_LEARNER_UIDS
        
        print("\n" + "=" * 60)
        print("开始运行HGC-CD-KT综合测试")
        print(f"测试学习者: {learner_uids}")
        print("=" * 60)
        
        # 初始化引擎
        if not self.initialize_engines():
            return {
                'success': False,
                'error': '引擎初始化失败',
                'all_tests_completed': False
            }
        
        all_results = {}
        
        try:
            # 测试路径1：已有单一学习者
            print("\n" + "=" * 60)
            print("测试路径1: 已有单一学习者")
            print("=" * 60)
            path1_result = self.test_path1_existing_single_learner(learner_uids[0])
            all_results['path1_existing_single'] = path1_result
            
            # 测试路径2：已有多个学习者
            print("\n" + "=" * 60)
            print("测试路径2: 已有多个学习者")
            print("=" * 60)
            path2_result = self.test_path2_existing_multiple_learners(learner_uids)
            all_results['path2_existing_multiple'] = path2_result
            
            # 测试路径3：新单一学习者
            print("\n" + "=" * 60)
            print("测试路径3: 新单一学习者")
            print("=" * 60)
            path3_result = self.test_path3_new_single_learner(learner_uids[0])
            all_results['path3_new_single'] = path3_result
            
            # 测试路径4：新多个学习者
            print("\n" + "=" * 60)
            print("测试路径4: 新多个学习者")
            print("=" * 60)
            path4_result = self.test_path4_new_multiple_learners(learner_uids)
            all_results['path4_new_multiple'] = path4_result
            
            # 统计结果
            successful_paths = []
            failed_paths = []
            
            for path_name, result in all_results.items():
                if result.get('success', False):
                    successful_paths.append(path_name)
                else:
                    failed_paths.append(path_name)
            
            final_result = {
                'all_tests_completed': True,
                'successful_paths': successful_paths,
                'failed_paths': failed_paths,
                'success_rate': f"{len(successful_paths)}/{len(all_results)}",
                'detailed_results': all_results,
                'timestamp': datetime.now().isoformat()
            }
            
            # 输出总结
            print("\n" + "=" * 60)
            print("HGC-CD-KT综合测试总结")
            print("=" * 60)
            print(f"总测试路径数: {len(all_results)}")
            print(f"成功路径数: {len(successful_paths)}")
            print(f"失败路径数: {len(failed_paths)}")
            
            if failed_paths:
                print(f"失败的路径: {failed_paths}")
            
            for path_name, result in all_results.items():
                status = "✅ 成功" if result.get('success', False) else "❌ 失败"
                print(f"{path_name}: {status}")
                if not result.get('success', False):
                    print(f"  错误: {result.get('error', '未知错误')}")
            
            print("=" * 60)
            
            return final_result
            
        except Exception as e:
            print(f"运行综合测试时发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'all_tests_completed': False,
                'error': str(e),
                'completed_results': all_results
            }


def main():
    """主函数：运行综合测试"""
    print("\n" + "=" * 60)
    print("HGC-CD-KT推理引擎综合测试")
    print("=" * 60)
    
    # 创建测试器
    tester = HGC_CD_KT_IntegratedTester(device='cpu')
    
    # 运行所有测试
    test_results = tester.run_all_tests(TEST_LEARNER_UIDS)
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    if test_results.get('all_tests_completed', False):
        print(f"✅ 综合测试完成: {test_results['success_rate']} 路径成功")
    else:
        print("❌ 综合测试未完成")
        print(f"错误: {test_results.get('error', '未知错误')}")
    
    return test_results


if __name__ == '__main__':
    # 运行综合测试
    results = main()
    
    # 根据测试结果退出
    if results.get('all_tests_completed', False) and len(results.get('successful_paths', [])) == 4:
        print("\n🎉 所有测试路径都成功完成！")
        sys.exit(0)
    else:
        print("\n⚠️  部分测试路径失败，请检查日志")
        sys.exit(1)