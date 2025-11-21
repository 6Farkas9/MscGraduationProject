# LearnerDataService.py 简化版
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

from Data.LearnerRepository import learner_repo

class LearnerDataService:
    """学习者数据服务层 - 简化版"""
    
    def __init__(self):
        self.repo = learner_repo
    
    def save_kt_inference_results(self, kt_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        保存KT推理结果
        
        Args:
            kt_results: [{
                'learner_id': 'uid',
                'concept_mastery': {cpt_uid1: mastery1, cpt_uid2: mastery2, ...}
            }]
        """
        print("💾 保存KT推理结果...")
        
        # 直接使用repository的批量保存方法
        result = self.repo.save_batch_learner_states(kt_results)
        
        stats = {
            'total_processed': len(kt_results),
            'successfully_saved': result,
            'failed': len(kt_results) - result,
            'success_rate': result / len(kt_results) if kt_results else 0
        }
        
        print(f"✅ KT推理结果保存完成: {stats}")
        return stats

# 创建全局实例
learner_data_service = LearnerDataService()