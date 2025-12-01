# kt_engine.py
import os
import torch
import torch.nn as nn
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

# 导入配置
from app.core.config import path_config

# 导入模型定义
from DeepLearning.Model.KT import KT
from DeepLearning.hyperparams.hyperparameter import hyperparams

# 导入Repository
from app.repositories.kt_repository import kt_repository
from app.repositories.embedding_repository import embedding_repository
from app.repositories.learner_repository import learner_repository
from app.repositories.base_repository import BaseRepository

logger = logging.getLogger(__name__)

class KTEngine:
    """KT模型推理引擎 - 专注于知识追踪计算"""
    
    def __init__(self, device: str = None):
        """
        初始化KT引擎
        
        Args:
            device: 计算设备，默认为超参数配置的设备
        """
        self.device = device or hyperparams.device
        self.model = None
        self.is_initialized = False
        
        # 数据缓存
        self.embedding_cache = {}
        
        # 知识点映射
        self.concept_mapping = None
        self.concept_uid_order = []  # 知识点UID的顺序列表
        self.concept_id_to_uid = {}  # 知识点ID到UID的映射
        
        # 学习单元映射
        self.qusunt_mapping = None
        self.qusunt_uid_order = []
        
        logger.info(f"KT引擎初始化，设备: {self.device}")
    
    def initialize(self) -> bool:
        """
        初始化KT模型
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            if self.is_initialized:
                logger.info("KT引擎已经初始化")
                return True
            
            logger.info("开始初始化KT引擎...")
            
            # 初始化知识点映射
            self._initialize_concept_mapping()
            
            # 初始化学习单元映射
            self._initialize_qusunt_mapping()
            
            # 获取概念映射关系（用于模型）
            concept_mapping = self._get_concept_mapping_for_model()
            
            # 初始化KT模型
            self._initialize_kt_model(concept_mapping)
            
            # 加载模型权重
            self._load_model_weights()
            
            # 验证模型
            self._validate_model()
            
            self.is_initialized = True
            logger.info("KT引擎初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"KT引擎初始化失败: {e}")
            self.is_initialized = False
            return False
    
    def _initialize_concept_mapping(self):
        """初始化知识点映射"""
        logger.info("初始化知识点映射...")
        
        # 获取知识点UID到ID的映射（按id顺序）
        base_repo = BaseRepository()
        self.concept_mapping = base_repo.get_concept_uid_to_id_mapping()
        self.concept_num = len(self.concept_mapping)
        
        # 创建知识点顺序列表（按ID从小到大）
        sorted_items = sorted(self.concept_mapping.items(), key=lambda x: x[1])
        self.concept_uid_order = [uid for uid, _ in sorted_items]
        
        # 创建ID到UID的映射
        self.concept_id_to_uid = {id: uid for uid, id in self.concept_mapping.items()}
        
        logger.info(f"知识点映射初始化完成: {self.concept_num} 个知识点")
    
    def _initialize_qusunt_mapping(self):
        """初始化学习单元映射"""
        logger.info("初始化学习单元映射...")
        
        try:
            # 获取所有学习单元UID（包括题目）
            query = "SELECT uid FROM Units"
            results = BaseRepository().execute_custom_mysql_query(query)
            
            qusunt_uids = [result['uid'] for result in results]
            self.qusunt_num = len(qusunt_uids)
            self.qusunt_uid_order = qusunt_uids
            
            # 创建UID到索引的映射
            self.qusunt_mapping = {uid: idx for idx, uid in enumerate(qusunt_uids)}
            
            # 获取学习单元类型
            self._initialize_unit_types()
            
            logger.info(f"学习单元映射初始化完成: {self.qusunt_num} 个学习单元")
            
        except Exception as e:
            logger.error(f"初始化学习单元映射失败: {e}")
            self.qusunt_mapping = {}
            self.qusunt_uid_order = []
            self.qusunt_num = 0
    
    def _initialize_unit_types(self):
        """初始化学习单元类型"""
        try:
            query = "SELECT uid, type FROM Units"
            results = BaseRepository().execute_custom_mysql_query(query)
            
            self.unit_types = {}
            for result in results:
                self.unit_types[result['uid']] = result.get('type', 'unknown')
            
            # 类型映射为数字
            type_mapping = {
                'video': 0, 'vr': 1, 'ar': 2, 
                'interact': 3, 'cooperate': 4, 'question': 5
            }
            self.unit_type_indices = {}
            for uid, unit_type in self.unit_types.items():
                self.unit_type_indices[uid] = type_mapping.get(unit_type, 5)
            
            logger.info(f"学习单元类型初始化完成: {len(self.unit_types)} 个单元")
            
        except Exception as e:
            logger.error(f"初始化学习单元类型失败: {e}")
            self.unit_types = {}
            self.unit_type_indices = {}
    
    def _get_concept_mapping_for_model(self) -> Dict[int, List[int]]:
        """
        获取模型需要的概念映射
        
        返回格式: {qusunt_idx: [cpt_idx1, cpt_idx2, ...]}
        """
        try:
            # 获取学习单元-知识点映射
            query = "SELECT unt_uid, cpt_uid FROM Unt_Cpt"
            results = BaseRepository().execute_custom_mysql_query(query)
            
            concept_mapping = {}
            
            for result in results:
                unt_uid = result['unt_uid']
                cpt_uid = result['cpt_uid']
                
                # 转换为索引
                if unt_uid in self.qusunt_mapping and cpt_uid in self.concept_mapping:
                    unt_idx = self.qusunt_mapping[unt_uid]
                    cpt_idx = self.concept_mapping[cpt_uid] - 1  # ID从1开始，索引从0开始
                    
                    if unt_idx not in concept_mapping:
                        concept_mapping[unt_idx] = []
                    concept_mapping[unt_idx].append(cpt_idx)
            
            logger.info(f"概念映射构建完成: {len(concept_mapping)} 个学习单元有知识点映射")
            return concept_mapping
            
        except Exception as e:
            logger.error(f"获取概念映射失败: {e}")
            return {}
    
    def _initialize_kt_model(self, concept_mapping: Dict[int, List[int]]):
        """初始化KT模型"""
        logger.info("初始化KT模型...")
        
        # 使用超参数配置
        embedding_dim = hyperparams.hgc_embedding_dim
        concept_num = self.concept_num
        
        self.model = KT(
            embedding_dim=embedding_dim,
            concept_num=concept_num,
            concept_mapping=concept_mapping
        ).to(self.device)
        
        logger.info(f"KT模型初始化完成: embedding_dim={embedding_dim}, concept_num={concept_num}")
    
    def _load_model_weights(self):
        """加载训练好的模型权重"""
        logger.info("加载KT模型权重...")
        
        # 模型权重路径
        save_dir = hyperparams.train_save_dir
        final_dir = os.path.join(save_dir, "final_models")
        kt_path = os.path.join(final_dir, "kt_best_model.pth")
        
        if not os.path.exists(kt_path):
            logger.warning(f"模型权重文件不存在: {kt_path}，使用随机初始化的模型")
            return
        
        try:
            checkpoint = torch.load(kt_path, map_location=self.device, weights_only=False)
            
            # 根据训练脚本的保存格式处理权重
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 训练脚本保存的完整检查点格式
                state_dict = checkpoint['model_state_dict']
                logger.info(f"加载完整检查点，epoch: {checkpoint.get('epoch', 'unknown')}")
            else:
                # 直接保存的模型权重
                state_dict = checkpoint
            
            # 加载权重
            load_result = self.model.load_state_dict(state_dict, strict=True)
            logger.info(f"KT模型权重加载成功")
            
            if load_result.missing_keys:
                logger.warning(f"缺失键: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                logger.warning(f"意外键: {load_result.unexpected_keys}")
                
        except Exception as e:
            logger.error(f"加载模型权重失败: {e}")
            logger.warning("使用随机初始化的模型")
        
        # 记录模型参数信息
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型参数总数: {total_params:,}")
    
    def _validate_model(self):
        """验证模型是否能正常推理"""
        logger.info("验证模型推理能力...")
        
        try:
            # 创建简单的测试数据
            batch_size = 2
            seq_len = 5
            embedding_dim = hyperparams.hgc_embedding_dim
            
            h_lrn_batch = torch.randn(batch_size, embedding_dim, device=self.device)
            h_qusunt_batch = torch.randn(batch_size, seq_len, embedding_dim, device=self.device)
            h_cpt = torch.randn(self.concept_num, embedding_dim, device=self.device)
            
            lrn_indices = torch.arange(batch_size, device=self.device)
            qusunt_seq_indices = torch.randint(0, self.qusunt_num, (batch_size, seq_len), device=self.device)
            add1 = torch.randn(batch_size, seq_len, device=self.device)
            add2 = torch.randn(batch_size, seq_len, device=self.device)
            type_indices = torch.randint(0, 6, (batch_size, seq_len), device=self.device)
            seq_masks = torch.ones(batch_size, seq_len, device=self.device)
            prediction_masks = torch.ones(batch_size, seq_len, device=self.device)
            
            with torch.no_grad():
                self.model.eval()
                
                # 测试标准前向传播
                predictions, ability = self.model(
                    h_lrn_batch=h_lrn_batch,
                    h_qusunt_batch=h_qusunt_batch,
                    h_cpt=h_cpt,
                    lrn_indices=lrn_indices,
                    qusunt_seq_indices=qusunt_seq_indices,
                    add1=add1,
                    add2=add2,
                    type_indices=type_indices,
                    seq_mask=seq_masks,
                    prediction_masks=prediction_masks,
                    use_cd_optimization=False,
                    use_contrastive=False
                )
                
                logger.info(f"标准前向传播测试:")
                logger.info(f"  predictions shape: {predictions.shape}")
                logger.info(f"  ability shape: {ability.shape}")
                
                # 测试get_concept_mastery方法
                concept_mastery = self.model.get_concept_mastery(
                    h_lrn_batch=h_lrn_batch,
                    h_qusunt_batch=h_qusunt_batch,
                    h_cpt=h_cpt,
                    lrn_indices=lrn_indices,
                    qusunt_seq_indices=qusunt_seq_indices,
                    add1=add1,
                    add2=add2,
                    type_indices=type_indices,
                    seq_mask=seq_masks,
                    qus_num=10  # 假设有10个题目
                )
                
                logger.info(f"知识点掌握程度测试: concept_mastery shape={concept_mastery.shape}")
                logger.info(f"concept_mastery 统计: mean={concept_mastery.mean().item():.6f}, " 
                          f"min={concept_mastery.min().item():.6f}, max={concept_mastery.max().item():.6f}")
                
                return True
                
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            return False
    
    def _get_embeddings(self, learner_uid: str = None, learner_uids: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从数据库获取所需的嵌入向量
        
        Args:
            learner_uid: 单个学习者UID
            learner_uids: 多个学习者UID
            
        Returns:
            Tuple: (学习者嵌入, 学习单元嵌入, 知识点嵌入)
        """
        cache_key = "embeddings_all"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        logger.info("从数据库获取嵌入向量...")
        
        try:
            # 获取所有学习单元嵌入
            qusunt_embeddings = embedding_repository.get_embeddings_by_entity_type('unt')
            
            # 构建学习单元嵌入矩阵
            h_qusunt_list = []
            for q_emb in qusunt_embeddings:
                if q_emb and q_emb.get('embedding'):
                    embedding_array = np.array(q_emb['embedding'])
                    h_qusunt_list.append(embedding_array)
                else:
                    # 如果找不到嵌入，使用零向量
                    embedding_dim = hyperparams.hgc_embedding_dim
                    h_qusunt_list.append(np.zeros(embedding_dim))
            
            # 获取所有知识点嵌入
            concept_embeddings = embedding_repository.get_embeddings_by_entity_type('cpt')
            
            # 构建知识点嵌入矩阵 - 按ID顺序
            h_cpt_list = []
            concept_emb_dict = {emb['uid']: np.array(emb['embedding']) for emb in concept_embeddings}
            
            for concept_uid in self.concept_uid_order:
                if concept_uid in concept_emb_dict:
                    h_cpt_list.append(concept_emb_dict[concept_uid])
                else:
                    # 如果找不到对应的知识点嵌入，使用零向量
                    embedding_dim = hyperparams.hgc_embedding_dim
                    h_cpt_list.append(np.zeros(embedding_dim))
                    logger.warning(f"知识点 {concept_uid} 的嵌入向量未找到")
            
            # 转换为Tensor
            h_qusunt = torch.tensor(np.array(h_qusunt_list), dtype=torch.float32, device=self.device)
            h_cpt = torch.tensor(np.array(h_cpt_list), dtype=torch.float32, device=self.device)
            
            # 学习者嵌入会在调用时按需获取
            h_lrn = None  # 将在具体调用时获取
            
            result = (h_lrn, h_qusunt, h_cpt)
            self.embedding_cache[cache_key] = result
            
            logger.info(f"嵌入向量加载完成: 学习单元={len(h_qusunt)}, 知识点={len(h_cpt)}")
            return result
            
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            # 返回默认值
            embedding_dim = hyperparams.hgc_embedding_dim
            h_qusunt = torch.zeros(self.qusunt_num, embedding_dim, device=self.device)
            h_cpt = torch.zeros(self.concept_num, embedding_dim, device=self.device)
            return (None, h_qusunt, h_cpt)
    
    def _simulate_cd_ability(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        模拟CD能力矩阵
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            
        Returns:
            模拟的CD能力矩阵 [batch_size, seq_len, concept_num]
        """
        # 创建随机的模拟能力矩阵（值在0-1之间）
        cd_ability = torch.rand(batch_size, seq_len, self.concept_num, device=self.device)
        
        # 添加一些偏置，使部分知识点掌握程度更高
        for i in range(batch_size):
            # 随机选择一些知识点掌握程度更高
            high_mastery_concepts = torch.randint(0, self.concept_num, (3,))
            cd_ability[i, :, high_mastery_concepts] = 0.8 + 0.2 * torch.rand(seq_len, 3, device=self.device)
        
        logger.debug(f"模拟CD能力矩阵: shape={cd_ability.shape}")
        return cd_ability
    
    def _prepare_kt_inputs(self, learner_uids: List[str], learner_embeddings: List[torch.Tensor] = None, 
                          max_seq_len: int = 50) -> Dict[str, Any]:
        """
        准备KT模型输入数据
        
        Args:
            learner_uids: 学习者UID列表
            learner_embeddings: 学习者嵌入列表（模式2），如果为None则从数据库获取
            max_seq_len: 最大序列长度
            
        Returns:
            KT模型输入数据字典
        """
        try:
            # 获取学习单元和知识点嵌入
            _, h_qusunt, h_cpt = self._get_embeddings()
            
            # 批量获取交互序列数据
            sequences_data = kt_repository.get_learner_interaction_sequences_batch(learner_uids, max_seq_len)
            
            batch_size = len(learner_uids)
            
            # 确定实际的最大序列长度
            actual_max_seq_len = 0
            for data in sequences_data.values():
                actual_max_seq_len = max(actual_max_seq_len, data['seq_len'])
            
            if actual_max_seq_len == 0:
                raise ValueError("没有有效的交互序列数据")
            
            # 使用实际最大长度或限制长度
            effective_max_seq_len = min(actual_max_seq_len, max_seq_len) if max_seq_len else actual_max_seq_len
            
            # 准备批次数据张量
            qusunt_seq_indices = torch.zeros(batch_size, effective_max_seq_len, dtype=torch.long, device=self.device)
            add1_seq = torch.zeros(batch_size, effective_max_seq_len, dtype=torch.float32, device=self.device)
            add2_seq = torch.zeros(batch_size, effective_max_seq_len, dtype=torch.float32, device=self.device)
            type_indices_seq = torch.zeros(batch_size, effective_max_seq_len, dtype=torch.long, device=self.device)
            seq_masks = torch.zeros(batch_size, effective_max_seq_len, dtype=torch.float32, device=self.device)
            prediction_masks_seq = torch.zeros(batch_size, effective_max_seq_len, dtype=torch.float32, device=self.device)
            
            # 记录每个学习者的实际序列长度
            learner_seq_lengths = []
            
            # 获取学习者嵌入
            if learner_embeddings:
                # 模式2：使用传入的嵌入
                h_lrn_batch = torch.stack(learner_embeddings)
            else:
                # 模式1：从数据库获取需要的学习者嵌入
                required_learner_embeddings = embedding_repository.get_embeddings_by_uids(
                    learner_uids, return_format="list"
                )
                h_lrn_batch_list = []
                for learner_uid, l_emb in zip(learner_uids, required_learner_embeddings):
                    if l_emb and l_emb.get('embedding'):
                        h_lrn = torch.tensor(np.array(l_emb['embedding']), dtype=torch.float32, device=self.device)
                    else:
                        # 如果找不到学习者嵌入，使用零向量
                        embedding_dim = hyperparams.hgc_embedding_dim
                        h_lrn = torch.zeros(embedding_dim, device=self.device)
                        logger.warning(f"学习者 {learner_uid} 的嵌入向量未找到")
                    h_lrn_batch_list.append(h_lrn)
                
                h_lrn_batch = torch.stack(h_lrn_batch_list)
            
            # 构建序列数据
            valid_learners = 0
            for i, learner_uid in enumerate(learner_uids):
                seq_data = sequences_data.get(learner_uid)
                if not seq_data:
                    learner_seq_lengths.append(0)
                    continue
                
                sequence = seq_data['sequence']
                seq_len = seq_data['seq_len']
                
                if seq_len == 0:
                    learner_seq_lengths.append(0)
                    continue
                
                # 记录实际长度（考虑截断）
                actual_len = min(seq_len, effective_max_seq_len)
                learner_seq_lengths.append(actual_len)
                
                # 只填充实际有数据的位置
                for j in range(actual_len):
                    # 学习单元索引
                    unt_uid = sequence[0][j]
                    if unt_uid in self.qusunt_mapping:
                        qusunt_seq_indices[i, j] = self.qusunt_mapping[unt_uid]
                    
                    # 数值特征
                    add1_seq[i, j] = sequence[1][j]
                    add2_seq[i, j] = sequence[2][j]
                    
                    # 类型索引
                    type_indices_seq[i, j] = self.unit_type_indices.get(unt_uid, 5)
                    
                    # 预测掩码
                    prediction_masks_seq[i, j] = sequence[5][j]
                    
                    # 序列掩码
                    seq_masks[i, j] = 1.0
                
                valid_learners += 1
            
            if valid_learners == 0:
                raise ValueError("没有有效的学习者序列数据")
            
            # 模拟CD能力矩阵
            simulated_cd_ability = self._simulate_cd_ability(batch_size, effective_max_seq_len)
            
            inputs = {
                'h_lrn_batch': h_lrn_batch,
                'h_qusunt': h_qusunt,
                'h_cpt': h_cpt,
                'lrn_indices': torch.arange(batch_size, device=self.device),
                'qusunt_seq_indices': qusunt_seq_indices,
                'add1': add1_seq,
                'add2': add2_seq,
                'type_indices': type_indices_seq,
                'seq_masks': seq_masks,
                'prediction_masks': prediction_masks_seq,
                'cd_ability': simulated_cd_ability,
                'effective_max_seq_len': effective_max_seq_len,
                'learner_seq_lengths': learner_seq_lengths
            }
            
            logger.info(f"KT输入数据准备完成: 批次大小={batch_size}, 有效学习者={valid_learners}, 序列长度={effective_max_seq_len}")
            return inputs
            
        except Exception as e:
            logger.error(f"准备KT输入数据失败: {e}")
            raise
    
    def _format_kt_results(self, ability_matrix: torch.Tensor, learner_uids: List[str], 
                          learner_seq_lengths: List[int]) -> List[Dict[str, Any]]:
        """
        格式化KT结果，与Inference_HGC_CD_KT格式一致
        
        Args:
            ability_matrix: 能力矩阵 [batch_size, seq_len, concept_num]
            learner_uids: 学习者UID列表
            learner_seq_lengths: 每个学习者的实际序列长度
            
        Returns:
            格式化的KT结果列表
        """
        results = []
        
        for i, learner_uid in enumerate(learner_uids):
            if i >= len(ability_matrix):
                continue
            
            seq_len = learner_seq_lengths[i] if i < len(learner_seq_lengths) else 0
            
            # 获取最后一个有效时间步的能力向量
            if seq_len > 0:
                # 取最后一个有效时间步
                ability_vector = ability_matrix[i, seq_len-1].cpu().numpy()
            else:
                # 如果没有有效数据，使用最后一个时间步
                ability_vector = ability_matrix[i, -1].cpu().numpy()
                logger.warning(f"学习者 {learner_uid} 序列长度为0，使用最后一个时间步")
            
            # 构建知识点掌握程度字典（按知识点UID顺序）
            concept_mastery_dict = {}
            for concept_idx in range(self.concept_num):
                concept_uid = self.concept_uid_order[concept_idx]
                mastery_value = float(ability_vector[concept_idx])
                concept_mastery_dict[concept_uid] = mastery_value
            
            results.append({
                'learner_id': learner_uid,
                'concept_mastery': concept_mastery_dict
            })
        
        return results
    
    def compute_single_learner_concept_mastery(self, learner_uid: str) -> Optional[Dict[str, Any]]:
        """
        计算单个学习者的知识点掌握程度（模式1）
        
        Args:
            learner_uid: 学习者UID
            
        Returns:
            Dict包含知识点掌握程度字典和相关信息
        """
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return None
            
            logger.info(f"计算单个学习者知识点掌握程度: {learner_uid}")
            
            # 验证学习者是否有交互记录
            if not kt_repository.validate_learner_has_interactions(learner_uid):
                logger.error(f"学习者 {learner_uid} 没有交互记录")
                return None
            
            # 准备输入数据
            inputs = self._prepare_kt_inputs([learner_uid])
            
            # 设置CD优化能力
            if inputs['cd_ability'] is not None:
                logger.info("设置模拟CD优化能力进行能力融合")
                self.model.set_cd_optimized_ability(inputs['cd_ability'], qus_num=0)
            
            # 获取学习单元嵌入批次
            batch_size, seq_len = inputs['qusunt_seq_indices'].shape
            embedding_dim = inputs['h_qusunt'].shape[1]
            
            # 构建学习单元嵌入批次
            qusunt_indices_flat = inputs['qusunt_seq_indices'].view(-1)
            h_qusunt_batch = inputs['h_qusunt'][qusunt_indices_flat].view(batch_size, seq_len, embedding_dim)
            
            # 模型推理
            with torch.no_grad():
                self.model.eval()
                
                concept_mastery = self.model.get_concept_mastery(
                    h_lrn_batch=inputs['h_lrn_batch'],
                    h_qusunt_batch=h_qusunt_batch,
                    h_cpt=inputs['h_cpt'],
                    lrn_indices=inputs['lrn_indices'],
                    qusunt_seq_indices=inputs['qusunt_seq_indices'],
                    add1=inputs['add1'],
                    add2=inputs['add2'],
                    type_indices=inputs['type_indices'],
                    seq_mask=inputs['seq_masks'],
                    qus_num=0
                )
            
            logger.debug(f"能力矩阵形状: {concept_mastery.shape}")
            
            # 格式化结果
            formatted_results = self._format_kt_results(
                concept_mastery, 
                [learner_uid], 
                inputs['learner_seq_lengths']
            )
            
            if formatted_results:
                result = formatted_results[0]
                result['concept_count'] = self.concept_num
                result['timestamp'] = datetime.now().isoformat()
                result['sequence_length'] = inputs['learner_seq_lengths'][0]
                
                logger.info(f"单个学习者知识点掌握程度计算完成")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"计算单个学习者知识点掌握程度失败 {learner_uid}: {e}")
            return None
    
    def compute_multiple_learners_concept_mastery(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        计算多个学习者的知识点掌握程度（模式1）
        
        Args:
            learner_uids: 学习者UID列表
            
        Returns:
            Dict包含所有学习者的知识点掌握程度
        """
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return {'success': False, 'error': '引擎初始化失败', 'results': {}}
            
            logger.info(f"计算多个学习者知识点掌握程度: {len(learner_uids)} 个学习者")
            
            if not learner_uids:
                return {'success': True, 'results': [], 'total_count': 0, 'success_count': 0}
            
            # 过滤有交互记录的学习者
            valid_learner_uids = [uid for uid in learner_uids if kt_repository.validate_learner_has_interactions(uid)]
            
            if not valid_learner_uids:
                logger.error("没有找到任何有交互记录的学习者")
                return {
                    'success': False, 
                    'error': '没有找到任何有交互记录的学习者',
                    'total_count': len(learner_uids),
                    'success_count': 0,
                    'results': []
                }
            
            # 准备输入数据
            inputs = self._prepare_kt_inputs(valid_learner_uids)
            
            # 设置CD优化能力
            if inputs['cd_ability'] is not None:
                logger.info("设置模拟CD优化能力进行能力融合")
                self.model.set_cd_optimized_ability(inputs['cd_ability'], qus_num=0)
            
            # 获取学习单元嵌入批次
            batch_size, seq_len = inputs['qusunt_seq_indices'].shape
            embedding_dim = inputs['h_qusunt'].shape[1]
            
            # 构建学习单元嵌入批次
            qusunt_indices_flat = inputs['qusunt_seq_indices'].view(-1)
            h_qusunt_batch = inputs['h_qusunt'][qusunt_indices_flat].view(batch_size, seq_len, embedding_dim)
            
            # 模型推理
            with torch.no_grad():
                self.model.eval()
                
                concept_mastery = self.model.get_concept_mastery(
                    h_lrn_batch=inputs['h_lrn_batch'],
                    h_qusunt_batch=h_qusunt_batch,
                    h_cpt=inputs['h_cpt'],
                    lrn_indices=inputs['lrn_indices'],
                    qusunt_seq_indices=inputs['qusunt_seq_indices'],
                    add1=inputs['add1'],
                    add2=inputs['add2'],
                    type_indices=inputs['type_indices'],
                    seq_mask=inputs['seq_masks'],
                    qus_num=0
                )
            
            logger.debug(f"能力矩阵形状: {concept_mastery.shape}")
            
            # 格式化结果
            formatted_results = self._format_kt_results(
                concept_mastery, 
                valid_learner_uids, 
                inputs['learner_seq_lengths']
            )
            
            logger.info(f"多个学习者知识点掌握程度计算完成: {len(formatted_results)} 成功")
            return {
                'success': True,
                'total_count': len(learner_uids),
                'valid_count': len(valid_learner_uids),
                'success_count': len(formatted_results),
                'results': formatted_results
            }
            
        except Exception as e:
            logger.error(f"计算多个学习者知识点掌握程度失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_count': len(learner_uids),
                'success_count': 0,
                'results': []
            }
    
    def compute_concept_mastery_with_embeddings(self, learner_embeddings: List[torch.Tensor], 
                                               learner_uids: List[str]) -> Dict[str, Any]:
        """
        使用提供的学习者嵌入计算知识点掌握程度（模式2）
        
        Args:
            learner_embeddings: 学习者嵌入列表
            learner_uids: 对应的学习者UID列表（必须有交互数据）
            
        Returns:
            Dict包含知识点掌握程度结果
        """
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return {'success': False, 'error': '引擎初始化失败', 'results': {}}
            
            if len(learner_embeddings) != len(learner_uids):
                return {
                    'success': False,
                    'error': '学习者嵌入数量与UID数量不匹配',
                    'total_count': len(learner_embeddings),
                    'success_count': 0,
                    'results': []
                }
            
            logger.info(f"使用提供的嵌入计算知识点掌握程度: {len(learner_embeddings)} 个学习者")
            
            # 验证学习者是否有交互记录
            valid_learner_uids = [uid for uid in learner_uids if kt_repository.validate_learner_has_interactions(uid)]
            valid_indices = [i for i, uid in enumerate(learner_uids) if uid in valid_learner_uids]
            valid_embeddings = [learner_embeddings[i] for i in valid_indices]
            
            if not valid_learner_uids:
                logger.error("没有找到任何有交互记录的学习者")
                return {
                    'success': False,
                    'error': '没有找到任何有交互记录的学习者',
                    'total_count': len(learner_uids),
                    'success_count': 0,
                    'results': []
                }
            
            # 准备输入数据（模式2）
            inputs = self._prepare_kt_inputs(valid_learner_uids, valid_embeddings)
            
            # 对于新学习者，不设置CD优化能力
            self.model.set_cd_optimized_ability(None, qus_num=0)
            
            # 获取学习单元嵌入批次
            batch_size, seq_len = inputs['qusunt_seq_indices'].shape
            embedding_dim = inputs['h_qusunt'].shape[1]
            
            # 构建学习单元嵌入批次
            qusunt_indices_flat = inputs['qusunt_seq_indices'].view(-1)
            h_qusunt_batch = inputs['h_qusunt'][qusunt_indices_flat].view(batch_size, seq_len, embedding_dim)
            
            # 模型推理
            with torch.no_grad():
                self.model.eval()
                
                concept_mastery = self.model.get_concept_mastery(
                    h_lrn_batch=inputs['h_lrn_batch'],
                    h_qusunt_batch=h_qusunt_batch,
                    h_cpt=inputs['h_cpt'],
                    lrn_indices=inputs['lrn_indices'],
                    qusunt_seq_indices=inputs['qusunt_seq_indices'],
                    add1=inputs['add1'],
                    add2=inputs['add2'],
                    type_indices=inputs['type_indices'],
                    seq_mask=inputs['seq_masks'],
                    qus_num=0
                )
            
            logger.debug(f"能力矩阵形状: {concept_mastery.shape}")
            
            # 格式化结果
            formatted_results = self._format_kt_results(
                concept_mastery, 
                valid_learner_uids, 
                inputs['learner_seq_lengths']
            )
            
            logger.info(f"使用嵌入计算知识点掌握程度完成: {len(formatted_results)} 个学习者")
            return {
                'success': True,
                'total_count': len(learner_uids),
                'valid_count': len(valid_learner_uids),
                'success_count': len(formatted_results),
                'results': formatted_results
            }
            
        except Exception as e:
            logger.error(f"使用嵌入计算知识点掌握程度失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_count': len(learner_uids),
                'success_count': 0,
                'results': []
            }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态信息"""
        return {
            'initialized': self.is_initialized,
            'device': self.device,
            'model_loaded': self.model is not None,
            'concept_count': self.concept_num if hasattr(self, 'concept_num') else 0,
            'qusunt_count': self.qusunt_num if hasattr(self, 'qusunt_num') else 0,
            'embedding_cache_size': len(self.embedding_cache)
        }


# 全局引擎实例
_kt_engine_instance = None

def get_kt_engine() -> KTEngine:
    """获取全局KT引擎实例"""
    global _kt_engine_instance
    if _kt_engine_instance is None:
        _kt_engine_instance = KTEngine()
    return _kt_engine_instance

def compute_single_learner_concept_mastery(learner_uid: str) -> Optional[Dict[str, Any]]:
    """
    计算单个学习者知识点掌握程度的便捷函数
    
    Args:
        learner_uid: 学习者UID
        
    Returns:
        Optional[Dict]: 知识点掌握程度结果
    """
    engine = get_kt_engine()
    return engine.compute_single_learner_concept_mastery(learner_uid)

def compute_multiple_learners_concept_mastery(learner_uids: List[str]) -> Dict[str, Any]:
    """
    计算多个学习者知识点掌握程度的便捷函数
    
    Args:
        learner_uids: 学习者UID列表
        
    Returns:
        Dict: 包含所有学习者知识点掌握程度的结果
    """
    engine = get_kt_engine()
    return engine.compute_multiple_learners_concept_mastery(learner_uids)

def compute_concept_mastery_with_embeddings(learner_embeddings: List[torch.Tensor], 
                                          learner_uids: List[str]) -> Dict[str, Any]:
    """
    使用提供的学习者嵌入计算知识点掌握程度的便捷函数
    
    Args:
        learner_embeddings: 学习者嵌入列表
        learner_uids: 对应的学习者UID列表（必须有交互数据）
        
    Returns:
        Dict: 知识点掌握程度结果
    """
    engine = get_kt_engine()
    return engine.compute_concept_mastery_with_embeddings(learner_embeddings, learner_uids)

def initialize_engine() -> bool:
    """初始化KT引擎的便捷函数"""
    engine = get_kt_engine()
    return engine.initialize()

def get_engine_status() -> Dict[str, Any]:
    """获取引擎状态的便捷函数"""
    engine = get_kt_engine()
    return engine.get_engine_status()


# 测试代码
def test_kt_engine():
    """测试KT引擎"""
    
    print("=== KT引擎测试 ===")
    
    try:
        # 初始化引擎
        engine = KTEngine(device='cpu')
        
        if not engine.initialize():
            print("❌ 引擎初始化失败")
            return False
        
        print("✅ 引擎初始化成功")
        
        # 真实存在的学习者UID（与cd_engine相同）
        test_learner_uids = [
            "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
            "lrn_004a9c3f5bf246faab3d390ce716e658"
        ]
        
        # 测试1：模式1 - 单个已有学习者
        print("\n--- 测试模式1: 单个学习者 ---")
        result = engine.compute_single_learner_concept_mastery(test_learner_uids[0])
        
        if result:
            print(f"✅ 学习者 {result['learner_id']} 计算成功")
            concept_mastery = result['concept_mastery']
            
            # 详细检查输出
            print(f"   知识点数: {result.get('concept_count', len(concept_mastery))}")
            print(f"   序列长度: {result.get('sequence_length', '未知')}")
            print(f"   掌握程度字典长度: {len(concept_mastery)}")
            
            # 检查知识点顺序
            concept_uids = list(concept_mastery.keys())
            print(f"   前5个知识点UID: {concept_uids[:5]}")
            print(f"   对应的掌握程度: {[concept_mastery[uid] for uid in concept_uids[:5]]}")
            
            # 统计信息
            values = list(concept_mastery.values())
            non_zero_count = sum(1 for x in values if abs(x) > 0.001)
            print(f"   非零值(>0.001)数量: {non_zero_count}")
            print(f"   平均值: {sum(values)/len(values):.6f}")
            print(f"   最小值: {min(values):.6f}")
            print(f"   最大值: {max(values):.6f}")
            
            if non_zero_count == 0:
                print("   ⚠️ 输出全零，可能需要检查模型")
        else:
            print("❌ 单个学习者计算失败")
        
        # 测试模式1：多个学习者
        print("\n--- 测试模式1: 多个学习者 ---")
        results = engine.compute_multiple_learners_concept_mastery(test_learner_uids)
        
        if results['success']:
            print(f"✅ 批量计算成功: {results['success_count']}/{results['total_count']}")
            
            for result in results['results']:
                if 'concept_mastery' in result:
                    concept_mastery = result['concept_mastery']
                    values = list(concept_mastery.values())
                    non_zero_count = sum(1 for x in values if abs(x) > 0.001)
                    print(f"   {result['learner_id']}: 知识点数={len(concept_mastery)}, 非零值={non_zero_count}")
                    
                    if non_zero_count == 0:
                        print(f"     警告: 输出全零!")
                        # 显示前几个值
                        concept_uids = list(concept_mastery.keys())
                        print(f"     前3个知识点UID: {concept_uids[:3]}")
                        print(f"     对应的值: {[concept_mastery[uid] for uid in concept_uids[:3]]}")
        else:
            print(f"❌ 批量计算失败: {results.get('error', '未知错误')}")
        
        # 测试模式2：新学习者
        print("\n--- 测试模式2: 新学习者 ---")
        
        embedding_dim = hyperparams.hgc_embedding_dim
        new_embeddings = [
            torch.randn(embedding_dim, device='cpu'),
            torch.randn(embedding_dim, device='cpu')
        ]
        
        new_results = engine.compute_concept_mastery_with_embeddings(new_embeddings, test_learner_uids)
        
        if new_results['success']:
            print(f"✅ 新学习者计算成功: {new_results['success_count']}/{new_results['total_count']}")
            
            for result in new_results['results']:
                if 'concept_mastery' in result:
                    concept_mastery = result['concept_mastery']
                    values = list(concept_mastery.values())
                    non_zero_count = sum(1 for x in values if abs(x) > 0.001)
                    print(f"   {result['learner_id']}: 知识点数={len(concept_mastery)}, 非零值={non_zero_count}")
        else:
            print(f"❌ 新学习者计算失败: {new_results.get('error', '未知错误')}")
        
        # 显示引擎状态
        status = engine.get_engine_status()
        print(f"\n📊 引擎状态: 初始化={status['initialized']}, 知识点数={status['concept_count']}, 学习单元数={status['qusunt_count']}")
        
        print("\n=== KT引擎测试完成 ===")
        return True
        
    except Exception as e:
        print(f"\n💥 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # 设置日志
    import logging
    logging.getLogger().setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console_handler)

    success = test_kt_engine()
    
    if success:
        print("\n🎉 KT引擎测试成功！")
    else:
        print("\n❌ KT引擎测试失败！")