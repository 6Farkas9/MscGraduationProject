# cd_engine.py
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
from DeepLearning.Model.CD import CD
from DeepLearning.hyperparams.hyperparameter import hyperparams

# 导入Repository
from app.repositories.cd_repository import cd_repository
from app.repositories.embedding_repository import embedding_repository
from app.repositories.learner_repository import learner_repository

logger = logging.getLogger(__name__)

class CDEngine:
    """CD模型推理引擎 - 专注于认知诊断计算"""
    
    def __init__(self, device: str = None):
        """
        初始化CD引擎
        
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
        
        logger.info(f"CD引擎初始化，设备: {self.device}")
    
    def initialize(self) -> bool:
        """
        初始化CD模型
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            if self.is_initialized:
                logger.info("CD引擎已经初始化")
                return True
            
            logger.info("开始初始化CD引擎...")
            
            # 初始化知识点映射
            self._initialize_concept_mapping()
            
            # 初始化CD模型
            self._initialize_cd_model()
            
            # 加载模型权重
            self._load_model_weights()
            
            # 验证模型
            self._validate_model()
            
            self.is_initialized = True
            logger.info("CD引擎初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"CD引擎初始化失败: {e}")
            self.is_initialized = False
            return False
    
    def _initialize_concept_mapping(self):
        """初始化知识点映射"""
        logger.info("初始化知识点映射...")
        
        # 获取知识点UID到ID的映射（按id顺序）
        self.concept_mapping = cd_repository.get_concept_uid_to_id_mapping()
        self.concept_num = len(self.concept_mapping)
        
        # 创建ID到UID的反向映射
        self.id_to_concept = {id: uid for uid, id in self.concept_mapping.items()}
        
        logger.info(f"知识点映射初始化完成: {self.concept_num} 个知识点")
    
    def _initialize_cd_model(self):
        """初始化CD模型"""
        logger.info("初始化CD模型...")
        
        # 使用超参数配置
        embedding_dim = hyperparams.hgc_embedding_dim
        concept_num = self.concept_num
        
        self.model = CD(
            embedding_dim=embedding_dim,
            concept_num=concept_num
        ).to(self.device)
        
        logger.info(f"CD模型初始化完成: embedding_dim={embedding_dim}, concept_num={concept_num}")
    
    def _load_model_weights(self):
        """加载训练好的模型权重"""
        logger.info("加载CD模型权重...")
        
        # 模型权重路径
        save_dir = hyperparams.train_save_dir
        final_dir = os.path.join(save_dir, "final_models")
        cd_path = os.path.join(final_dir, "cd_best_model.pth")
        
        if not os.path.exists(cd_path):
            logger.warning(f"模型权重文件不存在: {cd_path}，使用随机初始化的模型")
            return
        
        try:
            checkpoint = torch.load(cd_path, map_location=self.device, weights_only=False)
            
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
            logger.info(f"CD模型权重加载成功")
            
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
            h_qus = torch.randn(10, embedding_dim, device=self.device)
            h_cpt = torch.randn(self.concept_num, embedding_dim, device=self.device)
            qus_seq_indices = torch.randint(0, 10, (batch_size, seq_len), device=self.device)
            qus_seq_masks = torch.ones(batch_size, seq_len, device=self.device)
            
            with torch.no_grad():
                self.model.eval()
                
                # 测试标准前向传播
                predictions = self.model(
                    h_lrn_batch=h_lrn_batch,
                    h_qus=h_qus,
                    h_cpt=h_cpt,
                    qus_seq_indices=qus_seq_indices,
                    qus_seq_masks=qus_seq_masks,
                    return_ability=False,
                    use_kt_optimization=False
                )
                
                logger.info(f"标准前向传播测试: predictions shape={predictions.shape}")
                logger.info(f"predictions 统计: mean={predictions.mean().item():.6f}, "
                          f"min={predictions.min().item():.6f}, max={predictions.max().item():.6f}")
                
                # 测试get_ability_matrix方法
                ability_matrix = self.model.get_ability_matrix(
                    h_lrn_batch=h_lrn_batch,
                    h_qus=h_qus,
                    h_cpt=h_cpt,
                    unt_seq_indices=qus_seq_indices,
                    seq_masks=qus_seq_masks,
                    unt_num=0
                )
                
                logger.info(f"能力矩阵测试: ability_matrix shape={ability_matrix.shape}")
                logger.info(f"ability_matrix 统计: mean={ability_matrix.mean().item():.6f}, "
                          f"min={ability_matrix.min().item():.6f}, max={ability_matrix.max().item():.6f}")
                
                if ability_matrix.mean().item() == 0 and ability_matrix.std().item() == 0:
                    logger.warning("能力矩阵输出全零，可能有问题")
                
                return True
                
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            return False
    
    def _get_embeddings(self, required_question_uids: List[str]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        从数据库获取所需的嵌入向量
        
        Args:
            required_question_uids: 需要的题目UID列表
            
        Returns:
            Tuple: (题目嵌入, 知识点嵌入, 题目UID到索引的映射)
        """
        cache_key = f"embeddings_q{len(required_question_uids)}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        logger.info(f"从数据库获取嵌入向量: 题目={len(required_question_uids)}")
        
        try:
            # 获取需要的题目嵌入
            question_embeddings = embedding_repository.get_embeddings_by_uids(
                required_question_uids, return_format="list"
            )
            
            # 获取所有知识点嵌入
            concept_embeddings = embedding_repository.get_embeddings_by_entity_type('cpt')
            
            # 构建题目嵌入矩阵和映射
            h_qus_list = []
            qus_uid_to_idx = {}
            
            for idx, q_emb in enumerate(question_embeddings):
                if q_emb and q_emb.get('embedding'):
                    embedding_array = np.array(q_emb['embedding'])
                    h_qus_list.append(embedding_array)
                    qus_uid_to_idx[q_emb['uid']] = idx
                else:
                    logger.warning(f"题目 {q_emb.get('uid', 'unknown')} 的嵌入向量未找到")
            
            # 如果没有找到任何题目嵌入，创建空矩阵
            if not h_qus_list:
                embedding_dim = hyperparams.hgc_embedding_dim
                h_qus_list = [np.zeros(embedding_dim)]
                logger.warning("没有找到任何题目嵌入，使用零向量")
            
            # 构建知识点嵌入矩阵 - 按ID顺序
            h_cpt_list = []
            concept_emb_dict = {emb['uid']: np.array(emb['embedding']) for emb in concept_embeddings}
            
            for concept_id in range(1, self.concept_num + 1):
                concept_uid = self.id_to_concept.get(concept_id)
                if concept_uid and concept_uid in concept_emb_dict:
                    h_cpt_list.append(concept_emb_dict[concept_uid])
                else:
                    # 如果找不到对应的知识点嵌入，使用零向量
                    embedding_dim = hyperparams.hgc_embedding_dim
                    h_cpt_list.append(np.zeros(embedding_dim))
                    logger.warning(f"知识点 {concept_uid} 的嵌入向量未找到")
            
            # 转换为Tensor
            h_qus = torch.tensor(np.array(h_qus_list), dtype=torch.float32, device=self.device)
            h_cpt = torch.tensor(np.array(h_cpt_list), dtype=torch.float32, device=self.device)
            
            result = (h_qus, h_cpt, qus_uid_to_idx)
            self.embedding_cache[cache_key] = result
            
            logger.info(f"嵌入向量加载完成: 题目={len(h_qus)}, 知识点={len(h_cpt)}")

            logger.debug(f"=== 嵌入向量检查 ===")
            logger.debug(f"从数据库获取的题目嵌入数量: {len(question_embeddings)}")
            for i, q_emb in enumerate(question_embeddings[:3]):  # 只检查前3个
                if q_emb and q_emb.get('embedding'):
                    emb_array = np.array(q_emb['embedding'])
                    logger.debug(f"题目 {q_emb['uid']}: shape={emb_array.shape}, "
                                f"mean={emb_array.mean():.6f}, std={emb_array.std():.6f}")

            logger.debug(f"从数据库获取的知识点嵌入数量: {len(concept_embeddings)}")
            # 检查知识点嵌入
            for i in range(min(3, len(h_cpt_list))):
                logger.debug(f"知识点 {i+1}: shape={h_cpt_list[i].shape}, "
                            f"mean={h_cpt_list[i].mean():.6f}, std={h_cpt_list[i].std():.6f}")

            return result
            
        except Exception as e:
            logger.error(f"获取嵌入向量失败: {e}")
            # 返回默认值避免解包错误
            embedding_dim = hyperparams.hgc_embedding_dim
            h_qus = torch.zeros(1, embedding_dim, device=self.device)
            h_cpt = torch.zeros(self.concept_num, embedding_dim, device=self.device)
            qus_uid_to_idx = {}
            return (h_qus, h_cpt, qus_uid_to_idx)
    
    def _get_kt_ability_matrix(self, learner_uids: List[str]) -> Optional[torch.Tensor]:
        """
        获取学习者的KT能力矩阵
        
        Args:
            learner_uids: 学习者UID列表
            
        Returns:
            KT能力矩阵 [batch_size, 1, concept_num]，如果没有KT结果则返回None
        """
        try:
            # 获取KT结果
            kt_results = learner_repository.get_kt_results_by_uids(learner_uids, return_format="list")
            
            ability_matrix = []
            has_kt_data = False
            
            for learner_uid, kt_data in zip(learner_uids, kt_results):
                if kt_data and kt_data.get('KT'):
                    # 有KT结果，按知识点ID顺序构建能力向量
                    ability_vector = np.zeros(self.concept_num)
                    kt_dict = kt_data['KT']
                    
                    for concept_id in range(1, self.concept_num + 1):
                        concept_uid = self.id_to_concept.get(concept_id)
                        if concept_uid and concept_uid in kt_dict:
                            ability_vector[concept_id - 1] = kt_dict[concept_uid]
                    
                    # 添加时间维度 [1, concept_num]
                    ability_matrix_3d = ability_vector.reshape(1, 1, -1)
                    ability_matrix.append(ability_matrix_3d)
                    has_kt_data = True
                    logger.debug(f"学习者 {learner_uid} 有KT结果，构建3D能力矩阵")
                else:
                    # 没有KT结果，使用零向量 [1, 1, concept_num]
                    ability_matrix_3d = np.zeros((1, 1, self.concept_num))
                    ability_matrix.append(ability_matrix_3d)
            
            if not has_kt_data:
                logger.info("所有学习者都没有KT结果，跳过能力融合")
                return None
                
            # 拼接成 [batch_size, 1, concept_num]
            ability_tensor = torch.tensor(np.concatenate(ability_matrix, axis=0), dtype=torch.float32, device=self.device)
            logger.info(f"KT能力矩阵构建完成: {ability_tensor.shape}")
            return ability_tensor
            
        except Exception as e:
            logger.error(f"获取KT能力矩阵失败: {e}")
            return None
    
    def _prepare_cd_inputs(self, learner_uids: List[str], learner_embeddings: List[torch.Tensor] = None):
        """
        准备CD模型输入数据
        
        Args:
            learner_uids: 学习者UID列表
            learner_embeddings: 学习者嵌入列表（模式2），如果为None则从数据库获取
            
        Returns:
            CD模型输入数据字典
        """
        try:
            # 对于单个学习者，不限制序列长度；对于多个学习者，取实际最大长度
            max_seq_len = None if len(learner_uids) == 1 else 50
            
            # 构建题目序列并提取涉及的题目
            sequence_data = cd_repository.build_question_sequences(learner_uids, max_seq_len)
            sequences = sequence_data['sequences']
            actual_max_seq_len = sequence_data['actual_max_seq_len']
            
            # 提取所有涉及的题目UID
            required_question_uids = sequence_data['all_question_uids']
            
            if not required_question_uids:
                logger.error("没有找到任何题目交互记录")
                raise ValueError("没有找到任何题目交互记录")
            
            # 获取嵌入向量
            h_qus, h_cpt, qus_uid_to_idx = self._get_embeddings(required_question_uids)
            
            # 准备批次数据 - 使用实际最大长度
            batch_size = len(learner_uids)
            effective_max_seq_len = actual_max_seq_len
            
            # 使用实际长度创建张量，不填充空白
            qus_seq_indices = torch.zeros(batch_size, effective_max_seq_len, dtype=torch.long, device=self.device)
            qus_seq_masks = torch.zeros(batch_size, effective_max_seq_len, dtype=torch.float32, device=self.device)
            
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
            
            # 构建题目序列索引（保持时序）
            valid_learners = 0
            for i, learner_uid in enumerate(learner_uids):
                seq_data = sequences.get(learner_uid)
                if not seq_data:
                    continue
                    
                qus_seq = seq_data['qus_seq']
                seq_len = len(qus_seq)
                
                if seq_len == 0:
                    continue
                    
                # 只填充实际有数据的位置
                for j in range(seq_len):
                    qus_uid = qus_seq[j]
                    if qus_uid in qus_uid_to_idx:
                        qus_seq_indices[i, j] = qus_uid_to_idx[qus_uid]
                        qus_seq_masks[i, j] = 1.0
                
                valid_learners += 1
            
            if valid_learners == 0:
                logger.error("没有有效的学习者序列数据")
                raise ValueError("没有有效的学习者序列数据")
            
            # 获取KT能力矩阵
            kt_ability = self._get_kt_ability_matrix(learner_uids)
            
            inputs = {
                'h_lrn_batch': h_lrn_batch,
                'h_qus': h_qus,
                'h_cpt': h_cpt,
                'qus_seq_indices': qus_seq_indices,
                'qus_seq_masks': qus_seq_masks,
                'kt_ability': kt_ability,
                'actual_seq_len': effective_max_seq_len
            }
            
            logger.info(f"CD输入数据准备完成: 批次大小={batch_size}, 有效学习者={valid_learners}, 实际序列长度={effective_max_seq_len}")
            return inputs
            
        except Exception as e:
            logger.error(f"准备CD输入数据失败: {e}")
            raise
    
    def compute_single_learner_concept_mastery(self, learner_uid: str) -> Optional[Dict[str, Any]]:
        """
        计算单个学习者的知识点掌握程度
        
        Args:
            learner_uid: 学习者UID
            
        Returns:
            Dict包含知识点掌握程度向量和相关信息
        """
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return None
            
            logger.info(f"计算单个学习者知识点掌握程度: {learner_uid}")
            
            # 验证学习者是否有交互记录
            if not cd_repository.validate_learner_has_interactions(learner_uid):
                logger.error(f"学习者 {learner_uid} 没有交互记录")
                return None
            
            # 准备输入数据
            inputs = self._prepare_cd_inputs([learner_uid])
            
            # 设置KT优化能力
            if inputs['kt_ability'] is not None:
                logger.info("设置KT优化能力进行能力融合")
                self.model.set_kt_optimized_ability(inputs['kt_ability'], unt_num=0)
            
            # 模型推理
            with torch.no_grad():
                self.model.eval()

                # 调试：检查模型内部状态
                logger.debug("=== 模型推理调试 ===")

                ability_matrix = self.model.get_ability_matrix(
                    h_lrn_batch=inputs['h_lrn_batch'],
                    h_qus=inputs['h_qus'],
                    h_cpt=inputs['h_cpt'],
                    unt_seq_indices=inputs['qus_seq_indices'],
                    seq_masks=inputs['qus_seq_masks'],
                    unt_num=0
                )

                logger.debug(f"ability_matrix shape: {ability_matrix.shape}")
                logger.debug(f"ability_matrix 统计: mean={ability_matrix.mean().item():.6f}, "
                            f"std={ability_matrix.std().item():.6f}, "
                            f"min={ability_matrix.min().item():.6f}, max={ability_matrix.max().item():.6f}, "
                            f"非零值数量: {(ability_matrix.abs() > 0.001).sum().item()}")
            
            # 检查输出
            logger.debug(f"能力矩阵形状: {ability_matrix.shape}")
            logger.debug(f"能力矩阵统计: mean={ability_matrix.mean().item():.6f}, "
                        f"min={ability_matrix.min().item():.6f}, max={ability_matrix.max().item():.6f}")
            
            # 修改结果提取部分
            # 提取能力向量并转换为列表
            # 注意：现在只取有效时间步的最后一步
            actual_seq_len = inputs.get('actual_seq_len', 1)
            if actual_seq_len > 0:
                # 找到最后一个有效的时间步
                last_valid_step = -1
                for step in range(actual_seq_len - 1, -1, -1):
                    if inputs['qus_seq_masks'][0, step] > 0.5:
                        last_valid_step = step
                        break
                
                if last_valid_step >= 0:
                    ability_vector = ability_matrix[0, last_valid_step].cpu().numpy().tolist()
                else:
                    ability_vector = ability_matrix[0, -1].cpu().numpy().tolist()
            else:
                ability_vector = ability_matrix[0, -1].cpu().numpy().tolist()
            
            # 检查输出是否全零
            non_zero_count = sum(1 for x in ability_vector if abs(x) > 0.001)
            if non_zero_count == 0:
                logger.warning("能力向量输出全零或接近零")
            
            result = {
                'learner_uid': learner_uid,
                'concept_mastery_vector': ability_vector,
                'concept_count': self.concept_num,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"单个学习者知识点掌握程度计算完成")
            return result
            
        except Exception as e:
            logger.error(f"计算单个学习者知识点掌握程度失败 {learner_uid}: {e}")
            return None
    
    def compute_multiple_learners_concept_mastery(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        计算多个学习者的知识点掌握程度
        
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
                return {'success': True, 'results': {}, 'total_count': 0, 'success_count': 0}
            
            # 过滤有交互记录的学习者
            valid_learner_uids = [uid for uid in learner_uids if cd_repository.validate_learner_has_interactions(uid)]
            
            if not valid_learner_uids:
                logger.error("没有找到任何有交互记录的学习者")
                return {
                    'success': False, 
                    'error': '没有找到任何有交互记录的学习者',
                    'total_count': len(learner_uids),
                    'success_count': 0,
                    'results': {}
                }
            
            # 准备输入数据
            inputs = self._prepare_cd_inputs(valid_learner_uids)
            
            # 设置KT优化能力
            if inputs['kt_ability'] is not None:
                logger.info("设置KT优化能力进行能力融合")
                self.model.set_kt_optimized_ability(inputs['kt_ability'], unt_num=0)
            
            # 模型推理
            with torch.no_grad():
                self.model.eval()
                ability_matrix = self.model.get_ability_matrix(
                    h_lrn_batch=inputs['h_lrn_batch'],
                    h_qus=inputs['h_qus'],
                    h_cpt=inputs['h_cpt'],
                    unt_seq_indices=inputs['qus_seq_indices'],
                    seq_masks=inputs['qus_seq_masks'],
                    unt_num=0
                )
            
            logger.debug(f"能力矩阵形状: {ability_matrix.shape}")
            
            # 处理结果
            results = {}
            success_count = 0
            
            for i, learner_uid in enumerate(valid_learner_uids):
                if i < len(ability_matrix):
                    # 取每个学习者最后一个时间步的能力向量
                    ability_vector = ability_matrix[i, -1].cpu().numpy().tolist()
                    
                    results[learner_uid] = {
                        'concept_mastery_vector': ability_vector,
                        'concept_count': self.concept_num,
                        'timestamp': datetime.now().isoformat()
                    }
                    success_count += 1
                else:
                    logger.warning(f"学习者 {learner_uid} 的能力计算失败: 索引超出范围")
                    results[learner_uid] = {
                        'error': '索引超出范围',
                        'timestamp': datetime.now().isoformat()
                    }
            
            logger.info(f"多个学习者知识点掌握程度计算完成: {success_count} 成功")
            return {
                'success': True,
                'total_count': len(learner_uids),
                'valid_count': len(valid_learner_uids),
                'success_count': success_count,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"计算多个学习者知识点掌握程度失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_count': len(learner_uids),
                'success_count': 0,
                'results': {}
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
                    'results': {}
                }
            
            logger.info(f"使用提供的嵌入计算知识点掌握程度: {len(learner_embeddings)} 个学习者")
            
            # 验证学习者是否有交互记录
            valid_learner_uids = [uid for uid in learner_uids if cd_repository.validate_learner_has_interactions(uid)]
            valid_indices = [i for i, uid in enumerate(learner_uids) if uid in valid_learner_uids]
            valid_embeddings = [learner_embeddings[i] for i in valid_indices]
            
            if not valid_learner_uids:
                logger.error("没有找到任何有交互记录的学习者")
                return {
                    'success': False,
                    'error': '没有找到任何有交互记录的学习者',
                    'total_count': len(learner_uids),
                    'success_count': 0,
                    'results': {}
                }
            
            # 准备输入数据（模式2）
            inputs = self._prepare_cd_inputs(valid_learner_uids, valid_embeddings)
            
            # 对于新学习者，不设置KT优化能力（跳过能力融合）
            self.model.set_kt_optimized_ability(None, unt_num=0)
            
            # 模型推理
            with torch.no_grad():
                self.model.eval()
                ability_matrix = self.model.get_ability_matrix(
                    h_lrn_batch=inputs['h_lrn_batch'],
                    h_qus=inputs['h_qus'],
                    h_cpt=inputs['h_cpt'],
                    unt_seq_indices=inputs['qus_seq_indices'],
                    seq_masks=inputs['qus_seq_masks'],
                    unt_num=0
                )
            
            logger.debug(f"能力矩阵形状: {ability_matrix.shape}")
            
            # 处理结果
            results = {}
            success_count = 0
            
            for i, learner_uid in enumerate(valid_learner_uids):
                if i < len(ability_matrix):
                    ability_vector = ability_matrix[i, -1].cpu().numpy().tolist()
                    
                    results[learner_uid] = {
                        'concept_mastery_vector': ability_vector,
                        'concept_count': self.concept_num,
                        'timestamp': datetime.now().isoformat(),
                        'is_new_learner': True
                    }
                    success_count += 1
            
            logger.info(f"使用嵌入计算知识点掌握程度完成: {len(results)} 个学习者")
            return {
                'success': True,
                'total_count': len(learner_uids),
                'valid_count': len(valid_learner_uids),
                'success_count': success_count,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"使用嵌入计算知识点掌握程度失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_count': len(learner_uids),
                'success_count': 0,
                'results': {}
            }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态信息"""
        return {
            'initialized': self.is_initialized,
            'device': self.device,
            'model_loaded': self.model is not None,
            'concept_count': self.concept_num if self.concept_mapping else 0,
            'embedding_cache_size': len(self.embedding_cache)
        }


# 全局引擎实例
_cd_engine_instance = None

def get_cd_engine() -> CDEngine:
    """获取全局CD引擎实例"""
    global _cd_engine_instance
    if _cd_engine_instance is None:
        _cd_engine_instance = CDEngine()
    return _cd_engine_instance

def compute_single_learner_concept_mastery(learner_uid: str) -> Optional[Dict[str, Any]]:
    """
    计算单个学习者知识点掌握程度的便捷函数
    
    Args:
        learner_uid: 学习者UID
        
    Returns:
        Optional[Dict]: 知识点掌握程度结果
    """
    engine = get_cd_engine()
    return engine.compute_single_learner_concept_mastery(learner_uid)

def compute_multiple_learners_concept_mastery(learner_uids: List[str]) -> Dict[str, Any]:
    """
    计算多个学习者知识点掌握程度的便捷函数
    
    Args:
        learner_uids: 学习者UID列表
        
    Returns:
        Dict: 包含所有学习者知识点掌握程度的结果
    """
    engine = get_cd_engine()
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
    engine = get_cd_engine()
    return engine.compute_concept_mastery_with_embeddings(learner_embeddings, learner_uids)

def initialize_engine() -> bool:
    """初始化CD引擎的便捷函数"""
    engine = get_cd_engine()
    return engine.initialize()

def get_engine_status() -> Dict[str, Any]:
    """获取引擎状态的便捷函数"""
    engine = get_cd_engine()
    return engine.get_engine_status()


# 测试代码
def test_cd_engine():
    """测试CD引擎"""
    
    print("=== CD引擎测试 ===")
    
    try:
        # 初始化引擎
        engine = CDEngine(device='cpu')
        
        if not engine.initialize():
            print("❌ 引擎初始化失败")
            return False
        
        print("✅ 引擎初始化成功")
        
        # 真实存在的学习者UID
        test_learner_uids = [
            "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
            "lrn_004a9c3f5bf246faab3d390ce716e658"
        ]
        
        # 测试1：模式1 - 单个已有学习者
        print("\n--- 测试模式1: 单个学习者 ---")
        result = engine.compute_single_learner_concept_mastery(test_learner_uids[0])
        
        if result:
            print(f"✅ 学习者 {result['learner_uid']} 计算成功")
            ability_vector = result['concept_mastery_vector']
            
            # 详细检查输出
            non_zero_count = sum(1 for x in ability_vector if abs(x) > 0.001)
            print(f"   知识点数: {result['concept_count']}")
            print(f"   掌握程度向量长度: {len(ability_vector)}")
            print(f"   非零值(>0.001)数量: {non_zero_count}")
            print(f"   平均值: {sum(ability_vector)/len(ability_vector):.6f}")
            print(f"   最小值: {min(ability_vector):.6f}")
            print(f"   最大值: {max(ability_vector):.6f}")
            
            # 显示前几个值
            print(f"   前10个值: {ability_vector[:10]}")
            
            if non_zero_count == 0:
                print("   ⚠️ 输出全零，可能需要检查模型")
            else:
                # 找出非零值的位置和值
                non_zero_indices = [i for i, x in enumerate(ability_vector) if abs(x) > 0.001]
                print(f"   非零值位置(前5个): {non_zero_indices[:5]}")
                print(f"   对应的值: {[ability_vector[i] for i in non_zero_indices[:5]]}")
        else:
            print("❌ 单个学习者计算失败")
        
        # 测试模式1：多个学习者
        print("\n--- 测试模式1: 多个学习者 ---")
        results = engine.compute_multiple_learners_concept_mastery(test_learner_uids)
        
        if results['success']:
            print(f"✅ 批量计算成功: {results['success_count']}/{results['total_count']}")
            
            for uid, data in results['results'].items():
                if 'concept_mastery_vector' in data:
                    ability_vector = data['concept_mastery_vector']
                    non_zero_count = sum(1 for x in ability_vector if abs(x) > 0.001)
                    print(f"   {uid}: 知识点数={data['concept_count']}, 非零值={non_zero_count}")
                else:
                    print(f"   {uid}: 失败 - {data.get('error', '未知错误')}")
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
            
            for uid, data in new_results['results'].items():
                if 'concept_mastery_vector' in data:
                    ability_vector = data['concept_mastery_vector']
                    non_zero_count = sum(1 for x in ability_vector if abs(x) > 0.001)
                    print(f"   {uid}: 知识点数={data['concept_count']}, 非零值={non_zero_count}, 新学习者={data.get('is_new_learner', False)}")
        else:
            print(f"❌ 新学习者计算失败: {new_results.get('error', '未知错误')}")
        
        # 显示引擎状态
        status = engine.get_engine_status()
        print(f"\n📊 引擎状态: 初始化={status['initialized']}, 知识点数={status['concept_count']}, 缓存大小={status['embedding_cache_size']}")
        
        print("\n=== CD引擎测试完成 ===")
        return True
        
    except Exception as e:
        print(f"\n💥 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':

    # 1. 导入logging
    import logging
    # 2. 设置根日志记录器为DEBUG
    logging.getLogger().setLevel(logging.INFO)
    # 3. 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console_handler)

    success = test_cd_engine()
    
    if success:
        print("\n🎉 CD引擎测试成功！")
    else:
        print("\n❌ CD引擎测试失败！")