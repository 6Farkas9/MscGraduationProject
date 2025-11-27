# BackEnd/app/engine/hgc_engine.py
import sys
import os
import torch
import torch.nn as nn
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import scipy.sparse as sparse
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入模型定义
from DeepLearning.Model.HGC import LearnerEncoder
from DeepLearning.hyperparams.hyperparameter import hyperparams

# 导入Repository
from app.repositories.hgc_repository import hgc_repository

logger = logging.getLogger(__name__)

class HGCEngine:
    """HGC模型推理引擎 - 专注于学习者嵌入计算"""
    
    def __init__(self, device: str = None):
        """
        初始化HGC引擎
        
        Args:
            device: 计算设备，默认为超参数配置的设备
        """
        self.device = device or hyperparams.device
        self.model = None
        self.is_initialized = False
        
        # 数据缓存
        self.learner_data_cache = {}
        self.meta_path_cache = {}
        
        logger.info(f"HGC引擎初始化，设备: {self.device}")
    
    def initialize(self) -> bool:
        """
        初始化HGC模型
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            if self.is_initialized:
                logger.info("HGC引擎已经初始化")
                return True
            
            logger.info("开始初始化HGC引擎...")
            
            # 初始化学习者编码器
            self._initialize_learner_encoder()
            
            # 加载模型权重
            self._load_model_weights()
            
            self.is_initialized = True
            logger.info("HGC引擎初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"HGC引擎初始化失败: {e}")
            self.is_initialized = False
            return False
    
    def _initialize_learner_encoder(self):
        """初始化学习者编码器 - 适配新版HGC模型"""
        logger.info("初始化学习者编码器...")
        
        # 使用超参数配置
        embedding_dim = hyperparams.hgc_embedding_dim
        
        # 新版HGC使用自适应投影，不再需要输入维度参数
        self.model = LearnerEncoder(
            embedding_dim=embedding_dim
            # 不再需要lrn_input_dim参数
        ).to(self.device)
        
        logger.info(f"学习者编码器初始化完成: embedding_dim={embedding_dim}")
        logger.info(f"使用自适应投影，支持任意输入维度")
    
    def _load_model_weights(self):
        """加载训练好的模型权重 - 适配新版HGC模型结构"""
        logger.info("加载HGC模型权重...")
        
        # 模型权重路径
        save_dir = hyperparams.train_save_dir
        final_dir = os.path.join(save_dir, "final_models")
        hgc_path = os.path.join(final_dir, "hgc_best_model.pth")
        
        if not os.path.exists(hgc_path):
            logger.warning(f"模型权重文件不存在: {hgc_path}，使用随机初始化的模型")
            return
        
        try:
            checkpoint = torch.load(hgc_path, map_location=self.device)
            
            # 根据训练脚本的保存格式处理权重
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 训练脚本保存的完整检查点格式
                state_dict = checkpoint['model_state_dict']
                logger.info(f"加载完整检查点，epoch: {checkpoint.get('epoch', 'unknown')}")
            else:
                # 直接保存的模型权重
                state_dict = checkpoint
            
            # 新版HGC模型权重映射
            # 训练脚本保存的是完整的HGC模型，需要提取learner_encoder部分
            learner_encoder_state_dict = {}
            
            for key, value in state_dict.items():
                # 处理不同的权重命名方式
                if key.startswith('learner_encoder.'):
                    # 新版HGC的权重格式
                    new_key = key.replace('learner_encoder.', '')
                    learner_encoder_state_dict[new_key] = value
                elif key.startswith('lrn_proj.') or key.startswith('lrn_gcn_') or key.startswith('lrn_attn.') or key.startswith('output_norm.'):
                    # 直接是学习者编码器的权重（没有learner_encoder前缀）
                    learner_encoder_state_dict[key] = value
                elif key.startswith('model_hgc_state_dict'):
                    # 如果是嵌套的模型状态字典
                    nested_state_dict = value
                    for nested_key, nested_value in nested_state_dict.items():
                        if nested_key.startswith('learner_encoder.'):
                            new_nested_key = nested_key.replace('learner_encoder.', '')
                            learner_encoder_state_dict[new_nested_key] = nested_value
            
            if learner_encoder_state_dict:
                # 检查模型结构是否匹配
                model_keys = set(self.model.state_dict().keys())
                loaded_keys = set(learner_encoder_state_dict.keys())
                
                missing_keys = model_keys - loaded_keys
                unexpected_keys = loaded_keys - model_keys
                
                if missing_keys:
                    logger.warning(f"权重文件中缺少以下键: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"权重文件中有意外的键: {unexpected_keys}")
                
                # 加载权重
                load_result = self.model.load_state_dict(learner_encoder_state_dict, strict=False)
                logger.info(f"学习者编码器权重加载成功")
                logger.info(f"成功加载: {len(load_result.missing_keys)}个缺失键, {len(load_result.unexpected_keys)}个意外键")
            else:
                logger.warning("未找到学习者编码器权重，使用随机初始化")
                
        except Exception as e:
            logger.error(f"加载模型权重失败: {e}")
            logger.error(f"权重文件路径: {hgc_path}")
            logger.error(f"设备: {self.device}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            logger.warning("使用随机初始化的模型")
        
        if learner_encoder_state_dict:
            # 记录模型参数信息
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"模型参数统计 - 总数: {total_params:,}, 可训练: {trainable_params:,}")
    
    def _get_learner_data(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        获取学习者的HGC数据
        
        Args:
            learner_uids: 学习者UID列表
            
        Returns:
            Dict: 学习者数据
        """
        cache_key = tuple(sorted(learner_uids))
        if cache_key in self.learner_data_cache:
            return self.learner_data_cache[cache_key]
        
        logger.info(f"获取 {len(learner_uids)} 个学习者的HGC数据")
        
        if len(learner_uids) == 1:
            # 单个学习者
            data = hgc_repository.get_data_for_single_learner(learner_uids[0])
            result = {
                'target_learner_uids': [learner_uids[0]],
                'learner_entities': {
                    'units': {learner_uids[0]: data.get('interacted_units', [])},
                    'topics': {learner_uids[0]: data.get('learner_topics', [])},
                    'courses': {learner_uids[0]: data.get('learner_courses', [])}
                },
                'all_entities': {
                    'units': data.get('interacted_units', []),
                    'topics': data.get('learner_topics', []),
                    'courses': data.get('learner_courses', [])
                },
                'interaction_records': data.get('interaction_records', [])
            }
        else:
            # 多个学习者
            result = hgc_repository.get_data_for_multiple_learners(learner_uids)
        
        self.learner_data_cache[cache_key] = result
        return result
    
    def _build_learner_init_matrix(self, learner_data: Dict[str, Any]) -> torch.Tensor:
        """
        构建学习者初始化矩阵
        
        Args:
            learner_data: 学习者数据
            
        Returns:
            torch.Tensor: 学习者初始化矩阵
        """
        learner_uids = learner_data['target_learner_uids']
        all_units = learner_data['all_entities']['units']
        
        # 创建单位到索引的映射
        unit_to_idx = {unit: idx for idx, unit in enumerate(all_units)}
        
        # 初始化矩阵
        lrn_num = len(learner_uids)
        unit_num = len(all_units)
        init_matrix = torch.zeros((lrn_num, unit_num), dtype=torch.float)
        
        # 填充矩阵
        for lrn_idx, lrn_uid in enumerate(learner_uids):
            interacted_units = learner_data['learner_entities']['units'].get(lrn_uid, [])
            for unit_uid in interacted_units:
                if unit_uid in unit_to_idx:
                    unit_idx = unit_to_idx[unit_uid]
                    init_matrix[lrn_idx, unit_idx] = 1.0
        
        # 归一化
        self._normalize_matrix(init_matrix)
        
        return init_matrix
    
    def _normalize_matrix(self, matrix: torch.Tensor):
        """归一化矩阵"""
        row_sum = matrix.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum != 0, row_sum, torch.ones_like(row_sum))
        matrix.div_(row_sum)
    
    def _build_meta_path_lul(self, learner_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建L-U-L元路径"""
        learner_uids = learner_data['target_learner_uids']
        all_units = learner_data['all_entities']['units']
        
        # 创建映射
        lrn_to_idx = {uid: idx for idx, uid in enumerate(learner_uids)}
        unit_to_idx = {uid: idx for idx, uid in enumerate(all_units)}
        
        # 构建L-U矩阵
        lrn_unit_edges = []
        for lrn_uid, units in learner_data['learner_entities']['units'].items():
            if lrn_uid in lrn_to_idx:
                lrn_idx = lrn_to_idx[lrn_uid]
                for unit_uid in units:
                    if unit_uid in unit_to_idx:
                        unit_idx = unit_to_idx[unit_uid]
                        lrn_unit_edges.append((lrn_idx, unit_idx))
        
        # 构建L-U-L元路径邻接矩阵
        lrn_num = len(learner_uids)
        unit_num = len(all_units)
        
        if lrn_unit_edges:
            # 构建L-U稀疏矩阵
            rows, cols = zip(*lrn_unit_edges)
            data = np.ones(len(rows))
            A_lu = sparse.coo_matrix((data, (rows, cols)), shape=(lrn_num, unit_num))
            
            # 计算L-U-L: A_lu * A_lu^T
            A_lul = A_lu.dot(A_lu.T)
            
            # 添加自环并归一化
            A_lul_normalized = self._normalize_sparse_matrix(A_lul, add_self_loop=True)
            
            # 转换为边索引和权重
            return self._sparse_to_edge_index_weight(A_lul_normalized)
        else:
            return self._create_empty_edges(lrn_num)
    
    def _build_meta_path_lcl(self, learner_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建L-C-L元路径"""
        learner_uids = learner_data['target_learner_uids']
        all_courses = learner_data['all_entities']['courses']
        
        # 创建映射
        lrn_to_idx = {uid: idx for idx, uid in enumerate(learner_uids)}
        course_to_idx = {uid: idx for idx, uid in enumerate(all_courses)}
        
        # 构建L-C矩阵
        lrn_course_edges = []
        for lrn_uid, courses in learner_data['learner_entities']['courses'].items():
            if lrn_uid in lrn_to_idx:
                lrn_idx = lrn_to_idx[lrn_uid]
                for course_uid in courses:
                    if course_uid in course_to_idx:
                        course_idx = course_to_idx[course_uid]
                        lrn_course_edges.append((lrn_idx, course_idx))
        
        # 构建L-C-L元路径
        lrn_num = len(learner_uids)
        course_num = len(all_courses)
        
        if lrn_course_edges:
            rows, cols = zip(*lrn_course_edges)
            data = np.ones(len(rows))
            A_lc = sparse.coo_matrix((data, (rows, cols)), shape=(lrn_num, course_num))
            
            A_lcl = A_lc.dot(A_lc.T)
            A_lcl_normalized = self._normalize_sparse_matrix(A_lcl, add_self_loop=True)
            
            return self._sparse_to_edge_index_weight(A_lcl_normalized)
        else:
            return self._create_empty_edges(lrn_num)
    
    def _build_meta_path_ltl(self, learner_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建L-T-L元路径"""
        learner_uids = learner_data['target_learner_uids']
        all_topics = learner_data['all_entities']['topics']
        
        # 创建映射
        lrn_to_idx = {uid: idx for idx, uid in enumerate(learner_uids)}
        topic_to_idx = {uid: idx for idx, uid in enumerate(all_topics)}
        
        # 构建L-T矩阵
        lrn_topic_edges = []
        for lrn_uid, topics in learner_data['learner_entities']['topics'].items():
            if lrn_uid in lrn_to_idx:
                lrn_idx = lrn_to_idx[lrn_uid]
                for topic_uid in topics:
                    if topic_uid in topic_to_idx:
                        topic_idx = topic_to_idx[topic_uid]
                        lrn_topic_edges.append((lrn_idx, topic_idx))
        
        # 构建L-T-L元路径
        lrn_num = len(learner_uids)
        topic_num = len(all_topics)
        
        if lrn_topic_edges:
            rows, cols = zip(*lrn_topic_edges)
            data = np.ones(len(rows))
            A_lt = sparse.coo_matrix((data, (rows, cols)), shape=(lrn_num, topic_num))
            
            A_ltl = A_lt.dot(A_lt.T)
            A_ltl_normalized = self._normalize_sparse_matrix(A_ltl, add_self_loop=True)
            
            return self._sparse_to_edge_index_weight(A_ltl_normalized)
        else:
            return self._create_empty_edges(lrn_num)
    
    def _normalize_sparse_matrix(self, sparse_matrix: sparse.spmatrix, add_self_loop: bool = True) -> sparse.spmatrix:
        """归一化稀疏矩阵"""
        if add_self_loop:
            n = sparse_matrix.shape[0]
            identity = sparse.identity(n, format='csr')
            sparse_matrix = sparse_matrix + identity
        
        # 计算度矩阵
        row_sum = np.array(sparse_matrix.sum(axis=1)).flatten()
        row_sum = np.maximum(row_sum, 1e-6)
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(row_sum))
        
        # 归一化: D^(-1/2) * A * D^(-1/2)
        normalized = D_inv_sqrt.dot(sparse_matrix).dot(D_inv_sqrt)
        
        return normalized
    
    def _sparse_to_edge_index_weight(self, sparse_matrix: sparse.spmatrix) -> Tuple[torch.Tensor, torch.Tensor]:
        """从稀疏矩阵提取边索引和权重"""
        coo = sparse_matrix.tocoo()
        edge_index = torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long)
        edge_weight = torch.tensor(coo.data, dtype=torch.float)
        return edge_index, edge_weight
    
    def _create_empty_edges(self, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建空的边（只有自环）"""
        # 创建自环
        indices = torch.arange(num_nodes, dtype=torch.long)
        edge_index = torch.stack([indices, indices])
        edge_weight = torch.ones(num_nodes, dtype=torch.float)
        return edge_index, edge_weight
    
    def compute_single_learner_embedding(self, learner_uid: str) -> Optional[Dict[str, Any]]:
        """
        计算单个学习者的HGC嵌入表达
        
        Args:
            learner_uid: 学习者UID
            
        Returns:
            Optional[Dict]: 包含学习者嵌入的字典，失败返回None
        """
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return None
            
            logger.info(f"计算单个学习者嵌入: {learner_uid}")
            
            # 获取学习者数据
            learner_data = self._get_learner_data([learner_uid])
            
            # 构建输入数据
            lrn_init = self._build_learner_init_matrix(learner_data).to(self.device)
            p_lul = tuple(x.to(self.device) for x in self._build_meta_path_lul(learner_data))
            p_lcl = tuple(x.to(self.device) for x in self._build_meta_path_lcl(learner_data))
            p_ltl = tuple(x.to(self.device) for x in self._build_meta_path_ltl(learner_data))
            
            # 模型推理
            with torch.no_grad():
                self.model.eval()
                learner_embedding = self.model(
                    lrn_init=lrn_init,
                    p_lul=p_lul,
                    p_lcl=p_lcl, 
                    p_ltl=p_ltl,
                    device=self.device
                )
            
            # 提取目标学习者的嵌入（索引0位置）
            target_embedding = learner_embedding[0].cpu().numpy().tolist()
            
            result = {
                'learner_uid': learner_uid,
                'embedding': target_embedding,
                'embedding_dim': len(target_embedding),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"单个学习者嵌入计算完成: {learner_uid}, 维度: {len(target_embedding)}")
            return result
            
        except Exception as e:
            logger.error(f"计算单个学习者嵌入失败 {learner_uid}: {e}")
            return None
    
    def compute_multiple_learners_embedding(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        计算多个学习者的HGC嵌入表达
        
        Args:
            learner_uids: 学习者UID列表
            
        Returns:
            Dict: 包含所有学习者嵌入的字典
        """
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return {'success': False, 'error': '引擎初始化失败', 'results': {}}
            
            logger.info(f"计算多个学习者嵌入: {len(learner_uids)} 个学习者")
            
            if not learner_uids:
                return {'success': True, 'results': {}}
            
            # 获取学习者数据
            learner_data = self._get_learner_data(learner_uids)
            
            # 构建输入数据
            lrn_init = self._build_learner_init_matrix(learner_data).to(self.device)
            p_lul = tuple(x.to(self.device) for x in self._build_meta_path_lul(learner_data))
            p_lcl = tuple(x.to(self.device) for x in self._build_meta_path_lcl(learner_data))
            p_ltl = tuple(x.to(self.device) for x in self._build_meta_path_ltl(learner_data))
            
            # 模型推理
            with torch.no_grad():
                self.model.eval()
                learner_embeddings = self.model(
                    lrn_init=lrn_init,
                    p_lul=p_lul,
                    p_lcl=p_lcl,
                    p_ltl=p_ltl,
                    device=self.device
                )
            
            # 提取所有学习者的嵌入（按输入顺序）
            results = {}
            success_count = 0
            for idx, learner_uid in enumerate(learner_uids):
                if idx < len(learner_embeddings):
                    learner_embedding = learner_embeddings[idx].cpu().numpy().tolist()
                    results[learner_uid] = {
                        'embedding': learner_embedding,
                        'embedding_dim': len(learner_embedding),
                        'timestamp': datetime.now().isoformat()
                    }
                    success_count += 1
                else:
                    logger.warning(f"学习者 {learner_uid} 的嵌入计算失败: 索引超出范围")
                    results[learner_uid] = {
                        'error': '索引超出范围',
                        'timestamp': datetime.now().isoformat()
                    }
            
            logger.info(f"多个学习者嵌入计算完成: {success_count}/{len(learner_uids)} 成功")
            return {
                'success': True,
                'total_count': len(learner_uids),
                'success_count': success_count,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"计算多个学习者嵌入失败: {e}")
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
            'cache_size': len(self.learner_data_cache)
        }


# 全局引擎实例
_hgc_engine_instance = None

def get_hgc_engine() -> HGCEngine:
    """获取全局HGC引擎实例"""
    global _hgc_engine_instance
    if _hgc_engine_instance is None:
        _hgc_engine_instance = HGCEngine()
    return _hgc_engine_instance

def compute_single_learner_embedding(learner_uid: str) -> Optional[Dict[str, Any]]:
    """
    计算单个学习者嵌入的便捷函数
    
    Args:
        learner_uid: 学习者UID
        
    Returns:
        Optional[Dict]: 学习者嵌入结果
    """
    engine = get_hgc_engine()
    return engine.compute_single_learner_embedding(learner_uid)

def compute_multiple_learners_embedding(learner_uids: List[str]) -> Dict[str, Any]:
    """
    计算多个学习者嵌入的便捷函数
    
    Args:
        learner_uids: 学习者UID列表
        
    Returns:
        Dict: 包含所有学习者嵌入的结果
    """
    engine = get_hgc_engine()
    return engine.compute_multiple_learners_embedding(learner_uids)

def initialize_engine() -> bool:
    """初始化HGC引擎的便捷函数"""
    engine = get_hgc_engine()
    return engine.initialize()

def get_engine_status() -> Dict[str, Any]:
    """获取引擎状态的便捷函数"""
    engine = get_hgc_engine()
    return engine.get_engine_status()


# 测试代码
if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试HGC引擎
    print("=== HGC引擎测试 ===")
    
    # 初始化引擎
    engine = HGCEngine(device='cpu')
    status = engine.get_engine_status()
    print(f"引擎状态: {status}")
    
    # 初始化引擎
    if engine.initialize():
        print("引擎初始化成功")
        
        # 测试单个学习者（使用示例UID）
        test_learner_uid = "lrn_004a9c3f5bf246faab3d390ce716e658"  # 替换为实际存在的UID
        result = engine.compute_single_learner_embedding(test_learner_uid)
        if result:
            print(f"单个学习者测试成功: {result['learner_uid']}, 嵌入维度: {result['embedding_dim']}")
        else:
            print("单个学习者测试失败")
        
        # 测试多个学习者
        test_learner_uids = [
            "lrn_004a9c3f5bf246faab3d390ce716e658",
            "lrn_00a6f6e5a1e84e9d9f3b3c3a3d3e3f3a"  # 替换为实际存在的UID
        ]
        results = engine.compute_multiple_learners_embedding(test_learner_uids)
        print(f"多个学习者测试: 成功 {results['success_count']}/{results['total_count']}")
        
        # 显示详细结果
        for uid, data in results['results'].items():
            if 'embedding' in data:
                print(f"  {uid}: 嵌入维度 {data['embedding_dim']}")
            else:
                print(f"  {uid}: 失败 - {data.get('error', '未知错误')}")
    
    else:
        print("引擎初始化失败")
    
    print("=== HGC引擎测试完成 ===")