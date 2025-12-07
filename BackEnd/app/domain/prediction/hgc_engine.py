# BackEnd/app/domain/prediction/hgc_engine.py
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sparse

# 确保 DeepLearning 目录在 sys.path 中
from app.core.settings import path_settings  # noqa: F401

# HGC 模型与超参数
from DeepLearning.Model.HGC import LearnerEncoder
from DeepLearning.hyperparams.hyperparameter import hyperparams

# 基类与仓库
from app.domain.common.base_engine import BaseEngine
from app.data_access.prediction.hgc_repository import HGCRepository

logger = logging.getLogger(__name__)


class HGCEngine(BaseEngine):
    """
    HGC 模型推理引擎 - 负责学习者嵌入计算

    主要职责：
    - 使用 HGCRepository 取数，构造：
        - 学习者初始化向量（L-U）
        - 三种 meta-path：L-U-L、L-C-L、L-T-L
    - 调用 LearnerEncoder 得到学习者嵌入
    - 对外统一暴露 analyze(learner_uids) 接口，返回 {uid: embedding} 形式
    """

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(device=device or hyperparams.device)
        self.model: Optional[nn.Module] = None

        # 数据访问
        self._repository = HGCRepository()

        # 缓存
        self.learner_data_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self.meta_path_cache: Dict[Any, Any] = {}

        logger.info("HGCEngine 初始化，设备: %s", self.device)

    # ---------------------------------------------------------------------
    # 初始化与模型加载
    # ---------------------------------------------------------------------
    def initialize(self) -> bool:
        """
        初始化 HGC 模型

        Returns:
            bool: 初始化是否成功
        """
        try:
            if self.is_initialized:
                logger.info("HGC 引擎已经初始化")
                return True

            logger.info("开始初始化 HGC 引擎...")

            # 初始化学习者编码器
            self._initialize_learner_encoder()

            # 加载训练好的模型权重
            self._load_model_weights()

            self.is_initialized = True
            logger.info("HGC 引擎初始化完成")
            return True
        except Exception as exc:
            logger.error("HGC 引擎初始化失败: %s", exc)
            self.is_initialized = False
            return False

    def _initialize_learner_encoder(self) -> None:
        """初始化学习者编码器（LearnerEncoder）"""
        logger.info("初始化学习者编码器...")

        embedding_dim = hyperparams.hgc_embedding_dim

        # LearnerEncoder 使用自适应投影，不需要额外输入维度
        self.model = LearnerEncoder(
            embedding_dim=embedding_dim
        ).to(self.device)

        logger.info("学习者编码器初始化完成: embedding_dim=%d", embedding_dim)
        logger.info("使用自适应投影，支持任意输入维度")

    def _load_model_weights(self) -> None:
        """加载训练好的模型权重（只提取 LearnerEncoder 部分）"""
        logger.info("加载 HGC 模型权重...")

        save_dir = hyperparams.train_save_dir
        final_dir = os.path.join(save_dir, "final_models")
        hgc_path = os.path.join(final_dir, "hgc_best_model.pth")

        if not os.path.exists(hgc_path):
            logger.warning("模型权重文件不存在: %s，使用随机初始化的模型", hgc_path)
            return

        try:
            checkpoint = torch.load(hgc_path, map_location=self.device)

            # 兼容训练脚本保存的不同格式
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                logger.info(
                    "加载完整检查点，epoch: %s",
                    checkpoint.get("epoch", "unknown"),
                )
            else:
                # 直接保存 state_dict 的情况
                state_dict = checkpoint

            # 提取 LearnerEncoder 部分的权重
            learner_encoder_state_dict: Dict[str, torch.Tensor] = {}

            for key, value in state_dict.items():
                # 格式 1：带前缀 learner_encoder.xxx
                if key.startswith("learner_encoder."):
                    new_key = key.replace("learner_encoder.", "")
                    learner_encoder_state_dict[new_key] = value
                # 格式 2：直接就是编码器内部层（训练脚本中单独保存）
                elif (
                    key.startswith("lrn_proj.")
                    or key.startswith("lrn_attn.")
                    or key.startswith("output_norm.")
                ):
                    learner_encoder_state_dict[key] = value
                # 格式 3：嵌套在 model_hgc_state_dict 下
                elif key.startswith("model_hgc_state_dict"):
                    nested_state_dict = value
                    for nested_key, nested_value in nested_state_dict.items():
                        if nested_key.startswith("learner_encoder."):
                            new_nested_key = nested_key.replace(
                                "learner_encoder.", ""
                            )
                            learner_encoder_state_dict[new_nested_key] = nested_value

            if learner_encoder_state_dict:
                model_keys = set(self.model.state_dict().keys())
                loaded_keys = set(learner_encoder_state_dict.keys())

                missing_keys = model_keys - loaded_keys
                unexpected_keys = loaded_keys - model_keys

                if missing_keys:
                    logger.warning("权重文件中缺少以下键: %s", missing_keys)
                if unexpected_keys:
                    logger.warning("权重文件中存在未使用的键: %s", unexpected_keys)

                load_result = self.model.load_state_dict(
                    learner_encoder_state_dict, strict=False
                )
                logger.info("学习者编码器权重加载成功")
                logger.info(
                    "state_dict 加载结果: missing_keys=%d, unexpected_keys=%d",
                    len(load_result.missing_keys),
                    len(load_result.unexpected_keys),
                )
            else:
                logger.warning(
                    "未从检查点中找到任何 LearnerEncoder 相关权重，将使用随机初始化参数"
                )
        except Exception as exc:
            logger.error("加载 HGC 模型权重失败: %s", exc)
            logger.error("权重文件路径: %s", hgc_path)
            logger.error("设备: %s", self.device)
            import traceback

            logger.error("详细错误: %s", traceback.format_exc())
            logger.warning("使用随机初始化的模型")

        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            logger.info(
                "模型参数统计 - 总数: %d, 可训练: %d", total_params, trainable_params
            )

    # ---------------------------------------------------------------------
    # 数据获取与矩阵构建（全部通过 HGCRepository，避免直接耦合 DB）
    # ---------------------------------------------------------------------
    def _get_learner_data(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        从 HGCRepository 获取多个学习者的 HGC 所需数据

        返回结构（多学习者）：
            {
                "target_learner_uids": [...],
                "learner_entities": {
                    "units": {uid: [unit_uid, ...], ...},
                    "topics": {...},
                    "courses": {...},
                },
                "all_entities": {
                    "units": [...],
                    "topics": [...],
                    "courses": [...],
                },
                "related_learners": [...],
                "interaction_records": [...],
                "strategy_used": "intersection" | "union",
            }
        单学习者场景下会做一层包装，保持结构一致。
        """
        cache_key = tuple(sorted(learner_uids))
        if cache_key in self.learner_data_cache:
            return self.learner_data_cache[cache_key]

        logger.info("获取 %d 个学习者的 HGC 数据", len(learner_uids))

        if len(learner_uids) == 1:
            uid = learner_uids[0]
            data = self._repository.get_data_for_single_learner(uid)
            result = {
                "target_learner_uids": [uid],
                "learner_entities": {
                    "units": {uid: data.get("interacted_units", [])},
                    "topics": {uid: data.get("learner_topics", [])},
                    "courses": {uid: data.get("learner_courses", [])},
                },
                "all_entities": {
                    "units": data.get("interacted_units", []),
                    "topics": data.get("learner_topics", []),
                    "courses": data.get("learner_courses", []),
                },
                "related_learners": data.get("related_learners", []),
                "interaction_records": data.get("interaction_records", []),
                "strategy_used": data.get("strategy_used", "unknown"),
            }
        else:
            result = self._repository.get_data_for_multiple_learners(learner_uids)

        self.learner_data_cache[cache_key] = result
        return result

    # -------------------- dense 初始化向量 --------------------
    def _build_learner_init_matrix(self, learner_data: Dict[str, Any]) -> torch.Tensor:
        """
        构建学习者初始化矩阵（L-U 矩阵）

        Returns:
            torch.Tensor: [lrn_num, unit_num]
        """
        learner_uids = learner_data["target_learner_uids"]
        all_units = learner_data["all_entities"]["units"]

        unit_to_idx = {uid: idx for idx, uid in enumerate(all_units)}

        lrn_num = len(learner_uids)
        unit_num = len(all_units)
        init_matrix = torch.zeros((lrn_num, unit_num), dtype=torch.float32)

        for lrn_idx, lrn_uid in enumerate(learner_uids):
            interacted_units = learner_data["learner_entities"]["units"].get(
                lrn_uid, []
            )
            for unit_uid in interacted_units:
                if unit_uid in unit_to_idx:
                    unit_idx = unit_to_idx[unit_uid]
                    init_matrix[lrn_idx, unit_idx] = 1.0

        # 行归一化
        self._normalize_matrix(init_matrix)
        return init_matrix

    @staticmethod
    def _normalize_matrix(matrix: torch.Tensor) -> None:
        """对 dense 矩阵做行归一化（原地修改）"""
        row_sum = matrix.sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum != 0, row_sum, torch.ones_like(row_sum))
        matrix.div_(row_sum)

    # -------------------- 三种 meta-path 构建 --------------------
    def _build_meta_path_lul(
        self, learner_data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建 L-U-L 元路径"""
        learner_uids = learner_data["target_learner_uids"]
        all_units = learner_data["all_entities"]["units"]

        lrn_to_idx = {uid: idx for idx, uid in enumerate(learner_uids)}
        unit_to_idx = {uid: idx for idx, uid in enumerate(all_units)}

        lrn_unit_edges: List[Tuple[int, int]] = []
        for lrn_uid, units in learner_data["learner_entities"]["units"].items():
            if lrn_uid not in lrn_to_idx:
                continue
            lrn_idx = lrn_to_idx[lrn_uid]
            for unit_uid in units:
                if unit_uid in unit_to_idx:
                    unit_idx = unit_to_idx[unit_uid]
                    lrn_unit_edges.append((lrn_idx, unit_idx))

        lrn_num = len(learner_uids)
        unit_num = len(all_units)

        if lrn_unit_edges:
            rows, cols = zip(*lrn_unit_edges)
            data = np.ones(len(rows), dtype=np.float32)
            A_lu = sparse.coo_matrix((data, (rows, cols)), shape=(lrn_num, unit_num))

            A_lul = A_lu.dot(A_lu.T)
            A_lul_normalized = self._normalize_sparse_matrix(A_lul, add_self_loop=True)

            return self._sparse_to_edge_index_weight(A_lul_normalized)
        else:
            return self._create_empty_edges(lrn_num)

    def _build_meta_path_lcl(
        self, learner_data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建 L-C-L 元路径"""
        learner_uids = learner_data["target_learner_uids"]
        all_courses = learner_data["all_entities"]["courses"]

        lrn_to_idx = {uid: idx for idx, uid in enumerate(learner_uids)}
        course_to_idx = {uid: idx for idx, uid in enumerate(all_courses)}

        lrn_course_edges: List[Tuple[int, int]] = []
        for lrn_uid, courses in learner_data["learner_entities"]["courses"].items():
            if lrn_uid not in lrn_to_idx:
                continue
            lrn_idx = lrn_to_idx[lrn_uid]
            for course_uid in courses:
                if course_uid in course_to_idx:
                    course_idx = course_to_idx[course_uid]
                    lrn_course_edges.append((lrn_idx, course_idx))

        lrn_num = len(learner_uids)
        course_num = len(all_courses)

        if lrn_course_edges:
            rows, cols = zip(*lrn_course_edges)
            data = np.ones(len(rows), dtype=np.float32)
            A_lc = sparse.coo_matrix((data, (rows, cols)), shape=(lrn_num, course_num))

            A_lcl = A_lc.dot(A_lc.T)
            A_lcl_normalized = self._normalize_sparse_matrix(A_lcl, add_self_loop=True)

            return self._sparse_to_edge_index_weight(A_lcl_normalized)
        else:
            return self._create_empty_edges(lrn_num)

    def _build_meta_path_ltl(
        self, learner_data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建 L-T-L 元路径"""
        learner_uids = learner_data["target_learner_uids"]
        all_topics = learner_data["all_entities"]["topics"]

        lrn_to_idx = {uid: idx for idx, uid in enumerate(learner_uids)}
        topic_to_idx = {uid: idx for idx, uid in enumerate(all_topics)}

        lrn_topic_edges: List[Tuple[int, int]] = []
        for lrn_uid, topics in learner_data["learner_entities"]["topics"].items():
            if lrn_uid not in lrn_to_idx:
                continue
            lrn_idx = lrn_to_idx[lrn_uid]
            for topic_uid in topics:
                if topic_uid in topic_to_idx:
                    topic_idx = topic_to_idx[topic_uid]
                    lrn_topic_edges.append((lrn_idx, topic_idx))

        lrn_num = len(learner_uids)
        topic_num = len(all_topics)

        if lrn_topic_edges:
            rows, cols = zip(*lrn_topic_edges)
            data = np.ones(len(rows), dtype=np.float32)
            A_lt = sparse.coo_matrix((data, (rows, cols)), shape=(lrn_num, topic_num))

            A_ltl = A_lt.dot(A_lt.T)
            A_ltl_normalized = self._normalize_sparse_matrix(A_ltl, add_self_loop=True)

            return self._sparse_to_edge_index_weight(A_ltl_normalized)
        else:
            return self._create_empty_edges(lrn_num)

    # -------------------- 稀疏矩阵工具 --------------------
    @staticmethod
    def _normalize_sparse_matrix(
        sparse_matrix: sparse.spmatrix, add_self_loop: bool = True
    ) -> sparse.spmatrix:
        """对稀疏矩阵做 D^(-1/2) A D^(-1/2) 归一化"""
        if add_self_loop:
            n = sparse_matrix.shape[0]
            identity = sparse.identity(n, format="csr")
            sparse_matrix = sparse_matrix + identity

        row_sum = np.array(sparse_matrix.sum(axis=1)).flatten()
        row_sum = np.maximum(row_sum, 1e-6)
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(row_sum))

        return D_inv_sqrt.dot(sparse_matrix).dot(D_inv_sqrt)

    @staticmethod
    def _sparse_to_edge_index_weight(
        sparse_matrix: sparse.spmatrix,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从稀疏矩阵中提取 edge_index 和 edge_weight"""
        coo = sparse_matrix.tocoo()
        edge_index = torch.tensor(
            np.stack([coo.row, coo.col]), dtype=torch.long
        )  # [2, E]
        edge_weight = torch.tensor(coo.data, dtype=torch.float32)  # [E]
        return edge_index, edge_weight

    @staticmethod
    def _create_empty_edges(
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """没有任何连接时，构造只有自环的图"""
        indices = torch.arange(num_nodes, dtype=torch.long)
        edge_index = torch.stack([indices, indices])  # 自环
        edge_weight = torch.ones(num_nodes, dtype=torch.float32)
        return edge_index, edge_weight

    # ---------------------------------------------------------------------
    # 对外接口：单学习者 / 多学习者 / 统一 analyze
    # ---------------------------------------------------------------------
    def compute_single_learner_embedding(
        self, learner_uid: str
    ) -> Optional[Dict[str, Any]]:
        """
        计算单个学习者的 HGC 嵌入表达

        返回结构类似：
            {
                "learner_uid": uid,
                "embedding": [...],
                "embedding_dim": int,
                "timestamp": iso_str,
            }
        """
        # 直接复用多学习者逻辑，保证算法一致
        multi = self.compute_multiple_learners_embedding([learner_uid])
        if not multi.get("success", False):
            return None

        result_by_uid = multi.get("results", {})
        data = result_by_uid.get(learner_uid)
        if not data or "embedding" not in data:
            return None

        embedding = data["embedding"]
        return {
            "learner_uid": learner_uid,
            "embedding": embedding,
            "embedding_dim": data.get("embedding_dim", len(embedding)),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
        }

    def compute_multiple_learners_embedding(
        self, learner_uids: List[str]
    ) -> Dict[str, Any]:
        """
        计算多个学习者的 HGC 嵌入表达

        Returns:
            {
                "success": bool,
                "total_count": int,
                "success_count": int,
                "results": {
                    learner_uid: {
                        "embedding": [...],
                        "embedding_dim": int,
                        "timestamp": iso_str,
                    } | {
                        "error": str,
                        "timestamp": iso_str,
                    }
                }
            }
        """
        try:
            if not self.ensure_initialized():
                return {
                    "success": False,
                    "error": "引擎初始化失败",
                    "total_count": len(learner_uids),
                    "success_count": 0,
                    "results": {},
                }

            logger.info("计算多个学习者嵌入: %d 个学习者", len(learner_uids))

            if not learner_uids:
                return {
                    "success": True,
                    "total_count": 0,
                    "success_count": 0,
                    "results": {},
                }

            # 获取 HGC 数据
            learner_data = self._get_learner_data(learner_uids)

            # 构建输入
            lrn_init = self._build_learner_init_matrix(learner_data).to(self.device)
            p_lul = tuple(x.to(self.device) for x in self._build_meta_path_lul(learner_data))
            p_lcl = tuple(x.to(self.device) for x in self._build_meta_path_lcl(learner_data))
            p_ltl = tuple(x.to(self.device) for x in self._build_meta_path_ltl(learner_data))

            # 模型前向
            with torch.no_grad():
                self.model.eval()
                learner_embeddings = self.model(
                    lrn_init=lrn_init,
                    p_lul=p_lul,
                    p_lcl=p_lcl,
                    p_ltl=p_ltl,
                    device=self.device,
                )

            results: Dict[str, Any] = {}
            success_count = 0

            for idx, uid in enumerate(learner_uids):
                if idx < len(learner_embeddings):
                    emb = learner_embeddings[idx].cpu().numpy().tolist()
                    results[uid] = {
                        "embedding": emb,
                        "embedding_dim": len(emb),
                        "timestamp": datetime.now().isoformat(),
                    }
                    success_count += 1
                else:
                    logger.warning(
                        "学习者 %s 的嵌入计算失败: 索引超出范围 (idx=%d, len=%d)",
                        uid,
                        idx,
                        len(learner_embeddings),
                    )
                    results[uid] = {
                        "error": "索引超出范围",
                        "timestamp": datetime.now().isoformat(),
                    }

            logger.info(
                "多个学习者嵌入计算完成: %d/%d 成功",
                success_count,
                len(learner_uids),
            )
            return {
                "success": True,
                "total_count": len(learner_uids),
                "success_count": success_count,
                "results": results,
            }
        except Exception as exc:
            logger.error("计算多个学习者嵌入失败: %s", exc)
            import traceback

            logger.error("详细错误: %s", traceback.format_exc())
            return {
                "success": False,
                "error": str(exc),
                "total_count": len(learner_uids),
                "success_count": 0,
                "results": {},
            }

    # 统一给 prediction_pipeline 用的接口
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        统一接口：接受若干 learner_uids，返回包含 embedding 的结果字典。

        结构等同于 compute_multiple_learners_embedding 的返回值。
        """
        return self.compute_multiple_learners_embedding(learner_uids)


# ----------------------------------------------------------------------
# 模块级单例 & 便捷函数
# ----------------------------------------------------------------------
_hgc_engine_singleton: Optional[HGCEngine] = None


def get_hgc_engine() -> HGCEngine:
    global _hgc_engine_singleton
    if _hgc_engine_singleton is None:
        _hgc_engine_singleton = HGCEngine()
    return _hgc_engine_singleton


def analyze(learner_uids: List[str]) -> Dict[str, Any]:
    """
    便捷函数：直接调用全局 HGCEngine 的 analyze
    """
    engine = get_hgc_engine()
    return engine.analyze(learner_uids)


def compute_single_learner_embedding(learner_uid: str) -> Optional[Dict[str, Any]]:
    engine = get_hgc_engine()
    return engine.compute_single_learner_embedding(learner_uid)


def compute_multiple_learners_embedding(learner_uids: List[str]) -> Dict[str, Any]:
    engine = get_hgc_engine()
    return engine.compute_multiple_learners_embedding(learner_uids)


def initialize_engine() -> bool:
    engine = get_hgc_engine()
    return engine.initialize()


def get_engine_status() -> Dict[str, Any]:
    engine = get_hgc_engine()
    return engine.get_engine_status()
