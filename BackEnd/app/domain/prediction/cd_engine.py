# BackEnd/app/domain/prediction/cd_engine.py
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# 触发 DeepLearning 路径配置
from app.core.settings import path_settings  # noqa: F401

from DeepLearning.Model.CD import CD
from DeepLearning.hyperparams.hyperparameter import hyperparams

from app.domain.common.base_engine import BaseEngine
from app.data_access.prediction.cd_repository import CDRepository
from app.data_access.prediction.embedding_repository import EmbeddingRepository
from app.data_access.prediction.learner_repository import LearnerRepository

logger = logging.getLogger(__name__)


class CDEngine(BaseEngine):
    """CD 模型推理引擎 - 专注于认知诊断计算"""

    def __init__(self, device: Optional[str] = None):
        device = device or hyperparams.device
        super().__init__(device=device)

        # 仓库实例
        self.cd_repository = CDRepository()
        self.embedding_repository = EmbeddingRepository()
        self.learner_repository = LearnerRepository()

        # 数据缓存
        self.embedding_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]] = {}

        # 知识点映射
        self.concept_mapping: Optional[Dict[str, int]] = None
        self.concept_num: int = 0
        self.id_to_concept: Dict[int, str] = {}

    # ----------------- 初始化 -----------------

    def initialize(self) -> bool:
        """
        初始化 CD 模型
        """
        try:
            if self.is_initialized:
                logger.info("CD 引擎已经初始化")
                return True

            logger.info("开始初始化 CD 引擎.")

            # 初始化知识点映射
            self._initialize_concept_mapping()

            # 初始化 CD 模型
            self._initialize_cd_model()

            # 加载模型权重
            self._load_model_weights()

            # 验证模型
            self._validate_model()

            self.is_initialized = True
            logger.info("CD 引擎初始化完成")
            return True

        except Exception as exc:
            logger.error("CD 引擎初始化失败: %s", exc)
            self.is_initialized = False
            return False

    def _initialize_concept_mapping(self) -> None:
        """初始化知识点映射"""
        logger.info("初始化知识点映射.")

        self.concept_mapping = self.cd_repository.get_concept_uid_to_id_mapping()
        self.concept_num = len(self.concept_mapping or {})

        self.id_to_concept = {id_: uid for uid, id_ in (self.concept_mapping or {}).items()}

        logger.info("知识点映射初始化完成: %d 个知识点", self.concept_num)

    def _initialize_cd_model(self) -> None:
        """初始化 CD 模型"""
        logger.info("初始化 CD 模型.")

        embedding_dim = hyperparams.hgc_embedding_dim
        concept_num = self.concept_num

        self.model = CD(
            embedding_dim=embedding_dim,
            concept_num=concept_num,
        ).to(self.device)

        logger.info("CD 模型初始化完成: embedding_dim=%d, concept_num=%d", embedding_dim, concept_num)

    def _load_model_weights(self) -> None:
        """加载训练好的模型权重"""
        logger.info("加载 CD 模型权重.")

        save_dir = hyperparams.train_save_dir
        final_dir = os.path.join(save_dir, "final_models")
        cd_path = os.path.join(final_dir, "cd_best_model.pth")

        if not os.path.exists(cd_path):
            logger.warning("模型权重文件不存在: %s，使用随机初始化的模型", cd_path)
            return

        try:
            checkpoint = torch.load(cd_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                logger.info("加载完整检查点，epoch: %s", checkpoint.get("epoch", "unknown"))
            else:
                state_dict = checkpoint

            load_result = self.model.load_state_dict(state_dict, strict=True)
            logger.info("CD 模型权重加载成功")

            if load_result.missing_keys:
                logger.warning("缺失键: %s", load_result.missing_keys)
            if load_result.unexpected_keys:
                logger.warning("意外键: %s", load_result.unexpected_keys)

        except Exception as exc:
            logger.error("加载模型权重失败: %s", exc)
            logger.warning("使用随机初始化的模型")

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("模型参数总数: %d", total_params)

    def _validate_model(self) -> bool:
        """验证模型是否能正常推理"""
        logger.info("验证模型推理能力.")

        try:
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

                predictions = self.model(
                    h_lrn_batch=h_lrn_batch,
                    h_qus=h_qus,
                    h_cpt=h_cpt,
                    qus_seq_indices=qus_seq_indices,
                    qus_seq_masks=qus_seq_masks,
                    return_ability=False,
                    use_kt_optimization=False,
                )

                logger.info("标准前向传播测试: predictions shape=%s", predictions.shape)
                logger.info(
                    "predictions 统计: mean=%.6f, min=%.6f, max=%.6f",
                    predictions.mean().item(),
                    predictions.min().item(),
                    predictions.max().item(),
                )

                ability_matrix = self.model.get_ability_matrix(
                    h_lrn_batch=h_lrn_batch,
                    h_qus=h_qus,
                    h_cpt=h_cpt,
                    unt_seq_indices=qus_seq_indices,
                    seq_masks=qus_seq_masks,
                    unt_num=0,
                )

                logger.info("能力矩阵测试: ability_matrix shape=%s", ability_matrix.shape)
                logger.info(
                    "ability_matrix 统计: mean=%.6f, min=%.6f, max=%.6f",
                    ability_matrix.mean().item(),
                    ability_matrix.min().item(),
                    ability_matrix.max().item(),
                )

                if ability_matrix.mean().item() == 0 and ability_matrix.std().item() == 0:
                    logger.warning("能力矩阵输出全零，可能有问题")

            return True

        except Exception as exc:
            logger.error("模型验证失败: %s", exc)
            return False

    # ----------------- 嵌入获取 & KT 能力矩阵 -----------------

    def _get_embeddings(
        self, required_question_uids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        从数据库获取所需的嵌入向量（题目 + 知识点）
        """
        cache_key = f"embeddings_q{len(required_question_uids)}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        logger.info("从数据库获取嵌入向量: 题目=%d", len(required_question_uids))

        try:
            question_embeddings = self.embedding_repository.get_embeddings_by_uids(
                required_question_uids, return_format="list"
            )

            concept_embeddings = self.embedding_repository.get_embeddings_by_entity_type("cpt")

            h_qus_list: List[np.ndarray] = []
            qus_uid_to_idx: Dict[str, int] = {}

            for idx, q_emb in enumerate(question_embeddings):
                if q_emb and q_emb.get("embedding"):
                    embedding_array = np.array(q_emb["embedding"])
                    h_qus_list.append(embedding_array)
                    qus_uid_to_idx[q_emb["uid"]] = idx
                else:
                    logger.warning("题目 %s 的嵌入向量未找到", q_emb.get("uid", "unknown"))

            if not h_qus_list:
                embedding_dim = hyperparams.hgc_embedding_dim
                h_qus_list = [np.zeros(embedding_dim)]
                logger.warning("没有找到任何题目嵌入，使用零向量")

            h_cpt_list: List[np.ndarray] = []
            concept_emb_dict = {
                emb["uid"]: np.array(emb["embedding"]) for emb in concept_embeddings
            }

            for concept_id in range(1, self.concept_num + 1):
                concept_uid = self.id_to_concept.get(concept_id)
                if concept_uid and concept_uid in concept_emb_dict:
                    h_cpt_list.append(concept_emb_dict[concept_uid])
                else:
                    embedding_dim = hyperparams.hgc_embedding_dim
                    h_cpt_list.append(np.zeros(embedding_dim))
                    logger.warning("知识点 %s 的嵌入向量未找到", concept_uid)

            h_qus = torch.tensor(np.array(h_qus_list), dtype=torch.float32, device=self.device)
            h_cpt = torch.tensor(np.array(h_cpt_list), dtype=torch.float32, device=self.device)

            result = (h_qus, h_cpt, qus_uid_to_idx)
            self.embedding_cache[cache_key] = result

            logger.info("嵌入向量加载完成: 题目=%d, 知识点=%d", len(h_qus), len(h_cpt))
            return result

        except Exception as exc:
            logger.error("获取嵌入向量失败: %s", exc)
            embedding_dim = hyperparams.hgc_embedding_dim
            h_qus = torch.zeros(1, embedding_dim, device=self.device)
            h_cpt = torch.zeros(self.concept_num, embedding_dim, device=self.device)
            qus_uid_to_idx = {}
            return (h_qus, h_cpt, qus_uid_to_idx)

    def _get_kt_ability_matrix(self, learner_uids: List[str]) -> Optional[torch.Tensor]:
        """
        获取学习者的 KT 能力矩阵 [batch_size, 1, concept_num]
        """
        try:
            kt_results = self.learner_repository.get_kt_results_by_uids(
                learner_uids, return_format="list"
            )

            ability_matrix: List[np.ndarray] = []
            has_kt_data = False

            for learner_uid, kt_data in zip(learner_uids, kt_results):
                if kt_data and kt_data.get("KT"):
                    ability_vector = np.zeros(self.concept_num)
                    kt_dict = kt_data["KT"]

                    for concept_id in range(1, self.concept_num + 1):
                        concept_uid = self.id_to_concept.get(concept_id)
                        if concept_uid and concept_uid in kt_dict:
                            ability_vector[concept_id - 1] = kt_dict[concept_uid]

                    ability_matrix_3d = ability_vector.reshape(1, 1, -1)
                    ability_matrix.append(ability_matrix_3d)
                    has_kt_data = True
                    logger.debug("学习者 %s 有 KT 结果，构建 3D 能力矩阵", learner_uid)
                else:
                    ability_matrix_3d = np.zeros((1, 1, self.concept_num))
                    ability_matrix.append(ability_matrix_3d)

            if not has_kt_data:
                logger.info("所有学习者都没有 KT 结果，跳过能力融合")
                return None

            ability_tensor = torch.tensor(
                np.concatenate(ability_matrix, axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            logger.info("KT 能力矩阵构建完成: %s", ability_tensor.shape)
            return ability_tensor

        except Exception as exc:
            logger.error("获取 KT 能力矩阵失败: %s", exc)
            return None

    # ----------------- 输入准备（模式 1 / 2 共享） -----------------

    def _prepare_cd_inputs(
        self,
        learner_uids: List[str],
        learner_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        准备 CD 模型输入数据（保持原有逻辑，只换成实例化的 Repository）
        """
        try:
            max_seq_len = None if len(learner_uids) == 1 else 50

            sequence_data = self.cd_repository.build_question_sequences(learner_uids, max_seq_len)
            sequences = sequence_data["sequences"]
            actual_max_seq_len = sequence_data["actual_max_seq_len"]
            required_question_uids = sequence_data["all_question_uids"]

            if not required_question_uids:
                logger.error("没有找到任何题目交互记录")
                raise ValueError("没有找到任何题目交互记录")

            h_qus, h_cpt, qus_uid_to_idx = self._get_embeddings(required_question_uids)

            batch_size = len(learner_uids)
            effective_max_seq_len = actual_max_seq_len

            qus_seq_indices = torch.zeros(
                batch_size, effective_max_seq_len, dtype=torch.long, device=self.device
            )
            qus_seq_masks = torch.zeros(
                batch_size, effective_max_seq_len, dtype=torch.float32, device=self.device
            )

            learner_seq_lengths: List[int] = []

            # 学习者嵌入
            if learner_embeddings:
                h_lrn_batch = torch.stack(learner_embeddings)
            else:
                required_learner_embeddings = self.embedding_repository.get_embeddings_by_uids(
                    learner_uids, return_format="list"
                )
                h_lrn_batch_list: List[torch.Tensor] = []
                for learner_uid, l_emb in zip(learner_uids, required_learner_embeddings):
                    if l_emb and l_emb.get("embedding"):
                        h_lrn = torch.tensor(
                            np.array(l_emb["embedding"]),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    else:
                        embedding_dim = hyperparams.hgc_embedding_dim
                        h_lrn = torch.zeros(embedding_dim, device=self.device)
                        logger.warning("学习者 %s 的嵌入向量未找到", learner_uid)
                    h_lrn_batch_list.append(h_lrn)

                h_lrn_batch = torch.stack(h_lrn_batch_list)

            valid_learners = 0
            for i, learner_uid in enumerate(learner_uids):
                seq_data = sequences.get(learner_uid)
                if not seq_data:
                    learner_seq_lengths.append(0)
                    continue

                qus_seq = seq_data["qus_seq"]
                seq_len = seq_data["seq_len"]

                if seq_len == 0:
                    learner_seq_lengths.append(0)
                    continue

                learner_seq_lengths.append(seq_len)

                for j in range(seq_len):
                    qus_uid = qus_seq[j]
                    if qus_uid in qus_uid_to_idx:
                        qus_seq_indices[i, j] = qus_uid_to_idx[qus_uid]
                        qus_seq_masks[i, j] = 1.0

                valid_learners += 1

            if valid_learners == 0:
                logger.error("没有有效的学习者序列数据")
                raise ValueError("没有有效的学习者序列数据")

            kt_ability = self._get_kt_ability_matrix(learner_uids)

            inputs = {
                "h_lrn_batch": h_lrn_batch,
                "h_qus": h_qus,
                "h_cpt": h_cpt,
                "qus_seq_indices": qus_seq_indices,
                "qus_seq_masks": qus_seq_masks,
                "kt_ability": kt_ability,
                "actual_max_seq_len": effective_max_seq_len,
                "learner_seq_lengths": learner_seq_lengths,
            }

            logger.info(
                "CD 输入数据准备完成: 批次大小=%d, 有效学习者=%d, 实际最大序列长度=%d",
                batch_size,
                valid_learners,
                effective_max_seq_len,
            )
            return inputs

        except Exception as exc:
            logger.error("准备 CD 输入数据失败: %s", exc)
            raise

    # ----------------- 推理：模式 1（已有学习者） -----------------

    def compute_single_learner_concept_mastery(
        self, learner_uid: str
    ) -> Optional[Dict[str, Any]]:
        """
        计算单个已有学习者的知识点掌握程度（模式 1）
        """
        try:
            if not self.ensure_initialized():
                return None

            logger.info("计算单个学习者知识点掌握程度: %s", learner_uid)

            if not self.cd_repository.validate_learner_has_interactions(learner_uid):
                logger.error("学习者 %s 没有交互记录", learner_uid)
                return None

            inputs = self._prepare_cd_inputs([learner_uid])

            if inputs["kt_ability"] is not None:
                logger.info("设置 KT 优化能力进行能力融合")
                self.model.set_kt_optimized_ability(inputs["kt_ability"], unt_num=0)

            with torch.no_grad():
                self.model.eval()
                ability_matrix = self.model.get_ability_matrix(
                    h_lrn_batch=inputs["h_lrn_batch"],
                    h_qus=inputs["h_qus"],
                    h_cpt=inputs["h_cpt"],
                    unt_seq_indices=inputs["qus_seq_indices"],
                    seq_masks=inputs["qus_seq_masks"],
                    unt_num=0,
                )

            actual_seq_len = inputs.get("actual_max_seq_len", 1)
            if actual_seq_len > 0:
                last_valid_step = -1
                for step in range(actual_seq_len - 1, -1, -1):
                    if inputs["qus_seq_masks"][0, step] > 0.5:
                        last_valid_step = step
                        break
                if last_valid_step >= 0:
                    ability_vector = ability_matrix[0, last_valid_step].cpu().numpy().tolist()
                else:
                    ability_vector = ability_matrix[0, -1].cpu().numpy().tolist()
            else:
                ability_vector = ability_matrix[0, -1].cpu().numpy().tolist()

            non_zero_count = sum(1 for x in ability_vector if abs(x) > 0.001)
            if non_zero_count == 0:
                logger.warning("能力向量输出全零或接近零")

            result = {
                "learner_uid": learner_uid,
                "concept_mastery_vector": ability_vector,
                "concept_count": self.concept_num,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info("单个学习者知识点掌握程度计算完成")
            return result

        except Exception as exc:
            logger.error("计算单个学习者知识点掌握程度失败 %s: %s", learner_uid, exc)
            return None

    def compute_multiple_learners_concept_mastery(
        self, learner_uids: List[str]
    ) -> Dict[str, Any]:
        """
        计算多个已有学习者的知识点掌握程度（模式 1）
        """
        try:
            if not self.ensure_initialized():
                return {"success": False, "error": "引擎初始化失败", "results": {}}

            logger.info("计算多个学习者知识点掌握程度: %d 个学习者", len(learner_uids))

            if not learner_uids:
                return {
                    "success": True,
                    "results": {},
                    "total_count": 0,
                    "success_count": 0,
                }

            valid_learner_uids = [
                uid
                for uid in learner_uids
                if self.cd_repository.validate_learner_has_interactions(uid)
            ]

            if not valid_learner_uids:
                logger.error("没有找到任何有交互记录的学习者")
                return {
                    "success": False,
                    "error": "没有找到任何有交互记录的学习者",
                    "total_count": len(learner_uids),
                    "success_count": 0,
                    "results": {},
                }

            inputs = self._prepare_cd_inputs(valid_learner_uids)

            if inputs["kt_ability"] is not None:
                logger.info("设置 KT 优化能力进行能力融合")
                self.model.set_kt_optimized_ability(inputs["kt_ability"], unt_num=0)

            with torch.no_grad():
                self.model.eval()
                ability_matrix = self.model.get_ability_matrix(
                    h_lrn_batch=inputs["h_lrn_batch"],
                    h_qus=inputs["h_qus"],
                    h_cpt=inputs["h_cpt"],
                    unt_seq_indices=inputs["qus_seq_indices"],
                    seq_masks=inputs["qus_seq_masks"],
                    unt_num=0,
                )

            results: Dict[str, Any] = {}
            success_count = 0

            for i, learner_uid in enumerate(valid_learner_uids):
                if i < len(ability_matrix):
                    seq_len = (
                        inputs.get("learner_seq_lengths", [])[i]
                        if i < len(inputs.get("learner_seq_lengths", []))
                        else 0
                    )

                    if seq_len > 0:
                        ability_vector = ability_matrix[i, seq_len - 1].cpu().numpy().tolist()
                    else:
                        ability_vector = ability_matrix[i, -1].cpu().numpy().tolist()
                        logger.warning("学习者 %s 序列长度为 0，使用最后一个时间步", learner_uid)

                    results[learner_uid] = {
                        "concept_mastery_vector": ability_vector,
                        "concept_count": self.concept_num,
                        "timestamp": datetime.now().isoformat(),
                        "actual_seq_len": seq_len,
                    }
                    success_count += 1
                else:
                    logger.warning("学习者 %s 的能力计算失败: 索引超出范围", learner_uid)
                    results[learner_uid] = {
                        "error": "索引超出范围",
                        "timestamp": datetime.now().isoformat(),
                    }

            logger.info("多个学习者知识点掌握程度计算完成: %d 成功", success_count)
            return {
                "success": True,
                "total_count": len(learner_uids),
                "valid_count": len(valid_learner_uids),
                "success_count": success_count,
                "results": results,
            }

        except Exception as exc:
            logger.error("计算多个学习者知识点掌握程度失败: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "total_count": len(learner_uids),
                "success_count": 0,
                "results": {},
            }

    # ----------------- 推理：模式 2（新学习者，用传入的嵌入） -----------------

    def compute_concept_mastery_with_embeddings(
        self,
        learner_embeddings: List[torch.Tensor],
        learner_uids: List[str],
    ) -> Dict[str, Any]:
        """
        使用提供的学习者嵌入计算知识点掌握程度（模式 2）
        —— 主要用于你描述的“新学习者”流程
        """
        try:
            if not self.ensure_initialized():
                return {"success": False, "error": "引擎初始化失败", "results": {}}

            if len(learner_embeddings) != len(learner_uids):
                return {
                    "success": False,
                    "error": "学习者嵌入数量与 UID 数量不匹配",
                    "total_count": len(learner_embeddings),
                    "success_count": 0,
                    "results": {},
                }

            logger.info("使用提供的嵌入计算知识点掌握程度: %d 个学习者", len(learner_embeddings))

            valid_learner_uids = [
                uid
                for uid in learner_uids
                if self.cd_repository.validate_learner_has_interactions(uid)
            ]
            valid_indices = [i for i, uid in enumerate(learner_uids) if uid in valid_learner_uids]
            valid_embeddings = [learner_embeddings[i] for i in valid_indices]

            if not valid_learner_uids:
                logger.error("没有找到任何有交互记录的学习者")
                return {
                    "success": False,
                    "error": "没有找到任何有交互记录的学习者",
                    "total_count": len(learner_uids),
                    "success_count": 0,
                    "results": {},
                }

            inputs = self._prepare_cd_inputs(valid_learner_uids, valid_embeddings)

            # 新学习者：不做 KT 融合
            self.model.set_kt_optimized_ability(None, unt_num=0)

            with torch.no_grad():
                self.model.eval()
                ability_matrix = self.model.get_ability_matrix(
                    h_lrn_batch=inputs["h_lrn_batch"],
                    h_qus=inputs["h_qus"],
                    h_cpt=inputs["h_cpt"],
                    unt_seq_indices=inputs["qus_seq_indices"],
                    seq_masks=inputs["qus_seq_masks"],
                    unt_num=0,
                )

            results: Dict[str, Any] = {}
            success_count = 0

            for i, learner_uid in enumerate(valid_learner_uids):
                if i < len(ability_matrix):
                    seq_len = (
                        inputs.get("learner_seq_lengths", [])[i]
                        if i < len(inputs.get("learner_seq_lengths", []))
                        else 0
                    )

                    if seq_len > 0:
                        ability_vector = ability_matrix[i, seq_len - 1].cpu().numpy().tolist()
                    else:
                        ability_vector = ability_matrix[i, -1].cpu().numpy().tolist()
                        logger.warning("新学习者 %s 序列长度为 0，使用最后一个时间步", learner_uid)

                    results[learner_uid] = {
                        "concept_mastery_vector": ability_vector,
                        "concept_count": self.concept_num,
                        "timestamp": datetime.now().isoformat(),
                        "is_new_learner": True,
                        "actual_seq_len": seq_len,
                    }
                    success_count += 1

            logger.info("使用嵌入计算知识点掌握程度完成: %d 个学习者", len(results))
            return {
                "success": True,
                "total_count": len(learner_uids),
                "valid_count": len(valid_learner_uids),
                "success_count": success_count,
                "results": results,
            }

        except Exception as exc:
            logger.error("使用嵌入计算知识点掌握程度失败: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "total_count": len(learner_uids),
                "success_count": 0,
                "results": {},
            }

    # ----------------- 统一对外接口 -----------------

    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        统一对外接口：
        - 对于“已有学习者”预测，直接调用该方法即可
        - 内部走模式 1（使用 KT 结果做能力融合）
        """
        return self.compute_multiple_learners_concept_mastery(learner_uids)

    def get_engine_status(self) -> Dict[str, Any]:
        base = super().get_engine_status()
        base.update(
            {
                "concept_count": self.concept_num if self.concept_mapping else 0,
                "embedding_cache_size": len(self.embedding_cache),
            }
        )
        return base


# ----------------- 全局实例 & 便捷函数 -----------------

_cd_engine_instance: Optional[CDEngine] = None


def get_cd_engine() -> CDEngine:
    global _cd_engine_instance
    if _cd_engine_instance is None:
        _cd_engine_instance = CDEngine()
    return _cd_engine_instance


def analyze(learner_uids: List[str]) -> Dict[str, Any]:
    """
    推荐给上层的统一调用接口（CD 引擎）：
    - 输入：学习者 UID 列表
    - 输出：CD 知识点掌握结果（模式 1）
    """
    engine = get_cd_engine()
    return engine.analyze(learner_uids)


def compute_single_learner_concept_mastery(
    learner_uid: str,
) -> Optional[Dict[str, Any]]:
    engine = get_cd_engine()
    return engine.compute_single_learner_concept_mastery(learner_uid)


def compute_multiple_learners_concept_mastery(
    learner_uids: List[str],
) -> Dict[str, Any]:
    engine = get_cd_engine()
    return engine.compute_multiple_learners_concept_mastery(learner_uids)


def compute_concept_mastery_with_embeddings(
    learner_embeddings: List[torch.Tensor],
    learner_uids: List[str],
) -> Dict[str, Any]:
    engine = get_cd_engine()
    return engine.compute_concept_mastery_with_embeddings(learner_embeddings, learner_uids)


def initialize_engine() -> bool:
    engine = get_cd_engine()
    return engine.initialize()


def get_engine_status() -> Dict[str, Any]:
    engine = get_cd_engine()
    return engine.get_engine_status()
