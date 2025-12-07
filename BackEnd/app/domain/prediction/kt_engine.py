# BackEnd/app/domain/prediction/kt_engine.py
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# 触发 DeepLearning 相关路径注册（替代旧的 app.core.config.path_config）
from app.core.settings import path_settings  # noqa: F401

from DeepLearning.Model.KT import KT
from DeepLearning.hyperparams.hyperparameter import hyperparams

from app.domain.common.base_engine import BaseEngine
from app.data_access.prediction.kt_repository import KTRepository
from app.data_access.prediction.embedding_repository import EmbeddingRepository
from app.data_access.prediction.learner_repository import LearnerRepository

logger = logging.getLogger(__name__)


class KTEngine(BaseEngine):
    """
    KT 模型推理引擎 - 知识追踪 & 能力融合

    说明：
    - 保持原 kt_engine.py 的算法流程（概念映射、学习单元序列构造、最后时间步能力向量）
    - 仅将依赖从旧的 app.repositories.* 迁移到新的 data_access 层
    - 增加统一对外接口 analyze(learner_uids: List[str])
    """

    def __init__(self, device: Optional[str] = None) -> None:
        device = device or hyperparams.device
        super().__init__(device=device)

        # 仓库实例
        self.kt_repository = KTRepository()
        self.embedding_repository = EmbeddingRepository()
        self.learner_repository = LearnerRepository()

        # 数据缓存
        self.embedding_cache: Dict[str, Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]] = {}

        # 知识点映射
        self.concept_mapping: Optional[Dict[str, int]] = None  # {cpt_uid: id (1-based)}
        self.concept_uid_order: List[str] = []  # 按 id 排序后的 uid 列表
        self.concept_id_to_uid: Dict[int, str] = {}

        # 学习单元映射
        self.qusunt_mapping: Dict[str, int] = {}  # {unt_uid: index}
        self.qusunt_uid_order: List[str] = []
        self.qusunt_num: int = 0

        # 学习单元类型
        self.unit_types: Dict[str, str] = {}          # {unt_uid: type_str}
        self.unit_type_indices: Dict[str, int] = {}   # {unt_uid: type_idx}
        self.unit_type_num: int = 6                   # 固定 6 种类型

        logger.info("%s 初始化，设备: %s", self.__class__.__name__, self.device)

    # ----------------------------------------------------------------------
    # 初始化相关
    # ----------------------------------------------------------------------
    def initialize(self) -> bool:
        """
        初始化 KT 模型：
        - 知识点映射
        - 学习单元映射、类型
        - 概念映射（单元 -> 知识点索引列表）
        - 模型构建 + 权重加载 + 简单验证
        """
        try:
            if self.is_initialized:
                logger.info("KT 引擎已经初始化")
                return True

            logger.info("开始初始化 KT 引擎")

            # 1. 知识点映射
            self._initialize_concept_mapping()

            # 2. 学习单元映射 & 类型
            self._initialize_qusunt_mapping()

            # 3. 概念映射（供 KT 模型使用）
            concept_mapping_for_model = self._get_concept_mapping_for_model()

            # 4. 初始化 KT 模型
            self._initialize_kt_model(concept_mapping_for_model)

            # 5. 加载权重
            self._load_model_weights()

            # 6. 验证模型
            self._validate_model()

            self.is_initialized = True
            logger.info("KT 引擎初始化完成")
            return True

        except Exception as exc:
            logger.error("KT 引擎初始化失败: %s", exc)
            self.is_initialized = False
            return False

    def _initialize_concept_mapping(self) -> None:
        """
        初始化知识点 UID->ID 映射

        说明：
        - 沿用原 kt_engine 逻辑：从 MySQL Concepts 表读取 uid, id
        - id 从 1 开始，因此后续使用时减一作为索引
        """
        logger.info("初始化知识点映射")

        sql = "SELECT uid, id FROM Concepts ORDER BY id ASC"
        try:
            rows = self.kt_repository.execute_custom_mysql_query(sql)
        except Exception as exc:
            logger.error("获取概念映射失败: %s", exc)
            rows = []

        self.concept_mapping = {row["uid"]: int(row["id"]) for row in rows}
        self.concept_num = len(self.concept_mapping)

        sorted_items = sorted(self.concept_mapping.items(), key=lambda x: x[1])
        self.concept_uid_order = [uid for uid, _ in sorted_items]
        self.concept_id_to_uid = {id_: uid for uid, id_ in self.concept_mapping.items()}

        logger.info("知识点映射初始化完成: %d 个知识点", self.concept_num)

    def _initialize_qusunt_mapping(self) -> None:
        """
        初始化学习单元 UID->索引 + 类型信息
        """
        logger.info("初始化学习单元映射")

        try:
            sql = "SELECT uid, type FROM Units"
            rows = self.kt_repository.execute_custom_mysql_query(sql)
        except Exception as exc:
            logger.error("初始化学习单元映射失败: %s", exc)
            rows = []

        qusunt_uids: List[str] = [row["uid"] for row in rows]
        self.qusunt_uid_order = qusunt_uids
        self.qusunt_num = len(qusunt_uids)
        self.qusunt_mapping = {uid: idx for idx, uid in enumerate(qusunt_uids)}

        # 类型映射
        type_mapping = {
            "video": 0,
            "vr": 1,
            "ar": 2,
            "interact": 3,
            "cooperate": 4,
            "question": 5,
        }
        self.unit_types = {row["uid"]: row.get("type", "unknown") for row in rows}
        self.unit_type_indices = {
            uid: type_mapping.get(tp, 5) for uid, tp in self.unit_types.items()
        }

        logger.info(
            "学习单元映射初始化完成: %d 个学习单元，类型数: %d",
            self.qusunt_num,
            len(set(self.unit_types.values())),
        )

    def _get_concept_mapping_for_model(self) -> Dict[int, List[int]]:
        """
        构建 KT 模型需要的概念映射：

        返回：
            {qusunt_idx: [cpt_idx1, cpt_idx2, ...]}
        """
        logger.info("构建模型用概念映射")

        if not self.concept_mapping or not self.qusunt_mapping:
            logger.warning("概念映射或学习单元映射为空，返回空映射")
            return {}

        try:
            sql = "SELECT unt_uid, cpt_uid FROM Unit_Concept"
            rows = self.kt_repository.execute_custom_mysql_query(sql)
        except Exception as exc:
            logger.error("获取 Unit_Concept 失败: %s", exc)
            return {}

        concept_mapping: Dict[int, List[int]] = {}

        for row in rows:
            unt_uid = row["unt_uid"]
            cpt_uid = row["cpt_uid"]

            if unt_uid in self.qusunt_mapping and cpt_uid in self.concept_mapping:
                unt_idx = self.qusunt_mapping[unt_uid]
                cpt_idx = self.concept_mapping[cpt_uid] - 1  # 概念 id 从 1 开始

                concept_mapping.setdefault(unt_idx, []).append(cpt_idx)

        logger.info("概念映射构建完成: %d 个学习单元有知识点映射", len(concept_mapping))
        return concept_mapping

    def _initialize_kt_model(self, concept_mapping: Dict[int, List[int]]) -> None:
        """
        初始化 KT 模型实例
        """
        logger.info("初始化 KT 模型")

        embedding_dim = hyperparams.hgc_embedding_dim
        concept_num = self.concept_num

        self.model = KT(
            embedding_dim=embedding_dim,
            concept_num=concept_num,
            concept_mapping=concept_mapping,
        ).to(self.device)

        logger.info("KT 模型初始化完成: embedding_dim=%d, concept_num=%d", embedding_dim, concept_num)

    def _load_model_weights(self) -> None:
        """
        加载训练好的 KT 模型权重
        """
        logger.info("加载 KT 模型权重")

        save_dir = hyperparams.train_save_dir
        final_dir = os.path.join(save_dir, "final_models")
        kt_path = os.path.join(final_dir, "kt_best_model.pth")

        if not os.path.exists(kt_path):
            logger.warning("模型权重文件不存在: %s，使用随机初始化的模型", kt_path)
            return

        try:
            checkpoint = torch.load(kt_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                logger.info("加载完整检查点，epoch: %s", checkpoint.get("epoch", "unknown"))
            else:
                state_dict = checkpoint

            load_result = self.model.load_state_dict(state_dict, strict=True)
            logger.info("KT 模型权重加载成功")

            if load_result.missing_keys:
                logger.warning("缺失键: %s", load_result.missing_keys)
            if load_result.unexpected_keys:
                logger.warning("意外键: %s", load_result.unexpected_keys)

        except Exception as exc:
            logger.error("加载模型权重失败: %s", exc)
            logger.warning("使用随机初始化的模型")

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("KT 模型参数总数: %d", total_params)

    def _validate_model(self) -> bool:
        """
        验证模型能正常前向推理
        """
        logger.info("验证 KT 模型推理能力")

        try:
            batch_size = 2
            seq_len = 5
            embedding_dim = hyperparams.hgc_embedding_dim

            h_lrn_batch = torch.randn(batch_size, embedding_dim, device=self.device)
            h_qusunt_batch = torch.randn(batch_size, seq_len, embedding_dim, device=self.device)
            h_cpt = torch.randn(self.concept_num, embedding_dim, device=self.device)

            lrn_indices = torch.arange(batch_size, device=self.device)
            qusunt_seq_indices = torch.randint(0, max(1, self.qusunt_num), (batch_size, seq_len), device=self.device)
            add1 = torch.randn(batch_size, seq_len, device=self.device)
            add2 = torch.randn(batch_size, seq_len, device=self.device)
            type_indices = torch.randint(0, self.unit_type_num, (batch_size, seq_len), device=self.device)
            seq_masks = torch.ones(batch_size, seq_len, device=self.device)
            prediction_masks = torch.ones(batch_size, seq_len, device=self.device)

            with torch.no_grad():
                self.model.eval()
                ability_matrix = self.model.get_concept_mastery(
                    h_lrn_batch=h_lrn_batch,
                    h_qusunt_batch=h_qusunt_batch,
                    h_cpt=h_cpt,
                    lrn_indices=lrn_indices,
                    qusunt_seq_indices=qusunt_seq_indices,
                    add1=add1,
                    add2=add2,
                    type_indices=type_indices,
                    seq_mask=seq_masks,
                    qus_num=0,
                )

            logger.info("模型验证成功，能力矩阵形状: %s", tuple(ability_matrix.shape))
            return True

        except Exception as exc:
            logger.error("模型验证失败: %s", exc)
            return False

    # ----------------------------------------------------------------------
    # 嵌入 & 能力矩阵辅助
    # ----------------------------------------------------------------------
    def _get_embeddings(
        self,
        learner_uid: Optional[str] = None,
        learner_uids: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        从数据库获取学习单元 & 知识点嵌入（学习者嵌入在 _prepare_kt_inputs 中单独处理）

        返回:
            (None, h_qusunt [unt_num, dim], h_cpt [concept_num, dim])
        """
        cache_key = "embeddings_all"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        logger.info("从数据库获取学习单元 / 知识点嵌入")

        try:
            qusunt_embeddings = self.embedding_repository.get_embeddings_by_entity_type("unt")

            h_qusunt_list: List[np.ndarray] = []
            for q_emb in qusunt_embeddings:
                if q_emb and q_emb.get("embedding"):
                    h_qusunt_list.append(np.array(q_emb["embedding"]))
                else:
                    embedding_dim = hyperparams.hgc_embedding_dim
                    h_qusunt_list.append(np.zeros(embedding_dim))

            concept_embeddings = self.embedding_repository.get_embeddings_by_entity_type("cpt")
            concept_emb_dict = {
                emb["uid"]: np.array(emb["embedding"]) for emb in concept_embeddings if emb.get("embedding")
            }

            h_cpt_list: List[np.ndarray] = []
            for concept_uid in self.concept_uid_order:
                if concept_uid in concept_emb_dict:
                    h_cpt_list.append(concept_emb_dict[concept_uid])
                else:
                    embedding_dim = hyperparams.hgc_embedding_dim
                    h_cpt_list.append(np.zeros(embedding_dim))
                    logger.warning("知识点 %s 的嵌入未找到，使用零向量", concept_uid)

            h_qusunt = torch.tensor(np.array(h_qusunt_list), dtype=torch.float32, device=self.device)
            h_cpt = torch.tensor(np.array(h_cpt_list), dtype=torch.float32, device=self.device)

            h_lrn = None
            result = (h_lrn, h_qusunt, h_cpt)
            self.embedding_cache[cache_key] = result

            logger.info("嵌入加载完成: 学习单元=%d, 知识点=%d", len(h_qusunt_list), len(h_cpt_list))
            return result

        except Exception as exc:
            logger.error("获取嵌入向量失败: %s", exc)
            embedding_dim = hyperparams.hgc_embedding_dim
            h_qusunt = torch.zeros(1, embedding_dim, device=self.device)
            h_cpt = torch.zeros(self.concept_num, embedding_dim, device=self.device)
            return (None, h_qusunt, h_cpt)

    def _simulate_cd_ability(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        模拟 CD 能力矩阵（保持原逻辑，仅用于当前版本占位）

        返回:
            [batch_size, seq_len, concept_num]
        """
        cd_ability = torch.rand(batch_size, seq_len, self.concept_num, device=self.device)

        for i in range(batch_size):
            if self.concept_num <= 0:
                continue
            high_mastery_concepts = torch.randint(0, self.concept_num, (3,), device=self.device)
            cd_ability[i, :, high_mastery_concepts] = 0.8 + 0.2 * torch.rand(
                seq_len, 3, device=self.device
            )

        logger.debug("模拟 CD 能力矩阵: shape=%s", tuple(cd_ability.shape))
        return cd_ability

    # ----------------------------------------------------------------------
    # 输入构建
    # ----------------------------------------------------------------------
    def _prepare_kt_inputs(
        self,
        learner_uids: List[str],
        learner_embeddings: Optional[List[torch.Tensor]] = None,
        max_seq_len: int = 50,
    ) -> Dict[str, Any]:
        """
        准备 KT 模型输入数据

        learner_embeddings:
            - None: 模式 1（已有学习者，从 Embeddings 集合获取学习者嵌入）
            - 非 None: 模式 2（新学习者，使用上游 HGC/CD 计算好的嵌入）
        """
        if not learner_uids:
            raise ValueError("learner_uids 不能为空")

        # 学习单元 & 知识点嵌入
        _, h_qusunt, h_cpt = self._get_embeddings()

        # 批量交互序列
        sequences_data = self.kt_repository.get_learner_interaction_sequences_batch(
            learner_uids, max_seq_len
        )

        batch_size = len(learner_uids)
        actual_max_seq_len = 0
        for data in sequences_data.values():
            actual_max_seq_len = max(actual_max_seq_len, data["seq_len"])

        if actual_max_seq_len == 0:
            raise ValueError("没有有效的交互序列数据")

        effective_max_seq_len = (
            min(actual_max_seq_len, max_seq_len) if max_seq_len else actual_max_seq_len
        )

        # 序列张量
        qusunt_seq_indices = torch.zeros(
            batch_size, effective_max_seq_len, dtype=torch.long, device=self.device
        )
        add1_seq = torch.zeros(
            batch_size, effective_max_seq_len, dtype=torch.float32, device=self.device
        )
        add2_seq = torch.zeros(
            batch_size, effective_max_seq_len, dtype=torch.float32, device=self.device
        )
        type_indices_seq = torch.zeros(
            batch_size, effective_max_seq_len, dtype=torch.long, device=self.device
        )
        seq_masks = torch.zeros(
            batch_size, effective_max_seq_len, dtype=torch.float32, device=self.device
        )
        prediction_masks_seq = torch.zeros(
            batch_size, effective_max_seq_len, dtype=torch.float32, device=self.device
        )

        learner_seq_lengths: List[int] = []

        # 学习者嵌入
        if learner_embeddings is not None:
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
                    logger.warning("学习者 %s 的嵌入未找到，使用零向量", learner_uid)
                h_lrn_batch_list.append(h_lrn)
            h_lrn_batch = torch.stack(h_lrn_batch_list)

        valid_learners = 0

        for i, learner_uid in enumerate(learner_uids):
            seq_info = sequences_data.get(learner_uid)
            if not seq_info or seq_info["seq_len"] == 0:
                learner_seq_lengths.append(0)
                continue

            seq = seq_info["sequence"]
            (
                unt_uids,
                add1s,
                add2s,
                is_questions,
                results,
                prediction_masks,
                next_results,
            ) = seq

            seq_len = min(len(unt_uids), effective_max_seq_len)
            learner_seq_lengths.append(seq_len)

            for j in range(seq_len):
                unt_uid = unt_uids[j]
                qusunt_idx = self.qusunt_mapping.get(unt_uid, 0)

                qusunt_seq_indices[i, j] = qusunt_idx
                add1_seq[i, j] = float(add1s[j])
                add2_seq[i, j] = float(add2s[j])
                type_indices_seq[i, j] = self.unit_type_indices.get(unt_uid, 5)
                prediction_masks_seq[i, j] = float(prediction_masks[j])
                seq_masks[i, j] = 1.0

            valid_learners += 1

        if valid_learners == 0:
            raise ValueError("没有有效的学习者序列数据")

        # 当前版本仍然使用模拟的 CD 能力矩阵（保持原逻辑）
        simulated_cd_ability = self._simulate_cd_ability(
            batch_size, effective_max_seq_len
        )

        inputs: Dict[str, Any] = {
            "h_lrn_batch": h_lrn_batch,
            "h_qusunt": h_qusunt,
            "h_cpt": h_cpt,
            "lrn_indices": torch.arange(batch_size, device=self.device),
            "qusunt_seq_indices": qusunt_seq_indices,
            "add1": add1_seq,
            "add2": add2_seq,
            "type_indices": type_indices_seq,
            "seq_masks": seq_masks,
            "prediction_masks": prediction_masks_seq,
            "cd_ability": simulated_cd_ability,
            "effective_max_seq_len": effective_max_seq_len,
            "learner_seq_lengths": learner_seq_lengths,
        }

        logger.info(
            "KT 输入数据准备完成: 批次大小=%d, 有效学习者=%d, 序列长度=%d",
            batch_size,
            valid_learners,
            effective_max_seq_len,
        )

        return inputs

    # ----------------------------------------------------------------------
    # 结果格式化
    # ----------------------------------------------------------------------
    def _format_kt_results(
        self,
        ability_matrix: torch.Tensor,
        learner_uids: List[str],
        learner_seq_lengths: List[int],
    ) -> List[Dict[str, Any]]:
        """
        格式化 KT 输出为 {learner_id, concept_mastery} 结构
        """
        results: List[Dict[str, Any]] = []

        for i, learner_uid in enumerate(learner_uids):
            if i >= len(ability_matrix):
                continue

            seq_len = learner_seq_lengths[i] if i < len(learner_seq_lengths) else 0

            if seq_len > 0:
                ability_vector = ability_matrix[i, seq_len - 1].cpu().numpy()
            else:
                ability_vector = ability_matrix[i, -1].cpu().numpy()
                logger.warning("学习者 %s 序列长度为 0，使用最后一个时间步", learner_uid)

            concept_mastery_dict: Dict[str, float] = {}
            for concept_idx in range(self.concept_num):
                concept_uid = self.concept_uid_order[concept_idx]
                mastery_value = float(ability_vector[concept_idx])
                concept_mastery_dict[concept_uid] = mastery_value

            results.append(
                {
                    "learner_id": learner_uid,
                    "concept_mastery": concept_mastery_dict,
                }
            )

        return results

    # ----------------------------------------------------------------------
    # 推理：模式 1（已有学习者）
    # ----------------------------------------------------------------------
    def compute_single_learner_concept_mastery(
        self, learner_uid: str
    ) -> Optional[Dict[str, Any]]:
        """
        计算单个学习者的知识点掌握程度（模式 1）

        - 使用数据库中的 KT 序列
        - 使用嵌入仓库中的学习者向量
        - 使用模拟的 CD 能力矩阵进行能力融合（保持原逻辑）
        """
        try:
            if not self.ensure_initialized():
                return None

            logger.info("计算单个学习者知识点掌握程度: %s", learner_uid)

            if not self.kt_repository.validate_learner_has_interactions(learner_uid):
                logger.error("学习者 %s 没有交互记录", learner_uid)
                return None

            inputs = self._prepare_kt_inputs([learner_uid])

            if inputs["cd_ability"] is not None:
                logger.info("设置模拟 CD 能力进行能力融合")
                self.model.set_cd_optimized_ability(inputs["cd_ability"], qus_num=0)

            batch_size, seq_len = inputs["qusunt_seq_indices"].shape
            embedding_dim = inputs["h_qusunt"].shape[1]

            qusunt_indices_flat = inputs["qusunt_seq_indices"].view(-1)
            h_qusunt_batch = inputs["h_qusunt"][qusunt_indices_flat].view(
                batch_size, seq_len, embedding_dim
            )

            with torch.no_grad():
                self.model.eval()
                concept_mastery = self.model.get_concept_mastery(
                    h_lrn_batch=inputs["h_lrn_batch"],
                    h_qusunt_batch=h_qusunt_batch,
                    h_cpt=inputs["h_cpt"],
                    lrn_indices=inputs["lrn_indices"],
                    qusunt_seq_indices=inputs["qusunt_seq_indices"],
                    add1=inputs["add1"],
                    add2=inputs["add2"],
                    type_indices=inputs["type_indices"],
                    seq_mask=inputs["seq_masks"],
                    qus_num=0,
                )

            formatted = self._format_kt_results(
                concept_mastery,
                [learner_uid],
                inputs["learner_seq_lengths"],
            )

            if not formatted:
                return None

            result = formatted[0]
            result.update(
                {
                    "concept_count": self.concept_num,
                    "sequence_length": inputs["learner_seq_lengths"][0],
                    "timestamp": datetime.now().isoformat(),
                }
            )

            logger.info("单个学习者知识点掌握程度计算完成")
            return result

        except Exception as exc:
            logger.error("计算单个学习者知识点掌握程度失败 %s: %s", learner_uid, exc)
            return None

    def compute_multiple_learners_concept_mastery(
        self, learner_uids: List[str]
    ) -> Dict[str, Any]:
        """
        计算多个学习者的知识点掌握程度（模式 1）
        """
        try:
            if not self.ensure_initialized():
                return {
                    "success": False,
                    "error": "引擎初始化失败",
                    "results": [],
                    "total_count": len(learner_uids),
                    "success_count": 0,
                }

            logger.info("计算多个学习者知识点掌握程度: %d 个学习者", len(learner_uids))

            if not learner_uids:
                return {
                    "success": True,
                    "results": [],
                    "total_count": 0,
                    "success_count": 0,
                }

            valid_learner_uids = [
                uid
                for uid in learner_uids
                if self.kt_repository.validate_learner_has_interactions(uid)
            ]

            if not valid_learner_uids:
                logger.error("没有找到任何有交互记录的学习者")
                return {
                    "success": False,
                    "error": "没有找到任何有交互记录的学习者",
                    "total_count": len(learner_uids),
                    "success_count": 0,
                    "results": [],
                }

            inputs = self._prepare_kt_inputs(valid_learner_uids)

            if inputs["cd_ability"] is not None:
                logger.info("设置模拟 CD 能力进行能力融合")
                self.model.set_cd_optimized_ability(inputs["cd_ability"], qus_num=0)

            batch_size, seq_len = inputs["qusunt_seq_indices"].shape
            embedding_dim = inputs["h_qusunt"].shape[1]

            qusunt_indices_flat = inputs["qusunt_seq_indices"].view(-1)
            h_qusunt_batch = inputs["h_qusunt"][qusunt_indices_flat].view(
                batch_size, seq_len, embedding_dim
            )

            with torch.no_grad():
                self.model.eval()
                concept_mastery = self.model.get_concept_mastery(
                    h_lrn_batch=inputs["h_lrn_batch"],
                    h_qusunt_batch=h_qusunt_batch,
                    h_cpt=inputs["h_cpt"],
                    lrn_indices=inputs["lrn_indices"],
                    qusunt_seq_indices=inputs["qusunt_seq_indices"],
                    add1=inputs["add1"],
                    add2=inputs["add2"],
                    type_indices=inputs["type_indices"],
                    seq_mask=inputs["seq_masks"],
                    qus_num=0,
                )

            formatted_results = self._format_kt_results(
                concept_mastery,
                valid_learner_uids,
                inputs["learner_seq_lengths"],
            )

            logger.info("多个学习者知识点掌握程度计算完成: %d 成功", len(formatted_results))
            return {
                "success": True,
                "total_count": len(learner_uids),
                "valid_count": len(valid_learner_uids),
                "success_count": len(formatted_results),
                "results": formatted_results,
            }

        except Exception as exc:
            logger.error("计算多个学习者知识点掌握程度失败: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "total_count": len(learner_uids),
                "success_count": 0,
                "results": [],
            }

    # ----------------------------------------------------------------------
    # 推理：模式 2（新学习者，使用外部给定嵌入）
    # ----------------------------------------------------------------------
    def compute_concept_mastery_with_embeddings(
        self,
        learner_embeddings: List[torch.Tensor],
        learner_uids: List[str],
    ) -> Dict[str, Any]:
        """
        使用给定的学习者嵌入计算知识点掌握程度（模式 2）
        """
        try:
            if not self.ensure_initialized():
                return {
                    "success": False,
                    "error": "引擎初始化失败",
                    "total_count": len(learner_uids),
                    "success_count": 0,
                    "results": [],
                }

            if len(learner_embeddings) != len(learner_uids):
                return {
                    "success": False,
                    "error": "学习者嵌入数量与 UID 数量不匹配",
                    "total_count": len(learner_uids),
                    "success_count": 0,
                    "results": [],
                }

            logger.info(
                "使用提供嵌入计算知识点掌握程度: %d 个学习者", len(learner_embeddings)
            )

            valid_learner_uids = [
                uid
                for uid in learner_uids
                if self.kt_repository.validate_learner_has_interactions(uid)
            ]
            valid_indices = [
                i for i, uid in enumerate(learner_uids) if uid in valid_learner_uids
            ]
            valid_embeddings = [learner_embeddings[i] for i in valid_indices]

            if not valid_learner_uids:
                logger.error("没有找到任何有交互记录的学习者")
                return {
                    "success": False,
                    "error": "没有找到任何有交互记录的学习者",
                    "total_count": len(learner_uids),
                    "success_count": 0,
                    "results": [],
                }

            inputs = self._prepare_kt_inputs(valid_learner_uids, valid_embeddings)

            # 新学习者，不做 CD 能力融合
            self.model.set_cd_optimized_ability(None, qus_num=0)

            batch_size, seq_len = inputs["qusunt_seq_indices"].shape
            embedding_dim = inputs["h_qusunt"].shape[1]

            qusunt_indices_flat = inputs["qusunt_seq_indices"].view(-1)
            h_qusunt_batch = inputs["h_qusunt"][qusunt_indices_flat].view(
                batch_size, seq_len, embedding_dim
            )

            with torch.no_grad():
                self.model.eval()
                concept_mastery = self.model.get_concept_mastery(
                    h_lrn_batch=inputs["h_lrn_batch"],
                    h_qusunt_batch=h_qusunt_batch,
                    h_cpt=inputs["h_cpt"],
                    lrn_indices=inputs["lrn_indices"],
                    qusunt_seq_indices=inputs["qusunt_seq_indices"],
                    add1=inputs["add1"],
                    add2=inputs["add2"],
                    type_indices=inputs["type_indices"],
                    seq_mask=inputs["seq_masks"],
                    qus_num=0,
                )

            formatted_results = self._format_kt_results(
                concept_mastery,
                valid_learner_uids,
                inputs["learner_seq_lengths"],
            )

            logger.info("使用嵌入计算知识点掌握程度完成: %d 个学习者", len(formatted_results))
            return {
                "success": True,
                "total_count": len(learner_uids),
                "valid_count": len(valid_learner_uids),
                "success_count": len(formatted_results),
                "results": formatted_results,
            }

        except Exception as exc:
            logger.error("使用嵌入计算知识点掌握程度失败: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "total_count": len(learner_uids),
                "success_count": 0,
                "results": [],
            }

    # ----------------------------------------------------------------------
    # BaseEngine 接口实现
    # ----------------------------------------------------------------------
    def analyze(self, learner_uids: List[str]) -> Dict[str, Any]:
        """
        统一对外接口：
        - 已有学习者：直接调用该接口
        - 内部走模式 1（使用数据库中的交互序列 & 嵌入）
        """
        return self.compute_multiple_learners_concept_mastery(learner_uids)

    def get_engine_status(self) -> Dict[str, Any]:
        base = super().get_engine_status()
        base.update(
            {
                "concept_count": self.concept_num,
                "qusunt_num": self.qusunt_num,
                "embedding_cache_size": len(self.embedding_cache),
            }
        )
        return base


# ----------------------------------------------------------------------
# 单例封装 + 便捷函数
# ----------------------------------------------------------------------
_kt_engine_instance: Optional[KTEngine] = None


def get_kt_engine() -> KTEngine:
    global _kt_engine_instance
    if _kt_engine_instance is None:
        _kt_engine_instance = KTEngine()
    return _kt_engine_instance


def analyze(learner_uids: List[str]) -> Dict[str, Any]:
    engine = get_kt_engine()
    return engine.analyze(learner_uids)


def compute_single_learner_concept_mastery(
    learner_uid: str,
) -> Optional[Dict[str, Any]]:
    engine = get_kt_engine()
    return engine.compute_single_learner_concept_mastery(learner_uid)


def compute_multiple_learners_concept_mastery(
    learner_uids: List[str],
) -> Dict[str, Any]:
    engine = get_kt_engine()
    return engine.compute_multiple_learners_concept_mastery(learner_uids)


def compute_concept_mastery_with_embeddings(
    learner_embeddings: List[torch.Tensor],
    learner_uids: List[str],
) -> Dict[str, Any]:
    engine = get_kt_engine()
    return engine.compute_concept_mastery_with_embeddings(learner_embeddings, learner_uids)


def initialize_engine() -> bool:
    engine = get_kt_engine()
    return engine.initialize()


def get_engine_status() -> Dict[str, Any]:
    engine = get_kt_engine()
    return engine.get_engine_status()
