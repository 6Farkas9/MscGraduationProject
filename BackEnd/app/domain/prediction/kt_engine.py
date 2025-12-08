import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

# 触发 DeepLearning 相关路径注册 + 读取 KT 配置
from app.core.settings import path_settings, kt_settings  # noqa: F401

from DeepLearning.Model.KT import KT
from DeepLearning.hyperparams.hyperparameter import hyperparams

from app.domain.common.base_engine import BaseEngine
from app.data_access.prediction.kt_repository import KTRepository
from app.data_access.prediction.embedding_repository import EmbeddingRepository
from app.data_access.prediction.learner_repository import LearnerRepository

logger = logging.getLogger(__name__)


class KTEngine(BaseEngine):
    """
    KT 模型推理引擎 - 知识追踪 & CD 能力融合（t0 初始化语义）

    本次修正：
    1. cd_ability 只需要 CD 的最终能力（初始化值）：
        - 允许 2D: [batch, concept_num]
        - 兼容 3D: [batch, seq_len, concept_num]
    2. 若传入 2D（或 3D 且 seq_len=1），引擎内部自动广播到有效序列长度 T，
       形成常量能力轨迹 [batch, T, concept_num] 喂给模型。
       这等价于“t0 初始化值在全程参与融合”，与模型融合实现一致。
    3. 仍然保持“必须进行能力融合”：cd_ability 不传入直接报错。
    4. 输出：最后一个有效时间步 mastery + 最近 K 个有效时间步轨迹
    """

    def __init__(self, device: Optional[str] = None) -> None:
        device = device or hyperparams.device
        super().__init__(device=device)

        self.kt_repository = KTRepository()
        self.embedding_repository = EmbeddingRepository()
        self.learner_repository = LearnerRepository()

        self.embedding_cache: Dict[str, Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]] = {}

        self.concept_mapping: Optional[Dict[str, int]] = None  # {cpt_uid: id (1-based)}
        self.concept_uid_order: List[str] = []
        self.concept_id_to_uid: Dict[int, str] = {}
        self.concept_num: int = 0

        self.qusunt_mapping: Dict[str, int] = {}
        self.qusunt_uid_order: List[str] = []
        self.qusunt_num: int = 0

        self.unit_types: Dict[str, str] = {}
        self.unit_type_indices: Dict[str, int] = {}
        self.unit_type_num: int = 6

        logger.info("%s 初始化，设备: %s", self.__class__.__name__, self.device)

    # ----------------------------------------------------------------------
    # 初始化相关
    # ----------------------------------------------------------------------
    def initialize(self) -> bool:
        try:
            if self.is_initialized:
                logger.info("KT 引擎已经初始化")
                return True

            logger.info("开始初始化 KT 引擎")

            self._initialize_concept_mapping()
            self._initialize_qusunt_mapping()

            concept_mapping_for_model = self._get_concept_mapping_for_model()
            self._initialize_kt_model(concept_mapping_for_model)

            self._load_model_weights()
            self._validate_model()

            self.is_initialized = True
            logger.info("KT 引擎初始化完成")
            return True

        except Exception as exc:
            logger.error("KT 引擎初始化失败: %s", exc, exc_info=True)
            self.is_initialized = False
            return False

    def _initialize_concept_mapping(self) -> None:
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
                cpt_idx = self.concept_mapping[cpt_uid] - 1
                concept_mapping.setdefault(unt_idx, []).append(cpt_idx)

        logger.info("概念映射构建完成: %d 个学习单元有知识点映射", len(concept_mapping))
        return concept_mapping

    def _initialize_kt_model(self, concept_mapping: Dict[int, List[int]]) -> None:
        logger.info("初始化 KT 模型")
        embedding_dim = hyperparams.hgc_embedding_dim
        self.model = KT(
            embedding_dim=embedding_dim,
            concept_num=self.concept_num,
            concept_mapping=concept_mapping,
        ).to(self.device)
        logger.info(
            "KT 模型初始化完成: embedding_dim=%d, concept_num=%d",
            embedding_dim, self.concept_num
        )

    def _load_model_weights(self) -> None:
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
            logger.error("加载模型权重失败: %s", exc, exc_info=True)
            logger.warning("使用随机初始化的模型")

    def _validate_model(self) -> bool:
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
            logger.error("模型验证失败: %s", exc, exc_info=True)
            return False

    # ----------------------------------------------------------------------
    # 嵌入获取
    # ----------------------------------------------------------------------
    def _get_embeddings(
        self,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
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
                    h_qusunt_list.append(np.zeros(hyperparams.hgc_embedding_dim))

            concept_embeddings = self.embedding_repository.get_embeddings_by_entity_type("cpt")
            concept_emb_dict = {
                emb["uid"]: np.array(emb["embedding"])
                for emb in concept_embeddings if emb.get("embedding")
            }

            h_cpt_list: List[np.ndarray] = []
            for concept_uid in self.concept_uid_order:
                if concept_uid in concept_emb_dict:
                    h_cpt_list.append(concept_emb_dict[concept_uid])
                else:
                    h_cpt_list.append(np.zeros(hyperparams.hgc_embedding_dim))
                    logger.warning("知识点 %s 的嵌入未找到，使用零向量", concept_uid)

            h_qusunt = torch.tensor(np.array(h_qusunt_list), dtype=torch.float32, device=self.device)
            h_cpt = torch.tensor(np.array(h_cpt_list), dtype=torch.float32, device=self.device)

            result = (None, h_qusunt, h_cpt)
            self.embedding_cache[cache_key] = result
            logger.info("嵌入加载完成: 学习单元=%d, 知识点=%d", len(h_qusunt_list), len(h_cpt_list))
            return result

        except Exception as exc:
            logger.error("获取嵌入向量失败: %s", exc, exc_info=True)
            embedding_dim = hyperparams.hgc_embedding_dim
            return (
                None,
                torch.zeros(1, embedding_dim, device=self.device),
                torch.zeros(self.concept_num, embedding_dim, device=self.device),
            )

    # ----------------------------------------------------------------------
    # 输入构建（CD ability 只需最终结果）
    # ----------------------------------------------------------------------
    def _prepare_kt_inputs(
        self,
        learner_uids: List[str],
        learner_embeddings: Optional[List[torch.Tensor]] = None,
        cd_ability: Optional[torch.Tensor] = None,
        max_seq_len: int = 50,
    ) -> Dict[str, Any]:
        if not learner_uids:
            raise ValueError("learner_uids 不能为空")

        if cd_ability is None:
            raise ValueError("cd_ability 未传入，KT 不能进行能力融合")

        _, h_qusunt, h_cpt = self._get_embeddings()

        sequences_data = self.kt_repository.get_learner_interaction_sequences_batch(
            learner_uids, max_seq_len
        )

        batch_size = len(learner_uids)
        actual_max_seq_len = 0
        for data in sequences_data.values():
            actual_max_seq_len = max(actual_max_seq_len, data["seq_len"])

        if actual_max_seq_len == 0:
            raise ValueError("没有有效的交互序列数据")

        effective_max_seq_len = min(actual_max_seq_len, max_seq_len) if max_seq_len else actual_max_seq_len

        # ---------------- cd_ability 维度校验 + 广播成常量轨迹 ----------------
        if cd_ability.dim() == 2:
            # [B, C] -> [B, T, C]
            if cd_ability.size(0) != batch_size:
                raise ValueError(
                    f"cd_ability batch 维不匹配: cd={cd_ability.size(0)} vs batch={batch_size}"
                )
            if cd_ability.size(1) != self.concept_num:
                raise ValueError(
                    f"cd_ability concept 维不匹配: cd={cd_ability.size(1)} vs kt={self.concept_num}"
                )
            cd_ability_eff = cd_ability.unsqueeze(1).repeat(1, effective_max_seq_len, 1).to(self.device)

        elif cd_ability.dim() == 3:
            if cd_ability.size(0) != batch_size:
                raise ValueError(
                    f"cd_ability batch 维不匹配: cd={cd_ability.size(0)} vs batch={batch_size}"
                )
            if cd_ability.size(2) != self.concept_num:
                raise ValueError(
                    f"cd_ability concept 维不匹配: cd={cd_ability.size(2)} vs kt={self.concept_num}"
                )

            if cd_ability.size(1) == 1 and effective_max_seq_len > 1:
                # [B,1,C] -> [B,T,C]
                cd_ability_eff = cd_ability.repeat(1, effective_max_seq_len, 1).to(self.device)
            else:
                # 允许外部已经给了逐步能力（兼容旧行为）
                if cd_ability.size(1) < effective_max_seq_len:
                    raise ValueError(
                        f"cd_ability seq_len 太短: cd={cd_ability.size(1)} < kt_seq={effective_max_seq_len}"
                    )
                cd_ability_eff = cd_ability[:, :effective_max_seq_len, :].to(self.device)
        else:
            raise ValueError(f"cd_ability 维度必须为 2 或 3，但得到 {cd_ability.dim()}")

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
                    h_lrn = torch.zeros(hyperparams.hgc_embedding_dim, device=self.device)
                    logger.warning("学习者 %s 的嵌入未找到，使用零向量", learner_uid)
                h_lrn_batch_list.append(h_lrn)
            h_lrn_batch = torch.stack(h_lrn_batch_list)

        valid_learners = 0
        for i, learner_uid in enumerate(learner_uids):
            seq_info = sequences_data.get(learner_uid)
            if not seq_info or seq_info["seq_len"] == 0:
                learner_seq_lengths.append(0)
                continue

            unt_uids, add1s, add2s, _, _, prediction_masks, _ = seq_info["sequence"]

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
            "cd_ability": cd_ability_eff,
            "effective_max_seq_len": effective_max_seq_len,
            "learner_seq_lengths": learner_seq_lengths,
        }

        logger.info(
            "KT 输入数据准备完成: 批次大小=%d, 有效学习者=%d, 序列长度=%d",
            batch_size, valid_learners, effective_max_seq_len
        )
        return inputs

    # ----------------------------------------------------------------------
    # 输出格式化（最后一步 + 最后K个有效时间步）
    # ----------------------------------------------------------------------
    def _format_kt_results(
        self,
        ability_matrix: torch.Tensor,
        learner_uids: List[str],
        learner_seq_lengths: List[int],
        seq_masks: torch.Tensor,
        last_k_steps: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        k = max(1, int(last_k_steps))

        batch_size, seq_len, _ = ability_matrix.shape

        for i, learner_uid in enumerate(learner_uids):
            if i >= batch_size:
                continue

            seq_len_recorded = learner_seq_lengths[i] if i < len(learner_seq_lengths) else 0

            # 最后一个有效时间步
            last_valid_idx = -1
            for t in range(seq_len - 1, -1, -1):
                if seq_masks[i, t] > 0.5:
                    last_valid_idx = t
                    break
            if last_valid_idx < 0:
                last_valid_idx = seq_len - 1
                logger.warning("学习者 %s 没有有效的 seq_mask，使用最后一个时间步", learner_uid)

            ability_vector_last = ability_matrix[i, last_valid_idx].cpu().numpy()
            concept_mastery_last: Dict[str, float] = {
                self.concept_uid_order[c_idx]: float(ability_vector_last[c_idx])
                for c_idx in range(self.concept_num)
            }

            # 最后 K 个有效时间步
            collected_indices: List[int] = []
            for t in range(last_valid_idx, -1, -1):
                if seq_masks[i, t] > 0.5:
                    collected_indices.append(t)
                    if len(collected_indices) >= k:
                        break
            collected_indices = list(reversed(collected_indices))

            last_k_list: List[Dict[str, Any]] = []
            for t in collected_indices:
                vec_t = ability_matrix[i, t].cpu().numpy()
                cm_t = {
                    self.concept_uid_order[c_idx]: float(vec_t[c_idx])
                    for c_idx in range(self.concept_num)
                }
                last_k_list.append({"step_index": int(t), "concept_mastery": cm_t})

            results.append(
                {
                    "learner_id": learner_uid,
                    "sequence_length": int(seq_len_recorded),
                    "concept_mastery": concept_mastery_last,
                    "concept_mastery_last_k": last_k_list,
                }
            )

        return results

    # ----------------------------------------------------------------------
    # 推理：模式 1（已有学习者）
    # ----------------------------------------------------------------------
    def compute_single_learner_concept_mastery(
        self,
        learner_uid: str,
        cd_ability: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not self.ensure_initialized():
                return None

            logger.info("计算单个学习者知识点掌握程度: %s", learner_uid)

            if not self.kt_repository.validate_learner_has_interactions(learner_uid):
                logger.error("学习者 %s 没有交互记录", learner_uid)
                return None

            inputs = self._prepare_kt_inputs([learner_uid], cd_ability=cd_ability)

            # t0 初始化能力（常量轨迹），模型内部按每步门控使用
            self.model.set_cd_optimized_ability(inputs["cd_ability"], qus_num=0)

            batch_size, seq_len = inputs["qusunt_seq_indices"].shape
            embedding_dim = inputs["h_qusunt"].shape[1]
            qusunt_indices_flat = inputs["qusunt_seq_indices"].view(-1)
            h_qusunt_batch = inputs["h_qusunt"][qusunt_indices_flat].view(
                batch_size, seq_len, embedding_dim
            )

            with torch.no_grad():
                self.model.eval()
                ability_matrix = self.model.get_concept_mastery(
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
                ability_matrix=ability_matrix,
                learner_uids=[learner_uid],
                learner_seq_lengths=inputs["learner_seq_lengths"],
                seq_masks=inputs["seq_masks"],
                last_k_steps=kt_settings.history_steps,
            )
            if not formatted:
                return None

            result = formatted[0]
            result.update(
                {
                    "concept_count": self.concept_num,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return result

        except Exception as exc:
            logger.error("计算单个学习者知识点掌握程度失败 %s: %s", learner_uid, exc, exc_info=True)
            return None

    def compute_multiple_learners_concept_mastery(
        self,
        learner_uids: List[str],
        cd_ability: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        try:
            if not self.ensure_initialized():
                return {
                    "success": False,
                    "error": "引擎初始化失败",
                    "results": [],
                    "total_count": len(learner_uids),
                    "success_count": 0,
                }

            if not learner_uids:
                return {
                    "success": True,
                    "results": [],
                    "total_count": 0,
                    "success_count": 0,
                }

            valid_learner_uids = [
                uid for uid in learner_uids
                if self.kt_repository.validate_learner_has_interactions(uid)
            ]
            if not valid_learner_uids:
                return {
                    "success": False,
                    "error": "没有找到任何有交互记录的学习者",
                    "total_count": len(learner_uids),
                    "valid_count": 0,
                    "success_count": 0,
                    "results": [],
                }

            inputs = self._prepare_kt_inputs(
                valid_learner_uids,
                cd_ability=cd_ability,
            )

            self.model.set_cd_optimized_ability(inputs["cd_ability"], qus_num=0)

            batch_size, seq_len = inputs["qusunt_seq_indices"].shape
            embedding_dim = inputs["h_qusunt"].shape[1]
            qusunt_indices_flat = inputs["qusunt_seq_indices"].view(-1)
            h_qusunt_batch = inputs["h_qusunt"][qusunt_indices_flat].view(
                batch_size, seq_len, embedding_dim
            )

            with torch.no_grad():
                self.model.eval()
                ability_matrix = self.model.get_concept_mastery(
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
                ability_matrix=ability_matrix,
                learner_uids=valid_learner_uids,
                learner_seq_lengths=inputs["learner_seq_lengths"],
                seq_masks=inputs["seq_masks"],
                last_k_steps=kt_settings.history_steps,
            )

            return {
                "success": True,
                "total_count": len(learner_uids),
                "valid_count": len(valid_learner_uids),
                "success_count": len(formatted_results),
                "results": formatted_results,
            }

        except Exception as exc:
            logger.error("计算多个学习者知识点掌握程度失败: %s", exc, exc_info=True)
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
        cd_ability: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
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

            valid_learner_uids = [
                uid for uid in learner_uids
                if self.kt_repository.validate_learner_has_interactions(uid)
            ]
            valid_indices = [i for i, uid in enumerate(learner_uids) if uid in valid_learner_uids]
            valid_embeddings = [learner_embeddings[i] for i in valid_indices]

            if not valid_learner_uids:
                return {
                    "success": False,
                    "error": "没有找到任何有交互记录的学习者",
                    "total_count": len(learner_uids),
                    "success_count": 0,
                    "results": [],
                }

            inputs = self._prepare_kt_inputs(
                valid_learner_uids,
                learner_embeddings=valid_embeddings,
                cd_ability=cd_ability,
            )

            self.model.set_cd_optimized_ability(inputs["cd_ability"], qus_num=0)

            batch_size, seq_len = inputs["qusunt_seq_indices"].shape
            embedding_dim = inputs["h_qusunt"].shape[1]
            qusunt_indices_flat = inputs["qusunt_seq_indices"].view(-1)
            h_qusunt_batch = inputs["h_qusunt"][qusunt_indices_flat].view(
                batch_size, seq_len, embedding_dim
            )

            with torch.no_grad():
                self.model.eval()
                ability_matrix = self.model.get_concept_mastery(
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
                ability_matrix=ability_matrix,
                learner_uids=valid_learner_uids,
                learner_seq_lengths=inputs["learner_seq_lengths"],
                seq_masks=inputs["seq_masks"],
                last_k_steps=kt_settings.history_steps,
            )

            return {
                "success": True,
                "total_count": len(learner_uids),
                "valid_count": len(valid_learner_uids),
                "success_count": len(formatted_results),
                "results": formatted_results,
            }

        except Exception as exc:
            logger.error("使用嵌入计算知识点掌握程度失败: %s", exc, exc_info=True)
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
    def analyze(
        self,
        learner_uids: List[str],
        cd_ability: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return self.compute_multiple_learners_concept_mastery(
            learner_uids=learner_uids,
            cd_ability=cd_ability,
        )

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


def analyze(
    learner_uids: List[str],
    cd_ability: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    engine = get_kt_engine()
    return engine.analyze(learner_uids, cd_ability=cd_ability)


def compute_single_learner_concept_mastery(
    learner_uid: str,
    cd_ability: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, Any]]:
    engine = get_kt_engine()
    return engine.compute_single_learner_concept_mastery(learner_uid, cd_ability=cd_ability)


def compute_multiple_learners_concept_mastery(
    learner_uids: List[str],
    cd_ability: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    engine = get_kt_engine()
    return engine.compute_multiple_learners_concept_mastery(learner_uids, cd_ability=cd_ability)


def compute_concept_mastery_with_embeddings(
    learner_embeddings: List[torch.Tensor],
    learner_uids: List[str],
    cd_ability: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    engine = get_kt_engine()
    return engine.compute_concept_mastery_with_embeddings(
        learner_embeddings=learner_embeddings,
        learner_uids=learner_uids,
        cd_ability=cd_ability,
    )


def initialize_engine() -> bool:
    engine = get_kt_engine()
    return engine.initialize()


def get_engine_status() -> Dict[str, Any]:
    engine = get_kt_engine()
    return engine.get_engine_status()
