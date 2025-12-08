# BackEnd/app/domain/prediction/prediction_pipeline.py
"""
知识能力预测总线 PredictionPipeline

对外只暴露一个统一接口:
    analyze(learner_uids: List[str], is_new_learner: Optional[bool] = None)

职责：
- 根据学习者是否已有 KT 结果，自动选择流程：
    - 已有学习者：CD(模式1，使用历史KT) → KT(模式1)
    - 新学习者：  HGC → CD(模式2，使用外部嵌入) → KT(模式2)
- 支持混合场景：同一次调用中，部分 UID 走“已有”，部分 UID 走“新学习者”流程
- 返回结构简化为：success / errors / results（以学习者 uid 为键）

结果结构示例:
{
    "success": true,
    "errors": [],
    "results": {
        "learner_uid_1": {
            "kt": {
                "cpt_xxx": 0.83,
                "cpt_yyy": 0.21
            },
            "hgc": [0.1, 0.2, ...]  # 或 None
        },
        ...
    }
}

说明：
- 内部仍然会区分 existing / new 流程，但对外不再暴露该维度。
- CD / KT 的知识点集合理论上相同，这里通过统一 concept 顺序保证两者使用一致的顺序。
"""

import logging
from typing import List, Dict, Any, Optional

import torch

from app.domain.prediction.hgc_engine import get_hgc_engine
from app.domain.prediction.cd_engine import get_cd_engine
from app.domain.prediction.kt_engine import get_kt_engine
from app.data_access.prediction.learner_repository import LearnerRepository

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """HGC + CD + KT 的统一编排入口"""

    def __init__(self) -> None:
        # 引擎单例
        self.hgc_engine = get_hgc_engine()
        self.cd_engine = get_cd_engine()
        self.kt_engine = get_kt_engine()

        # 用于判断“是否已有 KT 结果”
        self.learner_repository = LearnerRepository()

        # 统一后的知识点顺序缓存
        self._unified_concept_uid_order: Optional[List[str]] = None

    # ---------------------------------------------------------------
    # 内部工具
    # ---------------------------------------------------------------
    def _initialize_engines(self) -> bool:
        """确保三个引擎都完成初始化"""
        try:
            logger.info("PredictionPipeline: 初始化所有引擎...")

            if not self.hgc_engine.ensure_initialized():
                logger.error("HGC 引擎初始化失败")
                return False
            if not self.cd_engine.ensure_initialized():
                logger.error("CD 引擎初始化失败")
                return False
            if not self.kt_engine.ensure_initialized():
                logger.error("KT 引擎初始化失败")
                return False

            logger.info("PredictionPipeline: 所有引擎初始化完成")
            return True
        except Exception as exc:
            logger.error("PredictionPipeline 初始化引擎异常: %s", exc, exc_info=True)
            return False

    def _split_learners_by_mode(
        self,
        learner_uids: List[str],
        is_new_learner: Optional[bool] = None,
    ) -> Dict[str, List[str]]:
        """
        根据 is_new_learner + LearnerRepository 中是否有 KT 结果，
        将学习者拆分为已有 / 新学习者两个子列表。

        is_new_learner:
            - True  -> 全部按新学习者处理
            - False -> 全部按已有学习者处理
            - None  -> 自动：有 KT 结果 => 已有，否则视为“新”
        """
        if not learner_uids:
            return {"existing": [], "new": []}

        # 强制模式：直接返回
        if is_new_learner is True:
            return {"existing": [], "new": list(learner_uids)}
        if is_new_learner is False:
            return {"existing": list(learner_uids), "new": []}

        # 自动模式：根据 LearnerRepository 中的 KT 字段判断
        try:
            kt_docs = self.learner_repository.get_kt_results_by_uids(
                learner_uids, return_format="list"
            )
        except Exception as exc:
            logger.error("自动划分已有/新学习者时查询 KT 失败: %s", exc, exc_info=True)
            # 查询失败时，保守起见全部按“已有”处理
            return {"existing": list(learner_uids), "new": []}

        existing: List[str] = []
        new: List[str] = []

        for uid, doc in zip(learner_uids, kt_docs):
            kt_data = (doc or {}).get("KT") or {}
            if kt_data:
                existing.append(uid)
            else:
                new.append(uid)

        logger.info(
            "自动划分学习者: 总数=%d, 已有=%d, 新=%d",
            len(learner_uids),
            len(existing),
            len(new),
        )

        return {"existing": existing, "new": new}

    def _build_hgc_embedding_tensors(
        self,
        hgc_result: Dict[str, Any],
        learner_uids: List[str],
        device: Optional[str] = None,
    ) -> List[torch.Tensor]:
        """
        将 HGC 引擎的嵌入结果转成按 learner_uids 顺序排列的 Tensor 列表
        """
        if not hgc_result.get("success", False):
            raise RuntimeError(f"HGC 计算失败: {hgc_result.get('error', 'unknown error')}")

        results_by_uid: Dict[str, Any] = hgc_result.get("results", {})
        tensors: List[torch.Tensor] = []

        for uid in learner_uids:
            if uid not in results_by_uid:
                raise KeyError(f"HGC 结果中缺少学习者 {uid}")
            emb_list = results_by_uid[uid].get("embedding")
            if emb_list is None:
                raise ValueError(f"HGC 结果中学习者 {uid} 没有 embedding 字段")

            tensor = torch.tensor(emb_list, dtype=torch.float32, device=device)
            tensors.append(tensor)

        return tensors

    # ---------------------------------------------------------------
    # 统一 concept 映射：让 CD / KT 的知识点顺序一致
    # ---------------------------------------------------------------
    def _get_unified_concept_uid_order(self) -> List[str]:
        """
        构建一个统一的知识点 uid 顺序：
        - 优先使用 CD 引擎的 concept_mapping（uid -> id），按 id 升序
        - 再和 KT 引擎的 concept_uid_order 取交集
        - 若一侧缺失，则退化为另一侧
        """
        if self._unified_concept_uid_order is not None:
            return self._unified_concept_uid_order

        cd_mapping = getattr(self.cd_engine, "concept_mapping", None) or {}
        kt_uid_order = getattr(self.kt_engine, "concept_uid_order", None) or []

        unified_order: List[str] = []

        if cd_mapping and kt_uid_order:
            kt_uid_set = set(kt_uid_order)
            sorted_cd = sorted(cd_mapping.items(), key=lambda x: x[1])  # (uid, id)
            unified_order = [uid for uid, _ in sorted_cd if uid in kt_uid_set]
        elif kt_uid_order:
            unified_order = list(kt_uid_order)
        elif cd_mapping:
            sorted_cd = sorted(cd_mapping.items(), key=lambda x: x[1])
            unified_order = [uid for uid, _ in sorted_cd]
        else:
            unified_order = []

        logger.info("统一 concept 顺序构建完成，知识点数量: %d", len(unified_order))

        self._unified_concept_uid_order = unified_order
        return unified_order

    # ---------------------------------------------------------------
    # 结果整理工具
    # ---------------------------------------------------------------
    def _extract_kt_by_learner(
        self,
        kt_result: Optional[Dict[str, Any]],
        unified_concepts: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        将 KT 引擎的返回整理为:
            { learner_uid: {concept_uid: value, ...}, ... }

        并且按 unified_concepts 的顺序插入，保证顺序一致。
        """
        learner_kt: Dict[str, Dict[str, float]] = {}

        if not kt_result or not kt_result.get("success", False):
            return learner_kt

        for item in kt_result.get("results", []):
            uid = item.get("learner_id")
            if not uid:
                continue
            concept_mastery = item.get("concept_mastery") or {}

            ordered_dict: Dict[str, float] = {}
            if unified_concepts:
                for c_uid in unified_concepts:
                    if c_uid in concept_mastery:
                        ordered_dict[c_uid] = float(concept_mastery[c_uid])
            else:
                # 没有统一顺序时，直接使用原有顺序
                ordered_dict = {k: float(v) for k, v in concept_mastery.items()}

            learner_kt[uid] = ordered_dict

        return learner_kt

    def _extract_hgc_embedding_list_by_learner(
        self,
        hgc_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Optional[list]]:
        """
        将 HGC 引擎的返回整理为:
            { learner_uid: embedding_list, ... }

        只保留嵌入表达本身（list），其余字段忽略。
        """
        if not hgc_result or not hgc_result.get("success", False):
            return {}

        results_by_uid = hgc_result.get("results", {}) or {}
        out: Dict[str, Optional[list]] = {}
        for uid, info in results_by_uid.items():
            if info is None:
                out[uid] = None
            else:
                out[uid] = info.get("embedding")
        return out

    # ---------------------------------------------------------------
    # 核心方法：分析流程
    # ---------------------------------------------------------------
    def analyze(
        self,
        learner_uids: List[str],
        is_new_learner: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        统一对外接口

        Args:
            learner_uids: 学习者 UID 列表
            is_new_learner:
                - True  -> 强制走“新学习者”流程（HGC -> CD/KT embedding 模式）
                - False -> 强制走“已有学习者”流程（CD/KT 普通模式）
                - None  -> 自动：
                    - 对有历史 KT 结果的学习者走“已有流程”
                    - 对没有 KT 结果的学习者走“新学习者流程”

        Returns:
            Dict[str, Any]: 以 "学习者 uid 为键" 的综合结果结构，仅包含 kt / hgc 内容。
        """
        if not learner_uids:
            return {
                "success": True,
                "errors": [],
                "results": {},
            }

        if not self._initialize_engines():
            return {
                "success": False,
                "errors": ["引擎初始化失败"],
                "results": {},
            }

        # 在引擎初始化之后获取统一的 concept 顺序，使 CD / KT 使用一致顺序
        unified_concepts = self._get_unified_concept_uid_order()

        # 划分已有 / 新学习者（仅用于内部流程控制）
        split = self._split_learners_by_mode(learner_uids, is_new_learner=is_new_learner)
        existing_uids = split["existing"]
        new_uids = split["new"]

        logger.info(
            "PredictionPipeline.analyze 调用: 总数=%d, existing=%d, new=%d",
            len(learner_uids),
            len(existing_uids),
            len(new_uids),
        )

        kt_result_existing: Optional[Dict[str, Any]] = None
        kt_result_new: Optional[Dict[str, Any]] = None

        cd_result_existing: Optional[Dict[str, Any]] = None  # 仅内部使用
        cd_result_new: Optional[Dict[str, Any]] = None       # 仅内部使用
        hgc_result_new: Optional[Dict[str, Any]] = None

        overall_success = True
        errors: List[str] = []

        # ----------------- 已有学习者路径：CD(模式1) → KT(模式1) -----------------
        if existing_uids:
            try:
                logger.info("PredictionPipeline: 已有学习者路径 -> CD.analyze")
                cd_result_existing = self.cd_engine.analyze(existing_uids)
                if not cd_result_existing.get("success", False):
                    overall_success = False
                    errors.append(f"CD existing 失败: {cd_result_existing.get('error', 'unknown')}")
                else:
                    logger.info(
                        "CD existing 完成: success_count=%s / total=%s",
                        cd_result_existing.get("success_count"),
                        cd_result_existing.get("total_count"),
                    )
            except Exception as exc:
                overall_success = False
                errors.append(f"CD existing 异常: {exc}")
                logger.error("CD existing 异常: %s", exc, exc_info=True)

            try:
                logger.info("PredictionPipeline: 已有学习者路径 -> KT.analyze")
                kt_result_existing = self.kt_engine.analyze(existing_uids)
                if not kt_result_existing.get("success", False):
                    overall_success = False
                    errors.append(f"KT existing 失败: {kt_result_existing.get('error', 'unknown')}")
                else:
                    logger.info(
                        "KT existing 完成: success_count=%s / total=%s",
                        kt_result_existing.get("success_count"),
                        kt_result_existing.get("total_count"),
                    )
            except Exception as exc:
                overall_success = False
                errors.append(f"KT existing 异常: {exc}")
                logger.error("KT existing 异常: %s", exc, exc_info=True)

        # ----------------- 新学习者路径：HGC → CD(模式2) → KT(模式2) -----------------
        embedding_tensors: List[torch.Tensor] = []
        if new_uids:
            try:
                logger.info("PredictionPipeline: 新学习者路径 -> HGC.analyze")
                hgc_result_new = self.hgc_engine.analyze(new_uids)
                # 将结果转成 embedding tensor 列表（和 new_uids 对齐）
                embedding_tensors = self._build_hgc_embedding_tensors(
                    hgc_result_new,
                    new_uids,
                    device=self.kt_engine.device,  # 与 KT 引擎保持同一设备
                )
            except Exception as exc:
                overall_success = False
                errors.append(f"HGC new 异常: {exc}")
                logger.error("HGC new 异常: %s", exc, exc_info=True)
                embedding_tensors = []

            if embedding_tensors:
                try:
                    logger.info("PredictionPipeline: 新学习者路径 -> CD.compute_concept_mastery_with_embeddings")
                    cd_result_new = self.cd_engine.compute_concept_mastery_with_embeddings(
                        learner_embeddings=embedding_tensors,
                        learner_uids=new_uids,
                    )
                    if not cd_result_new.get("success", False):
                        overall_success = False
                        errors.append(f"CD new 失败: {cd_result_new.get('error', 'unknown')}")
                    else:
                        logger.info(
                            "CD new 完成: success_count=%s / total=%s",
                            cd_result_new.get("success_count"),
                            cd_result_new.get("total_count"),
                        )
                except Exception as exc:
                    overall_success = False
                    errors.append(f"CD new 异常: {exc}")
                    logger.error("CD new 异常: %s", exc, exc_info=True)

                try:
                    logger.info("PredictionPipeline: 新学习者路径 -> KT.compute_concept_mastery_with_embeddings")
                    kt_result_new = self.kt_engine.compute_concept_mastery_with_embeddings(
                        learner_embeddings=embedding_tensors,
                        learner_uids=new_uids,
                    )
                    if not kt_result_new.get("success", False):
                        overall_success = False
                        errors.append(f"KT new 失败: {kt_result_new.get('error', 'unknown')}")
                    else:
                        logger.info(
                            "KT new 完成: success_count=%s / total=%s",
                            kt_result_new.get("success_count"),
                            kt_result_new.get("total_count"),
                        )
                except Exception as exc:
                    overall_success = False
                    errors.append(f"KT new 异常: {exc}")
                    logger.error("KT new 异常: %s", exc, exc_info=True)

        # ----------------- 将 KT / HGC 结果整理为按 learner_uid 的形式 -----------------
        kt_by_learner: Dict[str, Dict[str, float]] = {}
        if kt_result_existing:
            kt_by_learner.update(
                self._extract_kt_by_learner(kt_result_existing, unified_concepts)
            )
        if kt_result_new:
            kt_by_learner.update(
                self._extract_kt_by_learner(kt_result_new, unified_concepts)
            )

        hgc_embedding_by_learner: Dict[str, Optional[list]] = {}
        if hgc_result_new:
            hgc_embedding_by_learner.update(
                self._extract_hgc_embedding_list_by_learner(hgc_result_new)
            )

        # ----------------- 按学习者 uid 组装最终返回结构 -----------------
        final_results: Dict[str, Dict[str, Any]] = {}
        for uid in learner_uids:
            final_results[uid] = {
                "kt": kt_by_learner.get(uid, {}),
                "hgc": hgc_embedding_by_learner.get(uid),
            }

        result: Dict[str, Any] = {
            "success": overall_success,
            "errors": errors,
            "results": final_results,
        }

        return result


# ----------------------------------------------------------------------
# 模块级便捷函数（推荐上层直接使用）
# ----------------------------------------------------------------------
_pipeline_instance: Optional[PredictionPipeline] = None


def get_prediction_pipeline() -> PredictionPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = PredictionPipeline()
    return _pipeline_instance


def analyze(
    learner_uids: List[str],
    is_new_learner: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    统一对外函数接口：
        from app.domain.prediction.prediction_pipeline import analyze

        res = analyze(["uid1", "uid2"])         # 自动模式
        res = analyze(["uid1"], False)          # 强制按已有学习者处理
        res = analyze(["uid1", "uid2"], True)   # 强制走新学习者流程
    """
    pipeline = get_prediction_pipeline()
    return pipeline.analyze(learner_uids, is_new_learner=is_new_learner)
