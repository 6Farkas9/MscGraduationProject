# run_all_experiments_single_process_bs1.py
# 单进程复刻原训练脚本路径（train_cd_phase/train_kt_phase/evaluate_cd/evaluate_kt）
# 默认 batch_size=1、max_batch_size=1 优先稳定
# AUC/RMSE 在 evaluate_* 的同一次 forward 内计算（不额外 forward）

import os
import sys
import json
import time
import math
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -----------------------------
# 纯 numpy AUC / RMSE（避免 sklearn/scipy 内存常驻）
# -----------------------------
def auc_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.5
    y_true = y_true.astype(np.int32)
    if y_true.min() == y_true.max():
        return 0.5

    order = np.argsort(y_score)
    y_sorted = y_true[order]

    n_pos = int(y_sorted.sum())
    n_neg = y_sorted.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = np.arange(1, y_sorted.size + 1, dtype=np.float64)
    sum_ranks_pos = ranks[y_sorted == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def rmse_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    y_true = y_true.astype(np.float32)
    y_score = y_score.astype(np.float32)
    return float(np.sqrt(np.mean((y_true - y_score) ** 2)))


@dataclass
class Config:
    out_root: str = "results_single_process_bs1"
    seed: int = 2025

    # full 收敛（画loss趋势用）
    full_epochs: int = 20

    # 消融每个跑几轮
    ablation_epochs: int = 15

    # 关键：尽量降低 OOM 概率
    train_batch_size: int = 1
    eval_batch_size: int = 1
    max_batch_size: int = 1

    # w/o HGC 常数 embedding
    const_value: float = 0.5


def make_run_dir(root: str, name: str) -> str:
    os.makedirs(root, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(root, f"{ts}_{name}")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    cfg = Config()
    os.makedirs(cfg.out_root, exist_ok=True)

    # --------- 强制“只加载一次静态数据”（避免每个实验重复读SQL/大对象反复生成）---------
    # CompletePipelineTrainer.setup_models_and_data() 内部会调用 hgcdr.loadDatafromSql()
    # 我们在第一次前置加载后，把该函数替换为 no-op
    from DataReader.HGCDataReader import hgcdr

    if not hasattr(hgcdr, "_loaded_once"):
        print("[Preload] loading static data once via hgcdr.loadDatafromSql() ...", flush=True)
        hgcdr.loadDatafromSql()
        hgcdr._loaded_once = True
        print("[Preload] done.", flush=True)

    _orig_load = hgcdr.loadDatafromSql

    def _load_once_wrapper():
        # 后续不再重复加载
        return None

    hgcdr.loadDatafromSql = _load_once_wrapper

    # --------- 设置超参（与原训练脚本一致入口）---------
    from hyperparams.hyperparameter import hyperparams

    # 固定CPU（你原超参文件本来就是cpu）:contentReference[oaicite:2]{index=2}
    hyperparams.train_batch_size = cfg.train_batch_size
    hyperparams.train_eval_batch_size = cfg.eval_batch_size
    hyperparams.max_batch_size = cfg.max_batch_size

    # 你如果想更稳一点，可以把学习率调小一点点（可选）
    # hyperparams.train_learning_rate = 0.0005

    # --------- 实验计划 ---------
    plan = [
        ("full_convergence", "full", cfg.full_epochs),

        ("full", "full", cfg.ablation_epochs),
        ("w_o_hgc", "w_o_hgc", cfg.ablation_epochs),
        ("w_o_cd", "w_o_cd", cfg.ablation_epochs),
        ("w_o_kt", "w_o_kt", cfg.ablation_epochs),
        ("cd_only", "cd_only", cfg.ablation_epochs),
        ("kt_only", "kt_only", cfg.ablation_epochs),
    ]

    index: List[Dict[str, Any]] = []

    print("\n" + "#" * 80, flush=True)
    print("[Runner] Single-process experiments start (bs=1, max_batch_size=1)", flush=True)
    print(f"[Runner] out_root = {cfg.out_root}", flush=True)
    print("#" * 80 + "\n", flush=True)

    # --------- 每个实验：新建 trainer（模型重置），但静态数据不再重复加载 ---------
    for exp_i, (exp_name, mode, epochs) in enumerate(plan, start=1):
        out_dir = make_run_dir(cfg.out_root, exp_name)
        with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({"exp_name": exp_name, "mode": mode, "epochs": epochs, **asdict(cfg)}, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 80, flush=True)
        print(f"[Experiment {exp_i}/{len(plan)}] {exp_name} | mode={mode} | epochs={epochs}", flush=True)
        print(f"[Experiment {exp_i}/{len(plan)}] out_dir={out_dir}", flush=True)
        print("=" * 80, flush=True)

        t0 = time.time()
        ok = True
        err_msg = ""

        try:
            run_one_experiment(mode=mode, epochs=epochs, out_dir=out_dir, cfg=cfg)
        except Exception as e:
            ok = False
            err_msg = str(e)
            with open(os.path.join(out_dir, "ERROR.txt"), "w", encoding="utf-8") as f:
                f.write(err_msg + "\n\n" + traceback.format_exc())
            print(f"[FAILED] {exp_name}: {e}", flush=True)

        dt = time.time() - t0
        index.append({"exp_name": exp_name, "mode": mode, "epochs": epochs, "ok": ok, "time_sec": dt, "out_dir": out_dir, "error": err_msg})

        with open(os.path.join(cfg.out_root, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        print(f"[Experiment {exp_i}/{len(plan)}] {'SUCCESS' if ok else 'FAILED'} | {dt/60:.1f} min", flush=True)

    print("\n" + "#" * 80, flush=True)
    print("[Runner] All experiments finished.", flush=True)
    print(f"[Runner] index.json = {os.path.join(cfg.out_root, 'index.json')}", flush=True)
    print("#" * 80, flush=True)

    # 恢复原函数（可选）
    hgcdr.loadDatafromSql = _orig_load


def run_one_experiment(mode: str, epochs: int, out_dir: str, cfg: Config):
    import torch
    import random

    from Train_HGC_CD_KT import CompletePipelineTrainer
    from hyperparams.hyperparameter import hyperparams

    # 固定随机种子（可复现）
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # 确保每个实验的保存目录分开
    hyperparams.train_save_dir = out_dir

    # 新建 trainer（会初始化模型、数据集、dataloader、optimizer）
    trainer = CompletePipelineTrainer(resume_training=False)

    # 用于 w/o HGC：常数 embedding 缓存一次
    _const_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    if mode in {"w_o_hgc", "cd_only", "kt_only"}:
        def compute_const_embeddings():
            nonlocal _const_cache
            if _const_cache is not None:
                return _const_cache

            trainer.model_hgc.eval()
            with torch.no_grad():
                input_data = trainer._prepare_input_data()
                lrn_emb, unt_emb, cpt_emb = trainer.model_hgc(
                    input_data=input_data, device=trainer.device, return_dict=False
                )
            trainer.model_hgc.train()

            _const_cache = (
                torch.full_like(lrn_emb, cfg.const_value, device=trainer.device),
                torch.full_like(unt_emb, cfg.const_value, device=trainer.device),
                torch.full_like(cpt_emb, cfg.const_value, device=trainer.device),
            )
            print(f"  [Patch] w/o HGC -> constant={cfg.const_value} (cached)", flush=True)
            return _const_cache

        trainer.compute_hgc_embeddings = compute_const_embeddings

    # 模块开关
    cd_enabled = mode not in {"w_o_cd", "kt_only"}
    kt_enabled = mode not in {"w_o_kt", "cd_only"}

    print(f"[Run] device={trainer.device} | train_bs={hyperparams.train_batch_size} | eval_bs={hyperparams.train_eval_batch_size} | max_batch_size={hyperparams.max_batch_size}", flush=True)
    print(f"[Run] mode={mode} | cd_enabled={cd_enabled} | kt_enabled={kt_enabled}", flush=True)

    # 增强评估（复刻原 evaluate_* 逻辑 + 同一次forward收集auc/rmse）
    def evaluate_cd_plus():
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0
        evaluated_batches = 0

        probs: List[float] = []
        tgts: List[float] = []

        max_eval_batches = min(trainer.max_batch_size, len(trainer.cd_eval_loader))

        with torch.no_grad():
            lrn_emb, qusunt_emb, cpt_emb = trainer.compute_hgc_embeddings()

            for batch_idx, batch in enumerate(trainer.cd_eval_loader):
                if batch_idx >= max_eval_batches:
                    break
                try:
                    lrn_indices = batch['lrn_indices'].to(trainer.device, non_blocking=True)
                    qus_seq_indices = batch['qus_seq_indices'].to(trainer.device, non_blocking=True)
                    qus_seq_masks = batch['qus_seq_masks'].to(trainer.device, non_blocking=True).float()
                    results = batch['results'].to(trainer.device, non_blocking=True).float()

                    h_lrn_batch = lrn_emb[lrn_indices]

                    predictions = trainer.model_cd(
                        h_lrn_batch=h_lrn_batch,
                        h_qus=qusunt_emb[:trainer.cd_eval_dataset.qus_num],
                        h_cpt=cpt_emb,
                        qus_seq_indices=qus_seq_indices,
                        qus_seq_masks=qus_seq_masks,
                        return_ability=False,
                        use_kt_optimization=False
                    )

                    valid_predictions = predictions * qus_seq_masks
                    valid_targets = results * qus_seq_masks
                    loss = trainer.criterion(valid_predictions, valid_targets)

                    pred_binary = (predictions > 0.5).float()
                    correct = ((pred_binary == results) * qus_seq_masks).sum().item()

                    total_samples += qus_seq_masks.sum().item()
                    total_loss += float(loss.item())
                    total_correct += float(correct)
                    evaluated_batches += 1

                    # 在同一次forward里收集auc/rmse
                    pred_flat = predictions.detach().cpu().numpy().reshape(-1)
                    tgt_flat = results.detach().cpu().numpy().reshape(-1)
                    mask_flat = qus_seq_masks.detach().cpu().numpy().reshape(-1).astype(np.float32) > 0.5

                    if mask_flat.any():
                        probs.extend(pred_flat[mask_flat].tolist())
                        tgts.extend(tgt_flat[mask_flat].tolist())

                except Exception:
                    continue

        avg_loss = total_loss / evaluated_batches if evaluated_batches > 0 else 0.0
        acc = total_correct / total_samples if total_samples > 0 else 0.0

        y_true = np.array(tgts, dtype=np.float32)
        y_prob = np.array(probs, dtype=np.float32)
        auc = auc_fast(y_true, y_prob) if y_true.size > 0 else 0.5
        r = rmse_fast(y_true, y_prob)

        return avg_loss, acc, auc, r

    def evaluate_kt_plus():
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0.0
        evaluated_batches = 0

        probs: List[float] = []
        tgts: List[float] = []

        max_eval_batches = min(trainer.max_batch_size, len(trainer.kt_eval_loader))

        with torch.no_grad():
            lrn_emb, qusunt_emb, cpt_emb = trainer.compute_hgc_embeddings()

            for batch_idx, batch in enumerate(trainer.kt_eval_loader):
                if batch_idx >= max_eval_batches:
                    break
                try:
                    lrn_indices = batch['lrn_indices'].to(trainer.device, non_blocking=True)
                    qusunt_seq_indices = batch['qusunt_seq_indices'].to(trainer.device, non_blocking=True)
                    add1 = batch['add1'].to(trainer.device, non_blocking=True)
                    add2 = batch['add2'].to(trainer.device, non_blocking=True)
                    type_indices = batch['type_indices'].to(trainer.device, non_blocking=True)
                    seq_masks = batch['seq_masks'].to(trainer.device, non_blocking=True)
                    prediction_masks = batch['prediction_masks'].to(trainer.device, non_blocking=True)
                    next_results = batch['next_results'].to(trainer.device, non_blocking=True)

                    h_lrn_batch = lrn_emb[lrn_indices]

                    B, T = qusunt_seq_indices.shape
                    D = qusunt_emb.shape[1]
                    flat = qusunt_seq_indices.reshape(-1)
                    h_qusunt_batch = qusunt_emb[flat].view(B, T, D)

                    predictions, _ = trainer.model_kt(
                        h_lrn_batch=h_lrn_batch,
                        h_qusunt_batch=h_qusunt_batch,
                        h_cpt=cpt_emb,
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

                    # 复刻原 evaluate_kt 的处理逻辑
                    valid_predictions = predictions * prediction_masks.unsqueeze(-1)
                    valid_targets = next_results.unsqueeze(-1) * prediction_masks.unsqueeze(-1)

                    if len(valid_predictions.shape) == 3:
                        valid_predictions_mean = valid_predictions.mean(dim=-1)
                        valid_targets_mean = valid_targets.mean(dim=-1)
                    else:
                        valid_predictions_mean = valid_predictions
                        valid_targets_mean = valid_targets

                    valid_mask = prediction_masks.bool()
                    if valid_mask.any():
                        loss = trainer.criterion(
                            valid_predictions_mean[valid_mask],
                            valid_targets_mean[valid_mask]
                        )

                        pred_binary = (valid_predictions_mean > 0.5).float()
                        correct = ((pred_binary == valid_targets_mean) * valid_mask).sum().item()

                        total_samples += valid_mask.sum().item()
                        total_loss += float(loss.item())
                        total_correct += float(correct)

                        # 同一次forward收集auc/rmse（用 valid_predictions_mean 与 valid_targets_mean）
                        pred_flat = valid_predictions_mean.detach().cpu().numpy().reshape(-1)
                        tgt_flat = valid_targets_mean.detach().cpu().numpy().reshape(-1)
                        mask_flat = prediction_masks.detach().cpu().numpy().reshape(-1).astype(np.float32) > 0.5

                        if mask_flat.any():
                            probs.extend(pred_flat[mask_flat].tolist())
                            tgts.extend(tgt_flat[mask_flat].tolist())

                    evaluated_batches += 1

                except Exception:
                    continue

        avg_loss = total_loss / evaluated_batches if evaluated_batches > 0 else 0.0
        acc = total_correct / total_samples if total_samples > 0 else 0.0

        y_true = np.array(tgts, dtype=np.float32)
        y_prob = np.array(probs, dtype=np.float32)
        auc = auc_fast(y_true, y_prob) if y_true.size > 0 else 0.5
        r = rmse_fast(y_true, y_prob)

        return avg_loss, acc, auc, r

    history: Dict[str, Any] = {
        "mode": mode,
        "epochs": epochs,
        "cd_train_loss": [],
        "kt_train_loss": [],
        "combined_train_loss": [],
        "cd_val": [],
        "kt_val": [],
        "combined_val": [],
    }

    start = time.time()

    for ep in range(1, epochs + 1):
        print("\n" + "=" * 60, flush=True)
        print(f"[{mode}] Epoch {ep}/{epochs}", flush=True)
        print("=" * 60, flush=True)

        cd_loss = None
        kt_loss = None

        # ---- train phases (复刻原脚本) ----
        if cd_enabled:
            print(f"--- CD阶段训练 (第{ep}轮) ---", flush=True)
            # 如果KT关闭，不允许KT初始化
            cd_loss = trainer.train_cd_phase(epoch=ep, use_kt_initialization=False)
            history["cd_train_loss"].append(float(cd_loss))
        else:
            print("--- CD阶段训练 跳过 ---", flush=True)

        if kt_enabled:
            print(f"--- KT阶段训练 (第{ep}轮) ---", flush=True)
            # 如果CD关闭，强制 epoch_override=1，避免内部 epoch>1 触发 CD 优化能力逻辑
            epoch_for_kt = 1 if (mode in {"w_o_cd", "kt_only"}) else ep
            kt_loss = trainer.train_kt_phase(epoch=epoch_for_kt)
            history["kt_train_loss"].append(float(kt_loss))
        else:
            print("--- KT阶段训练 跳过 ---", flush=True)

        # combined train loss（对等口径）
        if cd_enabled and kt_enabled:
            comb_train = (cd_loss + kt_loss) / 2.0
        elif cd_enabled:
            comb_train = cd_loss
        else:
            comb_train = kt_loss
        history["combined_train_loss"].append(float(comb_train))
        print(f"[Train Combined] loss={comb_train:.6f}", flush=True)

        # ---- eval (同一次forward计算loss/acc/auc/rmse) ----
        print(f"--- 评估 (第{ep}轮) ---", flush=True)
        cd_val = None
        kt_val = None

        if cd_enabled:
            cd_vloss, cd_acc, cd_auc, cd_rmse = evaluate_cd_plus()
            cd_val = {"loss": cd_vloss, "acc": cd_acc, "auc": cd_auc, "rmse": cd_rmse}
            history["cd_val"].append(cd_val)
            print(f"  CD: loss={cd_vloss:.4f}, acc={cd_acc:.4f}, auc={cd_auc:.4f}, rmse={cd_rmse:.4f}", flush=True)

        if kt_enabled:
            kt_vloss, kt_acc, kt_auc, kt_rmse = evaluate_kt_plus()
            kt_val = {"loss": kt_vloss, "acc": kt_acc, "auc": kt_auc, "rmse": kt_rmse}
            history["kt_val"].append(kt_val)
            print(f"  KT: loss={kt_vloss:.4f}, acc={kt_acc:.4f}, auc={kt_auc:.4f}, rmse={kt_rmse:.4f}", flush=True)

        if cd_enabled and kt_enabled:
            comb = {
                "loss": (cd_val["loss"] + kt_val["loss"]) / 2.0,
                "acc": (cd_val["acc"] + kt_val["acc"]) / 2.0,
                "auc": (cd_val["auc"] + kt_val["auc"]) / 2.0,
                "rmse": (cd_val["rmse"] + kt_val["rmse"]) / 2.0,
            }
        elif cd_enabled:
            comb = dict(cd_val)
        else:
            comb = dict(kt_val)

        history["combined_val"].append(comb)
        print(f"[Val Combined] loss={comb['loss']:.4f}, acc={comb['acc']:.4f}, auc={comb['auc']:.4f}, rmse={comb['rmse']:.4f}", flush=True)

        with open(os.path.join(out_dir, "train_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start

    # 论文口径：full 最终输出以 KT 为准（你之前说的）
    if mode in {"full", "w_o_hgc", "w_o_cd", "kt_only"} and kt_enabled and len(history["kt_val"]) > 0:
        final = history["kt_val"][-1]
        final_output = "kt"
    elif cd_enabled and len(history["cd_val"]) > 0:
        final = history["cd_val"][-1]
        final_output = "cd"
    else:
        final = history["combined_val"][-1]
        final_output = "combined"

    summary = {
        "mode": mode,
        "epochs": epochs,
        "time_sec": elapsed,
        "final_output": final_output,
        "final_metrics": {
            "ACC": final["acc"],
            "AUC": final["auc"],
            "RMSE": final["rmse"],
            "VAL_LOSS": final["loss"],
        }
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "DONE.txt"), "w", encoding="utf-8") as f:
        f.write("OK\n")

    print(f"[DONE] mode={mode} saved to {out_dir} | {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
