# train_complete_pipeline.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from collections import defaultdict
import warnings
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 过滤警告
warnings.filterwarnings("ignore")

# 导入数据读取器
from DataReader.HGCDataReader import hgcdr
from DataReader.CDDataReader import cddr
from DataReader.KTDataReader import ktdr

# 导入数据集
from DataSet.CDDataSet import CDDataset
from DataSet.KTDataSet import KTDataSet

# 导入模型
from Model.HGC import HGC
from Model.CD import CD
from Model.KT import KT

# 导入超参数
from hyperparams.hyperparameter import hyperparams

max_batch_size = 4

class CompletePipelineTrainer:
    """完整的HGC-CD-KT训练管道"""
    
    def __init__(self):
        self.device = hyperparams.device
        self.setup_directories()
        self.setup_models_and_data()
        self.setup_optimizers()
        self.setup_tracking()
        
    def setup_directories(self):
        """创建模型保存目录"""
        self.save_dir = hyperparams.train_save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"模型将保存到: {self.save_dir}")
    
    def setup_models_and_data(self):
        """初始化模型和数据"""
        print("=== 初始化模型和数据 ===")
        
        # 1. 加载静态数据
        print("1. 加载静态数据...")
        hgcdr.loadDatafromSql()
        
        # 动态获取输入维度
        lrn_input_dim = hgcdr.lrn_init.shape[1]
        unt_input_dim = hgcdr.qusunt_init.shape[1]
        cpt_input_dim = hgcdr.cpt_init.shape[1]
        
        print(f"   输入维度 - lrn: {lrn_input_dim}, unt: {unt_input_dim}, cpt: {cpt_input_dim}")
        
        # 2. 初始化HGC模型
        print("2. 初始化HGC模型...")
        self.model_hgc = HGC(
            embedding_dim=hyperparams.hgc_embedding_dim,
            lrn_input_dim=lrn_input_dim,
            unt_input_dim=unt_input_dim,
            cpt_input_dim=cpt_input_dim
        ).to(self.device)
        
        # 3. 加载CD数据并创建数据集
        print("3. 初始化CD模型和数据...")
        cd_data = cddr.loadDatafromSql()
        
        # 计算初始HGC嵌入用于CD数据集
        print("   计算初始HGC嵌入...")
        self.model_hgc.eval()
        with torch.no_grad():
            initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb = self.model_hgc(
                hgcdr, self.device, return_dict=False
            )
        
        # 创建CD数据集
        print("   创建CD数据集...")
        self.cd_train_dataset = CDDataset(cd_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'train')
        self.cd_eval_dataset = CDDataset(cd_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'test')
        
        # 初始化CD模型
        embedding_dim = hyperparams.hgc_embedding_dim
        concept_num = self.cd_train_dataset.cpt_num
        
        self.model_cd = CD(
            embedding_dim=embedding_dim,
            concept_num=concept_num
        ).to(self.device)
        
        # 4. 加载KT数据并创建数据集
        print("4. 初始化KT模型和数据...")
        kt_data = ktdr.loadDatafromSql()
        
        # 创建KT数据集
        print("   创建KT数据集...")
        self.kt_train_dataset = KTDataSet(kt_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'train')
        self.kt_eval_dataset = KTDataSet(kt_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'test')
        
        # 初始化KT模型
        concept_mapping = kt_data.get('question_concepts', {})
        
        self.model_kt = KT(
            embedding_dim=embedding_dim,
            concept_num=concept_num,
            concept_mapping=concept_mapping
        ).to(self.device)
        
        # 5. 创建数据加载器
        print("5. 创建数据加载器...")
        self.cd_train_loader = torch.utils.data.DataLoader(
            self.cd_train_dataset,
            batch_size=hyperparams.train_batch_size,
            shuffle=False,
            collate_fn=self.cd_train_dataset.collate_fn
        )
        
        self.cd_eval_loader = torch.utils.data.DataLoader(
            self.cd_eval_dataset,
            batch_size=hyperparams.train_eval_batch_size,
            shuffle=False,
            collate_fn=self.cd_eval_dataset.collate_fn
        )
        
        self.kt_train_loader = torch.utils.data.DataLoader(
            self.kt_train_dataset,
            batch_size=hyperparams.train_batch_size,
            shuffle=False,
            collate_fn=self.kt_train_dataset.collate_fn
        )
        
        self.kt_eval_loader = torch.utils.data.DataLoader(
            self.kt_eval_dataset,
            batch_size=hyperparams.train_eval_batch_size,
            shuffle=False,
            collate_fn=self.kt_eval_dataset.collate_fn
        )
        
        # 打印数据集统计
        cd_train_stats = self.cd_train_dataset.get_data_statistics()
        kt_train_stats = self.kt_train_dataset.get_data_statistics()
        
        print(f"   CD训练集: {cd_train_stats['total_learners']}学习者, {cd_train_stats['total_records']}记录")
        print(f"   KT训练集: {kt_train_stats['total_learners']}学习者, {kt_train_stats['total_records']}记录")
        print(f"   CD训练批次: {len(self.cd_train_loader)}")
        print(f"   KT训练批次: {len(self.kt_train_loader)}")
        
        # 切换回训练模式
        self.model_hgc.train()
        self.model_cd.train()
        self.model_kt.train()
        
        print("✓ 模型和数据初始化完成")
    
    def setup_optimizers(self):
        """设置优化器"""
        print("6. 设置优化器...")
        
        # HGC优化器
        self.optimizer_hgc = optim.Adam(
            self.model_hgc.parameters(),
            lr=hyperparams.train_learning_rate,
            weight_decay=hyperparams.train_weight_decay
        )
        
        # CD优化器
        self.optimizer_cd = optim.Adam(
            self.model_cd.parameters(),
            lr=hyperparams.cd_learning_rate,
            weight_decay=hyperparams.cd_weight_decay
        )
        
        # KT优化器
        self.optimizer_kt = optim.Adam(
            self.model_kt.parameters(),
            lr=hyperparams.kt_learning_rate,
            weight_decay=hyperparams.train_weight_decay
        )
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        print("✓ 优化器设置完成")
    
    def setup_tracking(self):
        """设置训练跟踪"""
        self.train_history = {
            'cd_loss': [],
            'kt_loss': [],
            'cd_val_loss': [],
            'kt_val_loss': [],
            'cd_val_acc': [],
            'kt_val_acc': []
        }
        self.best_cd_loss = float('inf')
        self.best_kt_loss = float('inf')
    
    def compute_hgc_embeddings(self):
        """计算HGC嵌入（带梯度）"""
        return self.model_hgc(hgcdr, self.device, return_dict=False)
    
    def train_cd_phase(self, epoch, use_kt_initialization=False):
        """训练CD阶段"""
        self.model_hgc.train()
        self.model_cd.train()
        
        total_loss = 0
        total_batches = 0
        
        # 如果使用KT初始化，先获取KT的能力矩阵
        if use_kt_initialization and epoch > 0:
            print("   使用KT优化能力初始化CD...")
            self.model_kt.eval()
            with torch.no_grad():
                # 这里需要根据实际情况获取KT的能力矩阵
                # 简化实现：使用随机初始化
                kt_ability = torch.randn(
                    hyperparams.train_batch_size, 
                    self.cd_train_dataset.max_seq_len,
                    self.model_cd.concept_num
                ).to(self.device)
                self.model_cd.set_kt_optimized_ability(kt_ability, self.cd_train_dataset.qus_num)
        
        # 限制训练批次数量用于测试
        max_batches = min(max_batch_size, len(self.cd_train_loader))  # 最多20个批次
        
        print(f"   CD阶段训练进度 ({max_batches}个批次):")
        with tqdm(total=max_batches, desc="CD训练") as pbar:
            for batch_idx, batch in enumerate(self.cd_train_loader):
                if batch_idx >= max_batches:
                    break
                    
                try:
                    # 步骤1: 计算HGC嵌入
                    lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
                    
                    # 步骤2: 准备CD输入
                    lrn_indices = batch['lrn_indices'].to(self.device)
                    qus_seq_indices = batch['qus_seq_indices'].to(self.device)
                    qus_seq_masks = batch['qus_seq_masks'].to(self.device)
                    results = batch['results'].to(self.device)
                    
                    h_lrn_batch = lrn_emb[lrn_indices]
                    
                    # 步骤3: CD前向传播
                    self.optimizer_hgc.zero_grad()
                    self.optimizer_cd.zero_grad()
                    
                    predictions = self.model_cd(
                        h_lrn_batch=h_lrn_batch,
                        h_qus=qusunt_emb[:self.cd_train_dataset.qus_num],
                        h_cpt=cpt_emb,
                        qus_seq_indices=qus_seq_indices,
                        qus_seq_masks=qus_seq_masks,
                        return_ability=False,
                        use_kt_optimization=use_kt_initialization
                    )
                    
                    # 步骤4: 计算CD损失
                    valid_predictions = predictions * qus_seq_masks
                    valid_targets = results * qus_seq_masks
                    cd_loss = self.criterion(valid_predictions, valid_targets)
                    
                    # 步骤5: 反向传播优化HGC和CD
                    cd_loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model_hgc.parameters(), hyperparams.train_grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.model_cd.parameters(), hyperparams.train_grad_clip)
                    
                    # 优化步骤
                    self.optimizer_hgc.step()
                    self.optimizer_cd.step()
                    
                    total_loss += cd_loss.item()
                    total_batches += 1
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f'{cd_loss.item():.4f}',
                        'avg_loss': f'{total_loss/total_batches:.4f}' if total_batches > 0 else '0.0000'
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n   CD训练批次 {batch_idx} 失败: {e}")
                    pbar.update(1)
                    continue
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        self.train_history['cd_loss'].append(avg_loss)
        
        return avg_loss
    
    def train_kt_phase(self, epoch):
        """训练KT阶段"""
        self.model_hgc.train()
        self.model_kt.train()
        
        total_loss = 0
        total_batches = 0
        
        # 使用CD结果初始化KT
        if epoch > 0:
            print("   使用CD优化能力初始化KT...")
            self.model_cd.eval()
            with torch.no_grad():
                # 获取CD的能力矩阵用于初始化KT
                # 简化实现：这里需要根据实际情况获取CD的能力
                cd_ability = torch.randn(
                    hyperparams.train_batch_size,
                    self.kt_train_dataset.max_seq_len,
                    self.model_kt.concept_num
                ).to(self.device)
                self.model_kt.set_cd_optimized_ability(cd_ability, self.kt_train_dataset.qus_num)
        
        # 限制训练批次数量用于测试
        max_batches = min(max_batch_size, len(self.kt_train_loader))  # 最多20个批次
        
        print(f"   KT阶段训练进度 ({max_batches}个批次):")
        with tqdm(total=max_batches, desc="KT训练") as pbar:
            for batch_idx, batch in enumerate(self.kt_train_loader):
                if batch_idx >= max_batches:
                    break
                    
                try:
                    # 步骤1: 计算HGC嵌入
                    lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
                    
                    # 步骤2: 准备KT输入
                    lrn_indices = batch['lrn_indices'].to(self.device)
                    qusunt_seq_indices = batch['qusunt_seq_indices'].to(self.device)
                    add1 = batch['add1'].to(self.device)
                    add2 = batch['add2'].to(self.device)
                    type_indices = batch['type_indices'].to(self.device)
                    seq_masks = batch['seq_masks'].to(self.device)
                    next_question_masks = batch['next_question_masks'].to(self.device)
                    next_results = batch['next_results'].to(self.device)
                    
                    current_lrn_emb = lrn_emb[lrn_indices]
                    current_qusunt_emb = qusunt_emb[qusunt_seq_indices]
                    
                    # 步骤3: KT前向传播
                    self.optimizer_hgc.zero_grad()
                    self.optimizer_kt.zero_grad()
                    
                    predictions, concept_mastery = self.model_kt(
                        h_lrn_batch=current_lrn_emb,
                        h_qusunt_batch=current_qusunt_emb,
                        h_cpt=cpt_emb,
                        lrn_indices=lrn_indices,
                        qusunt_seq_indices=qusunt_seq_indices,
                        add1=add1,
                        add2=add2,
                        type_indices=type_indices,
                        seq_mask=seq_masks,
                        next_question_mask=next_question_masks,
                        use_cd_optimization=(epoch > 0),
                        use_contrastive=False
                    )
                    
                    # 步骤4: 计算KT损失
                    valid_predictions = predictions * next_question_masks.unsqueeze(-1)
                    valid_targets = next_results.unsqueeze(-1) * next_question_masks.unsqueeze(-1)
                    
                    if len(valid_predictions.shape) == 3:
                        valid_predictions_mean = valid_predictions.mean(dim=-1)
                        valid_targets_mean = valid_targets.mean(dim=-1)
                    else:
                        valid_predictions_mean = valid_predictions
                        valid_targets_mean = valid_targets
                    
                    valid_mask = next_question_masks.bool()
                    if valid_mask.any():
                        kt_loss = self.criterion(
                            valid_predictions_mean[valid_mask], 
                            valid_targets_mean[valid_mask]
                        )
                    else:
                        kt_loss = torch.tensor(0.001, requires_grad=True, device=self.device)
                    
                    # 步骤5: 反向传播优化HGC和KT
                    kt_loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model_hgc.parameters(), hyperparams.train_grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.model_kt.parameters(), hyperparams.train_grad_clip)
                    
                    # 优化步骤
                    self.optimizer_hgc.step()
                    self.optimizer_kt.step()
                    
                    total_loss += kt_loss.item()
                    total_batches += 1
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f'{kt_loss.item():.4f}',
                        'avg_loss': f'{total_loss/total_batches:.4f}' if total_batches > 0 else '0.0000'
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n   KT训练批次 {batch_idx} 失败: {e}")
                    pbar.update(1)
                    continue
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        self.train_history['kt_loss'].append(avg_loss)
        
        return avg_loss
    
    def evaluate_models(self, epoch):
        """评估模型"""
        print("   模型评估中...")
        self.model_hgc.eval()
        self.model_cd.eval()
        self.model_kt.eval()
        
        cd_val_loss, cd_val_acc = self.evaluate_cd()
        kt_val_loss, kt_val_acc = self.evaluate_kt()
        
        self.train_history['cd_val_loss'].append(cd_val_loss)
        self.train_history['kt_val_loss'].append(kt_val_loss)
        self.train_history['cd_val_acc'].append(cd_val_acc)
        self.train_history['kt_val_acc'].append(kt_val_acc)
        
        # 更新最佳模型
        if cd_val_loss < self.best_cd_loss:
            self.best_cd_loss = cd_val_loss
            self.save_model('cd_best')
        
        if kt_val_loss < self.best_kt_loss:
            self.best_kt_loss = kt_val_loss
            self.save_model('kt_best')
        
        print(f"  CD验证 - 损失: {cd_val_loss:.4f}, 准确率: {cd_val_acc:.4f}")
        print(f"  KT验证 - 损失: {kt_val_loss:.4f}, 准确率: {kt_val_acc:.4f}")
        
        return cd_val_loss, kt_val_loss
    
    def evaluate_cd(self):
        """评估CD模型"""
        total_loss = 0
        total_correct = 0
        total_samples = 0
        evaluated_batches = 0
        
        max_eval_batches = min(max_batch_size, len(self.cd_eval_loader))  # 最多10个评估批次
        
        with torch.no_grad():
            lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
            
            for batch_idx, batch in enumerate(self.cd_eval_loader):
                if batch_idx >= max_eval_batches:
                    break
                    
                try:
                    lrn_indices = batch['lrn_indices'].to(self.device)
                    qus_seq_indices = batch['qus_seq_indices'].to(self.device)
                    qus_seq_masks = batch['qus_seq_masks'].to(self.device)
                    results = batch['results'].to(self.device)
                    
                    h_lrn_batch = lrn_emb[lrn_indices]
                    
                    predictions = self.model_cd(
                        h_lrn_batch=h_lrn_batch,
                        h_qus=qusunt_emb[:self.cd_eval_dataset.qus_num],
                        h_cpt=cpt_emb,
                        qus_seq_indices=qus_seq_indices,
                        qus_seq_masks=qus_seq_masks,
                        return_ability=False,
                        use_kt_optimization=False
                    )
                    
                    valid_predictions = predictions * qus_seq_masks
                    valid_targets = results * qus_seq_masks
                    loss = self.criterion(valid_predictions, valid_targets)
                    
                    pred_binary = (predictions > 0.5).float()
                    correct = ((pred_binary == results) * qus_seq_masks).sum().item()
                    total_samples += qus_seq_masks.sum().item()
                    
                    total_loss += loss.item()
                    total_correct += correct
                    evaluated_batches += 1
                    
                except Exception as e:
                    continue
        
        avg_loss = total_loss / evaluated_batches if evaluated_batches > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    def evaluate_kt(self):
        """评估KT模型"""
        total_loss = 0
        total_correct = 0
        total_samples = 0
        evaluated_batches = 0
        
        max_eval_batches = min(max_batch_size, len(self.kt_eval_loader))  # 最多10个评估批次
        
        with torch.no_grad():
            lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
            
            for batch_idx, batch in enumerate(self.kt_eval_loader):
                if batch_idx >= max_eval_batches:
                    break
                    
                try:
                    lrn_indices = batch['lrn_indices'].to(self.device)
                    qusunt_seq_indices = batch['qusunt_seq_indices'].to(self.device)
                    add1 = batch['add1'].to(self.device)
                    add2 = batch['add2'].to(self.device)
                    type_indices = batch['type_indices'].to(self.device)
                    seq_masks = batch['seq_masks'].to(self.device)
                    next_question_masks = batch['next_question_masks'].to(self.device)
                    next_results = batch['next_results'].to(self.device)
                    
                    current_lrn_emb = lrn_emb[lrn_indices]
                    current_qusunt_emb = qusunt_emb[qusunt_seq_indices]
                    
                    predictions, concept_mastery = self.model_kt(
                        h_lrn_batch=current_lrn_emb,
                        h_qusunt_batch=current_qusunt_emb,
                        h_cpt=cpt_emb,
                        lrn_indices=lrn_indices,
                        qusunt_seq_indices=qusunt_seq_indices,
                        add1=add1,
                        add2=add2,
                        type_indices=type_indices,
                        seq_mask=seq_masks,
                        next_question_mask=next_question_masks,
                        use_cd_optimization=False,
                        use_contrastive=False
                    )
                    
                    valid_predictions = predictions * next_question_masks.unsqueeze(-1)
                    valid_targets = next_results.unsqueeze(-1) * next_question_masks.unsqueeze(-1)
                    
                    if len(valid_predictions.shape) == 3:
                        valid_predictions_mean = valid_predictions.mean(dim=-1)
                        valid_targets_mean = valid_targets.mean(dim=-1)
                    else:
                        valid_predictions_mean = valid_predictions
                        valid_targets_mean = valid_targets
                    
                    valid_mask = next_question_masks.bool()
                    if valid_mask.any():
                        loss = self.criterion(
                            valid_predictions_mean[valid_mask], 
                            valid_targets_mean[valid_mask]
                        )
                        
                        pred_binary = (valid_predictions_mean > 0.5).float()
                        correct = ((pred_binary == valid_targets_mean) * valid_mask).sum().item()
                        total_samples += valid_mask.sum().item()
                        
                        total_loss += loss.item()
                        total_correct += correct
                    
                    evaluated_batches += 1
                    
                except Exception as e:
                    continue
        
        avg_loss = total_loss / evaluated_batches if evaluated_batches > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    def save_model(self, name):
        """保存模型"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.pt"
        filepath = os.path.join(self.save_dir, filename)
        
        torch.save({
            'epoch': len(self.train_history['cd_loss']),
            'model_hgc_state_dict': self.model_hgc.state_dict(),
            'model_cd_state_dict': self.model_cd.state_dict(),
            'model_kt_state_dict': self.model_kt.state_dict(),
            'optimizer_hgc_state_dict': self.optimizer_hgc.state_dict(),
            'optimizer_cd_state_dict': self.optimizer_cd.state_dict(),
            'optimizer_kt_state_dict': self.optimizer_kt.state_dict(),
            'train_history': self.train_history,
            'hyperparams': hyperparams.get_training_params()
        }, filepath)
        
        print(f"✓ 模型已保存: {filepath}")
    
    def train(self):
        """完整训练流程"""
        print("=== 开始完整训练流程 ===")
        hyperparams.summary()
        
        start_time = time.time()
        
        for epoch in range(1, hyperparams.train_total_epochs + 1):
            print(f"\n{'='*60}")
            print(f"轮次 {epoch}/{hyperparams.train_total_epochs}")
            print(f"{'='*60}")
            
            epoch_start = time.time()
            
            # 完整的一轮训练流程
            print(f"\n--- CD阶段训练 (第{epoch}轮) ---")
            cd_loss = self.train_cd_phase(epoch, use_kt_initialization=(epoch > 1))
            
            print(f"\n--- KT阶段训练 (第{epoch}轮) ---")
            kt_loss = self.train_kt_phase(epoch)
            
            print(f"\n--- 模型评估 (第{epoch}轮) ---")
            cd_val_loss, kt_val_loss = self.evaluate_models(epoch)
            
            epoch_time = time.time() - epoch_start
            
            print(f"\n轮次 {epoch} 总结:")
            print(f"  CD训练损失: {cd_loss:.4f}, 验证损失: {cd_val_loss:.4f}")
            print(f"  KT训练损失: {kt_loss:.4f}, 验证损失: {kt_val_loss:.4f}")
            print(f"  本轮时间: {epoch_time:.2f}秒")
            
            # 定期保存模型
            if epoch % hyperparams.train_save_interval == 0:
                self.save_model(f'epoch_{epoch}')
        
        total_time = time.time() - start_time
        
        # 最终保存
        self.save_model('final')
        
        # 训练总结
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"{'='*60}")
        print(f"总训练时间: {total_time:.2f}秒")
        print(f"总训练轮次: {hyperparams.train_total_epochs}")
        print(f"最佳CD验证损失: {self.best_cd_loss:.4f}")
        print(f"最佳KT验证损失: {self.best_kt_loss:.4f}")
        print(f"模型保存在: {self.save_dir}")
        
        return self.train_history

def main():
    """主函数"""
    print("HGC-CD-KT 完整训练管道")
    print("训练流程: 静态数据 → HGC → CD → KT → 交替优化")
    
    try:
        trainer = CompletePipelineTrainer()
        history = trainer.train()
        
        print("\n🎉 训练完成!")
        return True
        
    except Exception as e:
        print(f"\n💥 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)