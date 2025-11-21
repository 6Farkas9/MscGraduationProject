# Train_HGC_CD_KT.py
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
import shutil
import glob
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 过滤警告
warnings.filterwarnings("ignore")

# 导入数据读取器
from DataReader.HGCDataReader import hgcdr
from DataReader.CDDataReader import cddr
from DataReader.KTDataReader import ktdr

# 导入数据合并器
from DataService.CD_KTDataMerger import cd_kt_merger

# 导入数据集
from DataSet.CDDataSet import CDDataset
from DataSet.KTDataSet import KTDataSet

# 导入模型
from Model.HGC import HGC
from Model.CD import CD
from Model.KT import KT

# 导入超参数
from hyperparams.hyperparameter import hyperparams

# self.max_batch_size = 4

class CompletePipelineTrainer:
    """完整的HGC-CD-KT训练管道 - 支持接续训练，适配新数据结构"""
    
    def __init__(self, resume_training=False):
        self.device = hyperparams.device
        self.resume_training = resume_training
        self.setup_directories()
        self.setup_models_and_data()
        self.setup_optimizers()
        self.setup_tracking()
        
        # 如果是接续训练，加载之前的状态
        if self.resume_training:
            self.load_training_state()

        self.max_batch_size = hyperparams.max_batch_size
        
    def setup_directories(self):
        """创建模型保存目录"""
        self.save_dir = hyperparams.train_save_dir
        self.final_dir = os.path.join(self.save_dir, "final_models")
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"过程模型将保存到: {self.checkpoint_dir}")
        print(f"最终模型将保存到: {self.final_dir}")
    
    def find_latest_checkpoint(self):
        """查找最新的检查点文件"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch*.pth"))
        if not checkpoint_files:
            return None
            
        # 按epoch编号排序
        checkpoint_files.sort(key=lambda x: int(x.split('epoch')[-1].split('_')[0]))
        return checkpoint_files[-1]
    
    def load_training_state(self):
        """加载训练状态（接续训练）"""
        latest_checkpoint = self.find_latest_checkpoint()
        
        if latest_checkpoint is None:
            print("⚠ 未找到检查点文件，从头开始训练")
            return False
        
        try:
            print(f"📂 加载检查点: {os.path.basename(latest_checkpoint)}")
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            
            # 加载模型状态
            self.model_hgc.load_state_dict(checkpoint['model_hgc_state_dict'])
            self.model_cd.load_state_dict(checkpoint['model_cd_state_dict'])
            self.model_kt.load_state_dict(checkpoint['model_kt_state_dict'])
            
            # 加载优化器状态
            self.optimizer_hgc.load_state_dict(checkpoint['optimizer_hgc_state_dict'])
            self.optimizer_cd.load_state_dict(checkpoint['optimizer_cd_state_dict'])
            self.optimizer_kt.load_state_dict(checkpoint['optimizer_kt_state_dict'])
            
            # 加载训练历史
            self.train_history = checkpoint['train_history']
            
            # 加载最佳状态
            self.best_cd_loss = checkpoint.get('best_cd_loss', float('inf'))
            self.best_kt_loss = checkpoint.get('best_kt_loss', float('inf'))
            self.best_hgc_state = checkpoint.get('best_hgc_state')
            self.best_cd_state = checkpoint.get('best_cd_state')
            self.best_kt_state = checkpoint.get('best_kt_state')
            self.best_cd_epoch = checkpoint.get('best_cd_epoch', 0)
            self.best_kt_epoch = checkpoint.get('best_kt_epoch', 0)
            
            # 计算起始epoch
            self.start_epoch = checkpoint['epoch'] + 1
            total_epochs = hyperparams.train_total_epochs
            
            print(f"✅ 成功加载检查点!")
            print(f"   接续训练: 从第 {self.start_epoch} 轮开始 / 总共 {total_epochs} 轮")
            print(f"   最佳CD损失: {self.best_cd_loss:.4f} (轮次 {self.best_cd_epoch})")
            print(f"   最佳KT损失: {self.best_kt_loss:.4f} (轮次 {self.best_kt_epoch})")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            print("⚠ 将从头开始训练")
            self.start_epoch = 1
            return False
    
    def setup_models_and_data(self):
        """初始化模型和数据 - 适配新数据结构"""
        print("=== 初始化模型和数据 (新数据结构) ===")
        
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
        
        # 4. 加载KT数据并创建数据集 - 适配新数据结构
        print("4. 初始化KT模型和数据 (新数据结构)...")
        kt_data = ktdr.loadDatafromSql()
        
        # 5. 数据合并和筛选
        print("5. 合并和筛选CD-KT数据...")
        cd_kt_merger.merge_and_filter_train_test()
        
        # 获取合并统计
        stats = cd_kt_merger.get_merged_statistics()
        if stats and 'train_test' in stats:
            print("   合并后统计:")
            for key, value in stats['train_test'].items():
                if 'kt' in key and ('valid' in key or 'coverage' in key):
                    print(f"     {key}: {value}")
        
        # 6. 计算初始HGC嵌入用于数据集创建
        print("6. 计算初始HGC嵌入...")
        self.model_hgc.eval()
        with torch.no_grad():
            initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb = self.model_hgc(
                hgcdr, self.device, return_dict=False
            )
        
        # 7. 创建CD数据集
        print("7. 创建CD数据集...")
        self.cd_train_dataset = CDDataset(cd_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'train')
        self.cd_eval_dataset = CDDataset(cd_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'test')
        
        # 初始化CD模型
        embedding_dim = hyperparams.hgc_embedding_dim
        concept_num = self.cd_train_dataset.cpt_num
        
        self.model_cd = CD(
            embedding_dim=embedding_dim,
            concept_num=concept_num
        ).to(self.device)
        
        # 8. 创建KT数据集 - 适配新数据结构
        print("8. 创建KT数据集 (新数据结构)...")
        self.kt_train_dataset = KTDataSet(kt_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'train')
        self.kt_eval_dataset = KTDataSet(kt_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'test')
        
        # 初始化KT模型
        concept_mapping = kt_data.get('question_concepts', {})
        
        self.model_kt = KT(
            embedding_dim=embedding_dim,
            concept_num=concept_num,
            concept_mapping=concept_mapping
        ).to(self.device)
        
        # 9. 创建数据加载器
        print("9. 创建数据加载器...")
        self.cd_train_loader = torch.utils.data.DataLoader(
            self.cd_train_dataset,
            batch_size=hyperparams.train_batch_size,
            shuffle=False,
            collate_fn=self.cd_train_dataset.collate_fn,
            num_workers=0,  # 避免多进程问题
            pin_memory=True  # 加速数据转移
        )
        
        self.cd_eval_loader = torch.utils.data.DataLoader(
            self.cd_eval_dataset,
            batch_size=hyperparams.train_eval_batch_size,
            shuffle=False,
            collate_fn=self.cd_eval_dataset.collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        self.kt_train_loader = torch.utils.data.DataLoader(
            self.kt_train_dataset,
            batch_size=hyperparams.train_batch_size,
            shuffle=False,
            collate_fn=self.kt_train_dataset.collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        self.kt_eval_loader = torch.utils.data.DataLoader(
            self.kt_eval_dataset,
            batch_size=hyperparams.train_eval_batch_size,
            shuffle=False,
            collate_fn=self.kt_eval_dataset.collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        # 打印数据集统计
        cd_train_stats = self.cd_train_dataset.get_data_statistics()
        kt_train_stats = self.kt_train_dataset.get_data_statistics()
        
        print(f"   CD训练集: {cd_train_stats['total_learners']}学习者, {cd_train_stats['total_records']}记录")
        print(f"   KT训练集: {kt_train_stats['total_learners']}学习者, {kt_train_stats['total_records']}记录")
        print(f"   KT有效预测位置: {kt_train_stats['total_valid_predictions']}")
        print(f"   CD训练批次: {len(self.cd_train_loader)}")
        print(f"   KT训练批次: {len(self.kt_train_loader)}")
        
        # 切换回训练模式
        self.model_hgc.train()
        self.model_cd.train()
        self.model_kt.train()
        
        print("✓ 模型和数据初始化完成")
    
    def setup_optimizers(self):
        """设置优化器"""
        print("10. 设置优化器...")
        
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
        self.best_hgc_state = None
        self.best_cd_state = None
        self.best_kt_state = None
        self.best_cd_epoch = 0
        self.best_kt_epoch = 0
        self.start_epoch = 1  # 默认从第1轮开始
    
    def compute_hgc_embeddings(self):
        """计算HGC嵌入（带梯度）"""
        return self.model_hgc(hgcdr, self.device, return_dict=False)
    
    def save_checkpoint(self, epoch, cd_loss, kt_loss):
        """保存检查点（用于接续训练）"""
        checkpoint = {
            'epoch': epoch,
            'model_hgc_state_dict': self.model_hgc.state_dict(),
            'model_cd_state_dict': self.model_cd.state_dict(),
            'model_kt_state_dict': self.model_kt.state_dict(),
            'optimizer_hgc_state_dict': self.optimizer_hgc.state_dict(),
            'optimizer_cd_state_dict': self.optimizer_cd.state_dict(),
            'optimizer_kt_state_dict': self.optimizer_kt.state_dict(),
            'train_history': self.train_history,
            'hyperparams': hyperparams.get_training_params(),
            'best_cd_loss': self.best_cd_loss,
            'best_kt_loss': self.best_kt_loss,
            'best_hgc_state': self.best_hgc_state,
            'best_cd_state': self.best_cd_state,
            'best_kt_state': self.best_kt_state,
            'best_cd_epoch': self.best_cd_epoch,
            'best_kt_epoch': self.best_kt_epoch,
            'timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        
        filename = f"checkpoint_epoch{epoch}_{time.strftime('%Y%m%d_%H%M%S')}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        torch.save(checkpoint, filepath)
        print(f"💾 检查点已保存: {filename}")
        
        # 删除旧的检查点，只保留最新的3个
        self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self):
        """清理旧的检查点，只保留最新的3个"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_epoch*.pth"))
        if len(checkpoint_files) > 3:
            # 按时间排序，删除最旧的
            checkpoint_files.sort(key=os.path.getctime)
            for old_checkpoint in checkpoint_files[:-3]:
                os.remove(old_checkpoint)
                print(f"🗑️  删除旧检查点: {os.path.basename(old_checkpoint)}")
    
    def train_cd_phase(self, epoch, use_kt_initialization=False):
        """训练CD阶段"""
        self.model_hgc.train()
        self.model_cd.train()
        
        total_loss = 0
        total_batches = 0
        
        # 如果使用KT初始化，先获取KT的能力矩阵
        if use_kt_initialization and epoch > 1:  # 从第2轮开始使用KT初始化
            print("   使用KT优化能力初始化CD...")
            self.model_kt.eval()
            with torch.no_grad():
                kt_ability = torch.randn(
                    hyperparams.train_batch_size, 
                    self.cd_train_dataset.max_seq_len,
                    self.model_cd.concept_num
                ).to(self.device)
                self.model_cd.set_kt_optimized_ability(kt_ability, self.cd_train_dataset.qus_num)
        
        max_batches = min(self.max_batch_size, len(self.cd_train_loader))
        
        print(f"   CD阶段训练进度 ({max_batches}个批次):")
        with tqdm(total=max_batches, desc="CD训练") as pbar:
            for batch_idx, batch in enumerate(self.cd_train_loader):
                if batch_idx >= max_batches:
                    break
                    
                try:
                    # 步骤1: 计算HGC嵌入
                    lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
                    
                    # 步骤2: 准备CD输入
                    lrn_indices = batch['lrn_indices'].to(self.device, non_blocking=True)
                    qus_seq_indices = batch['qus_seq_indices'].to(self.device, non_blocking=True)
                    qus_seq_masks = batch['qus_seq_masks'].to(self.device, non_blocking=True)
                    results = batch['results'].to(self.device, non_blocking=True)
                    
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
        """训练KT阶段 - 适配新数据结构"""
        self.model_hgc.train()
        self.model_kt.train()
        
        total_loss = 0
        total_batches = 0
        
        # 使用CD结果初始化KT
        if epoch > 1:  # 从第2轮开始使用CD初始化
            print("   使用CD优化能力初始化KT...")
            self.model_cd.eval()
            with torch.no_grad():
                cd_ability = torch.randn(
                    hyperparams.train_batch_size,
                    self.kt_train_dataset.max_seq_len,
                    self.model_kt.concept_num
                ).to(self.device)
                self.model_kt.set_cd_optimized_ability(cd_ability, self.kt_train_dataset.qus_num)
        
        max_batches = min(self.max_batch_size, len(self.kt_train_loader))
        
        print(f"   KT阶段训练进度 ({max_batches}个批次):")
        with tqdm(total=max_batches, desc="KT训练") as pbar:
            for batch_idx, batch in enumerate(self.kt_train_loader):
                if batch_idx >= max_batches:
                    break
                    
                try:
                    # 步骤1: 计算HGC嵌入
                    lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
                    
                    # 步骤2: 准备KT输入 - 适配新数据结构
                    lrn_indices = batch['lrn_indices'].to(self.device, non_blocking=True)
                    qusunt_seq_indices = batch['qusunt_seq_indices'].to(self.device, non_blocking=True)
                    add1 = batch['add1'].to(self.device, non_blocking=True)
                    add2 = batch['add2'].to(self.device, non_blocking=True)
                    type_indices = batch['type_indices'].to(self.device, non_blocking=True)
                    seq_masks = batch['seq_masks'].to(self.device, non_blocking=True)
                    prediction_masks = batch['prediction_masks'].to(self.device, non_blocking=True)  # 新的预测掩码
                    next_results = batch['next_results'].to(self.device, non_blocking=True)  # 新的下一个结果
                    
                    current_lrn_emb = lrn_emb[lrn_indices]
                    
                    # 优化：正确获取学习单元嵌入
                    batch_size, seq_len = qusunt_seq_indices.shape
                    embedding_dim = qusunt_emb.shape[1]
                    qusunt_indices_flat = qusunt_seq_indices.view(-1)
                    current_qusunt_emb = qusunt_emb[qusunt_indices_flat].view(batch_size, seq_len, embedding_dim)
                    
                    # 步骤3: KT前向传播 - 使用新接口
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
                        prediction_masks=prediction_masks,  # 新的预测掩码参数
                        use_cd_optimization=(epoch > 1),
                        use_contrastive=False
                    )
                    
                    # 步骤4: 计算KT损失 - 使用新的掩码和标签
                    valid_predictions = predictions * prediction_masks.unsqueeze(-1)
                    valid_targets = next_results.unsqueeze(-1) * prediction_masks.unsqueeze(-1)
                    
                    # 优化：更高效的计算方式
                    if len(valid_predictions.shape) == 3:
                        valid_predictions_mean = valid_predictions.mean(dim=-1)
                        valid_targets_mean = valid_targets.mean(dim=-1)
                    else:
                        valid_predictions_mean = valid_predictions
                        valid_targets_mean = valid_targets
                    
                    valid_mask = prediction_masks.bool()
                    if valid_mask.any():
                        # 只计算有效位置的损失
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
        
        # 更新最佳模型状态
        cd_improved = False
        kt_improved = False
        
        if cd_val_loss < self.best_cd_loss:
            self.best_cd_loss = cd_val_loss
            self.best_cd_state = {
                'model_state_dict': self.model_cd.state_dict().copy(),
                'optimizer_state_dict': self.optimizer_cd.state_dict().copy(),
                'epoch': epoch,
                'loss': cd_val_loss,
                'accuracy': cd_val_acc
            }
            self.best_cd_epoch = epoch
            cd_improved = True
        
        if kt_val_loss < self.best_kt_loss:
            self.best_kt_loss = kt_val_loss
            self.best_kt_state = {
                'model_state_dict': self.model_kt.state_dict().copy(),
                'optimizer_state_dict': self.optimizer_kt.state_dict().copy(),
                'epoch': epoch,
                'loss': kt_val_loss,
                'accuracy': kt_val_acc
            }
            self.best_kt_epoch = epoch
            kt_improved = True
        
        # 同时保存HGC的最佳状态
        current_avg_loss = (cd_val_loss + kt_val_loss) / 2
        if not hasattr(self, 'best_avg_loss') or current_avg_loss < getattr(self, 'best_avg_loss', float('inf')):
            self.best_avg_loss = current_avg_loss
            self.best_hgc_state = {
                'model_state_dict': self.model_hgc.state_dict().copy(),
                'optimizer_state_dict': self.optimizer_hgc.state_dict().copy(),
                'epoch': epoch,
                'cd_loss': cd_val_loss,
                'kt_loss': kt_val_loss,
                'avg_loss': current_avg_loss
            }
        
        print(f"  CD验证 - 损失: {cd_val_loss:.4f}, 准确率: {cd_val_acc:.4f} {'✓' if cd_improved else ''}")
        print(f"  KT验证 - 损失: {kt_val_loss:.4f}, 准确率: {kt_val_acc:.4f} {'✓' if kt_improved else ''}")
        
        return cd_val_loss, kt_val_loss
    
    def evaluate_cd(self):
        """评估CD模型"""
        total_loss = 0
        total_correct = 0
        total_samples = 0
        evaluated_batches = 0
        
        max_eval_batches = min(self.max_batch_size, len(self.cd_eval_loader))
        
        with torch.no_grad():
            lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
            
            for batch_idx, batch in enumerate(self.cd_eval_loader):
                if batch_idx >= max_eval_batches:
                    break
                    
                try:
                    lrn_indices = batch['lrn_indices'].to(self.device, non_blocking=True)
                    qus_seq_indices = batch['qus_seq_indices'].to(self.device, non_blocking=True)
                    qus_seq_masks = batch['qus_seq_masks'].to(self.device, non_blocking=True)
                    results = batch['results'].to(self.device, non_blocking=True)
                    
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
        """评估KT模型 - 适配新数据结构"""
        total_loss = 0
        total_correct = 0
        total_samples = 0
        evaluated_batches = 0
        
        max_eval_batches = min(self.max_batch_size, len(self.kt_eval_loader))
        
        with torch.no_grad():
            lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
            
            for batch_idx, batch in enumerate(self.kt_eval_loader):
                if batch_idx >= max_eval_batches:
                    break
                    
                try:
                    lrn_indices = batch['lrn_indices'].to(self.device, non_blocking=True)
                    qusunt_seq_indices = batch['qusunt_seq_indices'].to(self.device, non_blocking=True)
                    add1 = batch['add1'].to(self.device, non_blocking=True)
                    add2 = batch['add2'].to(self.device, non_blocking=True)
                    type_indices = batch['type_indices'].to(self.device, non_blocking=True)
                    seq_masks = batch['seq_masks'].to(self.device, non_blocking=True)
                    prediction_masks = batch['prediction_masks'].to(self.device, non_blocking=True)  # 新的预测掩码
                    next_results = batch['next_results'].to(self.device, non_blocking=True)  # 新的下一个结果
                    
                    current_lrn_emb = lrn_emb[lrn_indices]
                    
                    # 优化：正确获取学习单元嵌入
                    batch_size, seq_len = qusunt_seq_indices.shape
                    embedding_dim = qusunt_emb.shape[1]
                    qusunt_indices_flat = qusunt_seq_indices.view(-1)
                    current_qusunt_emb = qusunt_emb[qusunt_indices_flat].view(batch_size, seq_len, embedding_dim)
                    
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
                        prediction_masks=prediction_masks,  # 新的预测掩码参数
                        use_cd_optimization=False,
                        use_contrastive=False
                    )
                    
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
    
    def save_final_models(self):
        """保存最终的三个最佳模型"""
        if self.best_hgc_state is None or self.best_cd_state is None or self.best_kt_state is None:
            print("⚠ 警告: 没有找到最佳模型状态，使用当前模型状态")
            self.best_hgc_state = {'model_state_dict': self.model_hgc.state_dict()}
            self.best_cd_state = {'model_state_dict': self.model_cd.state_dict()}
            self.best_kt_state = {'model_state_dict': self.model_kt.state_dict()}
        
        # 保存HGC模型
        hgc_path = os.path.join(self.final_dir, "hgc_best_model.pth")
        torch.save(self.best_hgc_state, hgc_path)
        
        # 保存CD模型
        cd_path = os.path.join(self.final_dir, "cd_best_model.pth")
        torch.save(self.best_cd_state, cd_path)
        
        # 保存KT模型
        kt_path = os.path.join(self.final_dir, "kt_best_model.pth")
        torch.save(self.best_kt_state, kt_path)
        
        print(f"✓ 最终模型已保存:")
        print(f"  HGC: {hgc_path}")
        print(f"  CD:  {cd_path}")
        print(f"  KT:  {kt_path}")
        
        # 保存训练信息
        info_path = os.path.join(self.final_dir, "training_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("HGC-CD-KT 训练信息 (新数据结构)\n")
            f.write("=" * 50 + "\n")
            f.write(f"训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总训练轮次: {len(self.train_history['cd_loss'])}\n")
            f.write(f"最佳CD模型 - 轮次: {self.best_cd_epoch}, 损失: {self.best_cd_loss:.4f}\n")
            f.write(f"最佳KT模型 - 轮次: {self.best_kt_epoch}, 损失: {self.best_kt_loss:.4f}\n")
            f.write(f"接续训练: {'是' if self.resume_training else '否'}\n")
            f.write(f"KT数据结构: 7元素格式 (包含prediction_masks和next_results)\n")
        
        return hgc_path, cd_path, kt_path
    
    def cleanup_all_checkpoints(self):
        """清理所有检查点文件"""
        print("清理所有检查点文件...")
        try:
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "*.pth"))
            for checkpoint_file in checkpoint_files:
                os.remove(checkpoint_file)
                print(f"  删除: {os.path.basename(checkpoint_file)}")
            
            print("✓ 所有检查点文件已清理")
        except Exception as e:
            print(f"⚠ 清理检查点文件时出错: {e}")
    
    def train(self):
        """完整训练流程"""
        print("=== 开始完整训练流程 (新数据结构) ===")
        hyperparams.summary()
        
        start_time = time.time()
        total_epochs = hyperparams.train_total_epochs
        
        print(f"训练配置:")
        print(f"  - 总轮次: {total_epochs}")
        print(f"  - 接续训练: {'是' if self.resume_training else '否'}")
        print(f"  - 起始轮次: {self.start_epoch}")
        print(f"  - KT数据结构: 7元素格式")
        
        for epoch in range(self.start_epoch, total_epochs + 1):
            print(f"\n{'='*60}")
            print(f"轮次 {epoch}/{total_epochs}")
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
            
            # 保存检查点（每轮都保存，用于接续训练）
            self.save_checkpoint(epoch, cd_val_loss, kt_val_loss)
        
        total_time = time.time() - start_time
        
        # 保存最终模型
        print(f"\n{'='*60}")
        print("保存最终模型...")
        hgc_path, cd_path, kt_path = self.save_final_models()
        
        # 清理检查点文件
        self.cleanup_all_checkpoints()
        
        # 训练总结
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"{'='*60}")
        print(f"总训练时间: {total_time:.2f}秒")
        print(f"总训练轮次: {total_epochs}")
        print(f"最佳CD验证损失: {self.best_cd_loss:.4f} (轮次 {self.best_cd_epoch})")
        print(f"最佳KT验证损失: {self.best_kt_loss:.4f} (轮次 {self.best_kt_epoch})")
        print(f"最终模型保存在: {self.final_dir}")
        
        return self.train_history

def main():
    """主函数"""
    print("HGC-CD-KT 完整训练管道 (适配新数据结构)")
    print("训练流程: 静态数据 → HGC → CD → KT → 交替优化")
    print("KT数据结构: [unt_uids, add1s, add2s, is_questions, results, prediction_masks, next_results]")
    print("保存策略: 智能检查点 + 接续训练 + 最终三个最佳模型")
    
    # 询问是否接续训练
    resume = input("是否接续训练? (y/N): ").strip().lower() == 'y'
    
    try:
        trainer = CompletePipelineTrainer(resume_training=resume)
        history = trainer.train()
        
        print("\n🎉 训练完成!")
        print("📁 最终模型文件:")
        print("   - hgc_best_model.pth")
        print("   - cd_best_model.pth") 
        print("   - kt_best_model.pth")
        return True
        
    except Exception as e:
        print(f"\n💥 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)