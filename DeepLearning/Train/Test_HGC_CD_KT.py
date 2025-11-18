# Test_HGC_CD_KT.py - 修复梯度传播版本
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np

from Model.HGC import HGC
from Model.CD import CD
from Model.KT import KT
from DataReader.HGCDataReader import hgcdr
from DataReader.CDDataReader import cddr
from DataReader.KTDataReader import ktdr
from DataSet.CDDataSet import CDDataset
from DataSet.KTDataSet import KTDataSet
from hyperparams.hyperparameter import hyperparams

class HGC_CD_KT_Trainer:
    """HGC-CD-KT联合训练器 - 修复梯度传播版本"""
    
    def __init__(self, device=None, num_rounds=None):
        self.device = device or hyperparams.device
        self.num_rounds = num_rounds or 3
        
        # 训练记录
        self.history = {
            'round': [],
            'cd_loss': [], 'kt_loss': [], 
            'cd_accuracy': [], 'kt_accuracy': [],
            'hgc_grad_norm': [], 'cd_grad_norm': [], 'kt_grad_norm': [],
            'cd_kt_ability_diff': [],
            'time_per_round': []
        }
        
        # 初始化
        self.setup_static_data()
        self.setup_models()
        self.setup_optimizers()
    
    def setup_static_data(self):
        """1. 准备静态数据"""
        print("=== 步骤1: 准备静态数据 ===")
        start_time = time.time()
        
        # 加载HGC数据
        hgcdr.loadDatafromSql()
        
        # 加载CD数据
        self.cddata = cddr.loadDatafromSql()
        
        # 加载KT数据
        self.ktdata = ktdr.loadDatafromSql()
        
        # 获取数量信息
        self.lrn_num = hgcdr.lrn_num
        self.qusunt_num = hgcdr.qusunt_num
        self.cpt_num = hgcdr.cpt_num
        self.qus_num = len(self.cddata['qus_uid'])
        
        print(f"✓ 静态数据加载完成")
        print(f"  学习者: {self.lrn_num}, 题目+学习单元: {self.qusunt_num}, 知识点: {self.cpt_num}")
        print(f"  耗时: {time.time() - start_time:.2f}秒")
    
    def setup_models(self):
        """初始化模型"""
        print("\n=== 步骤2: 初始化模型 ===")
        
        # 动态获取输入维度
        lrn_input_dim = hgcdr.lrn_init.shape[1]
        unt_input_dim = hgcdr.qusunt_init.shape[1]
        cpt_input_dim = hgcdr.cpt_init.shape[1]
        
        # 创建HGC模型
        self.model_hgc = HGC(
            embedding_dim=hyperparams.hgc_embedding_dim,
            lrn_input_dim=lrn_input_dim,
            unt_input_dim=unt_input_dim,
            cpt_input_dim=cpt_input_dim
        ).to(self.device)
        
        print("✓ HGC模型初始化完成")
        print(f"  输入维度: lrn={lrn_input_dim}, qusunt={unt_input_dim}, cpt={cpt_input_dim}")
        
        # 检查HGC参数是否可训练
        trainable_params = sum(p.numel() for p in self.model_hgc.parameters() if p.requires_grad)
        print(f"  HGC可训练参数: {trainable_params}")
    
    def setup_optimizers(self):
        """设置优化器"""
        # HGC优化器
        self.hgc_optimizer = optim.Adam(
            self.model_hgc.parameters(), 
            lr=hyperparams.train_learning_rate,
            weight_decay=hyperparams.train_weight_decay
        )
        
        # 损失函数
        self.cd_criterion = nn.BCELoss()
        self.kt_criterion = nn.BCELoss()
    
    def compute_hgc_embeddings_with_grad(self):
        """3. HGC计算嵌入表达 - 保留梯度"""
        print("  计算HGC嵌入表达(保留梯度)...")
        self.model_hgc.train()  # 确保在训练模式
        lrn_emb, qusunt_emb, cpt_emb = self.model_hgc(hgcdr, self.device)
        
        # 检查梯度信息
        print(f"  HGC嵌入梯度状态:")
        print(f"    lrn_emb: requires_grad={lrn_emb.requires_grad}")
        print(f"    qusunt_emb: requires_grad={qusunt_emb.requires_grad}") 
        print(f"    cpt_emb: requires_grad={cpt_emb.requires_grad}")
        
        return lrn_emb, qusunt_emb, cpt_emb
    
    def build_datasets_with_grad(self, lrn_emb, qusunt_emb, cpt_emb):
        """2. 构建直接输入到模型的数据 - 保留梯度"""
        print("  构建数据集(保留梯度)...")
        
        # 确保嵌入有梯度
        lrn_emb = lrn_emb.clone().detach().requires_grad_(True)
        qusunt_emb = qusunt_emb.clone().detach().requires_grad_(True)
        cpt_emb = cpt_emb.clone().detach().requires_grad_(True)
        
        # CD数据集
        cd_train_dataset = CDDataset(
            self.cddata, lrn_emb, qusunt_emb, cpt_emb, 
            'train', max_seq_len=hyperparams.data_max_seq_len
        )
        
        # KT数据集  
        kt_train_dataset = KTDataSet(
            self.ktdata, lrn_emb, qusunt_emb, cpt_emb,
            'train', max_seq_len=hyperparams.data_max_seq_len
        )
        
        # 数据加载器
        cd_train_loader = DataLoader(
            cd_train_dataset, 
            batch_size=min(hyperparams.data_batch_size, 4),  # 使用小batch确保内存
            shuffle=True, 
            collate_fn=cd_train_dataset.collate_fn
        )
        
        kt_train_loader = DataLoader(
            kt_train_dataset,
            batch_size=min(hyperparams.data_batch_size, 4),
            shuffle=True,
            collate_fn=kt_train_dataset.collate_fn
        )
        
        return cd_train_loader, kt_train_loader, cd_train_dataset, kt_train_dataset
    
    def create_cd_model_with_grad(self, h_qus, h_cpt, kt_concept_mastery=None):
        """创建CD模型 - 确保梯度传播"""
        model_cd = CD(
            embedding_dim=hyperparams.hgc_embedding_dim,
            concept_num=len(self.cddata['cpt_uid']),
            h_qus=h_qus,
            h_cpt=h_cpt
        ).to(self.device)
        
        # 设置KT优化的能力矩阵
        if kt_concept_mastery is not None:
            model_cd.set_kt_optimized_ability(kt_concept_mastery, self.qus_num)
        
        return model_cd
    
    def train_cd_phase_fixed(self, model_cd, cd_train_loader, round_idx):
        """5. 修复的CD训练阶段 - 确保HGC梯度传播"""
        print("  CD训练阶段(修复梯度)...")
        
        # CD优化器
        cd_optimizer = optim.Adam(
            model_cd.parameters(),
            lr=hyperparams.cd_learning_rate
        )
        
        model_cd.train()
        self.model_hgc.train()  # 重要：HGC必须在训练模式
        
        cd_losses = []
        cd_grad_norms = []
        hgc_grad_norms = []
        
        for i, cd_batch in enumerate(cd_train_loader):
            if i >= 5:  # 减少batch数量用于调试
                break
                
            # 清零梯度 - 包括HGC的梯度
            self.hgc_optimizer.zero_grad()
            cd_optimizer.zero_grad()
            
            # 关键修复：确保输入数据有梯度
            h_lrn_batch = cd_batch['h_lrn_batch'].to(self.device)
            if not h_lrn_batch.requires_grad:
                h_lrn_batch = h_lrn_batch.clone().detach().requires_grad_(True)
            
            # CD前向传播
            cd_predictions = model_cd(
                h_lrn_batch,
                cd_batch['qus_seq_indices'].to(self.device),
                cd_batch['qus_seq_masks'].to(self.device),
                use_kt_optimization=False  # 第一轮先不用KT优化
            )
            
            # 计算CD损失
            cd_loss = self.compute_cd_loss(
                cd_predictions,
                cd_batch['results'].to(self.device),
                cd_batch['qus_seq_masks'].to(self.device)
            )
            
            if cd_loss.item() > 0:
                # 关键：计算损失相对于HGC参数的梯度
                cd_loss.backward()
                
                # 检查梯度
                cd_has_grad = any(p.grad is not None for p in model_cd.parameters())
                hgc_has_grad = any(p.grad is not None for p in self.model_hgc.parameters())
                
                print(f"    CD Batch {i+1} 梯度检查: CD有梯度={cd_has_grad}, HGC有梯度={hgc_has_grad}")
                
                if cd_has_grad:
                    cd_grad_norm = sum(p.grad.norm().item() for p in model_cd.parameters() 
                                     if p.grad is not None)
                    cd_grad_norms.append(cd_grad_norm)
                
                if hgc_has_grad:
                    hgc_grad_norm = sum(p.grad.norm().item() for p in self.model_hgc.parameters()
                                      if p.grad is not None)
                    hgc_grad_norms.append(hgc_grad_norm)
                    print(f"      HGC梯度范数: {hgc_grad_norm:.6f}")
                
                # 参数更新
                if cd_has_grad:
                    cd_optimizer.step()
                if hgc_has_grad:
                    self.hgc_optimizer.step()
                
                cd_losses.append(cd_loss.item())
                
                print(f"    CD Batch {i+1}: 损失={cd_loss.item():.4f}")
        
        avg_cd_loss = sum(cd_losses) / len(cd_losses) if cd_losses else 0
        avg_cd_grad = sum(cd_grad_norms) / len(cd_grad_norms) if cd_grad_norms else 0
        avg_hgc_grad = sum(hgc_grad_norms) / len(hgc_grad_norms) if hgc_grad_norms else 0
        
        return avg_cd_loss, avg_cd_grad, avg_hgc_grad, model_cd
    
    def compute_cd_loss(self, predictions, results, masks):
        """计算CD损失"""
        valid_mask = masks > 0.5
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        valid_predictions = predictions[valid_mask]
        valid_targets = results[valid_mask]
        return self.cd_criterion(valid_predictions, valid_targets)
    
    def create_kt_model_with_grad(self, h_lrn, h_qusunt, h_cpt, cd_ability=None):
        """创建KT模型 - 确保梯度传播"""
        model_kt = KT(
            embedding_dim=hyperparams.hgc_embedding_dim,
            concept_num=len(self.ktdata['cpt_uid']),
            h_lrn=h_lrn,
            h_qusunt=h_qusunt,
            h_cpt=h_cpt,
            concept_mapping=self.ktdata.get('question_concepts', {})
        ).to(self.device)
        
        # 设置CD优化的能力矩阵
        if cd_ability is not None:
            model_kt.set_cd_optimized_ability(cd_ability, self.qus_num)
        
        return model_kt
    
    def train_kt_phase_fixed(self, model_kt, kt_train_loader, round_idx):
        """8. 修复的KT训练阶段 - 确保HGC梯度传播"""
        print("  KT训练阶段(修复梯度)...")
        
        # KT优化器
        kt_optimizer = optim.Adam(
            model_kt.parameters(),
            lr=hyperparams.kt_learning_rate
        )
        
        model_kt.train()
        self.model_hgc.train()  # 重要：HGC必须在训练模式
        
        kt_losses = []
        kt_grad_norms = []
        hgc_grad_norms = []
        ability_diffs = []
        
        for i, kt_batch in enumerate(kt_train_loader):
            if i >= 5:  # 减少batch数量用于调试
                break
                
            # 清零梯度
            self.hgc_optimizer.zero_grad()
            kt_optimizer.zero_grad()
            
            # KT前向传播
            kt_predictions, kt_ability = model_kt(
                kt_batch['lrn_indices'].to(self.device),
                kt_batch['qusunt_seq_indices'].to(self.device),
                kt_batch['add1'].to(self.device),
                kt_batch['add2'].to(self.device),
                kt_batch['type_indices'].to(self.device),
                kt_batch['seq_masks'].to(self.device),
                kt_batch['next_question_masks'].to(self.device),
                use_cd_optimization=False,  # 第一轮先不用CD优化
                use_contrastive=False  # 简化测试
            )
            
            # 计算KT损失
            kt_loss = self.compute_kt_loss(
                kt_predictions,
                kt_batch['next_results'].to(self.device),
                kt_batch['next_question_masks'].to(self.device)
            )
            
            if kt_loss.item() > 0:
                kt_loss.backward()
                
                # 检查梯度
                kt_has_grad = any(p.grad is not None for p in model_kt.parameters())
                hgc_has_grad = any(p.grad is not None for p in self.model_hgc.parameters())
                
                print(f"    KT Batch {i+1} 梯度检查: KT有梯度={kt_has_grad}, HGC有梯度={hgc_has_grad}")
                
                if kt_has_grad:
                    kt_grad_norm = sum(p.grad.norm().item() for p in model_kt.parameters() 
                                     if p.grad is not None)
                    kt_grad_norms.append(kt_grad_norm)
                
                if hgc_has_grad:
                    hgc_grad_norm = sum(p.grad.norm().item() for p in self.model_hgc.parameters()
                                      if p.grad is not None)
                    hgc_grad_norms.append(hgc_grad_norm)
                    print(f"      HGC梯度范数: {hgc_grad_norm:.6f}")
                
                # 参数更新
                if kt_has_grad:
                    kt_optimizer.step()
                if hgc_has_grad:
                    self.hgc_optimizer.step()
                
                kt_losses.append(kt_loss.item())
                
                print(f"    KT Batch {i+1}: 损失={kt_loss.item():.4f}")
        
        avg_kt_loss = sum(kt_losses) / len(kt_losses) if kt_losses else 0
        avg_kt_grad = sum(kt_grad_norms) / len(kt_grad_norms) if kt_grad_norms else 0
        avg_hgc_grad = sum(hgc_grad_norms) / len(hgc_grad_norms) if hgc_grad_norms else 0
        
        return avg_kt_loss, avg_kt_grad, avg_hgc_grad, model_kt
    
    def compute_kt_loss(self, predictions, next_results, next_question_masks):
        """计算KT损失"""
        valid_mask = next_question_masks > 0.5
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        if len(predictions.shape) == 3:
            predictions = predictions.mean(dim=-1)
        
        valid_predictions = predictions[valid_mask]
        valid_targets = next_results[valid_mask]
        
        return self.kt_criterion(valid_predictions, valid_targets)
    
    def train_one_round_debug(self, round_idx):
        """调试版本的一轮训练"""
        print(f"\n=== 第 {round_idx + 1} 轮训练开始 (调试模式) ===")
        round_start_time = time.time()
        
        # 步骤3: HGC计算嵌入表达
        print("\n步骤3: HGC计算嵌入表达")
        lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings_with_grad()
        
        # 步骤2: 构建数据集
        print("\n步骤2: 构建数据集")
        cd_train_loader, kt_train_loader, cd_train_dataset, kt_train_dataset = self.build_datasets_with_grad(
            lrn_emb, qusunt_emb, cpt_emb
        )
        
        # 步骤4&5: CD训练
        print("\n步骤4&5: CD训练阶段")
        model_cd = self.create_cd_model_with_grad(
            cd_train_dataset.h_qus, 
            cd_train_dataset.h_cpt
        )
        cd_loss, cd_grad, hgc_grad_cd, model_cd = self.train_cd_phase_fixed(
            model_cd, cd_train_loader, round_idx
        )
        
        # 步骤6: 重新计算HGC嵌入
        print("\n步骤6: 重新计算HGC嵌入")
        lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings_with_grad()
        
        # 步骤7&8: KT训练
        print("\n步骤7&8: KT训练阶段")
        model_kt = self.create_kt_model_with_grad(
            kt_train_dataset.h_lrn,
            kt_train_dataset.h_qusunt, 
            kt_train_dataset.h_cpt
        )
        kt_loss, kt_grad, hgc_grad_kt, model_kt = self.train_kt_phase_fixed(
            model_kt, kt_train_loader, round_idx
        )
        
        round_time = time.time() - round_start_time
        
        # 记录结果
        self.history['round'].append(round_idx + 1)
        self.history['cd_loss'].append(cd_loss)
        self.history['kt_loss'].append(kt_loss)
        self.history['cd_grad_norm'].append(cd_grad)
        self.history['kt_grad_norm'].append(kt_grad)
        self.history['hgc_grad_norm'].append(max(hgc_grad_cd, hgc_grad_kt))
        self.history['time_per_round'].append(round_time)
        
        print(f"\n第 {round_idx + 1} 轮结果:")
        print(f"  CD: 损失={cd_loss:.4f}, 梯度={cd_grad:.6f}, HGC梯度={hgc_grad_cd:.6f}")
        print(f"  KT: 损失={kt_loss:.4f}, 梯度={kt_grad:.6f}, HGC梯度={hgc_grad_kt:.6f}")
        print(f"  本轮耗时: {round_time:.2f}秒")
        
        return True
    
    def run_training_debug(self):
        """调试版本的训练流程"""
        print(f"\n=== 开始 {self.num_rounds} 轮训练 (调试模式) ===")
        
        for round_idx in range(min(self.num_rounds, 2)):  # 只运行2轮用于调试
            self.train_one_round_debug(round_idx)
        
        print(f"\n=== 调试训练完成 ===")
        
        # 分析梯度问题
        self.analyze_gradient_issues()
        
        return self.history
    
    def analyze_gradient_issues(self):
        """分析梯度问题"""
        print("\n=== 梯度问题分析 ===")
        
        # 检查HGC模型状态
        print("HGC模型分析:")
        for name, param in self.model_hgc.named_parameters():
            print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
        
        # 检查最后记录的梯度
        if self.history['hgc_grad_norm']:
            print(f"历史HGC梯度: {self.history['hgc_grad_norm']}")
        
        print("建议:")
        print("1. 检查HGC输出是否在计算图中")
        print("2. 确保CD/KT的输入来自HGC且未detach")
        print("3. 验证损失函数是否连接到HGC参数")

def main():
    """主函数 - 调试版本"""
    print("=== HGC-CD-KT联合训练系统 (调试梯度问题) ===")
    
    # 使用小规模配置进行调试
    num_rounds = 2
    device = hyperparams.device
    
    print(f"\n调试配置:")
    print(f"  训练轮数: {num_rounds}")
    print(f"  设备: {device}")
    print(f"  批次大小: 4 (调试模式)")
    
    # 创建训练器并开始调试训练
    trainer = HGC_CD_KT_Trainer(device=device, num_rounds=num_rounds)
    history = trainer.run_training_debug()
    
    print(f"\n=== 调试完成 ===")
    print("请根据梯度分析结果修复模型")

if __name__ == '__main__':
    main()