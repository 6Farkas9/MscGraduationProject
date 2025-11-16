# Test_HGC_CD_KT.py - 修复版本
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Model.HGC import HGC
from Model.CD import CD
from Model.KT import KT
from DataReader.HGCDataReader import hgcdr
from DataReader.CDDataReader import cddr
from DataReader.KTDataReader import ktdr
from DataSet.CDDataSet import CDDataset
from DataSet.KTDataSet import KTDataSet

class HGC_CD_KT_Trainer:
    """HGC-CD-KT联合训练器 - 修复版本"""
    
    def __init__(self, device='cpu', num_rounds=3):
        self.device = device
        self.num_rounds = num_rounds
        self.setup_models()
        self.setup_data()
        self.setup_optimizers()
        
        # 训练记录
        self.history = {
            'cd_loss': [], 'kt_loss': [], 'cd_accuracy': [], 'kt_accuracy': [],
            'hgc_grad_norm': [], 'cd_grad_norm': [], 'kt_grad_norm': [],
            'cd_kt_ability_diff': []  # 记录CD和KT能力差异
        }
    
    def setup_models(self):
        """初始化所有模型"""
        print("=== 初始化模型 ===")
        
        # 1. 加载HGC数据
        hgcdr.loadDatafromSql()
        
        # 动态获取输入维度
        lrn_input_dim = hgcdr.lrn_init.shape[1]
        unt_input_dim = hgcdr.untqus_init.shape[1]
        cpt_input_dim = hgcdr.cpt_init.shape[1]
        
        # 2. 创建HGC模型
        self.model_hgc = HGC(
            embedding_dim=64,
            lrn_input_dim=lrn_input_dim,
            unt_input_dim=unt_input_dim,
            cpt_input_dim=cpt_input_dim
        ).to(self.device)
        
        print("✓ HGC模型初始化完成")
    
    def setup_data(self):
        """加载所有数据"""
        print("\n=== 加载数据 ===")
        
        # 1. 加载CD数据
        self.cddata = cddr.loadDatafromSql()
        
        # 2. 加载KT数据
        self.ktdata = ktdr.loadDatafromSql()
        
        # 3. 计算HGC嵌入（初始）
        with torch.no_grad():
            self.lrn_emb, self.unt_emb, self.cpt_emb = self.model_hgc(hgcdr, self.device)
        
        # 4. 创建数据集
        self.cd_train_dataset = CDDataset(self.cddata, self.lrn_emb, self.unt_emb, self.cpt_emb, 'train', max_seq_len=128)
        self.cd_test_dataset = CDDataset(self.cddata, self.lrn_emb, self.unt_emb, self.cpt_emb, 'test', max_seq_len=128)
        
        self.kt_train_dataset = KTDataSet(self.ktdata, self.lrn_emb, self.unt_emb, self.cpt_emb, 'train', max_seq_len=128)
        self.kt_test_dataset = KTDataSet(self.ktdata, self.lrn_emb, self.unt_emb, self.cpt_emb, 'test', max_seq_len=128)


        
        # 5. 创建数据加载器
        self.cd_train_loader = DataLoader(self.cd_train_dataset, batch_size=4, shuffle=True, collate_fn=self.cd_train_dataset.collate_fn)
        self.kt_train_loader = DataLoader(self.kt_train_dataset, batch_size=4, shuffle=True, collate_fn=self.kt_train_dataset.collate_fn)
        
        print("✓ 数据加载完成")
        print(f"  CD训练集: {len(self.cd_train_dataset)} 学习者, {self.cd_train_dataset.get_data_statistics()}")
        print(f"  KT训练集: {len(self.kt_train_dataset)} 学习者, {self.kt_train_dataset.get_data_statistics()}")
    
    def setup_optimizers(self):
        """设置优化器"""
        # HGC优化器
        self.hgc_optimizer = optim.Adam(self.model_hgc.parameters(), lr=0.001, weight_decay=1e-5)
        
        # 损失函数
        self.cd_criterion = nn.BCELoss()
        self.kt_criterion = nn.BCELoss()
    
    def compute_cd_loss(self, predictions, results, masks):
        """计算CD损失 - 只对有效位置"""
        valid_mask = masks > 0.5
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        valid_predictions = predictions[valid_mask]
        valid_targets = results[valid_mask]
        return self.cd_criterion(valid_predictions, valid_targets)
    
    def compute_kt_loss(self, predictions, next_results, next_question_masks):
        """计算KT损失 - 只对下一个是题目的位置"""
        valid_mask = next_question_masks > 0.5
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # 根据KT模型输出处理预测值
        if len(predictions.shape) == 3:  # [batch, seq_len, concept_num]
            # 对概念维度取平均，得到每个时间步的总体预测
            predictions = predictions.mean(dim=-1)
        
        valid_predictions = predictions[valid_mask]
        valid_targets = next_results[valid_mask]
        
        return self.kt_criterion(valid_predictions, valid_targets)
    
    def update_embeddings(self):
        """更新HGC嵌入并传递给CD和KT模型"""
        # 重新计算HGC嵌入
        self.lrn_emb, self.unt_emb, self.cpt_emb = self.model_hgc(hgcdr, self.device)
        
        # 更新CD模型的嵌入
        unt_num = self.unt_emb.shape[0] - len(self.cddata['qus_uid'])
        h_qus = self.unt_emb[unt_num:]
        
        # 确保CD模型存在
        if hasattr(self, 'model_cd'):
            with torch.no_grad():
                self.model_cd.h_qus.data = h_qus.data
                self.model_cd.h_cpt.data = self.cpt_emb.data
        
        # 更新KT模型的嵌入
        if hasattr(self, 'model_kt'):
            with torch.no_grad():
                self.model_kt.h_lrn.data = self.lrn_emb.data
                self.model_kt.h_unt.data = self.unt_emb.data  
                self.model_kt.h_cpt.data = self.cpt_emb.data
    
    def train_one_round(self, round_idx):
        """训练一轮：CD → KT → 反馈优化 - 恢复到能工作的版本"""
        print(f"\n=== 第 {round_idx + 1} 轮训练开始 ===")
        
        # 1. 每轮重新计算HGC嵌入（不计算梯度，避免计算图冲突）
        with torch.no_grad():
            self.lrn_emb, self.unt_emb, self.cpt_emb = self.model_hgc(hgcdr, self.device)
        
        # 2. 每轮重新创建CD和KT模型
        unt_num = self.unt_emb.shape[0] - len(self.cddata['qus_uid'])
        h_qus = self.unt_emb[unt_num:]
        
        self.model_cd = CD(
            embedding_dim=64,
            concept_num=len(self.cddata['cpt_uid']),
            h_qus=h_qus,
            h_cpt=self.cpt_emb
        ).to(self.device)
        
        concept_mapping = self.ktdata.get('question_concepts', {})
        self.model_kt = KT(
            embedding_dim=64,
            concept_num=len(self.ktdata['cpt_uid']),
            h_lrn=self.lrn_emb,
            h_unt=self.unt_emb,
            h_cpt=self.cpt_emb,
            concept_mapping=concept_mapping
        ).to(self.device)
        
        # 3. 创建优化器
        cd_optimizer = optim.Adam(self.model_cd.parameters(), lr=0.001)
        kt_optimizer = optim.Adam(self.model_kt.parameters(), lr=0.001)
        
        # 4. CD训练阶段 - 使用原始版本
        print("阶段1: CD训练")
        cd_losses = []
        cd_grad_norms = []
        
        self.model_cd.train()
        self.model_hgc.eval()  # HGC在CD阶段不训练，避免计算图问题
        
        for i, cd_batch in enumerate(self.cd_train_loader):
            if i >= 2:
                break
                
            cd_optimizer.zero_grad()
            
            # CD前向传播
            cd_predictions = self.model_cd(
                cd_batch['h_lrn_batch'].to(self.device),
                cd_batch['qus_seq_indices'].to(self.device),
                cd_batch['qus_seq_masks'].to(self.device)
            )
            
            # 计算CD损失
            cd_loss = self.compute_cd_loss(
                cd_predictions,
                cd_batch['results'].to(self.device),
                cd_batch['qus_seq_masks'].to(self.device)
            )
            
            if cd_loss.item() > 0:
                cd_loss.backward()
                
                cd_grad_norm = sum(p.grad.norm().item() for p in self.model_cd.parameters() if p.grad is not None)
                cd_optimizer.step()
                
                cd_losses.append(cd_loss.item())
                cd_grad_norms.append(cd_grad_norm)
                print(f"  CD Batch {i+1}: 损失={cd_loss.item():.4f}, CD梯度={cd_grad_norm:.6f}")
        
        avg_cd_loss = sum(cd_losses) / len(cd_losses) if cd_losses else 0
        avg_cd_grad = sum(cd_grad_norms) / len(cd_grad_norms) if cd_grad_norms else 0
        
        # 5. KT训练阶段
        print("阶段2: KT训练")
        kt_losses = []
        kt_grad_norms = []
        ability_diffs = []
        
        self.model_kt.train()
        self.model_hgc.eval()  # HGC在KT阶段不训练
        
        for i, kt_batch in enumerate(self.kt_train_loader):
            if i >= 2:
                break
                
            kt_optimizer.zero_grad()
            
            # 使用CD能力初始化KT
            with torch.no_grad():
                cd_ability = self.model_cd.get_ability_matrix(
                    kt_batch['h_lrn_batch'].to(self.device),
                    kt_batch['unt_seq_indices'].to(self.device),
                    kt_batch['seq_masks'].to(self.device),
                    unt_num
                )
                self.model_kt.set_cd_optimized_ability(cd_ability, unt_num)
            
            # KT前向传播
            kt_predictions, kt_ability = self.model_kt(
                kt_batch['lrn_indices'].to(self.device),
                kt_batch['unt_seq_indices'].to(self.device),
                kt_batch['add1'].to(self.device),
                kt_batch['add2'].to(self.device),
                kt_batch['type_indices'].to(self.device),
                kt_batch['seq_masks'].to(self.device),
                kt_batch['next_question_masks'].to(self.device),
                use_cd_optimization=True,
                use_contrastive=False
            )
            
            ability_diff = torch.mean(torch.abs(kt_ability - cd_ability)).item()
            ability_diffs.append(ability_diff)
            
            # 计算KT损失
            kt_loss = self.compute_kt_loss(
                kt_predictions,
                kt_batch['next_results'].to(self.device),
                kt_batch['next_question_masks'].to(self.device)
            )
            
            if kt_loss.item() > 0:
                kt_loss.backward()
                
                kt_grad_norm = sum(p.grad.norm().item() for p in self.model_kt.parameters() if p.grad is not None)
                kt_optimizer.step()
                
                kt_losses.append(kt_loss.item())
                kt_grad_norms.append(kt_grad_norm)
                print(f"  KT Batch {i+1}: 损失={kt_loss.item():.4f}, KT梯度={kt_grad_norm:.6f}, 能力差异={ability_diff:.4f}")
        
        avg_kt_loss = sum(kt_losses) / len(kt_losses) if kt_losses else 0
        avg_kt_grad = sum(kt_grad_norms) / len(kt_grad_norms) if kt_grad_norms else 0
        avg_ability_diff = sum(ability_diffs) / len(ability_diffs) if ability_diffs else 0
        
        # 6. HGC单独训练阶段（如果需要）
        # 如果希望HGC也参与训练，可以在这里添加HGC的训练步骤
        # 但需要确保不重复使用计算图
        
        # 记录结果
        self.history['cd_loss'].append(avg_cd_loss)
        self.history['kt_loss'].append(avg_kt_loss)
        self.history['cd_grad_norm'].append(avg_cd_grad)
        self.history['kt_grad_norm'].append(avg_kt_grad)
        self.history['cd_kt_ability_diff'].append(avg_ability_diff)
        
        print(f"第 {round_idx + 1} 轮结果:")
        print(f"  CD: 损失={avg_cd_loss:.4f}, 梯度={avg_cd_grad:.6f}")
        print(f"  KT: 损失={avg_kt_loss:.4f}, 梯度={avg_kt_grad:.6f}")
        print(f"  CD-KT能力差异: {avg_ability_diff:.4f}")
    
    def run_training(self):
        """运行完整训练流程"""
        print(f"\n=== 开始 {self.num_rounds} 轮训练 ===")
        
        for round_idx in range(self.num_rounds):
            self.train_one_round(round_idx)
        
        self.plot_results()
    
    def plot_results(self):
        """绘制训练结果"""
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 3, 1)
        plt.plot(self.history['cd_loss'], 'b-', label='CD Loss', marker='o')
        plt.plot(self.history['kt_loss'], 'r-', label='KT Loss', marker='s')
        plt.title('Training Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 梯度曲线
        plt.subplot(2, 3, 2)
        plt.plot(self.history['cd_grad_norm'], 'b-', label='CD Gradient', marker='o')
        plt.plot(self.history['kt_grad_norm'], 'r-', label='KT Gradient', marker='s')
        plt.title('Gradient Norm')
        plt.xlabel('Round')
        plt.ylabel('Gradient Norm')
        plt.legend()
        plt.grid(True)
        
        # 能力差异曲线
        plt.subplot(2, 3, 3)
        plt.plot(self.history['cd_kt_ability_diff'], 'g-', label='CD-KT Ability Diff', marker='^')
        plt.title('CD-KT Ability Difference')
        plt.xlabel('Round')
        plt.ylabel('Ability Difference')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('hgc_cd_kt_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n=== 训练完成 ===")
        print(f"最终结果:")
        print(f"  CD损失: {self.history['cd_loss'][-1]:.4f}")
        print(f"  KT损失: {self.history['kt_loss'][-1]:.4f}")
        print(f"  CD-KT能力差异: {self.history['cd_kt_ability_diff'][-1]:.4f}")

def main():
    """主函数 - 修改num_rounds参数来控制训练轮数"""
    num_rounds = 3  # 修改这个参数来指定训练轮数
    
    print(f"开始HGC-CD-KT联合训练，轮数: {num_rounds}")
    
    trainer = HGC_CD_KT_Trainer(device='cpu', num_rounds=num_rounds)
    trainer.run_training()

if __name__ == '__main__':
    main()