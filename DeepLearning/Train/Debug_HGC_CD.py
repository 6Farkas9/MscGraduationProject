# Comprehensive_Test_HGC_CD_Pipeline_Fixed.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import defaultdict
import numpy as np

from DataReader.HGCDataReader import hgcdr
from DataReader.CDDataReader import cddr
from DataSet.CDDataSet import CDDataset
from Model.HGC import HGC
from Model.CD import CD
from hyperparams.hyperparameter import hyperparams

class GradientMonitor:
    """梯度监测器"""
    def __init__(self):
        self.gradient_history = defaultdict(list)
        
    def register_model(self, model, model_name):
        """注册模型以监测梯度"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name, model_name=model_name: 
                                  self._gradient_hook(grad, name, model_name))
    
    def _gradient_hook(self, grad, param_name, model_name):
        """梯度钩子函数"""
        if grad is not None:
            grad_norm = grad.norm().item()
            self.gradient_history[f"{model_name}.{param_name}"].append(grad_norm)
        return grad
    
    def get_gradient_summary(self):
        """获取梯度摘要"""
        summary = {}
        for param_path, norms in self.gradient_history.items():
            if norms:  # 只处理有梯度值的参数
                summary[param_path] = {
                    'current_norm': norms[-1],
                    'max_norm': max(norms),
                    'min_norm': min(norms),
                    'mean_norm': sum(norms) / len(norms)
                }
        return summary
    
    def print_gradient_report(self, step):
        """打印梯度报告"""
        print(f"\n=== 梯度报告 (步骤 {step}) ===")
        summary = self.get_gradient_summary()
        
        if not summary:
            print("  没有检测到梯度")
            return
        
        # 按模型分组
        model_grads = defaultdict(dict)
        for param_path, grad_info in summary.items():
            model_name, param_name = param_path.split('.', 1)
            model_grads[model_name][param_name] = grad_info
        
        for model_name, params in model_grads.items():
            print(f"\n  {model_name}模型梯度:")
            total_norm = 0
            active_params = 0
            for param_name, grad_info in params.items():
                current_norm = grad_info['current_norm']
                total_norm += current_norm
                if current_norm > 1e-8:
                    active_params += 1
                    status = "✅"
                else:
                    status = "⚠️ "
                print(f"    {status} {param_name}: {current_norm:.6f}")
            print(f"    总梯度范数: {total_norm:.6f}")
            print(f"    活跃参数: {active_params}/{len(params)}")

class ComprehensiveTester:
    """综合测试器 - 修复版本"""
    def __init__(self):
        self.device = hyperparams.device
        self.gradient_monitor = GradientMonitor()
        self.setup_models()
        
    def setup_models(self):
        """设置模型"""
        print("=== 初始化模型 ===")
        
        # 1. 加载HGC数据
        print("1. 加载HGC数据...")
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
        
        print(f"   HGC模型参数:")
        hgc_param_count = self.model_hgc.get_parameter_count()
        print(f"     总参数: {hgc_param_count['total_parameters']:,}")
        print(f"     可训练参数: {hgc_param_count['trainable_parameters']:,}")
        
        # 3. 加载CD数据
        print("3. 加载CD数据...")
        cd_data = cddr.loadDatafromSql()
        
        # 4. 创建初始HGC嵌入用于数据集创建
        print("4. 创建初始HGC嵌入...")
        self.model_hgc.eval()
        with torch.no_grad():
            initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb = self.model_hgc(
                hgcdr, self.device, return_dict=False
            )
        
        # 5. 创建CD数据集（使用初始嵌入）
        print("5. 创建CD数据集...")
        self.train_dataset = CDDataset(cd_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'train')
        self.test_dataset = CDDataset(cd_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'test')
        
        # 打印数据集统计
        train_stats = self.train_dataset.get_data_statistics()
        test_stats = self.test_dataset.get_data_statistics()
        print(f"   训练集: {train_stats['total_learners']}个学习者, {train_stats['total_records']}条记录")
        print(f"   测试集: {test_stats['total_learners']}个学习者, {test_stats['total_records']}条记录")
        
        # 6. 初始化CD模型
        print("6. 初始化CD模型...")
        embedding_dim = hyperparams.hgc_embedding_dim
        concept_num = self.train_dataset.cpt_num
        
        self.model_cd = CD(
            embedding_dim=embedding_dim,
            concept_num=concept_num
        ).to(self.device)
        
        print(f"   CD模型配置:")
        cd_model_info = self.model_cd.get_model_info()
        print(f"     嵌入维度: {cd_model_info['embedding_dim']}")
        print(f"     知识点数: {cd_model_info['concept_num']}")
        print(f"     DTR隐藏层: {cd_model_info['dtr_hidden_dims']}")
        
        cd_param_count = self.model_cd.get_parameter_count()
        print(f"     总参数: {cd_param_count['total_parameters']:,}")
        print(f"     可训练参数: {cd_param_count['trainable_parameters']:,}")
        
        # 7. 设置优化器
        print("7. 设置优化器...")
        self.optimizer_hgc = optim.Adam(
            self.model_hgc.parameters(), 
            lr=hyperparams.cd_learning_rate,
            weight_decay=hyperparams.cd_weight_decay
        )
        
        self.optimizer_cd = optim.Adam(
            self.model_cd.parameters(),
            lr=hyperparams.cd_learning_rate,
            weight_decay=hyperparams.cd_weight_decay
        )
        
        self.criterion = nn.BCELoss()
        
        # 8. 注册梯度监测
        print("8. 注册梯度监测...")
        self.gradient_monitor.register_model(self.model_hgc, "HGC")
        self.gradient_monitor.register_model(self.model_cd, "CD")
        
        print("✓ 模型初始化完成")
        
        # 切换回训练模式
        self.model_hgc.train()
        self.model_cd.train()
    
    def compute_hgc_embeddings(self):
        """计算HGC嵌入 - 每次训练步骤重新计算"""
        return self.model_hgc(hgcdr, self.device, return_dict=False)
    
    def train_step(self, batch, batch_idx):
        """单步训练 - 修复版本"""
        try:
            # 每次训练步骤重新计算HGC嵌入
            lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
            
            # 将数据移动到设备
            lrn_indices = batch['lrn_indices'].to(self.device)
            qus_seq_indices = batch['qus_seq_indices'].to(self.device)
            qus_seq_masks = batch['qus_seq_masks'].to(self.device)
            results = batch['results'].to(self.device)
            
            # 获取当前批次的学习者嵌入
            h_lrn_batch = lrn_emb[lrn_indices]
            
            # 前向传播
            self.optimizer_hgc.zero_grad()
            self.optimizer_cd.zero_grad()
            
            # CD前向传播
            predictions = self.model_cd(
                h_lrn_batch=h_lrn_batch,
                h_qus=qusunt_emb[:self.train_dataset.qus_num],
                h_cpt=cpt_emb,
                qus_seq_indices=qus_seq_indices,
                qus_seq_masks=qus_seq_masks,
                return_ability=False,
                use_kt_optimization=False
            )
            
            # 计算损失 - 只计算有效位置
            valid_predictions = predictions * qus_seq_masks
            valid_targets = results * qus_seq_masks
            loss = self.criterion(valid_predictions, valid_targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model_hgc.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.model_cd.parameters(), max_norm=1.0)
            
            # 优化步骤
            self.optimizer_hgc.step()
            self.optimizer_cd.step()
            
            return loss.item(), True
            
        except Exception as e:
            print(f"  训练步骤 {batch_idx} 失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, False
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        print(f"\n=== 训练阶段 (Epoch {epoch}) ===")
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=hyperparams.data_batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn
        )
        
        total_loss = 0
        total_batches = 0
        successful_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 限制批次数量用于测试
            if batch_idx >= 5:  # 减少批次数量避免计算图问题
                break
                
            loss_value, success = self.train_step(batch, batch_idx)
            
            if success:
                total_loss += loss_value
                successful_batches += 1
                
                # 每个批次都打印进度
                print(f"  批次 {batch_idx}: 损失 = {loss_value:.4f}")
                    
                # 每2个批次打印一次梯度报告
                if batch_idx % 2 == 0:
                    self.gradient_monitor.print_gradient_report(f"Epoch{epoch}_Batch{batch_idx}")
            
            total_batches += 1
        
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            print(f"  Epoch {epoch} 平均损失: {avg_loss:.4f} ({successful_batches}/{total_batches} 批次成功)")
        else:
            avg_loss = 0.0
            print(f"  Epoch {epoch} 无成功批次")
        
        return avg_loss
    
    def evaluate(self, epoch):
        """评估模型"""
        print(f"\n=== 评估阶段 (Epoch {epoch}) ===")
        
        self.model_hgc.eval()
        self.model_cd.eval()
        
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=hyperparams.data_batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn
        )
        
        total_loss = 0
        total_samples = 0
        total_correct = 0
        evaluated_batches = 0
        
        with torch.no_grad():
            # 评估时使用固定的HGC嵌入
            lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
            
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 3:  # 减少评估批次数量
                    break
                    
                try:
                    # 将数据移动到设备
                    lrn_indices = batch['lrn_indices'].to(self.device)
                    qus_seq_indices = batch['qus_seq_indices'].to(self.device)
                    qus_seq_masks = batch['qus_seq_masks'].to(self.device)
                    results = batch['results'].to(self.device)
                    
                    # 获取当前批次的学习者嵌入
                    h_lrn_batch = lrn_emb[lrn_indices]
                    
                    # 前向传播
                    predictions = self.model_cd(
                        h_lrn_batch=h_lrn_batch,
                        h_qus=qusunt_emb[:self.test_dataset.qus_num],
                        h_cpt=cpt_emb,
                        qus_seq_indices=qus_seq_indices,
                        qus_seq_masks=qus_seq_masks,
                        return_ability=False,
                        use_kt_optimization=False
                    )
                    
                    # 计算损失
                    valid_predictions = predictions * qus_seq_masks
                    valid_targets = results * qus_seq_masks
                    loss = self.criterion(valid_predictions, valid_targets)
                    total_loss += loss.item()
                    
                    # 计算准确率
                    pred_binary = (predictions > 0.5).float()
                    correct = ((pred_binary == results) * qus_seq_masks).sum().item()
                    total_correct += correct
                    total_samples += qus_seq_masks.sum().item()
                    
                    evaluated_batches += 1
                    
                except Exception as e:
                    print(f"  评估批次 {batch_idx} 失败: {e}")
                    continue
        
        if evaluated_batches > 0:
            avg_loss = total_loss / evaluated_batches
            accuracy = total_correct / total_samples if total_samples > 0 else 0
            print(f"  测试损失: {avg_loss:.4f}")
            print(f"  测试准确率: {accuracy:.4f} ({total_correct}/{int(total_samples)})")
        else:
            avg_loss = 0.0
            accuracy = 0.0
            print("  无成功评估批次")
        
        self.model_hgc.train()
        self.model_cd.train()
        
        return avg_loss, accuracy
    
    def check_gradient_flow(self):
        """检查梯度流动情况"""
        print(f"\n=== 梯度流动检查 ===")
        
        summary = self.gradient_monitor.get_gradient_summary()
        
        if not summary:
            print("  没有检测到梯度流动")
            return False
        
        # 检查关键模块的梯度
        hgc_has_grad = any('HGC' in key and summary[key]['current_norm'] > 1e-8 for key in summary.keys())
        cd_has_grad = any('CD' in key and summary[key]['current_norm'] > 1e-8 for key in summary.keys())
        
        print(f"  HGC模型梯度: {'✅ 正常' if hgc_has_grad else '❌ 异常'}")
        print(f"  CD模型梯度: {'✅ 正常' if cd_has_grad else '❌ 异常'}")
        
        # 检查具体参数
        critical_params = {
            'HGC': ['lrn_proj', 'cpt_proj', 'lrn_gcn_lul', 'unt_gcn_ulu'],
            'CD': ['dtr.l_p_lrn', 'dtr.l_d_qus', 'dtr.l_b_qus', 'mirt.scale']
        }
        
        for model_name, params in critical_params.items():
            print(f"\n  {model_name}关键参数梯度:")
            for param in params:
                param_found = False
                param_has_grad = False
                for key in summary.keys():
                    if model_name in key and param in key:
                        param_found = True
                        if summary[key]['current_norm'] > 1e-8:
                            param_has_grad = True
                        break
                
                status = "✅" if param_has_grad else "❌"
                found_status = "找到" if param_found else "未找到"
                grad_status = "有梯度" if param_has_grad else "无梯度"
                print(f"    {status} {param}: {found_status}, {grad_status}")
        
        return hgc_has_grad and cd_has_grad
    
    def run_comprehensive_test(self, num_epochs=2):
        """运行综合测试"""
        print("=== 开始综合测试 ===")
        start_time = time.time()
        
        # 训练前评估
        print("\n--- 初始评估 ---")
        initial_loss, initial_acc = self.evaluate(0)
        
        # 训练循环
        train_losses = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # 评估
            test_loss, test_acc = self.evaluate(epoch)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            
            # 检查梯度
            gradient_ok = self.check_gradient_flow()
            
            if not gradient_ok:
                print("⚠️  警告: 检测到梯度流动问题")
        
        # 最终评估和总结
        total_time = time.time() - start_time
        
        print(f"\n=== 测试总结 ===")
        print(f"总训练时间: {total_time:.2f}秒")
        print(f"训练轮次: {num_epochs}")
        print(f"初始损失: {initial_loss:.4f}, 最终损失: {test_losses[-1]:.4f}")
        print(f"初始准确率: {initial_acc:.4f}, 最终准确率: {test_accuracies[-1]:.4f}")
        
        # 梯度最终报告
        print(f"\n=== 最终梯度报告 ===")
        self.gradient_monitor.print_gradient_report("Final")
        
        # 检查训练效果
        loss_improved = initial_loss - test_losses[-1] > 0.01
        acc_improved = test_accuracies[-1] - initial_acc > 0.01
        gradients_ok = self.check_gradient_flow()
        
        print(f"\n=== 测试结果验证 ===")
        print(f"损失改善: {'✅' if loss_improved else '❌'} ({initial_loss:.4f} -> {test_losses[-1]:.4f})")
        print(f"准确率改善: {'✅' if acc_improved else '❌'} ({initial_acc:.4f} -> {test_accuracies[-1]:.4f})")
        print(f"梯度流动: {'✅' if gradients_ok else '❌'}")
        
        success = (gradients_ok and (loss_improved or acc_improved))
        
        if success:
            print("🎉 综合测试通过! 模型训练正常，梯度流动良好。")
        else:
            print("⚠️  综合测试发现一些问题，需要进一步检查。")
            
        return success

def main():
    """主函数"""
    print("HGC-CD 管道综合测试 (修复版本)")
    print("=" * 50)
    
    # 打印超参数配置
    hyperparams.summary()
    
    try:
        # 创建测试器
        tester = ComprehensiveTester()
        
        # 运行综合测试
        success = tester.run_comprehensive_test(num_epochs=2)
        
        if success:
            print("\n🎉 所有测试通过！模型可以正常训练。")
        else:
            print("\n❌ 测试发现问题，需要调试。")
            
    except Exception as e:
        print(f"\n💥 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)