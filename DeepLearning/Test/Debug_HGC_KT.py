# Debug_HGC_KT.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import defaultdict
import numpy as np
import warnings
import gc

# 过滤嵌套张量警告
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")

from DataReader.HGCDataReader import hgcdr
from DataReader.KTDataReader import ktdr
from DataSet.KTDataSet import KTDataSet
from Model.HGC import HGC
from Model.KT import KT
from hyperparams.hyperparameter import hyperparams

class GradientMonitor:
    """梯度监测器 - 增强版本"""
    def __init__(self):
        self.gradient_history = defaultdict(list)
        self.parameter_history = defaultdict(list)
        
    def register_model(self, model, model_name):
        """注册模型以监测梯度和参数"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 梯度钩子
                param.register_hook(lambda grad, name=name, model_name=model_name: 
                                  self._gradient_hook(grad, name, model_name))
                # 参数值记录
                self.parameter_history[f"{model_name}.{name}"] = []
    
    def _gradient_hook(self, grad, param_name, model_name):
        """梯度钩子函数"""
        if grad is not None:
            grad_norm = grad.norm().item()
            self.gradient_history[f"{model_name}.{param_name}"].append(grad_norm)
        return grad
    
    def record_parameters(self, model, model_name):
        """记录参数值"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_value = param.data.norm().item() if param.data is not None else 0
                self.parameter_history[f"{model_name}.{name}"].append(param_value)
    
    def get_gradient_summary(self):
        """获取梯度摘要"""
        summary = {}
        for param_path, norms in self.gradient_history.items():
            if norms:  # 只处理有梯度值的参数
                summary[param_path] = {
                    'current_norm': norms[-1],
                    'max_norm': max(norms),
                    'min_norm': min(norms),
                    'mean_norm': sum(norms) / len(norms),
                    'update_count': len(norms)
                }
        return summary
    
    def get_parameter_summary(self):
        """获取参数摘要"""
        summary = {}
        for param_path, values in self.parameter_history.items():
            if values:
                summary[param_path] = {
                    'current_norm': values[-1],
                    'max_norm': max(values),
                    'min_norm': min(values),
                    'mean_norm': sum(values) / len(values)
                }
        return summary
    
    def print_gradient_report(self, step):
        """打印梯度报告"""
        print(f"\n=== 梯度报告 (步骤 {step}) ===")
        grad_summary = self.get_gradient_summary()
        param_summary = self.get_parameter_summary()
        
        if not grad_summary:
            print("  没有检测到梯度")
            return
        
        # 按模型分组
        model_grads = defaultdict(dict)
        model_params = defaultdict(dict)
        
        for param_path, grad_info in grad_summary.items():
            model_name, param_name = param_path.split('.', 1)
            model_grads[model_name][param_name] = grad_info
        
        for param_path, param_info in param_summary.items():
            model_name, param_name = param_path.split('.', 1)
            model_params[model_name][param_name] = param_info
        
        for model_name in set(list(model_grads.keys()) + list(model_params.keys())):
            print(f"\n  {model_name}模型:")
            
            # 梯度信息
            if model_name in model_grads:
                print(f"    梯度信息:")
                total_grad_norm = 0
                active_grad_params = 0
                for param_name, grad_info in model_grads[model_name].items():
                    current_norm = grad_info['current_norm']
                    total_grad_norm += current_norm
                    if current_norm > 1e-8:
                        active_grad_params += 1
                        status = "✅"
                    else:
                        status = "⚠️ "
                    print(f"      {status} {param_name}: {current_norm:.6f} (更新{grad_info['update_count']}次)")
                print(f"      总梯度范数: {total_grad_norm:.6f}")
                print(f"      活跃梯度参数: {active_grad_params}/{len(model_grads[model_name])}")
            
            # 参数信息
            if model_name in model_params:
                print(f"    参数信息:")
                total_param_norm = 0
                for param_name, param_info in model_params[model_name].items():
                    current_norm = param_info['current_norm']
                    total_param_norm += current_norm
                    print(f"      📊 {param_name}: {current_norm:.6f}")
                print(f"      总参数范数: {total_param_norm:.6f}")
    
    def check_gradient_health(self):
        """检查梯度健康状况"""
        grad_summary = self.get_gradient_summary()
        
        if not grad_summary:
            return "❌ 无梯度"
        
        # 检查梯度消失/爆炸
        issues = []
        for param_path, grad_info in grad_summary.items():
            current_norm = grad_info['current_norm']
            if current_norm < 1e-10:
                issues.append(f"梯度消失: {param_path} ({current_norm:.2e})")
            elif current_norm > 1000:
                issues.append(f"梯度爆炸: {param_path} ({current_norm:.2e})")
        
        if not issues:
            return "✅ 梯度健康"
        else:
            return f"⚠️ 梯度问题: {', '.join(issues[:3])}"  # 只显示前3个问题

class MemoryManager:
    """内存管理器"""
    @staticmethod
    def clear_memory():
        """清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def get_memory_usage():
        """获取内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            return f"GPU: {allocated:.2f}GB / {cached:.2f}GB"
        else:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024**3
            return f"RAM: {memory_usage:.2f}GB"

class ComprehensiveTester:
    """HGC-KT综合测试器 - 符合预期流程版本"""
    def __init__(self):
        self.device = hyperparams.device
        self.gradient_monitor = GradientMonitor()
        self.memory_manager = MemoryManager()
        self.setup_models()
        
    def setup_models(self):
        """设置模型 - 只初始化，不预计算嵌入"""
        print("=== 初始化HGC-KT模型 ===")
        print(f"内存使用: {self.memory_manager.get_memory_usage()}")
        
        # 1. 加载HGC数据（静态数据）
        print("1. 加载HGC静态数据...")
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
        
        # 3. 加载KT数据
        print("3. 加载KT数据...")
        kt_data = ktdr.loadDatafromSql()
        
        # 4. 创建初始HGC嵌入用于数据集创建（仅用于数据集，不带梯度）
        print("4. 创建数据集...")
        self.model_hgc.eval()
        with torch.no_grad():
            initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb = self.model_hgc(
                hgcdr, self.device, return_dict=False
            )
        
        # 5. 创建KT数据集（使用初始嵌入）
        self.train_dataset = KTDataSet(kt_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'train')
        self.test_dataset = KTDataSet(kt_data, initial_lrn_emb, initial_qusunt_emb, initial_cpt_emb, 'test')
        
        # 打印数据集统计
        train_stats = self.train_dataset.get_data_statistics()
        test_stats = self.test_dataset.get_data_statistics()
        print(f"   训练集: {train_stats['total_learners']}个学习者, {train_stats['total_records']}条记录")
        print(f"   测试集: {test_stats['total_learners']}个学习者, {test_stats['total_records']}条记录")
        
        # 6. 初始化KT模型 - 适配新接口
        print("6. 初始化KT模型...")
        embedding_dim = hyperparams.hgc_embedding_dim
        concept_num = self.train_dataset.cpt_num
        concept_mapping = kt_data.get('question_concepts', {})
        
        self.model_kt = KT(
            embedding_dim=embedding_dim,
            concept_num=concept_num,
            concept_mapping=concept_mapping
        ).to(self.device)
        
        print(f"   KT模型配置:")
        kt_model_info = self.model_kt.get_model_info()
        for key, value in kt_model_info.items():
            print(f"     {key}: {value}")
        
        kt_param_count = self.model_kt.get_parameter_count()
        print(f"     总参数: {kt_param_count['total_parameters']:,}")
        print(f"     可训练参数: {kt_param_count['trainable_parameters']:,}")
        
        # 7. 设置优化器
        print("7. 设置优化器...")
        self.optimizer_hgc = optim.Adam(
            self.model_hgc.parameters(), 
            lr=hyperparams.kt_learning_rate,
            weight_decay=hyperparams.train_weight_decay
        )
        
        self.optimizer_kt = optim.Adam(
            self.model_kt.parameters(),
            lr=hyperparams.kt_learning_rate,
            weight_decay=hyperparams.train_weight_decay
        )
        
        self.criterion = nn.BCELoss()
        
        # 8. 注册梯度监测
        print("8. 注册梯度监测...")
        self.gradient_monitor.register_model(self.model_hgc, "HGC")
        self.gradient_monitor.register_model(self.model_kt, "KT")
        
        print("✓ HGC-KT模型初始化完成")
        print(f"内存使用: {self.memory_manager.get_memory_usage()}")
        
        # 切换回训练模式
        self.model_hgc.train()
        self.model_kt.train()
    
    def compute_hgc_embeddings_with_grad(self):
        """计算带梯度的HGC嵌入 - 每轮都重新计算"""
        # 关键修改：去掉torch.no_grad()，让HGC计算带梯度
        embeddings = self.model_hgc(hgcdr, self.device, return_dict=False)
        return embeddings
    
    def prepare_kt_inputs(self, batch, lrn_emb, qusunt_emb, cpt_emb):
        """准备KT模型输入数据 - 修复维度问题"""
        # 将数据移动到设备
        lrn_indices = batch['lrn_indices'].to(self.device)
        qusunt_seq_indices = batch['qusunt_seq_indices'].to(self.device)
        add1 = batch['add1'].to(self.device)
        add2 = batch['add2'].to(self.device)
        type_indices = batch['type_indices'].to(self.device)
        seq_masks = batch['seq_masks'].to(self.device)
        prediction_masks = batch['prediction_masks'].to(self.device)
        next_results = batch['next_results'].to(self.device)
        
        # 获取当前批次的学习者嵌入
        current_lrn_emb = lrn_emb[lrn_indices]  # [batch_size, embedding_dim]
        
        # 关键修复：正确获取学习单元嵌入
        batch_size, seq_len = qusunt_seq_indices.shape
        embedding_dim = qusunt_emb.shape[1]
        
        # 重塑qusunt_seq_indices以便索引
        qusunt_indices_flat = qusunt_seq_indices.view(-1)
        current_qusunt_emb = qusunt_emb[qusunt_indices_flat].view(batch_size, seq_len, embedding_dim)
        
        return {
            'lrn_indices': lrn_indices,
            'qusunt_seq_indices': qusunt_seq_indices,
            'add1': add1,
            'add2': add2,
            'type_indices': type_indices,
            'seq_masks': seq_masks,
            'prediction_masks': prediction_masks,
            'next_results': next_results,
            'current_lrn_emb': current_lrn_emb,
            'current_qusunt_emb': current_qusunt_emb,
            'cpt_emb': cpt_emb
        }
    
    def train_step(self, batch, batch_idx, epoch):
        """单步训练 - 每步都重新计算HGC嵌入（带梯度）"""
        try:
            # 清理内存
            self.memory_manager.clear_memory()
            
            # 关键修改：每步都重新计算带梯度的HGC嵌入
            lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings_with_grad()
            
            # 准备KT输入数据
            kt_inputs = self.prepare_kt_inputs(batch, lrn_emb, qusunt_emb, cpt_emb)
            
            # 前向传播
            self.optimizer_hgc.zero_grad()
            self.optimizer_kt.zero_grad()
            
            # KT前向传播 - 使用新接口
            predictions, concept_mastery = self.model_kt(
                h_lrn_batch=kt_inputs['current_lrn_emb'],
                h_qusunt_batch=kt_inputs['current_qusunt_emb'],
                h_cpt=kt_inputs['cpt_emb'],
                lrn_indices=kt_inputs['lrn_indices'],
                qusunt_seq_indices=kt_inputs['qusunt_seq_indices'],
                add1=kt_inputs['add1'],
                add2=kt_inputs['add2'],
                type_indices=kt_inputs['type_indices'],
                seq_mask=kt_inputs['seq_masks'],
                prediction_masks=kt_inputs['prediction_masks'],
                use_cd_optimization=False,
                use_contrastive=False
            )
            
            # 计算损失 - 使用新的掩码和标签
            valid_predictions = predictions * kt_inputs['prediction_masks'].unsqueeze(-1)
            valid_targets = kt_inputs['next_results'].unsqueeze(-1) * kt_inputs['prediction_masks'].unsqueeze(-1)
            
            # 对概念维度取平均
            if len(valid_predictions.shape) == 3:
                valid_predictions_mean = valid_predictions.mean(dim=-1)
                valid_targets_mean = valid_targets.mean(dim=-1)
            else:
                valid_predictions_mean = valid_predictions
                valid_targets_mean = valid_targets
            
            # 只计算有有效预测的位置
            valid_mask = kt_inputs['prediction_masks'].bool()
            if valid_mask.any():
                loss = self.criterion(
                    valid_predictions_mean[valid_mask], 
                    valid_targets_mean[valid_mask]
                )
            else:
                loss = torch.tensor(0.001, requires_grad=True, device=self.device)
            
            # 反向传播 - 关键：这会同时优化HGC和KT
            loss.backward()
            
            # 记录参数值
            self.gradient_monitor.record_parameters(self.model_hgc, "HGC")
            self.gradient_monitor.record_parameters(self.model_kt, "KT")
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model_hgc.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.model_kt.parameters(), max_norm=0.5)
            
            # 优化步骤 - 关键：同时更新HGC和KT
            self.optimizer_hgc.step()
            self.optimizer_kt.step()
            
            return loss.item(), True
            
        except Exception as e:
            print(f"  训练步骤 {batch_idx} 失败: {e}")
            import traceback
            traceback.print_exc()
            self.memory_manager.clear_memory()
            return 0.0, False
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        print(f"\n=== 训练阶段 (Epoch {epoch}) ===")
        print(f"内存使用: {self.memory_manager.get_memory_usage()}")
        
        # 创建数据加载器 - 使用更小的批次
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,  # 关键：使用批次大小为1避免内存爆炸
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn
        )
        
        total_loss = 0
        total_batches = 0
        successful_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 限制批次数量
            if batch_idx >= 1:  # 只训练1个批次避免内存问题
                break
                
            loss_value, success = self.train_step(batch, batch_idx, epoch)
            
            if success:
                total_loss += loss_value
                successful_batches += 1
                
                print(f"  批次 {batch_idx}: 损失 = {loss_value:.4f}")
                self.gradient_monitor.print_gradient_report(f"Epoch{epoch}_Batch{batch_idx}")
                grad_health = self.gradient_monitor.check_gradient_health()
                print(f"  梯度健康状况: {grad_health}")
                print(f"  内存使用: {self.memory_manager.get_memory_usage()}")
            
            total_batches += 1
            self.memory_manager.clear_memory()
        
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            print(f"  Epoch {epoch} 平均损失: {avg_loss:.4f}")
        else:
            avg_loss = 0.0
            print(f"  Epoch {epoch} 无成功批次")
        
        return avg_loss
    
    def evaluate(self, epoch):
        """评估模型"""
        print(f"\n=== 评估阶段 (Epoch {epoch}) ===")
        
        self.model_hgc.eval()
        self.model_kt.eval()
        
        # 评估时使用当前HGC模型计算嵌入
        with torch.no_grad():
            lrn_emb, qusunt_emb, cpt_emb = self.model_hgc(hgcdr, self.device, return_dict=False)
        
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn
        )
        
        total_loss = 0
        total_samples = 0
        total_correct = 0
        evaluated_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 1:
                    break
                    
                try:
                    # 准备KT输入数据
                    kt_inputs = self.prepare_kt_inputs(batch, lrn_emb, qusunt_emb, cpt_emb)
                    
                    predictions, concept_mastery = self.model_kt(
                        h_lrn_batch=kt_inputs['current_lrn_emb'],
                        h_qusunt_batch=kt_inputs['current_qusunt_emb'],
                        h_cpt=kt_inputs['cpt_emb'],
                        lrn_indices=kt_inputs['lrn_indices'],
                        qusunt_seq_indices=kt_inputs['qusunt_seq_indices'],
                        add1=kt_inputs['add1'],
                        add2=kt_inputs['add2'],
                        type_indices=kt_inputs['type_indices'],
                        seq_mask=kt_inputs['seq_masks'],
                        prediction_masks=kt_inputs['prediction_masks'],
                        use_cd_optimization=False,
                        use_contrastive=False
                    )
                    
                    valid_predictions = predictions * kt_inputs['prediction_masks'].unsqueeze(-1)
                    valid_targets = kt_inputs['next_results'].unsqueeze(-1) * kt_inputs['prediction_masks'].unsqueeze(-1)
                    
                    if len(valid_predictions.shape) == 3:
                        valid_predictions_mean = valid_predictions.mean(dim=-1)
                        valid_targets_mean = valid_targets.mean(dim=-1)
                    else:
                        valid_predictions_mean = valid_predictions
                        valid_targets_mean = valid_targets
                    
                    valid_mask = kt_inputs['prediction_masks'].bool()
                    if valid_mask.any():
                        loss = self.criterion(
                            valid_predictions_mean[valid_mask], 
                            valid_targets_mean[valid_mask]
                        )
                        total_loss += loss.item()
                        
                        pred_binary = (valid_predictions_mean > 0.5).float()
                        correct = ((pred_binary == valid_targets_mean) * valid_mask).sum().item()
                        total_correct += correct
                        total_samples += valid_mask.sum().item()
                    
                    evaluated_batches += 1
                    
                except Exception as e:
                    print(f"  评估批次 {batch_idx} 失败: {e}")
                    continue
        
        self.model_hgc.train()
        self.model_kt.train()
        
        if evaluated_batches > 0 and total_samples > 0:
            avg_loss = total_loss / evaluated_batches
            accuracy = total_correct / total_samples
            print(f"  测试损失: {avg_loss:.4f}")
            print(f"  测试准确率: {accuracy:.4f}")
            return avg_loss, accuracy
        else:
            print("  无成功评估批次")
            return 0.0, 0.0
    
    def check_gradient_flow(self):
        """检查梯度流动情况"""
        grad_summary = self.gradient_monitor.get_gradient_summary()
        
        if not grad_summary:
            print("  没有检测到梯度流动")
            return False
        
        hgc_has_grad = any('HGC' in key and grad_summary[key]['current_norm'] > 1e-8 for key in grad_summary.keys())
        kt_has_grad = any('KT' in key and grad_summary[key]['current_norm'] > 1e-8 for key in grad_summary.keys())
        
        print(f"  HGC模型梯度: {'✅ 正常' if hgc_has_grad else '❌ 异常'}")
        print(f"  KT模型梯度: {'✅ 正常' if kt_has_grad else '❌ 异常'}")
        
        return hgc_has_grad and kt_has_grad
    
    def run_comprehensive_test(self, num_epochs=2):
        """运行综合测试"""
        print("=== 开始HGC-KT综合测试 ===")
        print("流程: 静态数据 → HGC(带梯度) → KT → 损失 → 优化HGC+KT")
        start_time = time.time()
        
        initial_loss, initial_acc = self.evaluate(0)
        
        train_losses = []
        test_losses = []
        test_accuracies = []
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            test_loss, test_acc = self.evaluate(epoch)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            
            gradient_ok = self.check_gradient_flow()
            if not gradient_ok:
                print("⚠️  警告: 检测到梯度流动问题")
            
            self.memory_manager.clear_memory()
        
        total_time = time.time() - start_time
        
        print(f"\n=== HGC-KT测试总结 ===")
        print(f"总训练时间: {total_time:.2f}秒")
        print(f"训练轮次: {num_epochs}")
        print(f"初始损失: {initial_loss:.4f}, 最终损失: {test_losses[-1]:.4f}")
        print(f"初始准确率: {initial_acc:.4f}, 最终准确率: {test_accuracies[-1]:.4f}")
        
        self.gradient_monitor.print_gradient_report("Final")
        
        loss_improved = initial_loss > 0 and (initial_loss - test_losses[-1] > 0.001)
        acc_improved = test_accuracies[-1] - initial_acc > 0.001
        gradients_ok = self.check_gradient_flow()
        
        print(f"\n=== 测试结果验证 ===")
        print(f"损失改善: {'✅' if loss_improved else '❌'}")
        print(f"准确率改善: {'✅' if acc_improved else '❌'}")
        print(f"梯度流动: {'✅' if gradients_ok else '❌'}")
        print(f"HGC优化: {'✅' if any('HGC' in key for key in self.gradient_monitor.gradient_history.keys()) else '❌'}")
        
        success = gradients_ok
        if success:
            print("🎉 HGC-KT综合测试通过!")
        else:
            print("⚠️  HGC-KT综合测试发现问题")
            
        return success

def main():
    """主函数"""
    print("HGC-KT 管道综合测试 (符合预期流程版本)")
    print("=" * 60)
    print("流程说明:")
    print("  1. 准备静态数据 (HGC初始嵌入 + 元路径)")
    print("  2. 每轮训练:")
    print("     - 计算HGC嵌入(带梯度)")
    print("     - 计算KT结果")  
    print("     - 计算损失")
    print("     - 反向传播优化HGC和KT")
    print("=" * 60)
    
    hyperparams.summary()
    
    try:
        tester = ComprehensiveTester()
        success = tester.run_comprehensive_test(num_epochs=2)
        
        if success:
            print("\n🎉 HGC-KT所有测试通过！")
        else:
            print("\n❌ HGC-KT测试发现问题")
            
    except Exception as e:
        print(f"\n💥 测试错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)