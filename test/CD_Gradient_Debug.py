# CD_Gradient_Debug.py
import torch
import torch.nn as nn

class SimpleCD(nn.Module):
    """简化的CD模型用于调试"""
    def __init__(self, embedding_dim, concept_num):
        super(SimpleCD, self).__init__()
        self.embedding_dim = embedding_dim
        self.concept_num = concept_num
        
        # 简化的DTR模块
        self.ability_net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, concept_num),
            nn.Sigmoid()
        )
        
        self.difficulty_net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(), 
            nn.Linear(64, concept_num),
            nn.Sigmoid()
        )
        
        # 简化的MIRT
        self.alpha = nn.Parameter(torch.ones(concept_num))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_lrn, h_qus):
        # 计算能力和难度
        ability = self.ability_net(h_lrn)  # [batch_size, concept_num]
        difficulty = self.difficulty_net(h_qus)  # [batch_size, concept_num]
        
        # 简化的IRT公式
        interaction = torch.sum(self.alpha * ability * difficulty, dim=1)
        predictions = torch.sigmoid(self.beta * interaction)
        
        return predictions, ability

def debug_gradient_flow():
    """精确调试梯度流"""
    print("=== 精确梯度调试 ===")
    
    # 创建简单模型
    embedding_dim = 64
    concept_num = 50
    model = SimpleCD(embedding_dim, concept_num)
    
    # 创建模拟数据 - 确保有足够的方差
    batch_size = 4
    seq_len = 10
    
    # 创建有意义的输入数据
    h_lrn = torch.randn(batch_size, embedding_dim) * 0.1 + 0.5  # 集中在0.5附近
    h_qus = torch.randn(batch_size, embedding_dim) * 0.1 + 0.5
    
    # 创建有变化的目标数据
    targets = torch.randint(0, 2, (batch_size,)).float()
    targets = targets * 0.8 + 0.1  # 避免0和1的极端值
    
    print(f"输入数据统计:")
    print(f"  h_lrn: mean={h_lrn.mean():.3f}, std={h_lrn.std():.3f}")
    print(f"  h_qus: mean={h_qus.mean():.3f}, std={h_qus.std():.3f}")
    print(f"  targets: mean={targets.mean():.3f}")
    
    # 前向传播
    predictions, ability = model(h_lrn, h_qus)
    
    print(f"\n前向传播结果:")
    print(f"  predictions: {predictions.shape}, range=[{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  ability: {ability.shape}, range=[{ability.min():.3f}, {ability.max():.3f}]")
    
    # 计算损失
    loss = nn.BCELoss()(predictions, targets)
    print(f"  损失值: {loss.item():.6f}")
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 详细检查梯度
    print(f"\n梯度检查:")
    total_grad_norm = 0
    has_valid_gradient = False
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            
            if grad_norm > 1e-8:  # 有意义的梯度阈值
                has_valid_gradient = True
                print(f"  ✓ {name}: {param.shape} -> 梯度范数 = {grad_norm:.8f}")
            else:
                print(f"  ✗ {name}: {param.shape} -> 梯度范数 = {grad_norm:.8f} (太小)")
        elif param.requires_grad:
            print(f"  ✗ {name}: {param.shape} -> 无梯度")
    
    print(f"\n总梯度范数: {total_grad_norm:.8f}")
    
    if has_valid_gradient:
        print("🎉 梯度流正常!")
        
        # 测试参数更新
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        # 保存初始参数
        initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()
        
        # 更新参数
        optimizer.step()
        
        # 检查参数变化
        params_updated = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                change = (param.data - initial_params[name]).abs().max().item()
                if change > 1e-6:
                    params_updated = True
                    print(f"  ✓ {name}: 参数已更新 (变化: {change:.8f})")
        
        if params_updated:
            print("🎉 参数更新正常!")
        else:
            print("❌ 参数未更新!")
            
    else:
        print("❌ 梯度流异常!")
    
    return has_valid_gradient

if __name__ == '__main__':
    debug_gradient_flow()