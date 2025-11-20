import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

class HyperParameters:
    """
    超参数管理类 - 单例模式
    统一管理HGC、CD、KT模型及数据集的超参数
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HyperParameters, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_device()
            self._setup_hgc_params()
            self._setup_cd_params() 
            self._setup_kt_params()
            self._setup_data_params()
            self._setup_training_params()
            self._initialized = True
    
    def _setup_device(self):
        """设置设备：优先使用CUDA，其次CPU"""
        # if torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        #     self.device_name = 'cuda'
        #     print(f"使用设备: CUDA ({torch.cuda.get_device_name()})")
        # else:
        #     self.device = torch.device('cpu')
        #     self.device_name = 'cpu'
        #     print("使用设备: CPU")
        self.device = torch.device('cpu')
        self.device_name = 'cpu'
    
    def _setup_hgc_params(self):
        """HGC模型超参数"""
        # 通用嵌入维度
        self.hgc_embedding_dim = 64
        
        # GCN配置
        self.hgc_gcn_layers = 2
        self.hgc_activation_slope = 0.2
        
        # 投影层配置
        self.hgc_proj_hidden_dim = 128
        
        # 注意力配置
        self.hgc_attention_heads = 4
        self.hgc_dropout_rate = 0.1
    
    def _setup_cd_params(self):
        """CD模型超参数"""
        # DTR模块配置
        self.cd_dtr_hidden_dims = [256, 128, 64]
        self.cd_dtr_dropout_rate = 0.2
        
        # MIRT模块配置
        self.cd_mirt_scale_init = 1.0
        self.cd_mirt_clamp_range = [-20, 20]
        
        # 能力融合配置
        self.cd_ability_fusion_hidden_dim = 128
        self.cd_use_kt_optimization = True
        
        # 训练配置
        self.cd_learning_rate = 0.001
        self.cd_weight_decay = 1e-5
    
    def _setup_kt_params(self):
        """KT模型超参数"""
        # 对比学习配置
        self.kt_contrastive_temperature = 0.07
        self.kt_contrastive_hidden_dim = 256
        self.kt_momentum = 0.99
        
        # 时序配置
        self.kt_gru_hidden_dim = 32  # 双向GRU，实际隐藏层为64
        self.kt_attention_heads = 4
        
        # 知识点库检索配置
        self.kt_memory_units = 20
        self.kt_memory_dim = 32
        
        # 融合配置
        self.kt_fusion_hidden_dims = [512, 256]
        self.kt_fusion_dropout_rate = 0.3
        
        # 类型处理配置
        self.kt_type_embedding_dim = 64
        self.kt_type_specific_hidden_dim = 32
        
        # 训练配置
        self.kt_learning_rate = 0.001
        self.kt_use_cd_optimization = True
        self.kt_use_contrastive = True
    
    def _setup_data_params(self):
        """数据集超参数"""
        # 序列长度
        self.data_max_seq_len = 128
        self.data_batch_size = 4
        
        # 数据分割
        self.data_train_ratio = 0.8
        self.data_val_ratio = 0.1
        self.data_test_ratio = 0.1
        
        # 嵌入配置
        self.data_sentence_transformer_model = 'all-MiniLM-L6-v2'
        
        # 缓存配置
        self.data_use_cache = True
        self.data_cache_dir = './cache'
    
    def _setup_training_params(self):
        """训练超参数"""
        # 训练轮次配置
        self.train_total_epochs = 3  # 总训练轮次
        self.train_warmup_epochs = 1  # 预热轮次

        # 训练batch数量
        self.max_batch_size = 2
        
        # 批次配置
        self.train_batch_size = 2     # 小批次训练
        self.train_eval_batch_size = 2
        
        # 优化器配置
        self.train_learning_rate = 0.001
        self.train_weight_decay = 1e-5
        self.train_beta1 = 0.9
        self.train_beta2 = 0.999
        self.train_epsilon = 1e-8
        
        # 学习率调度
        self.train_lr_scheduler = 'cosine'
        self.train_min_lr = 1e-6
        
        # 早停配置
        self.train_patience = 10
        self.train_delta = 1e-4
        
        # 梯度配置
        self.train_grad_clip = 1.0
        self.train_accumulation_steps = 1
        
        # 模型保存配置
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 获取PT文件夹路径
        pt_folder_path = os.path.join(root_dir, 'PT')
        self.train_save_dir = pt_folder_path
        self.train_save_interval = 2  # 每隔多少轮保存一次
    
    def get_hgc_params(self):
        """获取HGC相关参数"""
        return {
            'embedding_dim': self.hgc_embedding_dim,
            'gcn_layers': self.hgc_gcn_layers,
            'activation_slope': self.hgc_activation_slope,
            'proj_hidden_dim': self.hgc_proj_hidden_dim,
            'attention_heads': self.hgc_attention_heads,
            'dropout_rate': self.hgc_dropout_rate
        }
    
    def get_cd_params(self):
        """获取CD相关参数"""
        return {
            'embedding_dim': self.hgc_embedding_dim,
            'dtr_hidden_dims': self.cd_dtr_hidden_dims,
            'dtr_dropout_rate': self.cd_dtr_dropout_rate,
            'mirt_scale_init': self.cd_mirt_scale_init,
            'mirt_clamp_range': self.cd_mirt_clamp_range,
            'ability_fusion_hidden_dim': self.cd_ability_fusion_hidden_dim,
            'use_kt_optimization': self.cd_use_kt_optimization,
            'learning_rate': self.cd_learning_rate,
            'weight_decay': self.cd_weight_decay
        }
    
    def get_kt_params(self):
        """获取KT相关参数"""
        return {
            'embedding_dim': self.hgc_embedding_dim,
            'contrastive_temperature': self.kt_contrastive_temperature,
            'contrastive_hidden_dim': self.kt_contrastive_hidden_dim,
            'momentum': self.kt_momentum,
            'gru_hidden_dim': self.kt_gru_hidden_dim,
            'attention_heads': self.kt_attention_heads,
            'memory_units': self.kt_memory_units,
            'memory_dim': self.kt_memory_dim,
            'fusion_hidden_dims': self.kt_fusion_hidden_dims,
            'fusion_dropout_rate': self.kt_fusion_dropout_rate,
            'type_embedding_dim': self.kt_type_embedding_dim,
            'type_specific_hidden_dim': self.kt_type_specific_hidden_dim,
            'learning_rate': self.kt_learning_rate,
            'use_cd_optimization': self.kt_use_cd_optimization,
            'use_contrastive': self.kt_use_contrastive
        }
    
    def get_data_params(self):
        """获取数据相关参数"""
        return {
            'max_seq_len': self.data_max_seq_len,
            'batch_size': self.data_batch_size,
            'train_ratio': self.data_train_ratio,
            'val_ratio': self.data_val_ratio,
            'test_ratio': self.data_test_ratio,
            'sentence_transformer_model': self.data_sentence_transformer_model,
            'use_cache': self.data_use_cache,
            'cache_dir': self.data_cache_dir
        }
    
    def get_training_params(self):
        """获取训练相关参数"""
        return {
            'total_epochs': self.train_total_epochs,
            'warmup_epochs': self.train_warmup_epochs,
            'batch_size': self.train_batch_size,
            'eval_batch_size': self.train_eval_batch_size,
            'learning_rate': self.train_learning_rate,
            'weight_decay': self.train_weight_decay,
            'beta1': self.train_beta1,
            'beta2': self.train_beta2,
            'epsilon': self.train_epsilon,
            'lr_scheduler': self.train_lr_scheduler,
            'min_lr': self.train_min_lr,
            'patience': self.train_patience,
            'delta': self.train_delta,
            'grad_clip': self.train_grad_clip,
            'accumulation_steps': self.train_accumulation_steps,
            'save_dir': self.train_save_dir,
            'save_interval': self.train_save_interval
        }
    
    def summary(self):
        """打印超参数摘要"""
        print("=" * 50)
        print("超参数配置摘要")
        print("=" * 50)
        print(f"设备: {self.device_name}")
        print(f"总训练轮次: {self.train_total_epochs}")
        print(f"批次大小: {self.train_batch_size}")
        print(f"嵌入维度: {self.hgc_embedding_dim}")
        print(f"学习率: {self.train_learning_rate}")
        print(f"模型保存目录: {self.train_save_dir}")
        print("=" * 50)

# 创建全局实例
hyperparams = HyperParameters()

if __name__ == '__main__':
    # 测试单例模式
    hp1 = HyperParameters()
    hp2 = HyperParameters()
    
    print(f"单例模式测试: {hp1 is hp2}")
    
    # 打印配置摘要
    hyperparams.summary()