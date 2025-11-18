# hyperparams.py
import torch

class HyperParameters:
    """
    统一超参数配置类
    为HGC、CD、KT三个模型提供统一的超参数管理
    """
    
    def __init__(self):
        # ==================== 设备配置 ====================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 训练设备，自动选择CUDA或CPU
        
        # ==================== 通用训练参数 ====================
        self.epochs = 100
        # 训练总轮数，用于控制训练迭代次数
        
        self.learning_rate = 0.001
        # 基础学习率，用于所有优化器的初始学习率
        
        self.batch_size = 32
        # 批次大小，控制每次训练使用的样本数量
        
        self.max_seq_len = 128
        # 最大序列长度，用于截断或填充序列数据
        
        self.weight_decay = 1e-5
        # 权重衰减，L2正则化系数，防止过拟合
        
        # ==================== HGC模型参数 ====================
        self.hgc_embedding_dim = 64
        # HGC嵌入维度，控制学习者、学习单元、知识点的嵌入向量大小
        
        self.hgc_gcn_layers = 3
        # HGC中GCN卷积层数，控制图卷积的深度
        
        self.hgc_proj_hidden_dim = 64
        # HGC投影层的隐藏维度，用于特征变换的中间层大小
        
        self.hgc_activation = 'leaky_relu'
        # HGC激活函数类型，可选relu、leaky_relu等
        
        self.hgc_activation_slope = 0.1
        # LeakyReLU激活函数的负斜率参数
        
        # ==================== CD模型参数 ====================
        self.cd_embedding_dim = 64
        # CD模型嵌入维度，与HGC保持一致
        
        self.cd_concept_num = 50
        # CD知识点数量，需要根据实际数据调整
        
        self.cd_dtr_hidden_dims = [256, 128]
        # DTR模块隐藏层维度，学生能力计算网络的层结构
        
        self.cd_dropout_rates = [0.2, 0.1, 0.1]
        # CD模型dropout率，分别对应不同层的丢弃概率
        
        self.cd_ability_fusion_hidden_dim = 128
        # 能力融合门控网络的隐藏层维度
        
        # ==================== KT模型参数 ====================
        self.kt_embedding_dim = 64
        # KT模型嵌入维度，与HGC保持一致
        
        self.kt_concept_num = 50
        # KT知识点数量，需要根据实际数据调整
        
        self.kt_hidden_dim = 256
        # KT基础隐藏层维度，用于投影层和编码器
        
        self.kt_temperature = 0.07
        # 对比学习温度参数，控制相似度分布的平滑程度
        
        self.kt_num_heads = 4
        # 多头注意力头数，用于自注意力和融合注意力
        
        self.kt_transformer_layers = 2
        # Transformer编码器层数，用于基础编码器和动量编码器
        
        self.kt_memory_units = 20
        # 知识点库检索模块的记忆单元数量
        
        self.kt_memory_dim = 32
        # 知识点记忆矩阵的维度
        
        self.kt_momentum = 0.99
        # 动量编码器的动量系数，控制参数更新速度
        
        self.kt_dropout_rates = [0.3, 0.2]
        # KT模型dropout率，分别用于不同层
        
        self.kt_ability_fusion_hidden_dim = 128
        # KT能力融合门控网络的隐藏层维度
        
        self.kt_type_embedding_dim = 64
        # 交互类型嵌入维度，用于6种交互类型的表示
        
        # ==================== 联合训练参数 ====================
        self.joint_training_rounds = 3
        # 联合训练轮数，控制HGC-CD-KT交替训练的次数
        
        self.ability_diff_threshold = 0.1
        # CD-KT能力差异阈值，用于监控两个模型的能力一致性
        
        self.contrastive_weight = 1.0
        # 对比学习损失权重，控制对比学习在总损失中的比重
        
    def update_from_data(self, hgcdr, cddata, ktdata):
        """
        根据实际数据更新超参数
        """
        # 更新知识点数量
        self.cd_concept_num = len(cddata['cpt_uid'])
        self.kt_concept_num = len(ktdata['cpt_uid'])
        
        # 更新输入维度（如果需要）
        if hasattr(hgcdr, 'lrn_init'):
            self.hgc_lrn_input_dim = hgcdr.lrn_init.shape[1]
        if hasattr(hgcdr, 'untqus_init'):
            self.hgc_unt_input_dim = hgcdr.untqus_init.shape[1]
        if hasattr(hgcdr, 'cpt_init'):
            self.hgc_cpt_input_dim = hgcdr.cpt_init.shape[1]