import torch
from torch_geometric.data import Dataset, Data, HeteroData
from typing import Tuple, Dict

# 虽然这里写着dataset，但是不能在这里创建dataset，应该在RR或者CD中创建针对任务要求的dataset

class MOOCDataset(Dataset):
    # 类属性存储共享图结构 (所有实例共享)
    _shared_graphs: Dict[str, Data] = None
    _shared_features: Dict[str, torch.Tensor] = None
    _shared_metadata: Dict = None
    
    @classmethod
    def initialize_from_datareader(cls, datareader):
        """初始化共享图结构和特征"""
        # 从DataReader加载数据
        ids, inits, graphs = datareader.load_data_from_db()
        
        # 存储共享图结构
        cls._shared_graphs = {
            'lsl': graphs[0],  # 学习者-场景-学习者
            'cc': graphs[1],   # 知识点-知识点
            'cac': graphs[2],  # 知识点-领域-知识点
            'csc': graphs[3],  # 知识点-场景-知识点
            'scs': graphs[4],  # 场景-知识点-场景
            'sls': graphs[5]   # 场景-学习者-场景
        }
        
        # 存储共享特征
        cls._shared_features = {
            'learners': inits[0],  # 学习者初始嵌入 [num_learners, feat_dim]
            'scenes': inits[1],    # 场景初始嵌入 [num_scenes, feat_dim]
            'concepts': inits[2]   # 知识点初始嵌入 [num_concepts, feat_dim]
        }
        
        # 存储元数据
        cls._shared_metadata = {
            'learner_ids': ids[0],
            'scene_ids': ids[1],
            'concept_ids': ids[2]
        }
    
    def __init__(self, mode='train', dynamic_seq_len=5):
        """
        Args:
            mode: 'train' 或 'test'
            dynamic_seq_len: 动态序列的最大长度
        """
        super().__init__()
        assert self._shared_graphs is not None, "必须先调用initialize_from_datareader"
        
        self.mode = mode
        self.dynamic_seq_len = dynamic_seq_len
        self.num_learners = len(self._shared_metadata['learner_ids'])
        
        # 加载用户特定数据 (示例，需根据实际数据调整)
        self.user_sequences = self._load_user_sequences()  # 加载用户学习序列
    
    def _load_user_sequences(self) -> Dict[str, list]:
        """加载用户学习序列 (示例实现)"""
        # 这里应该是从数据库或文件加载每个用户的学习序列
        # 返回格式: {learner_id: [scene_id1, scene_id2, ...]}
        return {}  # 实际实现需替换
    
    def len(self):
        return self.num_learners  # 基于用户数量
    
    def get(self, idx) -> Dict[str, any]:
        """返回一个用户的所有数据"""
        learner_id = self._shared_metadata['learner_ids'][idx]
        
        # 获取用户学习序列 (动态需求建模用)
        full_sequence = self.user_sequences.get(learner_id, [])
        
        # 截取或填充到固定长度
        if len(full_sequence) > self.dynamic_seq_len:
            sequence = full_sequence[-self.dynamic_seq_len:]
        else:
            sequence = full_sequence + [0] * (self.dynamic_seq_len - len(full_sequence))  # 用0填充
        
        # 转换为张量
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        seq_mask = torch.tensor([1]*len(full_sequence) + [0]*(self.dynamic_seq_len - len(full_sequence)), 
                          dtype=torch.bool)
        
        return {
            # 用户标识
            'learner_id': learner_id,
            
            # 静态特征
            'static_features': {
                'learner': self._shared_features['learners'][idx],
                'scenes': self._shared_features['scenes'],
                'concepts': self._shared_features['concepts']
            },
            
            # 动态序列
            'dynamic_sequence': sequence_tensor,
            'sequence_mask': seq_mask,
            
            # 图结构引用 (共享)
            'graphs': self._shared_graphs,
            
            # 元数据
            'metadata': {
                'scene_ids': self._shared_metadata['scene_ids'],
                'concept_ids': self._shared_metadata['concept_ids']
            }
        }