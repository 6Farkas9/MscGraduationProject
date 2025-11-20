# save_training_results.py
import sys
import os
import torch
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

# 导入数据服务
from DataService.EmbeddingDataService import embedding_data_service
from DataService.LearnerDataService import learner_data_service
from Data.EmbeddingRepository import embedding_repo
from Data.LearnerRepository import learner_repo

class TrainingResultsSaver:
    """保存训练结果到数据库 - 修正版"""
    
    def __init__(self, model_path):
        self.device = hyperparams.device
        self.model_path = model_path
        self.setup_models()
        self.setup_complete_data()
        
    def setup_models(self):
        """加载训练好的模型"""
        print("=== 加载训练好的模型 ===")
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 1. 加载静态数据获取输入维度
        print("1. 加载静态数据...")
        hgcdr.loadDatafromSql()
        
        # 动态获取输入维度
        lrn_input_dim = hgcdr.lrn_init.shape[1]
        unt_input_dim = hgcdr.qusunt_init.shape[1]
        cpt_input_dim = hgcdr.cpt_init.shape[1]
        
        print(f"   输入维度 - lrn: {lrn_input_dim}, unt: {unt_input_dim}, cpt: {cpt_input_dim}")
        
        # 2. 初始化并加载HGC模型
        print("2. 加载HGC模型...")
        self.model_hgc = HGC(
            embedding_dim=hyperparams.hgc_embedding_dim,
            lrn_input_dim=lrn_input_dim,
            unt_input_dim=unt_input_dim,
            cpt_input_dim=cpt_input_dim
        ).to(self.device)
        
        # 3. 加载模型状态
        print("3. 加载模型权重...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_hgc_state_dict' in checkpoint:
            self.model_hgc.load_state_dict(checkpoint['model_hgc_state_dict'])
            print("   ✓ 从检查点文件加载HGC权重")
        elif 'model_state_dict' in checkpoint:
            self.model_hgc.load_state_dict(checkpoint['model_state_dict'])
            print("   ✓ 从单个模型文件加载HGC权重")
        else:
            self.model_hgc.load_state_dict(checkpoint)
            print("   ✓ 直接加载模型权重")
        
        # 设置为评估模式
        self.model_hgc.eval()
        
        print("✓ 模型加载完成")
    
    def setup_complete_data(self):
        """设置完整数据 - 正确合并训练集和测试集"""
        print("=== 设置完整数据集 ===")
        
        # 1. 计算HGC嵌入
        print("1. 计算HGC嵌入...")
        with torch.no_grad():
            self.lrn_emb, self.qusunt_emb, self.cpt_emb = self.model_hgc(
                hgcdr, self.device, return_dict=False
            )
        
        # 2. 创建完整CD数据集
        print("2. 创建完整CD数据集...")
        cd_data = cddr.loadDatafromSql()
        
        # 正确合并CD数据：保持原有的数据结构
        complete_cd_data = self.create_complete_cd_data(cd_data)
        
        # 使用合并后的数据创建数据集
        self.cd_full_dataset = CDDataset(complete_cd_data, self.lrn_emb, self.qusunt_emb, self.cpt_emb, 'train')
        
        # 3. 创建完整KT数据集  
        print("3. 创建完整KT数据集...")
        kt_data = ktdr.loadDatafromSql()
        
        # 正确合并KT数据：保持原有的数据结构
        complete_kt_data = self.create_complete_kt_data(kt_data)
        
        self.kt_full_dataset = KTDataSet(complete_kt_data, self.lrn_emb, self.qusunt_emb, self.cpt_emb, 'train')
        
        # 4. 创建数据加载器
        print("4. 创建数据加载器...")
        self.cd_full_loader = torch.utils.data.DataLoader(
            self.cd_full_dataset,
            batch_size=hyperparams.train_eval_batch_size,
            shuffle=False,
            collate_fn=self.cd_full_dataset.collate_fn
        )
        
        self.kt_full_loader = torch.utils.data.DataLoader(
            self.kt_full_dataset,
            batch_size=hyperparams.train_eval_batch_size,
            shuffle=False,
            collate_fn=self.kt_full_dataset.collate_fn
        )
        
        print(f"   CD完整数据批次: {len(self.cd_full_loader)}")
        print(f"   KT完整数据批次: {len(self.kt_full_loader)}")
        print("✓ 数据设置完成")
    
    def create_complete_cd_data(self, original_data):
        """创建完整的CD数据，保持原有数据结构"""
        print("   合并CD训练集和测试集...")
        
        # 检查数据结构
        if 'train_data' not in original_data or 'test_data' not in original_data:
            print("   ⚠ 原始CD数据没有train_data/test_data结构，直接使用")
            return original_data
        
        # 创建完整数据，保持原有结构
        complete_data = original_data.copy()
        
        # 合并训练集和测试集
        train_data = original_data['train_data']
        test_data = original_data['test_data']
        
        # 合并学习者序列数据
        complete_sequences = {}
        
        # 合并训练集学习者
        if 'sequences' in train_data:
            for learner_id, seq_data in train_data['sequences'].items():
                complete_sequences[learner_id] = seq_data
        
        # 合并测试集学习者（如果测试集学习者不在训练集中，则添加）
        if 'sequences' in test_data:
            for learner_id, seq_data in test_data['sequences'].items():
                if learner_id in complete_sequences:
                    # 如果学习者已存在，可能需要合并序列（这里简单使用测试集覆盖）
                    complete_sequences[learner_id] = seq_data
                else:
                    complete_sequences[learner_id] = seq_data
        
        # 更新统计数据
        complete_train_data = train_data.copy()
        complete_train_data['sequences'] = complete_sequences
        
        # 更新统计信息
        if 'statistics' in complete_train_data:
            train_stats = complete_train_data['statistics']
            test_stats = test_data.get('statistics', {})
            
            complete_train_data['statistics'] = {
                'total_learners': len(complete_sequences),
                'total_records': train_stats.get('total_records', 0) + test_stats.get('total_records', 0),
                'max_sequence_length': max(
                    train_stats.get('max_sequence_length', 0),
                    test_stats.get('max_sequence_length', 0)
                )
            }
        
        # 创建完整数据结构
        complete_data['train_data'] = complete_train_data
        complete_data['complete_data'] = complete_train_data  # 额外保存完整数据引用
        
        print(f"   CD数据合并完成: {len(complete_sequences)}个学习者")
        return complete_data
    
    def create_complete_kt_data(self, original_data):
        """创建完整的KT数据，保持原有数据结构"""
        print("   合并KT训练集和测试集...")
        
        # 检查数据结构
        if 'train_data' not in original_data or 'test_data' not in original_data:
            print("   ⚠ 原始KT数据没有train_data/test_data结构，直接使用")
            return original_data
        
        # 创建完整数据，保持原有结构
        complete_data = original_data.copy()
        
        # 合并训练集和测试集
        train_data = original_data['train_data']
        test_data = original_data['test_data']
        
        # 合并序列数据
        complete_sequences = {}
        
        # 合并训练集序列
        if 'sequences' in train_data:
            for learner_id, seq_data in train_data['sequences'].items():
                complete_sequences[learner_id] = seq_data
        
        # 合并测试集序列
        if 'sequences' in test_data:
            for learner_id, seq_data in test_data['sequences'].items():
                if learner_id in complete_sequences:
                    # 如果学习者已存在，合并序列数据
                    existing_seq = complete_sequences[learner_id]
                    # 这里需要根据实际数据结构进行合并
                    # 假设序列数据是列表，直接拼接
                    if isinstance(existing_seq, list) and isinstance(seq_data, list):
                        complete_sequences[learner_id] = existing_seq + seq_data
                    else:
                        # 其他情况使用测试集数据
                        complete_sequences[learner_id] = seq_data
                else:
                    complete_sequences[learner_id] = seq_data
        
        # 更新训练数据
        complete_train_data = train_data.copy()
        complete_train_data['sequences'] = complete_sequences
        
        # 更新统计信息
        if 'statistics' in complete_train_data:
            train_stats = complete_train_data['statistics']
            test_stats = test_data.get('statistics', {})
            
            complete_train_data['statistics'] = {
                'total_learners': len(complete_sequences),
                'total_records': train_stats.get('total_records', 0) + test_stats.get('total_records', 0),
                'max_sequence_length': max(
                    train_stats.get('max_sequence_length', 0),
                    test_stats.get('max_sequence_length', 0)
                )
            }
        
        # 创建完整数据结构
        complete_data['train_data'] = complete_train_data
        complete_data['complete_data'] = complete_train_data
        
        print(f"   KT数据合并完成: {len(complete_sequences)}个学习者")
        return complete_data
    
    def save_hgc_embeddings_fast(self):
        """快速保存HGC嵌入向量 - 绕过服务层直接使用Repository"""
        print("\n=== 保存HGC嵌入向量 (优化版) ===")
        
        # 确保HGC嵌入已经计算
        if not hasattr(self, 'lrn_emb') or self.lrn_emb is None:
            with torch.no_grad():
                self.lrn_emb, self.qusunt_emb, self.cpt_emb = self.model_hgc(
                    hgcdr, self.device, return_dict=False
                )
        
        # 直接使用Repository批量保存，避免服务层的额外开销
        embeddings_dict = {
            'learner': {},
            'unit': {}, 
            'concept': {}
        }
        
        # 构建学习者嵌入字典
        print("1. 准备学习者嵌入...")
        for i in tqdm(range(len(self.lrn_emb)), desc="学习者嵌入"):
            uid = str(i)
            embeddings_dict['learner'][uid] = self.lrn_emb[i].detach().cpu()
        
        # 构建学习单元嵌入字典
        print("2. 准备学习单元嵌入...")
        for i in tqdm(range(len(self.qusunt_emb)), desc="学习单元嵌入"):
            uid = str(i)
            embeddings_dict['unit'][uid] = self.qusunt_emb[i].detach().cpu()
        
        # 构建知识点嵌入字典
        print("3. 准备知识点嵌入...")
        for i in tqdm(range(len(self.cpt_emb)), desc="知识点嵌入"):
            uid = str(i)
            embeddings_dict['concept'][uid] = self.cpt_emb[i].detach().cpu()
        
        # 批量保存到数据库
        print("4. 批量保存到数据库...")
        result = embedding_repo.save_embeddings(embeddings_dict)
        
        stats = type('Stats', (), {
            'total_count': result.get('upserted_count', 0) + result.get('modified_count', 0),
            'learner_count': len(embeddings_dict['learner']),
            'unit_count': len(embeddings_dict['unit']),
            'concept_count': len(embeddings_dict['concept']),
            'embedding_dim': self.lrn_emb.shape[1],
            'creation_time': self._get_current_time()
        })()
        
        print(f"✓ HGC嵌入向量保存完成:")
        print(f"  学习者: {stats.learner_count}个")
        print(f"  学习单元: {stats.unit_count}个") 
        print(f"  知识点: {stats.concept_count}个")
        print(f"  嵌入维度: {stats.embedding_dim}")
        
        return stats
    
    def compute_and_save_cd_kt_results(self):
        """计算并保存CD和KT结果"""
        print("\n=== 计算CD和KT结果 ===")
        
        # 先初始化CD和KT模型
        self.initialize_cd_kt_models()
        
        kt_results = []
        cd_ability_cache = {}  # 缓存CD能力结果
        
        # 第一阶段: 计算CD能力
        print("1. 计算CD能力...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.cd_full_loader, desc="CD计算")):
                try:
                    lrn_indices = batch['lrn_indices'].to(self.device)
                    qus_seq_indices = batch['qus_seq_indices'].to(self.device)
                    qus_seq_masks = batch['qus_seq_masks'].to(self.device)
                    
                    h_lrn_batch = self.lrn_emb[lrn_indices]
                    
                    # CD前向传播
                    predictions, cd_ability = self.model_cd(
                        h_lrn_batch=h_lrn_batch,
                        h_qus=self.qusunt_emb[:self.cd_full_dataset.qus_num],
                        h_cpt=self.cpt_emb,
                        qus_seq_indices=qus_seq_indices,
                        qus_seq_masks=qus_seq_masks,
                        return_ability=True,
                        use_kt_optimization=False
                    )
                    
                    # 缓存CD能力结果
                    for i, lrn_idx in enumerate(lrn_indices):
                        learner_id = str(lrn_idx.item())
                        cd_ability_cache[learner_id] = cd_ability[i].detach().cpu()
                        
                except Exception as e:
                    print(f"⚠ CD计算批次 {batch_idx} 失败: {e}")
                    continue
        
        # 第二阶段: 计算KT知识状态
        print("2. 计算KT知识状态...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.kt_full_loader, desc="KT计算")):
                try:
                    lrn_indices = batch['lrn_indices'].to(self.device)
                    qusunt_seq_indices = batch['qusunt_seq_indices'].to(self.device)
                    add1 = batch['add1'].to(self.device)
                    add2 = batch['add2'].to(self.device)
                    type_indices = batch['type_indices'].to(self.device)
                    seq_masks = batch['seq_masks'].to(self.device)
                    next_question_masks = batch['next_question_masks'].to(self.device)
                    
                    current_lrn_emb = self.lrn_emb[lrn_indices]
                    current_qusunt_emb = self.qusunt_emb[qusunt_seq_indices]
                    
                    # KT前向传播
                    predictions, concept_mastery = self.model_kt(
                        h_lrn_batch=current_lrn_emb,
                        h_qusunt_batch=current_qusunt_emb,
                        h_cpt=self.cpt_emb,
                        lrn_indices=lrn_indices,
                        qusunt_seq_indices=qusunt_seq_indices,
                        add1=add1,
                        add2=add2,
                        type_indices=type_indices,
                        seq_mask=seq_masks,
                        next_question_mask=next_question_masks,
                        use_cd_optimization=True,  # 使用CD优化
                        use_contrastive=False
                    )
                    
                    # 保存KT结果
                    for i, lrn_idx in enumerate(lrn_indices):
                        learner_id = str(lrn_idx.item())
                        kt_results.append({
                            'learner_id': learner_id,
                            'concept_mastery': concept_mastery[i].detach().cpu(),
                            'metadata': {
                                'batch_index': batch_idx,
                                'sample_index': i,
                                'model_type': 'KT',
                                'computation_type': 'full_data',
                                'used_cd_optimization': True
                            }
                        })
                        
                except Exception as e:
                    print(f"⚠ KT计算批次 {batch_idx} 失败: {e}")
                    continue
        
        # 第三阶段: 批量保存结果到数据库
        print("3. 保存结果到数据库...")
        
        # 批量保存KT结果
        kt_batch_data = []
        for result in kt_results:
            kt_batch_data.append({
                'learner_id': result['learner_id'],
                'concept_mastery': result['concept_mastery'],
                'model_type': 'KT',
                'metadata': result['metadata']
            })
        
        # 使用Repository直接批量保存
        kt_saved_ids = learner_repo.save_batch_learner_states(kt_batch_data)
        
        # 批量保存CD结果
        cd_batch_data = []
        for learner_id, cd_ability in cd_ability_cache.items():
            cd_batch_data.append({
                'learner_id': f"{learner_id}_cd",
                'concept_mastery': cd_ability,
                'model_type': 'CD',
                'metadata': {
                    'purpose': 'kt_initialization',
                    'computation_type': 'full_data'
                }
            })
        
        cd_saved_ids = learner_repo.save_batch_learner_states(cd_batch_data)
        
        print(f"   KT结果保存: {len(kt_saved_ids)}个")
        print(f"   CD结果保存: {len(cd_saved_ids)}个")
        
        return {
            'cd_results_count': len(cd_saved_ids),
            'kt_results_count': len(kt_saved_ids)
        }
    
    def initialize_cd_kt_models(self):
        """初始化CD和KT模型"""
        print("初始化CD和KT模型...")
        
        # 从检查点加载CD和KT模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 初始化CD模型
        concept_num = self.cd_full_dataset.cpt_num
        self.model_cd = CD(
            embedding_dim=hyperparams.hgc_embedding_dim,
            concept_num=concept_num
        ).to(self.device)
        
        # 初始化KT模型
        kt_data = ktdr.loadDatafromSql()
        concept_mapping = kt_data.get('question_concepts', {})
        self.model_kt = KT(
            embedding_dim=hyperparams.hgc_embedding_dim,
            concept_num=concept_num,
            concept_mapping=concept_mapping
        ).to(self.device)
        
        # 加载权重
        if 'model_cd_state_dict' in checkpoint and 'model_kt_state_dict' in checkpoint:
            self.model_cd.load_state_dict(checkpoint['model_cd_state_dict'])
            self.model_kt.load_state_dict(checkpoint['model_kt_state_dict'])
            print("   ✓ 加载CD和KT模型权重")
        else:
            print("   ⚠ 检查点中没有CD和KT权重，使用随机初始化")
        
        self.model_cd.eval()
        self.model_kt.eval()
    
    def _get_current_time(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def save_all_results(self):
        """保存所有结果"""
        print("开始保存训练结果到数据库...")
        
        # 1. 快速保存HGC嵌入向量
        hgc_stats = self.save_hgc_embeddings_fast()
        
        # 2. 计算并保存CD和KT结果
        cd_kt_stats = self.compute_and_save_cd_kt_results()
        
        # 3. 生成总结报告
        print("\n" + "="*60)
        print("训练结果保存完成!")
        print("="*60)
        print("保存内容总结:")
        print(f"  HGC嵌入向量:")
        print(f"    - 学习者: {hgc_stats.learner_count}个")
        print(f"    - 学习单元: {hgc_stats.unit_count}个")
        print(f"    - 知识点: {hgc_stats.concept_count}个")
        print(f"    - 集合: Embeddings")
        
        print(f"  CD能力估计:")
        print(f"    - 学习者: {cd_kt_stats['cd_results_count']}个")
        print(f"    - 集合: Learners (标记为CD)")
        
        print(f"  KT知识状态:")
        print(f"    - 学习者: {cd_kt_stats['kt_results_count']}个") 
        print(f"    - 集合: Learners (标记为KT)")
        
        return {
            'hgc_stats': hgc_stats,
            'cd_kt_stats': cd_kt_stats
        }

def main():
    """主函数"""
    print("训练结果保存脚本 - 修正版")
    print("功能: 加载训练好的模型，计算完整数据结果，保存到MongoDB")
    
    # 模型文件路径
    model_dir = hyperparams.train_save_dir
    final_dir = os.path.join(model_dir, "final_models")
    
    # 查找HGC模型文件
    hgc_model_path = os.path.join(final_dir, "hgc_best_model.pth")
    if not os.path.exists(hgc_model_path):
        print(f"❌ HGC模型文件不存在: {hgc_model_path}")
        return False
    
    try:
        # 创建保存器并执行保存
        saver = TrainingResultsSaver(hgc_model_path)
        results = saver.save_all_results()
        
        print("\n🎉 训练结果保存完成!")
        print("📊 保存统计:")
        print(f"  - HGC嵌入: {results['hgc_stats'].total_count}个实体")
        print(f"  - CD能力: {results['cd_kt_stats']['cd_results_count']}个学习者")
        print(f"  - KT状态: {results['cd_kt_stats']['kt_results_count']}个学习者")
        
        return True
        
    except Exception as e:
        print(f"\n💥 保存过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)