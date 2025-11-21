# Inference_HGC_CD_KT.py
import sys
import os
import torch
import torch.nn as nn
import time
import numpy as np
from collections import defaultdict
import warnings
from tqdm import tqdm
import glob

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

# 导入数据服务层
from DataService.EmbeddingDataService import embedding_data_service
from DataService.LearnerDataService import learner_data_service

class CompletePipelineInference:
    """完整的HGC-CD-KT推理管道 - 保存所有计算结果到MongoDB"""
    
    def __init__(self, batch_size=32):
        self.device = hyperparams.device
        self.batch_size = batch_size
        self.setup_models_and_data()
        
    def setup_models_and_data(self):
        """初始化模型和数据 - 按照训练脚本的正确方式"""
        print("=== 初始化模型和数据 (完整数据) ===")
        
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
        
        # 3. 加载CD数据
        print("3. 加载CD数据...")
        cd_data = cddr.loadDatafromSql()
        
        # 4. 加载KT数据
        print("4. 加载KT数据...")
        kt_data = ktdr.loadDatafromSql()
        
        # 5. 数据合并和筛选完整数据
        print("5. 合并和筛选完整数据...")
        cd_complete_data, kt_complete_data = cd_kt_merger.merge_and_filter_complete_data()
        
        # 6. 计算初始HGC嵌入用于数据集创建
        print("6. 计算初始HGC嵌入...")
        self.model_hgc.eval()
        with torch.no_grad():
            self.initial_lrn_emb, self.initial_qusunt_emb, self.initial_cpt_emb = self.model_hgc(
                hgcdr, self.device, return_dict=False
            )
        
        # 7. 创建CD数据集 - 使用完整数据
        print("7. 创建CD数据集 (完整数据)...")
        self.cd_dataset = CDDataset(cd_data, self.initial_lrn_emb, self.initial_qusunt_emb, self.initial_cpt_emb, 'all')
        
        # 初始化CD模型
        embedding_dim = hyperparams.hgc_embedding_dim
        concept_num = self.cd_dataset.cpt_num
        
        self.model_cd = CD(
            embedding_dim=embedding_dim,
            concept_num=concept_num
        ).to(self.device)
        
        # 8. 创建KT数据集 - 使用完整数据
        print("8. 创建KT数据集 (完整数据)...")
        self.kt_dataset = KTDataSet(kt_data, self.initial_lrn_emb, self.initial_qusunt_emb, self.initial_cpt_emb, 'all')
        
        # 初始化KT模型
        concept_mapping = kt_data.get('question_concepts', {})
        
        self.model_kt = KT(
            embedding_dim=embedding_dim,
            concept_num=concept_num,
            concept_mapping=concept_mapping
        ).to(self.device)
        
        # 9. 创建数据加载器
        print("9. 创建数据加载器...")
        self.cd_loader = torch.utils.data.DataLoader(
            self.cd_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.cd_dataset.collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        self.kt_loader = torch.utils.data.DataLoader(
            self.kt_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.kt_dataset.collate_fn,
            num_workers=0,
            pin_memory=True
        )
        
        # 10. 获取UID映射
        print("10. 获取UID映射...")
        self.lrn_uid_mapping = hgcdr.lrn_uid  # {uid: idx}
        self.qusunt_uid_mapping = hgcdr.qusunt_uid  # {uid: idx}
        self.cpt_uid_mapping = hgcdr.cpt_uid  # {uid: idx}
        
        # 创建反向映射 {idx: uid}
        self.lrn_idx_to_uid = {idx: uid for uid, idx in self.lrn_uid_mapping.items()}
        self.qusunt_idx_to_uid = {idx: uid for uid, idx in self.qusunt_uid_mapping.items()}
        self.cpt_idx_to_uid = {idx: uid for uid, idx in self.cpt_uid_mapping.items()}
        
        # 使用merge_and_filter_complete_data的返回值来确定有效学习者UID
        self.valid_lrn_uids = list(cd_complete_data.keys())
        
        # 打印数据集统计
        cd_stats = self.cd_dataset.get_data_statistics()
        kt_stats = self.kt_dataset.get_data_statistics()
        
        print(f"   CD完整数据集: {cd_stats['total_learners']}学习者, {cd_stats['total_records']}记录")
        print(f"   KT完整数据集: {kt_stats['total_learners']}学习者, {kt_stats['total_records']}记录")
        print(f"   有效学习者: {len(self.valid_lrn_uids)}个")
        print(f"   知识点数量: {len(self.cpt_uid_mapping)}个")
        
        # 设置模型为评估模式
        self.model_hgc.eval()
        self.model_cd.eval()
        self.model_kt.eval()
        
        print("✅ 模型和数据初始化完成")
    
    def load_trained_models(self):
        """加载训练好的模型权重"""
        print("=== 加载训练好的模型权重 ===")
        
        # 获取模型保存路径
        save_dir = hyperparams.train_save_dir
        final_dir = os.path.join(save_dir, "final_models")
        
        # 检查模型文件是否存在
        hgc_path = os.path.join(final_dir, "hgc_best_model.pth")
        cd_path = os.path.join(final_dir, "cd_best_model.pth")
        kt_path = os.path.join(final_dir, "kt_best_model.pth")
        
        for path in [hgc_path, cd_path, kt_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"模型文件不存在: {path}")
        
        print(f"📂 加载模型权重从: {final_dir}")
        
        # 加载HGC模型权重
        print("1. 加载HGC模型权重...")
        hgc_checkpoint = torch.load(hgc_path, map_location=self.device)
        if 'model_state_dict' in hgc_checkpoint:
            self.model_hgc.load_state_dict(hgc_checkpoint['model_state_dict'])
        else:
            self.model_hgc.load_state_dict(hgc_checkpoint)
        
        # 加载CD模型权重
        print("2. 加载CD模型权重...")
        cd_checkpoint = torch.load(cd_path, map_location=self.device)
        if 'model_state_dict' in cd_checkpoint:
            self.model_cd.load_state_dict(cd_checkpoint['model_state_dict'])
        else:
            self.model_cd.load_state_dict(cd_checkpoint)
        
        # 加载KT模型权重
        print("3. 加载KT模型权重...")
        kt_checkpoint = torch.load(kt_path, map_location=self.device)
        if 'model_state_dict' in kt_checkpoint:
            self.model_kt.load_state_dict(kt_checkpoint['model_state_dict'])
        else:
            self.model_kt.load_state_dict(kt_checkpoint)
        
        print("✅ 所有模型权重加载完成")
    
    def compute_hgc_embeddings(self):
        """计算HGC嵌入（无梯度）"""
        with torch.no_grad():
            return self.model_hgc(hgcdr, self.device, return_dict=False)
    
    def save_embeddings_to_db(self, lrn_emb, qusunt_emb, cpt_emb):
        """保存HGC嵌入到数据库"""
        print("💾 保存HGC嵌入到数据库...")
        
        # 准备嵌入数据
        embeddings_data = {
            'lrn': {},
            'unt': {},
            'cpt': {}
        }
        
        # 学习者嵌入 - 只保存有效学习者
        for lrn_uid in self.valid_lrn_uids:
            lrn_idx = self.lrn_uid_mapping.get(lrn_uid)
            if lrn_idx is not None and lrn_idx < len(lrn_emb):
                embeddings_data['lrn'][lrn_uid] = lrn_emb[lrn_idx]
        
        # 学习单元嵌入 - 保存所有
        for idx in range(len(qusunt_emb)):
            uid = self.qusunt_idx_to_uid.get(idx)
            if uid is not None:
                embeddings_data['unt'][uid] = qusunt_emb[idx]
        
        # 知识点嵌入 - 保存所有
        for idx in range(len(cpt_emb)):
            uid = self.cpt_idx_to_uid.get(idx)
            if uid is not None:
                embeddings_data['cpt'][uid] = cpt_emb[idx]
        
        print(f"   准备保存: {len(embeddings_data['lrn'])} 个学习者, {len(embeddings_data['unt'])} 个学习单元, {len(embeddings_data['cpt'])} 个知识点")
        
        # 使用修复后的方法保存
        stats = embedding_data_service.save_embeddings_dict(embeddings_data)
        
        print(f"✅ HGC嵌入保存完成: {stats}")
        return stats
    
    def process_kt_results_batch(self, batch_lrn_indices, concept_mastery_batch):
        """处理KT结果批次"""
        batch_results = []
        
        for i, lrn_idx in enumerate(batch_lrn_indices):
            lrn_uid = self.lrn_idx_to_uid.get(lrn_idx.item())
            if lrn_uid is None or lrn_uid not in self.valid_lrn_uids:
                continue
            
            # 获取该学习者的知识点掌握程度
            learner_mastery = concept_mastery_batch[i]
            
            # 如果是序列数据，取最后一个时间步
            if len(learner_mastery.shape) > 1:
                learner_mastery = learner_mastery[-1]
            
            # 转换为UID映射格式
            concept_mastery_dict = {}
            for cpt_idx in range(len(learner_mastery)):
                cpt_uid = self.cpt_idx_to_uid.get(cpt_idx)
                if cpt_uid is not None:
                    concept_mastery_dict[cpt_uid] = float(learner_mastery[cpt_idx].item())
            
            batch_results.append({
                'learner_id': lrn_uid,
                'concept_mastery': concept_mastery_dict
            })
        
        return batch_results
    
    def run_inference(self):
        """运行完整推理流程"""
        print("=== 开始完整推理流程 ===")
        start_time = time.time()
        
        # 1. 加载训练好的模型权重
        self.load_trained_models()
        
        # 2. 计算HGC嵌入并保存
        print("\n--- 阶段1: 计算HGC嵌入 ---")
        lrn_emb, qusunt_emb, cpt_emb = self.compute_hgc_embeddings()
        embedding_stats = self.save_embeddings_to_db(lrn_emb, qusunt_emb, cpt_emb)
        
        # 3. CD-KT联合推理
        print("\n--- 阶段2: CD-KT联合推理 ---")
        
        total_kt_results = []
        processed_batches = 0
        
        # 处理KT数据获取知识点掌握程度
        print("   KT推理进度:")
        with tqdm(total=len(self.kt_loader), desc="KT推理") as pbar:
            for batch_idx, kt_batch in enumerate(self.kt_loader):
                try:
                    with torch.no_grad():
                        # KT推理
                        lrn_indices = kt_batch['lrn_indices'].to(self.device)
                        qusunt_seq_indices = kt_batch['qusunt_seq_indices'].to(self.device)
                        add1 = kt_batch['add1'].to(self.device)
                        add2 = kt_batch['add2'].to(self.device)
                        type_indices = kt_batch['type_indices'].to(self.device)
                        seq_masks = kt_batch['seq_masks'].to(self.device)
                        prediction_masks = kt_batch['prediction_masks'].to(self.device)
                        
                        current_lrn_emb = lrn_emb[lrn_indices]
                        
                        # 正确获取学习单元嵌入
                        batch_size, seq_len = qusunt_seq_indices.shape
                        embedding_dim = qusunt_emb.shape[1]
                        qusunt_indices_flat = qusunt_seq_indices.view(-1)
                        current_qusunt_emb = qusunt_emb[qusunt_indices_flat].view(batch_size, seq_len, embedding_dim)
                        
                        # 使用KT模型获取知识点掌握程度
                        concept_mastery = self.model_kt.get_concept_mastery(
                            h_lrn_batch=current_lrn_emb,
                            h_qusunt_batch=current_qusunt_emb,
                            h_cpt=cpt_emb,
                            lrn_indices=lrn_indices,
                            qusunt_seq_indices=qusunt_seq_indices,
                            add1=add1,
                            add2=add2,
                            type_indices=type_indices,
                            seq_mask=seq_masks,
                            qus_num=self.kt_dataset.qus_num
                        )
                        
                        # 处理KT结果
                        batch_kt_results = self.process_kt_results_batch(lrn_indices, concept_mastery)
                        total_kt_results.extend(batch_kt_results)
                    
                    processed_batches += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n⚠ KT推理批次 {batch_idx} 失败: {e}")
                    pbar.update(1)
                    continue
        
        # 4. 保存KT结果到数据库 - 使用专门的方法
        print("\n--- 阶段3: 保存KT结果 ---")
        print(f"   准备保存 {len(total_kt_results)} 个学习者的KT结果...")
        kt_stats = learner_data_service.save_kt_inference_results(
            kt_results=total_kt_results
        )
        
        total_time = time.time() - start_time
        
        # 推理总结
        print(f"\n{'='*60}")
        print("推理完成!")
        print(f"{'='*60}")
        print(f"总推理时间: {total_time:.2f}秒")
        print(f"KT处理批次: {processed_batches}/{len(self.kt_loader)}")
        print(f"KT结果保存: {kt_stats['successfully_saved']}/{len(total_kt_results)} 个学习者")
        print(f"嵌入保存: {embedding_stats.total_count} 个实体")
        print(f"有效学习者: {len(self.valid_lrn_uids)} 个")
        
        return {
            'embedding_stats': embedding_stats,
            'kt_stats': kt_stats,
            'total_time': total_time,
            'valid_learners': len(self.valid_lrn_uids)
        }
    
    def _get_current_time(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def main():
    """主函数"""
    print("HGC-CD-KT 完整推理管道")
    print("功能: 加载训练好的模型，计算所有数据结果并保存到MongoDB")
    print("保存内容:")
    print("  - HGC嵌入: Embeddings集合 (lrn/unt/cpt)")
    print("  - KT结果: Learners集合 (知识点掌握程度)")
    
    try:
        # 设置批量大小
        batch_size = 32  # 可根据需要调整
        
        # 创建推理器并运行
        inference = CompletePipelineInference(batch_size=batch_size)
        results = inference.run_inference()
        
        print("\n🎉 推理完成!")
        print("📊 结果统计:")
        print(f"  嵌入向量: {results['embedding_stats'].total_count} 个实体")
        print(f"  KT结果: {results['kt_stats']['successfully_saved']} 个学习者")
        print(f"  有效学习者: {results['valid_learners']} 个")
        print(f"  总时间: {results['total_time']:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"\n💥 推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


# {
#   "_id": 某种id格式,
#   "entity_type": "lrn",
#   "uid": "lrn_004a9c3f5bf246faab3d390ce716e658",
#   "embedding": [0.1,0.2,...],
#   "updated_time": "2025-11-21T14:23:32.032Z"
# }

# {
#   "_id": 某种id格式,
#   "uid": "lrn_004a9c3f5bf246faab3d390ce716e658",
#   "KT": {
#       "cpt_c75fd115df334248ae9a3bca1511036a": 0.9970404505729675,
#       "cpt_5b42d66cbd9a474dbf5f873ad82cfd89": 1,
#       "cpt_29b672e534d74009b29f45d9e3c269c3": 1.3097310735954437e-10,
#       "cpt_119608ea29824fee9fbdbeacffaa3393": 1,
#       "cpt_c68c355aa6d747b094f61f872d9f2049": 1,
#       "cpt_bd8d0d509983400d966aa2e0b871f781": 1,
#       ...
#     },
#   "updated_time": "2025-11-21T14:23:32.032Z"
# }