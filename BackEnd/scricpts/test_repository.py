# test_all_repositories.py
import logging
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_database_connection():
    """测试数据库连接"""
    logger.info("=" * 60)
    logger.info("测试数据库连接")
    logger.info("=" * 60)
    
    try:
        from app.core.database import get_mysql_connection, get_mongodb_database
        
        # 测试MySQL连接
        mysql_conn = get_mysql_connection()
        if mysql_conn and mysql_conn.open:
            logger.info("✓ MySQL数据库连接成功")
        else:
            logger.error("✗ MySQL数据库连接失败")
        
        # 测试MongoDB连接 - 修改测试逻辑
        try:
            mongodb_db = get_mongodb_database()
            # 通过执行简单命令来测试连接
            mongodb_db.command('ping')
            logger.info("✓ MongoDB数据库连接成功")
            logger.info(f"  - 数据库名称: {mongodb_db.name}")
            
            # 显示集合列表
            collections = mongodb_db.list_collection_names()
            logger.info(f"  - 集合数量: {len(collections)}")
            logger.info(f"  - 集合列表: {', '.join(collections[:5])}{'...' if len(collections) > 5 else ''}")
            
        except Exception as e:
            logger.error(f"✗ MongoDB数据库连接失败: {e}")
            
    except Exception as e:
        logger.error(f"✗ 数据库连接测试失败: {e}")

def test_hgc_repository():
    """测试HGC Repository"""
    logger.info("=" * 60)
    logger.info("测试 HGC Repository")
    logger.info("=" * 60)
    
    try:
        from app.repositories.hgc_repository import hgc_repository
        
        # 测试数据
        test_learner_uid = "lrn_004a9c3f5bf246faab3d390ce716e658"
        test_learner_uids = [
            "lrn_004a9c3f5bf246faab3d390ce716e658", 
            "lrn_efc1639eb8ec4421a7559ad4d2f9858c"
        ]
        
        logger.info(f"测试学习者UID: {test_learner_uid}")
        
        # 测试单个学习者
        hgc_data_single = hgc_repository.get_data_for_single_learner(test_learner_uid)
        if hgc_data_single:
            logger.info("✓ HGC单个学习者数据获取成功")
            logger.info(f"  - 交互单元数: {len(hgc_data_single['interacted_units'])}")
            logger.info(f"  - 涉及主题数: {len(hgc_data_single['learner_topics'])}")
            logger.info(f"  - 涉及课程数: {len(hgc_data_single['learner_courses'])}")
            logger.info(f"  - 相关学习者数: {len(hgc_data_single['related_learners'])}")
            logger.info(f"  - 交互记录数: {len(hgc_data_single['interaction_records'])}")
            logger.info(f"  - 使用策略: {hgc_data_single['strategy_used']}")
        else:
            logger.warning("✗ 未获取到HGC单个学习者数据")
        
        # 测试多个学习者
        hgc_data_multi = hgc_repository.get_data_for_multiple_learners(test_learner_uids)
        if hgc_data_multi:
            logger.info("✓ HGC多个学习者数据获取成功")
            logger.info(f"  - 目标学习者数: {len(hgc_data_multi['target_learner_uids'])}")
            logger.info(f"  - 相关学习者数: {len(hgc_data_multi['related_learners'])}")
            logger.info(f"  - 交互记录数: {len(hgc_data_multi['interaction_records'])}")
            logger.info(f"  - 使用策略: {hgc_data_multi['strategy_used']}")
        else:
            logger.warning("✗ 未获取到HGC多个学习者数据")
            
    except Exception as e:
        logger.error(f"✗ HGC Repository测试失败: {e}")

def test_cd_repository():
    """测试CD Repository"""
    logger.info("=" * 60)
    logger.info("测试 CD Repository")
    logger.info("=" * 60)
    
    try:
        from app.repositories.cd_repository import cd_repository
        
        test_learner_uid = "lrn_004a9c3f5bf246faab3d390ce716e658"
        test_learner_uids = [
            "lrn_004a9c3f5bf246faab3d390ce716e658", 
            "lrn_efc1639eb8ec4421a7559ad4d2f9858c"
        ]
        
        logger.info(f"测试学习者UID: {test_learner_uid}")
        
        # 测试单个学习者
        cd_data_single = cd_repository.get_data_for_single_learner(test_learner_uid)
        if cd_data_single:
            logger.info("✓ CD单个学习者数据获取成功")
            logger.info(f"  - 题目交互记录数: {cd_data_single['interaction_count']}")
            logger.info(f"  - 涉及题目数: {len(cd_data_single['involved_questions'])}")
            logger.info(f"  - 格式化交互记录数: {len(cd_data_single['question_interactions'])}")
            
            # 显示前几条交互记录
            if cd_data_single['question_interactions']:
                logger.info("  - 前2条交互记录:")
                for i, record in enumerate(cd_data_single['question_interactions'][:2]):
                    logger.info(f"    {i+1}. 学习者: {record[0]}, 题目: {record[1]}, 正确: {record[2]}")
        else:
            logger.warning("✗ 未获取到CD单个学习者数据")
        
        # 测试多个学习者
        cd_data_multi = cd_repository.get_data_for_multiple_learners(test_learner_uids)
        if cd_data_multi:
            logger.info("✓ CD多个学习者数据获取成功")
            logger.info(f"  - 成功获取数据的学习者数: {len([v for v in cd_data_multi.values() if v])}")
            for uid, data in cd_data_multi.items():
                if data:
                    logger.info(f"  - {uid}: {data['interaction_count']} 条交互记录")
        else:
            logger.warning("✗ 未获取到CD多个学习者数据")
            
        # 测试统计信息
        stats = cd_repository.get_interaction_statistics(test_learner_uid)
        if stats:
            logger.info("✓ CD交互统计信息获取成功")
            logger.info(f"  - 总交互次数: {stats['total_interactions']}")
            logger.info(f"  - 唯一题目数: {stats['unique_questions']}")
            logger.info(f"  - 正确率: {stats['accuracy_rate']:.2%}")
            
    except Exception as e:
        logger.error(f"✗ CD Repository测试失败: {e}")

def test_kt_repository():
    """测试KT Repository"""
    logger.info("=" * 60)
    logger.info("测试 KT Repository")
    logger.info("=" * 60)
    
    try:
        from app.repositories.kt_repository import kt_repository
        
        test_learner_uid = "lrn_004a9c3f5bf246faab3d390ce716e658"
        test_learner_uids = [
            "lrn_004a9c3f5bf246faab3d390ce716e658", 
            "lrn_efc1639eb8ec4421a7559ad4d2f9858c"
        ]
        
        logger.info(f"测试学习者UID: {test_learner_uid}")
        
        # 测试单个学习者
        kt_data_single = kt_repository.get_data_for_single_learner(test_learner_uid)
        if kt_data_single:
            logger.info("✓ KT单个学习者数据获取成功")
            logger.info(f"  - 学习单元交互记录数: {kt_data_single['interaction_count']}")
            logger.info(f"  - 涉及学习单元数: {len(kt_data_single['involved_units'])}")
            logger.info(f"  - 单元类型数: {len(kt_data_single['unit_types_mapping'])}")
            logger.info(f"  - 格式化交互记录数: {len(kt_data_single['unit_interactions'])}")
            
            # 显示前几条交互记录
            if kt_data_single['unit_interactions']:
                logger.info("  - 前2条交互记录:")
                for i, record in enumerate(kt_data_single['unit_interactions'][:2]):
                    logger.info(f"    {i+1}. 学习者: {record[0]}, 单元: {record[1]}, 信息1: {record[2]:.2f}")
        else:
            logger.warning("✗ 未获取到KT单个学习者数据")
        
        # 测试多个学习者
        kt_data_multi = kt_repository.get_data_for_multiple_learners(test_learner_uids)
        if kt_data_multi:
            logger.info("✓ KT多个学习者数据获取成功")
            logger.info(f"  - 成功获取数据的学习者数: {len([v for v in kt_data_multi.values() if v])}")
            for uid, data in kt_data_multi.items():
                if data:
                    logger.info(f"  - {uid}: {data['interaction_count']} 条交互记录")
        else:
            logger.warning("✗ 未获取到KT多个学习者数据")
            
        # 测试统计信息
        stats = kt_repository.get_interaction_statistics(test_learner_uid)
        if stats:
            logger.info("✓ KT交互统计信息获取成功")
            logger.info(f"  - 总交互次数: {stats['total_interactions']}")
            logger.info(f"  - 唯一单元数: {stats['unique_units']}")
            logger.info(f"  - 平均附加信息1: {stats['avg_additioninfo1']:.2f}")
            
    except Exception as e:
        logger.error(f"✗ KT Repository测试失败: {e}")

def test_embedding_repository():
    """测试Embedding Repository"""
    logger.info("=" * 60)
    logger.info("测试 Embedding Repository")
    logger.info("=" * 60)
    
    try:
        from app.repositories.embedding_repository import embedding_repository
        
        # 测试数据
        test_uid = "lrn_004a9c3f5bf246faab3d390ce716e658"
        test_uids = [
            "lrn_004a9c3f5bf246faab3d390ce716e658",
            "lrn_efc1639eb8ec4421a7559ad4d2f9858c",
            "unt_1234567890abcdef1234567890abcdef"
        ]
        
        logger.info(f"测试UID: {test_uid}")
        
        # 测试单个嵌入向量
        single_embedding = embedding_repository.get_embedding_by_uid(test_uid)
        if single_embedding:
            logger.info("✓ 单个嵌入向量获取成功")
            logger.info(f"  - UID: {single_embedding['uid']}")
            logger.info(f"  - 实体类型: {single_embedding['entity_type']}")
            logger.info(f"  - 嵌入维度: {len(single_embedding['embedding'])}")
            logger.info(f"  - 更新时间: {single_embedding['updated_time']}")
        else:
            logger.warning("✗ 未找到单个嵌入向量")
        
        # 测试批量获取嵌入向量
        list_embeddings = embedding_repository.get_embeddings_by_uids(test_uids, return_format="list")
        logger.info(f"✓ 列表格式获取成功，返回数量: {len(list_embeddings)}")
        
        found_count = sum(1 for e in list_embeddings if e['embedding'])
        logger.info(f"  - 成功找到嵌入向量的实体数: {found_count}/{len(list_embeddings)}")
        
        # 测试统计信息
        stats = embedding_repository.get_embedding_stats()
        if stats:
            logger.info("✓ 嵌入向量统计信息获取成功")
            logger.info(f"  - 总文档数: {stats['total_documents']}")
            for stat in stats['statistics_by_type']:
                logger.info(f"  - {stat['entity_type']}: {stat['count']} 个文档")
        
        # 测试存在性检查
        exists = embedding_repository.check_embedding_exists(test_uid)
        logger.info(f"  - UID {test_uid} 存在性: {exists}")
        
        # 测试最近更新的嵌入向量
        recent_embeddings = embedding_repository.get_recent_updated_embeddings(limit=2)
        logger.info(f"  - 最近更新数量: {len(recent_embeddings)}")
        
    except Exception as e:
        logger.error(f"✗ Embedding Repository测试失败: {e}")

def test_learner_repository():
    """测试Learner Repository"""
    logger.info("=" * 60)
    logger.info("测试 Learner Repository")
    logger.info("=" * 60)
    
    try:
        from app.repositories.learner_repository import learner_repository
        
        # 测试数据
        test_uid = "lrn_004a9c3f5bf246faab3d390ce716e658"
        test_uids = [
            "lrn_004a9c3f5bf246faab3d390ce716e658",
            "lrn_efc1639eb8ec4421a7559ad4d2f9858c",
            "lrn_test_nonexistent_uid"
        ]
        
        logger.info(f"测试学习者UID: {test_uid}")
        
        # 测试单个学习者的KT结果
        single_kt = learner_repository.get_kt_result_by_uid(test_uid)
        if single_kt:
            logger.info("✓ 单个学习者KT结果获取成功")
            logger.info(f"  - UID: {single_kt['uid']}")
            logger.info(f"  - KT知识点数量: {len(single_kt['KT'])}")
            logger.info(f"  - 更新时间: {single_kt['updated_time']}")
            
            # 显示前几个知识点掌握情况
            kt_items = list(single_kt['KT'].items())[:2]
            for i, (concept_id, mastery) in enumerate(kt_items):
                logger.info(f"  - 知识点{i+1}: {concept_id} = {mastery:.6f}")
        else:
            logger.warning("✗ 未找到单个学习者KT结果")
        
        # 测试批量获取KT结果
        list_kt_results = learner_repository.get_kt_results_by_uids(test_uids, return_format="list")
        logger.info(f"✓ 列表格式获取成功，返回数量: {len(list_kt_results)}")
        
        found_count = sum(1 for kt in list_kt_results if kt['KT'])
        logger.info(f"  - 成功找到KT结果的学习者数: {found_count}/{len(list_kt_results)}")
        
        # 测试统计信息
        stats = learner_repository.get_kt_statistics()
        if stats:
            logger.info("✓ KT统计信息获取成功")
            logger.info(f"  - 总学习者数: {stats['total_learners']}")
            if stats['statistics']:
                sample_stat = stats['statistics'][0]
                logger.info(f"  - 示例统计:")
                logger.info(f"    * 总知识点: {sample_stat['total_concepts']}")
                logger.info(f"    * 平均掌握度: {sample_stat['avg_mastery']:.4f}")
                logger.info(f"    * 高掌握度比例: {sample_stat['high_mastery_ratio']:.2%}")
        
        # 测试存在性检查
        exists = learner_repository.check_learner_exists(test_uid)
        logger.info(f"  - UID {test_uid} 存在性: {exists}")
        
        # 测试最近更新的学习者
        recent_learners = learner_repository.get_recent_updated_learners(limit=2)
        logger.info(f"  - 最近更新数量: {len(recent_learners)}")
        
    except Exception as e:
        logger.error(f"✗ Learner Repository测试失败: {e}")

def test_repository_integration():
    """测试Repository之间的集成"""
    logger.info("=" * 60)
    logger.info("测试 Repository 集成")
    logger.info("=" * 60)
    
    try:
        from app.repositories.hgc_repository import hgc_repository
        from app.repositories.embedding_repository import embedding_repository
        from app.repositories.learner_repository import learner_repository
        
        test_learner_uid = "lrn_004a9c3f5bf246faab3d390ce716e658"
        
        logger.info(f"集成测试学习者UID: {test_learner_uid}")
        
        # 1. 从HGC获取相关学习者
        hgc_data = hgc_repository.get_data_for_single_learner(test_learner_uid)
        if hgc_data and hgc_data['related_learners']:
            related_learners = hgc_data['related_learners'][:3]  # 取前3个相关学习者
            logger.info(f"✓ 从HGC获取到 {len(related_learners)} 个相关学习者")
            
            # 2. 获取这些学习者的嵌入向量
            if related_learners:
                embeddings = embedding_repository.get_embeddings_by_uids(related_learners, return_format="dict")
                found_embeddings = len([e for e in embeddings.values() if e['embedding']])
                logger.info(f"✓ 成功获取 {found_embeddings}/{len(related_learners)} 个相关学习者的嵌入向量")
            
            # 3. 获取这些学习者的KT结果
            if related_learners:
                kt_results = learner_repository.get_kt_results_by_uids(related_learners, return_format="dict")
                found_kt = len([k for k in kt_results.values() if k['KT']])
                logger.info(f"✓ 成功获取 {found_kt}/{len(related_learners)} 个相关学习者的KT结果")
        
        # 4. 测试数据一致性
        embedding_exists = embedding_repository.check_embedding_exists(test_learner_uid)
        learner_exists = learner_repository.check_learner_exists(test_learner_uid)
        logger.info(f"✓ 数据一致性检查:")
        logger.info(f"  - 嵌入向量存在: {embedding_exists}")
        logger.info(f"  - 学习者存在: {learner_exists}")
        
    except Exception as e:
        logger.error(f"✗ Repository集成测试失败: {e}")

def main():
    """主测试函数"""
    logger.info("开始综合测试所有Repository...")
    
    try:
        # 运行各个测试
        test_database_connection()
        test_hgc_repository()
        test_cd_repository()
        test_kt_repository()
        test_embedding_repository()
        test_learner_repository()
        test_repository_integration()
        
        logger.info("=" * 60)
        logger.info("🎉 所有Repository测试完成！")
        logger.info("=" * 60)
        
        # 总结信息
        logger.info("📊 测试总结:")
        logger.info("  - HGC Repository: 异构图表数据获取")
        logger.info("  - CD Repository: 认知诊断数据获取") 
        logger.info("  - KT Repository: 知识追踪数据获取")
        logger.info("  - Embedding Repository: 嵌入向量数据获取")
        logger.info("  - Learner Repository: 学习者KT结果获取")
        logger.info("  - 所有Repository均支持单体和批量数据操作")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 关闭数据库连接
        try:
            from app.core.database import close_db_connections
            close_db_connections()
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接时发生错误: {e}")

if __name__ == "__main__":
    main()