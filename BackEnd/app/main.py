# main.py
import logging
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_hgc_repository():
    """测试HGC Repository"""
    logger.info("=" * 50)
    logger.info("测试 HGC Repository")
    logger.info("=" * 50)
    
    try:
        # 获取一个测试学习者（这里假设数据库中至少有一个学习者）
        test_learner_uid = "lrn_004a9c3f5bf246faab3d390ce716e658"  # 替换为实际存在的学习者UID
        
        logger.info(f"测试学习者UID: {test_learner_uid}")
        
        # 测试获取HGC数据
        hgc_data = hgc_repository.get_data_for_single_learner(test_learner_uid)
        
        if hgc_data:
            logger.info("✓ HGC数据获取成功")
            logger.info(f"  - 交互单元数: {len(hgc_data['interacted_units'])}")
            logger.info(f"  - 涉及主题数: {len(hgc_data['learner_topics'])}")
            logger.info(f"  - 涉及课程数: {len(hgc_data['learner_courses'])}")
            logger.info(f"  - 相关学习者数: {len(hgc_data['related_learners'])}")
            logger.info(f"  - 交互记录数: {len(hgc_data['interaction_records'])}")
            logger.info(f"  - 学习者映射大小: {len(hgc_data['learner_uid_mapping'])}")
        else:
            logger.warning("✗ 未获取到HGC数据")
            
    except Exception as e:
        logger.error(f"✗ HGC Repository测试失败: {e}")

def test_cd_repository():
    """测试CD Repository"""
    logger.info("=" * 50)
    logger.info("测试 CD Repository")
    logger.info("=" * 50)
    
    try:
        test_learner_uid = "lrn_004a9c3f5bf246faab3d390ce716e658"  # 替换为实际存在的学习者UID
        
        logger.info(f"测试学习者UID: {test_learner_uid}")
        
        # 测试获取CD数据
        cd_data = cd_repository.get_data_for_single_learner(test_learner_uid)
        
        if cd_data:
            logger.info("✓ CD数据获取成功")
            logger.info(f"  - 题目交互记录数: {cd_data['interaction_count']}")
            logger.info(f"  - 格式化交互记录数: {len(cd_data['question_interactions'])}")
            logger.info(f"  - 总题目数: {len(cd_data['all_questions'])}")
            logger.info(f"  - 题目映射大小: {len(cd_data['question_uid_mapping'])}")
            
            # 显示前几条交互记录
            if cd_data['question_interactions']:
                logger.info("  - 前3条交互记录:")
                for i, record in enumerate(cd_data['question_interactions'][:3]):
                    logger.info(f"    {i+1}. {record}")
        else:
            logger.warning("✗ 未获取到CD数据")
            
    except Exception as e:
        logger.error(f"✗ CD Repository测试失败: {e}")

def test_kt_repository():
    """测试KT Repository"""
    logger.info("=" * 50)
    logger.info("测试 KT Repository")
    logger.info("=" * 50)
    
    try:
        test_learner_uid = "lrn_004a9c3f5bf246faab3d390ce716e658"  # 替换为实际存在的学习者UID
        
        logger.info(f"测试学习者UID: {test_learner_uid}")
        
        # 测试获取KT数据
        kt_data = kt_repository.get_data_for_single_learner(test_learner_uid)
        
        if kt_data:
            logger.info("✓ KT数据获取成功")
            logger.info(f"  - 学习单元交互记录数: {kt_data['interaction_count']}")
            logger.info(f"  - 格式化交互记录数: {len(kt_data['unit_interactions'])}")
            logger.info(f"  - 总学习单元数: {len(kt_data['all_units'])}")
            logger.info(f"  - 单元映射大小: {len(kt_data['unit_uid_mapping'])}")
            logger.info(f"  - 单元类型映射大小: {len(kt_data['unit_types_mapping'])}")
            
            # 显示前几条交互记录
            if kt_data['unit_interactions']:
                logger.info("  - 前3条交互记录:")
                for i, record in enumerate(kt_data['unit_interactions'][:3]):
                    logger.info(f"    {i+1}. {record}")
                    
            # 测试统计信息
            stats = kt_repository.get_interaction_statistics(test_learner_uid)
            if stats:
                logger.info("  - 交互统计信息:")
                logger.info(f"    * 总交互次数: {stats['total_interactions']}")
                logger.info(f"    * 唯一单元数: {stats['unique_units']}")
                logger.info(f"    * 平均附加信息1: {stats['avg_additioninfo1']:.2f}")
                logger.info(f"    * 平均附加信息2: {stats['avg_additioninfo2']:.2f}")
                
        else:
            logger.warning("✗ 未获取到KT数据")
            
    except Exception as e:
        logger.error(f"✗ KT Repository测试失败: {e}")

def test_base_functionality():
    """测试基础功能"""
    logger.info("=" * 50)
    logger.info("测试基础功能")
    logger.info("=" * 50)
    
    try:
        # 测试数据库连接
        from .core.database import get_mysql_connection
        connection = get_mysql_connection()
        if connection and connection.open:
            logger.info("✓ MySQL数据库连接成功")
        else:
            logger.error("✗ MySQL数据库连接失败")
            
        # 测试配置
        from .core.config import db_config
        mysql_config = db_config.get_mysql_config()
        logger.info("✓ 配置加载成功")
        logger.info(f"  - 数据库: {mysql_config['database']}")
        logger.info(f"  - 主机: {mysql_config['host']}")
        logger.info(f"  - 用户: {mysql_config['user']}")
        
    except Exception as e:
        logger.error(f"✗ 基础功能测试失败: {e}")

def test_multiple_learners():
    """测试多学习者功能"""
    logger.info("=" * 50)
    logger.info("测试多学习者功能")
    logger.info("=" * 50)
    
    try:
        # 获取一些测试学习者UID（这里需要替换为实际存在的UID）
        test_learner_uids = ["lrn_004a9c3f5bf246faab3d390ce716e658", "lrn_efc1639eb8ec4421a7559ad4d2f9858c"]  # 替换为实际存在的学习者UID
        
        logger.info(f"测试学习者UID列表: {test_learner_uids}")
        
        # 测试批量获取HGC数据
        hgc_batch_data = hgc_repository.get_data_for_multiple_learners(test_learner_uids)
        logger.info(f"✓ HGC批量数据获取: {len(hgc_batch_data)} 个学习者")
        
        # 测试批量获取CD数据
        cd_batch_data = cd_repository.get_data_for_multiple_learners(test_learner_uids)
        logger.info(f"✓ CD批量数据获取: {len(cd_batch_data)} 个学习者")
        
        # 测试批量获取KT数据
        kt_batch_data = kt_repository.get_data_for_multiple_learners(test_learner_uids)
        logger.info(f"✓ KT批量数据获取: {len(kt_batch_data)} 个学习者")
        
    except Exception as e:
        logger.error(f"✗ 多学习者测试失败: {e}")

def main():
    """主测试函数"""
    logger.info("开始测试后端数据访问层...")
    
    try:
        # 运行各个测试
        test_base_functionality()
        test_hgc_repository()
        test_cd_repository()
        test_kt_repository()
        test_multiple_learners()
        
        logger.info("=" * 50)
        logger.info("所有测试完成！")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        
    finally:
        # 关闭数据库连接
        from .core.database import close_db_connections
        close_db_connections()
        logger.info("数据库连接已关闭")

if __name__ == "__main__":
    # 使用相对导入
    from .repositories.hgc_repository import hgc_repository
    from .repositories.cd_repository import cd_repository
    from .repositories.kt_repository import kt_repository
    
    main()