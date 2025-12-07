# BackEnd/app/tests/test_prediction_pipeline.py
"""
PredictionPipeline 集成测试脚本

测试内容：
1. 已有单一学习者路径（不使用 HGC）:
   analyze([uid], is_new_learner=False)
2. 已有多个学习者路径（不使用 HGC）:
   analyze([uid1, uid2], is_new_learner=False)
3. 新单一学习者路径（强制走 HGC -> CD/KT embedding 模式）:
   analyze([uid], is_new_learner=True)
4. 新多个学习者路径:
   analyze([uid1, uid2], is_new_learner=True)

说明：
- 使用的 UID 为线上真实存在的学习者：
    - "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e"
    - "lrn_004a9c3f5bf246faab3d390ce716e658"
- 本脚本可以作为 pytest/ unittest 里的测试用例，也可以直接 python 执行。
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List

# 将项目根目录加入 Python 路径（以便在直接运行本文件时也能找到 app 包）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.domain.prediction.prediction_pipeline import analyze as pipeline_analyze  # noqa: E402

# 测试用学习者 UID（真实存在）
TEST_LEARNER_UIDS: List[str] = [
    "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
    "lrn_004a9c3f5bf246faab3d390ce716e658",
]

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def summarize_kt_result(kt_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    对 KT 结果做一个简单的统计摘要，便于快速观察执行效果
    """
    if not kt_result or not kt_result.get("success", False):
        return {
            "success": kt_result.get("success", False) if kt_result else False,
            "total_count": kt_result.get("total_count", 0) if kt_result else 0,
            "success_count": kt_result.get("success_count", 0) if kt_result else 0,
            "message": kt_result.get("error", "no data") if kt_result else "no data",
        }

    results = kt_result.get("results", [])
    learner_count = len(results)

    # 统计每个学习者的非零能力数量
    non_zero_stats = {}
    for item in results:
        uid = item.get("learner_id", "unknown")
        concept_mastery = item.get("concept_mastery", {})
        values = list(concept_mastery.values()) if concept_mastery else []
        non_zero_count = sum(1 for v in values if abs(v) > 1e-3)
        non_zero_stats[uid] = {
            "concept_count": len(values),
            "non_zero_count": non_zero_count,
        }

    return {
        "success": True,
        "total_count": kt_result.get("total_count", learner_count),
        "success_count": kt_result.get("success_count", learner_count),
        "non_zero_stats": non_zero_stats,
    }


def run_single_existing_test(learner_uid: str) -> Dict[str, Any]:
    """
    测试：已有单一学习者（不使用 HGC）
    """
    logger.info("=== 测试：已有单一学习者 (existing single) ===")
    logger.info("learner_uid = %s", learner_uid)

    result = pipeline_analyze([learner_uid], is_new_learner=False)

    kt_existing = result["kt_results"]["existing"]
    summary = summarize_kt_result(kt_existing)

    logger.info("existing single KT 摘要: %s", summary)
    return {
        "pipeline_result": result,
        "kt_summary": summary,
    }


def run_multiple_existing_test(learner_uids: List[str]) -> Dict[str, Any]:
    """
    测试：已有多个学习者（不使用 HGC）
    """
    logger.info("=== 测试：已有多个学习者 (existing multiple) ===")
    logger.info("learner_uids = %s", learner_uids)

    result = pipeline_analyze(learner_uids, is_new_learner=False)

    kt_existing = result["kt_results"]["existing"]
    summary = summarize_kt_result(kt_existing)

    logger.info("existing multiple KT 摘要: %s", summary)
    return {
        "pipeline_result": result,
        "kt_summary": summary,
    }


def run_single_new_test(learner_uid: str) -> Dict[str, Any]:
    """
    测试：新单一学习者（强制走 HGC -> CD/KT embedding 流程）

    注意：这里的“新学习者”是指流程意义上的新（使用 HGC + embedding），
    不要求数据库中真的不存在历史 KT 记录。
    """
    logger.info("=== 测试：新单一学习者 (new single) ===")
    logger.info("learner_uid = %s", learner_uid)

    result = pipeline_analyze([learner_uid], is_new_learner=True)

    kt_new = result["kt_results"]["new"]
    summary = summarize_kt_result(kt_new)

    logger.info("new single KT 摘要: %s", summary)
    return {
        "pipeline_result": result,
        "kt_summary": summary,
    }


def run_multiple_new_test(learner_uids: List[str]) -> Dict[str, Any]:
    """
    测试：新多个学习者（强制走 HGC -> CD/KT embedding 流程）
    """
    logger.info("=== 测试：新多个学习者 (new multiple) ===")
    logger.info("learner_uids = %s", learner_uids)

    result = pipeline_analyze(learner_uids, is_new_learner=True)

    kt_new = result["kt_results"]["new"]
    summary = summarize_kt_result(kt_new)

    logger.info("new multiple KT 摘要: %s", summary)
    return {
        "pipeline_result": result,
        "kt_summary": summary,
    }


def main() -> Dict[str, Any]:
    """
    作为脚本执行时的主入口：
    依次跑 4 条路径，并给出一个简单的总结。
    """
    print("\n" + "=" * 60)
    print("PredictionPipeline 综合测试")
    print("=" * 60)
    print("测试学习者: ", TEST_LEARNER_UIDS)
    print("=" * 60)

    all_results: Dict[str, Any] = {}

    try:
        # 1. 已有单一学习者
        print("\n[1/4] 已有单一学习者 (existing single)")
        res1 = run_single_existing_test(TEST_LEARNER_UIDS[0])
        all_results["existing_single"] = res1

        # 2. 已有多个学习者
        print("\n[2/4] 已有多个学习者 (existing multiple)")
        res2 = run_multiple_existing_test(TEST_LEARNER_UIDS)
        all_results["existing_multiple"] = res2

        # 3. 新单一学习者
        print("\n[3/4] 新单一学习者 (new single)")
        res3 = run_single_new_test(TEST_LEARNER_UIDS[0])
        all_results["new_single"] = res3

        # 4. 新多个学习者
        print("\n[4/4] 新多个学习者 (new multiple)")
        res4 = run_multiple_new_test(TEST_LEARNER_UIDS)
        all_results["new_multiple"] = res4

        # 统计整体成功情况
        success_paths = []
        failed_paths = []

        for name, res in all_results.items():
            kt_summary = res.get("kt_summary", {})
            if kt_summary.get("success"):
                success_paths.append(name)
            else:
                failed_paths.append(name)

        print("\n" + "=" * 60)
        print("PredictionPipeline 综合测试总结")
        print("=" * 60)
        print("时间:", datetime.now().isoformat())
        print(f"总路径数: {len(all_results)}")
        print(f"成功路径: {success_paths}")
        print(f"失败路径: {failed_paths}")
        print("=" * 60)

        return {
            "all_results": all_results,
            "success_paths": success_paths,
            "failed_paths": failed_paths,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as exc:
        print("\n测试过程中发生异常:", exc)
        import traceback

        traceback.print_exc()
        return {
            "all_results": all_results,
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    results = main()

    # 简单的退出码控制：所有路径都成功则退出 0，否则退出 1
    if results.get("failed_paths"):
        print("\n⚠️  部分路径失败，请检查日志")
        sys.exit(1)
    else:
        print("\n🎉 所有测试路径均成功！")
        sys.exit(0)
