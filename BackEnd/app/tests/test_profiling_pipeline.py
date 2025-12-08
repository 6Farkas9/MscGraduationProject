# app/tests/test_profiling_pipeline.py
# -*- coding: utf-8 -*-

"""
这是一个简单的功能性测试（functional test），用于验证 profiling_pipeline
能否正常运行、并返回预期结构，而不依赖 pytest 或 unittest 框架。

运行方式：
    python app/tests/test_profiling_pipeline.py
"""

from pprint import pprint

from app.domain.profiling.profiling_pipeline import analyze


def check_result_structure(result, learner_uids):
    """
    基础结构检查，只确保关键字段存在，不关心具体业务值。
    """
    if not isinstance(result, dict):
        print("[ERROR] 返回结果不是 dict")
        return False

    # 必须包含所有 uid
    for uid in learner_uids:
        if uid not in result:
            print(f"[ERROR] 返回结果缺少 uid: {uid}")
            return False

        block = result[uid]
        if not isinstance(block, dict):
            print(f"[ERROR] {uid} 对应结构不是 dict")
            return False

        if "overall" not in block or "details" not in block:
            print(f"[ERROR] {uid} 缺少 overall 或 details 字段")
            return False

        if not isinstance(block["overall"], dict):
            print(f"[ERROR] {uid}.overall 不是 dict")
            return False
        if not isinstance(block["details"], dict):
            print(f"[ERROR] {uid}.details 不是 dict")
            return False

        # details 里至少要有一个维度
        if len(block["details"]) == 0:
            print(f"[ERROR] {uid}.details 没有任何维度内容")
            return False

    return True


def main():
    learner_uids = [
        "lrn_51efbdbcf8844c478bbbb3ab7ad8e64e",
        # "lrn_004a9c3f5bf246faab3d390ce716e658",
    ]

    print("=== 开始测试 ProfilingPipeline ===\n")

    # 调用 pipeline
    result = analyze(learner_uids)

    # 结构检查
    print(">>> 检查返回结果结构...")
    ok = check_result_structure(result, learner_uids)
    if not ok:
        print("\n[FAILED] profiling_pipeline 的结构验证失败。\n")
        return

    print("[OK] 结构检查通过。\n")

    # 打印整体画像
    print(">>> Overall（总体画像标签）结果：")
    for uid in learner_uids:
        print(f"\n--- Learner: {uid} ---")
        pprint(result[uid]["overall"])

    # 打印某个维度的 details（示例：attention_allocation）
    example_dim = "attention_allocation"
    print(f"\n>>> Details（示例维度: {example_dim}）")
    for uid in learner_uids:
        dim_data = result[uid]["details"].get(example_dim)
        print(f"\n--- {uid} ---")
        pprint(dim_data)

    print("\n=== ProfilingPipeline 功能测试结束 ===")


if __name__ == "__main__":
    main()
