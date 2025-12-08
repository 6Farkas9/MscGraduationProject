# scripts/generate_profiles_for_all_learners.py
# -*- coding: utf-8 -*-

"""
批量为系统中的每一个学习者生成画像，并保存到 MongoDB 中的 Learners 集合。

- 数据来源：MLS.Learners_bak
- 数据去向：MLS.Learners
- 画像结果来源：app.domain.profiling.profiling_pipeline.analyze

重要特性：
- 按批次处理，每批分析完立即写入，降低中途崩溃损失；
- 使用多进程加速分析；
- 带进度条；
- 断点续跑：通过 checkpoint 文件记录已完成 uid；
- 自动重试：批次级别重试；
- 错误日志独立保存在脚本同级目录。
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Any

from concurrent.futures import ProcessPoolExecutor, as_completed

from pymongo import MongoClient, UpdateOne
from tqdm import tqdm


# ========= 确保可以 import app =========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.domain.profiling.profiling_pipeline import analyze  # noqa: E402
from app.core.settings import profiling_settings  # noqa: E402


# ========= 基本配置 =========
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "MLS"
BAK_COLLECTION = "Learners_bak"
TARGET_COLLECTION = "Learners"

BATCH_SIZE = 50          # 每批处理多少个学习者
NUM_WORKERS = 4          # 进程数量（根据机器 CPU 调整）
MAX_RETRIES = 3          # 每个批次最多重试次数
DEBUG_LIMIT = 0          # 测试时只处理前 3 个学习者；正式跑全量时改为 0 或 None

# 文件路径（保存在脚本同级）
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "profiles_checkpoint.txt")
ERROR_LOG_FILE = os.path.join(SCRIPT_DIR, "profiling_errors.log")
FAILED_BATCHES_FILE = os.path.join(SCRIPT_DIR, "profiling_failed_batches.log")


# ========= 日志初始化 =========
logger = logging.getLogger("profiling_batch")
logger.setLevel(logging.INFO)

# 控制台输出
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# 错误日志文件
file_handler = logging.FileHandler(ERROR_LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.ERROR)
file_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


# ========= MongoDB 连接 =========
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
col_bak = db[BAK_COLLECTION]
col_target = db[TARGET_COLLECTION]


# ========= 工具函数 =========
def load_checkpoint(path: str) -> List[str]:
    """从 checkpoint 文件读取已完成的 uid 列表。"""
    if not os.path.exists(path):
        return []
    uids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                uids.append(line)
    return uids


def append_checkpoint(path: str, uids: List[str]) -> None:
    """将新完成的一批 uid 追加写入 checkpoint。"""
    if not uids:
        return
    with open(path, "a", encoding="utf-8") as f:
        for uid in uids:
            f.write(f"{uid}\n")


def write_failed_batch_log(batch_uids: List[str], reason: str) -> None:
    """记录最终失败的批次到单独文件。"""
    with open(FAILED_BATCHES_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"[{datetime.utcnow().isoformat()}] "
            f"FAILED BATCH ({len(batch_uids)} learners): {batch_uids}\n"
            f"REASON: {reason}\n\n"
        )


def setup_target_collection(processed_uids_exist: bool) -> None:
    """
    根据是否断点续跑决定对 Learners 集合的处理：

    - 若 checkpoint 为空（无已完成 uid）：
        视为首次运行：
        - 若 Learners 存在则删除；
        - 重新创建 Learners 集合；
        - 创建必要索引。

    - 若 checkpoint 非空：
        视为断点续跑：
        - 不删除集合；
        - 再次调用 create_index（幂等），确保索引存在。
    """
    existing_collections = db.list_collection_names()

    if not processed_uids_exist:
        # 首次运行：删旧建新
        if TARGET_COLLECTION in existing_collections:
            logger.info(
                "检测到已有集合 '%s'，将删除后重新创建。",
                TARGET_COLLECTION,
            )
            db.drop_collection(TARGET_COLLECTION)
        else:
            logger.info("集合 '%s' 不存在，将创建新集合。", TARGET_COLLECTION)
    else:
        logger.info(
            "检测到 checkpoint 中已有已处理 UID，进入断点续跑模式，不删除集合。"
        )

    # 无论是否断点续跑，确保索引存在（幂等）
    logger.info("正在为集合 '%s' 创建/确认索引...", TARGET_COLLECTION)
    col_target.create_index("uid", unique=True)
    col_target.create_index("updated_time")
    # 通配索引：方便后续对 profiles 内多种字段组合查询
    col_target.create_index([("profiles.$**", 1)])
    logger.info("索引创建/确认完成。")


def analyze_batch(uids: List[str]) -> Dict[str, Any]:
    """
    供子进程调用的分析函数。

    返回：
    {
        "ok": bool,
        "result": {uid: {...}},  # ok=True 时存在
        "error": str,            # ok=False 时存在
        "traceback": str,        # ok=False 时存在
    }
    """
    try:
        result = analyze(uids)
        return {"ok": True, "result": result}
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def write_batch_to_db(
    batch_docs: List[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
) -> None:
    """
    将一个批次的分析结果写入 Learners 集合。

    - batch_docs：来自 Learners_bak 的原始文档（仅使用 uid / KT）
    - results：profiling_pipeline.analyze 的返回结果
    """
    ops = []
    now = datetime.utcnow()

    for bak_doc in batch_docs:
        uid = bak_doc["uid"]
        profile_result = results.get(uid, {})

        # overall -> profiles
        profiles = profile_result.get("overall", {})

        new_doc = {
            "uid": uid,
            "KT": bak_doc.get("KT", {}),
            "updated_time": now,
            "profiles": profiles,
        }

        # 使用 upsert 是为了：
        # - 首次运行时插入新文档；
        # - 断点续跑时覆盖已有文档（保证幂等）。
        ops.append(
            UpdateOne(
                {"uid": uid},
                {"$set": new_doc},
                upsert=True,
            )
        )

    if ops:
        col_target.bulk_write(ops)


# ========= 主流程 =========
def main() -> None:
    logger.info("========== 开始批量生成所有学习者画像 ==========")

    # 1. 读取 checkpoint（断点续跑）
    processed_uids = set(load_checkpoint(CHECKPOINT_FILE))
    checkpoint_exists = len(processed_uids) > 0
    if checkpoint_exists:
        logger.info("已从 checkpoint 读取到 %d 个已完成的学习者。", len(processed_uids))
    else:
        logger.info("未发现有效 checkpoint，视为首次运行。")

    # 2. 处理 Learners 集合（删除/创建 & 索引）
    setup_target_collection(processed_uids_exist=checkpoint_exists)

    # 3. 从 Learners_bak 中读取所有需要处理的学习者
    total_learners = col_bak.count_documents({})
    logger.info("Learners_bak 中共 %d 个学习者。", total_learners)

    # 仅处理尚未完成的 uid
    docs_to_process: List[Dict[str, Any]] = []
    cursor = col_bak.find({}, {"_id": 1, "uid": 1, "KT": 1})
    for doc in cursor:
        uid = doc.get("uid")
        if not uid:
            continue
        if uid in processed_uids:
            continue
        docs_to_process.append(doc)

    # 如果设置了 DEBUG_LIMIT，则仅取前 N 个学习者用于测试
    if DEBUG_LIMIT and DEBUG_LIMIT > 0:
        docs_to_process = docs_to_process[:DEBUG_LIMIT]
        logger.info(
            "DEBUG 模式：仅测试前 %d 个学习者。",
            len(docs_to_process),
        )

    if not docs_to_process:
        logger.info("没有需要处理的学习者，任务结束。")
        return

    total_to_process = len(docs_to_process)
    logger.info("本次运行需要处理 %d 个学习者。", total_to_process)

    # 4. 按批次切分
    tasks: List[Dict[str, Any]] = []
    current_batch: List[Dict[str, Any]] = []
    for doc in docs_to_process:
        current_batch.append(doc)
        if len(current_batch) >= BATCH_SIZE:
            uids = [d["uid"] for d in current_batch]
            tasks.append({"uids": uids, "docs": current_batch, "attempt": 0})
            current_batch = []
    if current_batch:
        uids = [d["uid"] for d in current_batch]
        tasks.append({"uids": uids, "docs": current_batch, "attempt": 0})

    logger.info("共分为 %d 个批次，批次大小 = %d。", len(tasks), BATCH_SIZE)

    # 5. 多进程执行 + 重试 + 进度条
    remaining_tasks = tasks
    wave = 1
    processed_count = 0

    pbar = tqdm(
        total=total_to_process,
        desc="Profiling learners",
        unit="learner",
    )

    while remaining_tasks:
        logger.info(
            "开始第 %d 轮处理，本轮批次数量：%d",
            wave,
            len(remaining_tasks),
        )

        new_tasks: List[Dict[str, Any]] = []

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_map = {
                executor.submit(analyze_batch, t["uids"]): t
                for t in remaining_tasks
            }

            for future in as_completed(future_map):
                task = future_map[future]
                uids = task["uids"]
                batch_docs = task["docs"]

                try:
                    res = future.result()
                except Exception as e:  # noqa: BLE001
                    # 子进程未知异常
                    task["attempt"] += 1
                    err_msg = (
                        f"子进程执行异常（批次大小 {len(uids)}，"
                        f"attempt={task['attempt']}）：{e}"
                    )
                    logger.error(err_msg)
                    logger.error(traceback.format_exc())

                    if task["attempt"] < MAX_RETRIES:
                        new_tasks.append(task)
                    else:
                        write_failed_batch_log(uids, err_msg)
                    continue

                if not res.get("ok"):
                    # analyze_batch 内部捕获到异常
                    task["attempt"] += 1
                    err = res.get("error", "unknown error")
                    tb = res.get("traceback", "")
                    logger.error(
                        "分析批次失败（批次大小 %d，attempt=%d）：%s",
                        len(uids),
                        task["attempt"],
                        err,
                    )
                    logger.error(tb)

                    if task["attempt"] < MAX_RETRIES:
                        new_tasks.append(task)
                    else:
                        write_failed_batch_log(
                            uids,
                            f"analyze failed after {MAX_RETRIES} attempts: {err}",
                        )
                    continue

                # 成功拿到分析结果，写入数据库
                results = res["result"]
                try:
                    write_batch_to_db(batch_docs, results)
                except Exception as e2:  # noqa: BLE001
                    task["attempt"] += 1
                    err_msg = (
                        f"写入数据库失败（批次大小 {len(uids)}，"
                        f"attempt={task['attempt']}）：{e2}"
                    )
                    logger.error(err_msg)
                    logger.error(traceback.format_exc())

                    if task["attempt"] < MAX_RETRIES:
                        new_tasks.append(task)
                    else:
                        write_failed_batch_log(uids, err_msg)
                    continue

                # 完成一个批次
                processed_count += len(uids)
                pbar.update(len(uids))
                append_checkpoint(CHECKPOINT_FILE, uids)

        remaining_tasks = new_tasks
        wave += 1

    pbar.close()

    logger.info(
        "全部处理完成：本次共成功处理 %d 个学习者。", processed_count
    )
    logger.info("若存在最终失败的批次，可查看文件：%s", FAILED_BATCHES_FILE)
    logger.info("错误日志记录在：%s", ERROR_LOG_FILE)
    logger.info("========== 学习者画像生成任务结束 ==========")


if __name__ == "__main__":
    main()
