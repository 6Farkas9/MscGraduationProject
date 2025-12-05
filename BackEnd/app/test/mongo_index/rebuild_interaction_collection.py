# -*- coding: utf-8 -*-
"""
一次性重建 Interaction 并从 Interaction_bak 全量导入（带单行进度条）
"""

from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError
from tqdm import tqdm

# ========= 你只需要改这里 =========
MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "MLS"
SRC_COLLECTION = "Interaction_bak"
DST_COLLECTION = "Interaction"
BATCH_SIZE = 5000
# =================================


def recreate_collection(db, coll_name: str):
    if coll_name in db.list_collection_names():
        print(f"[init] {coll_name} 已存在，drop 中...")
        db[coll_name].drop()

    print(f"[init] 创建新集合 {coll_name} ...")
    db.create_collection(coll_name)
    return db[coll_name]


def ensure_indexes(coll):
    print("[index] create idx_lrn_verb_course")
    coll.create_index(
        [("_lrn_uid", ASCENDING), ("verb.id", ASCENDING), ("_course_uid", ASCENDING)],
        name="idx_lrn_verb_course",
        background=False,
    )

    print("[index] create idx_course_verb_lrn")
    coll.create_index(
        [("_course_uid", ASCENDING), ("verb.id", ASCENDING), ("_lrn_uid", ASCENDING)],
        name="idx_course_verb_lrn",
        background=False,
    )

    print("[index] create idx_course_lrn")
    coll.create_index(
        [("_course_uid", ASCENDING), ("_lrn_uid", ASCENDING)],
        name="idx_course_lrn",
        background=False,
    )


def copy_in_batches_with_progress(src, dst, batch_size: int) -> int:
    """
    用 _id 递增分页全量复制，tqdm 单行进度条显示进度。
    """
    # 用精确 count 会更直观（但对超大表会慢一点）；你不想慢可改用 estimated_document_count()
    # try:
    #     total_docs = src.count_documents({})
    # except Exception:
    #     total_docs = src.estimated_document_count()

    total_docs = 17070102

    last_id = None
    total_inserted = 0

    with tqdm(total=total_docs, unit="docs", dynamic_ncols=True, desc="Copy Interaction_bak -> Interaction") as pbar:
        while True:
            q = {}
            if last_id is not None:
                q["_id"] = {"$gt": last_id}

            docs = list(
                src.find(q, no_cursor_timeout=True)
                   .sort([("_id", ASCENDING)])
                   .limit(batch_size)
            )
            if not docs:
                break

            try:
                dst.insert_many(docs, ordered=False)
            except BulkWriteError as e:
                # 若重复跑脚本，可能 duplicate _id；忽略 11000，其他错误抛出
                write_errors = e.details.get("writeErrors", [])
                non_dup = [we for we in write_errors if we.get("code") != 11000]
                if non_dup:
                    raise

            n = len(docs)
            total_inserted += n
            last_id = docs[-1]["_id"]

            pbar.update(n)

            # 在进度条右侧显示一些关键指标（不会换行刷屏）
            pbar.set_postfix_str(f"batch={n}, inserted={total_inserted}")

    return total_inserted


def main():
    print(f"[config] Mongo={MONGO_HOST}:{MONGO_PORT}, DB={DB_NAME}")
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]

    if SRC_COLLECTION not in db.list_collection_names():
        raise RuntimeError(f"源集合不存在: {SRC_COLLECTION}（请确认你已手动将 Interaction 改名为 Interaction_bak）")

    src = db[SRC_COLLECTION]

    dst = recreate_collection(db, DST_COLLECTION)
    ensure_indexes(dst)

    print("[copy] 开始全量导入（原封不动）...")
    inserted = copy_in_batches_with_progress(src, dst, BATCH_SIZE)
    print(f"\n[copy] 导入完成，插入约 {inserted} 条")

    print(f"[check] src estimated={src.estimated_document_count()}, dst estimated={dst.estimated_document_count()}")
    print("[finish] OK")


if __name__ == "__main__":
    main()
