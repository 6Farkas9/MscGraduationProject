# resource_orchestration_repository.py
# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional

from app.data_access.base.mongodb_base_repository import MongoDBBaseRepository


class ResourceOrchestrationRepository(MongoDBBaseRepository):
    """
    资源编排 / 匹配专用 Mongo 仓库。
    - 底层使用 MongoDBBaseRepository 封装好的 MongoDBOperator
    - 提供对 Fragments 资源池的通用访问方法
    - 不涉及任何“匹配算法”或“打分逻辑”
    """

    FRAGMENTS_COLLECTION = "Fragments"

    def __init__(self, *args, **kwargs) -> None:
        """
        保持与 MongoDBBaseRepository 一致的构造方式：
        - 若上层需要传入自定义 MongoDBOperator，可通过 *args / **kwargs 透传；
        - 若不传，则使用默认的 MongoDBOperator。
        """
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # 通用查询
    # ------------------------------------------------------------------
    def get_fragments_by_filter(
        self,
        query: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        按任意 Mongo 查询条件取 Fragments 文档。
        仅负责拼 query & 调用底层，不做任何分析逻辑。

        :param query: Mongo 查询字典
        :param limit: 限制返回数量（在 Python 侧截断）
        """
        docs = self.get_documents(self.FRAGMENTS_COLLECTION, query or {})
        if limit is not None and limit > 0 and len(docs) > limit:
            return docs[:limit]
        return docs

    # ------------------------------------------------------------------
    # 按概念 / 类型等条件做结构化查询（仍然是“数据访问”，不打分）
    # ------------------------------------------------------------------
    def get_fragments_by_concepts_and_types(
        self,
        concept_names: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        extra_filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        根据知识点名称列表 + 资源基础类型 + 额外过滤条件，查询 Fragments。

        说明：
        - concept_names 匹配的是文档中 concepts.name 字段；
        - types 匹配文档中的 type 字段；
        - extra_filter 可用于注入任意其他 Mongo 查询条件（例如难度、功能等）。
        """
        query: Dict[str, Any] = dict(extra_filter or {})

        if concept_names:
            query["concepts.name"] = {"$in": concept_names}

        if types:
            query["type"] = {"$in": types}

        docs = self.get_documents(self.FRAGMENTS_COLLECTION, query)
        if limit is not None and limit > 0 and len(docs) > limit:
            return docs[:limit]
        return docs

    # ------------------------------------------------------------------
    # Fallback：取全部（慎用，一般会配合 limit）
    # ------------------------------------------------------------------
    def get_all_fragments(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取全部资源片段（仅用于兜底，推荐务必加上 limit）。
        """
        return self.get_fragments_by_filter(query=None, limit=limit)
