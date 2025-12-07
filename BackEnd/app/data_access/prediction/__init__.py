# BackEnd/app/data_access/prediction/__init__.py
"""
prediction 数据访问子模块

职责：
- 提供“动态知识能力预测”子域相关的数据访问能力
- 当前主要包含：
    - LearnerRepository：学习者及其 KT 结果等
    - EmbeddingRepository：各类实体（学习者 / 题目 / 知识点等）的嵌入向量
    - hgc_repository / cd_repository / kt_repository（后续会补）
"""
