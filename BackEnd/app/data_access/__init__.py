# BackEnd/app/data_access/__init__.py
"""
data_access 包

职责：
- 统一存放所有数据访问层（Repository）的实现
- 下属按子目录划分不同业务域：
    - base：通用基类和工具（MySQL / MongoDB 基类、Mixin 等）
    - prediction：动态知识能力预测相关仓库（HGC / CD / KT / Learner / Embedding）
    - profiling：多维画像相关仓库（11 个画像维度）
"""
