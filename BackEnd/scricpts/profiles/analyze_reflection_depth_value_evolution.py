# -*- coding: utf-8 -*-
"""
分析维度：反思深度与价值观演变（Reflective Depth & Value Evolution）

脚本目标（与画像设计文档对齐）：
--------------------------------------------------
对应《论文.txt》中“画像设计”里关于：
【反思深度与价值观演变（Reflective Depth & Value Evolution）】一项，强调：
- 学习者在活动后的反思频率与内容深度；
- 对“元宇宙”“学习价值”等概念的理解与态度是否随时间演变；
- 反思之后是否会去做“拓展活动”，体现“反思驱动的行动”。

本脚本采用的行为代理与论文依据：
--------------------------------------------------
1）“多次反思文本 + 概念共现 → 价值观与概念理解演变”的思路
   - Hsu et al. 2023（LEARNER-C）在“教育元宇宙”课程中：
     * 收集学生在不同时间点提交的多次反思文本和课堂讨论文本；
     * 构建词–文档矩阵 X，计算词共现矩阵 X^T X 以及学生共现矩阵 X X^T，
       将“metaverse / value / experience / us”等词作为核心节点，
       分析其在共现网络中的中心性与连接变化，用来刻画学生价值观和概念理解的变化。
   - 该思路表明：
     * “围绕核心概念的词语使用情况 + 时间切片对比”可以作为价值观/概念理解演变的行为代理。
   → 本脚本在没有教师讲稿原文与完整网络可视化的前提下，做一个“轻量共现代理”：
     - 针对元宇宙价值相关的关键词（如“元宇宙 / metaverse / 价值 / value / 体验 / experience / 我们 / us”），
       统计其在学习者反思文本中的出现频率和覆盖度；
     - 将同一 (学习者, 课程) 的反思文本按时间划分为“早期”与“后期”两段，
       比较两段中的“价值语汇使用评分”，构造“价值观演变评分（value_evolution_score）”。

2）“日志/文本数据用于刻画情感、价值、态度”的方法论依据
   - Lampropoulos & Lappas 2025 等关于 XR+LA/EDM 的系统综述指出：
     * 在 VR/AR/元宇宙环境的学习分析中，典型数据源包括：
       交互日志、行为轨迹、文本/对话内容等；
     * 多篇文献使用这些数据分析学习者的参与度、动机、情感与态度等。
   - 更广义的 LA/EDM 综合调查也明确：
     * LMS、MOOC、虚拟环境等系统记录的“阅读/书写、发帖、任务执行”等行为日志
       是建模学习者状态和态度（如持续参与意愿、自我调节、价值感知）的重要基础。
   → 因此，本脚本使用：
     - xAPI 中 verb = reflected-on-activity 的 result.response 文本作为反思内容；
     - 其频率与文本特征（长度、多样性、价值关键词密度）作为“反思深度”代理；
     - 配合时间切片对比，用于刻画“价值与态度是否发生演变”。

3）“反思驱动的行动”：反思后是否探索拓展
   - 在你的画像设计中，明确提出：
     * reflected-on-activity 与 explored-extension 联合使用，
       “反思后是否去做拓展”可以体现“反思驱动的行动”和对学习价值的内化程度。
   → 本脚本据此定义：
     - 在一定时间窗口内（默认 30 分钟），“一次反思”后是否出现“explored-extension”事件；
     - 统计“反思 → 拓展”的配对比例，形成 reflection_to_action_rate，
       作为“把反思转化为行动”的行为指标。

4）“多指标综合 → 反思与价值观演变指数”的设计思路
   - LEARNER-C 的原始工作更偏向网络结构分析与描述性比较，并未给出固定的“分档标签”。
   - 为适配你的画像框架，本脚本在保留“概念共现 + 时间比较”核心思想的基础上，
     设计了一个可计算的综合指数：
       reflection_index = f(反思频率, 反思文本深度, 价值观演变程度, 反思驱动行动比例)
     并将其归一化为 [0,1]，再据此划分为三档标签：
       - 低水平反思者（浅层/不稳定反思）；
       - 中等反思者（有一定深度但价值演变有限或不稳定）；
       - 成长型价值反思者（反思频率与深度较高，且价值语汇随时间显著增加）。
   - 与原文的差异说明：
     * 原文使用完整的词共现网络和中心性指标，本脚本改为：
       - 用“价值关键词密度 + 覆盖度”作为“核心节点聚集度”的轻量代理；
       - 用“早期 vs 后期价值分数差值”作为网络结构变化的简化量化指标；
       - 再结合其他行为特征（频率、反思驱动行动）做综合评分与分档。
     * 改动原因：
       - 实际部署中不一定具备教师讲稿与完整网络分析组件；
       - 希望脚本在只依赖 xAPI 文本字段的前提下即可执行，同时保持与 LEARNER-C 的核心思想一致。

5）分类标签设计与 LearnerProfile 对齐
   - xAPI 生成脚本中，画像维度列表包含 "reflection_depth"，并基于粗粒度统计给出反思相关分数，
     用于合成 LearnerProfile.global_profile.reflection_depth.score。
   - 因此，本脚本：
     - 使用行为侧的 reflection_index_norm（[0,1]）作为“细粒度反思与价值观演变水平”；
     - 将其分为三档，并为每档给出明确的文本标签说明；
     - 在学习者层面对 reflection_index_norm 做平均，得到 global_reflection_index，
       与 LearnerProfile.global_profile.reflection_depth.score 计算皮尔逊相关，
       用于粗略验证细粒度分析与人设维度的一致性。

脚本功能总结：
--------------------------------------------------
1. 从 MongoDB 中读取：
   - 细粒度 xAPI 行为：MLS.Interaction 集合；
   - 学习者画像：MLS.LearnerProfile 集合（特别是 global_profile.reflection_depth.score）。

2. 对每个 (学习者, 课程)：
   - 使用 verb = reflected-on-activity 的事件：
     * 统计反思次数（反思频率）；
     * 对每条反思文本进行分词，计算：
       - 文本长度（词数）；
       - 词汇多样性（type-token ratio）；
       - “元宇宙/学习价值”等价值关键词密度；
     * 将所有反思按时间排序，划分为“早期”和“后期”，
       比较两段的“价值语汇使用评分”，得到 value_evolution_score（后期 - 早期）。
   - 使用 verb = explored-extension 的事件：
     * 在同一 (学习者, 课程) 内，寻找每条反思之后的“拓展探索”行为（默认 30 分钟窗口）；
     * 统计“反思 → 拓展”的配对比例 reflection_to_action_rate。

3. 将上述指标归一化并构造综合指数：
   - freq_norm：课程内（学习者间）反思次数归一化；
   - depth_norm：基于文本长度、多样性与价值关键词密度综合得到的反思深度评分归一化；
   - value_growth_norm：仅考虑“正向”价值观演变（value_evolution_score > 0 的部分）并归一化；
   - action_norm：reflection_to_action_rate 归一化；
   - 最终得到 reflection_index 和其归一化版本 reflection_index_norm ∈ [0,1]。

4. 基于 reflection_index_norm 和 value_evolution_score：
   - 将每个 (学习者, 课程) 划分为三档反思与价值观演变类型，并给出中文解释标签。

5. 与人设对比：
   - 对每个学习者，把其在所有课程上的 reflection_index_norm 取平均，
     得到行为侧 global_reflection_index；
   - 与 LearnerProfile.global_profile.reflection_depth.score 做皮尔逊相关，
     用于粗略验证行为分析与人设中反思维度的一致性。

6. 数据库存储接口（不在 main() 中调用）：
   - 定义 save_reflection_analysis_to_db(db, reflection_results) 函数，
     演示如何把结果写入 MLS.ReflectionValueAnalysis 集合，但默认不调用。
   - 你可以在需要时手动取消注释进行持久化。
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
from math import sqrt
from collections import defaultdict
import random
import re

from tqdm import tqdm

# 可选中文分词：若环境中安装了 jieba，则优先使用；否则使用简单规则分词
try:
    import jieba  # type: ignore
except ImportError:  # pragma: no cover - 环境可能没有安装 jieba
    jieba = None

# ===================== 配置区域 =====================

MONGO_HOST = "localhost"
MONGO_PORT = 27017
DB_NAME = "MLS"
XAPI_COLLECTION = "Interaction"          # 细粒度行为集合（xAPI_interaction_profile.py 生成）
PROFILE_COLLECTION = "LearnerProfile"    # 人设集合（infer_persona_for_course 写入）
REFLECTION_COLLECTION = "ReflectionValueAnalysis"  # 反思深度与价值观演变分析结果集合（仅接口，不在 main 中调用）

# 采样学习者数量：随机选取至多 N_SAMPLE 个学习者进行分析
# 若 N_SAMPLE <= 0 或大于实际可选数量，则使用所有学习者
N_SAMPLE = 3000

VERB_BASE = "https://legend-meta.com/xapi/verb/"

VERBS = {
    # 反思事件：用于提取反思文本
    "reflected_on_activity": VERB_BASE + "reflected-on-activity",
    # 拓展探索事件：用于判断“反思后是否去做拓展”
    "explored_extension": VERB_BASE + "explored-extension",
}

# 反思之后统计“反思驱动拓展”时使用的时间窗口（秒）
REFLECTION_ACTION_WINDOW_SECONDS = 30 * 60  # 30 分钟

# 反思文本中关注的“元宇宙 / 学习价值”相关关键词
# 设计依据：
#   - Hsu et al. 2023 中共现网络的核心节点包括 “metaverse / us / value / experience”等，
#     这里在此基础上扩展部分与学习价值和意义相关的词汇。
VALUE_KEYWORDS = {
    # 英文
    "metaverse", "value", "values", "experience", "meaning", "ethics",
    "privacy", "fairness", "community", "identity", "learning", "future",
    # 中文
    "元宇宙", "价值", "体验", "意义", "伦理", "隐私", "公平", "社群", "身份", "学习", "未来", "贡献",
    # 与“我们”相关的共同体视角
    "我们", "us", "一起", "集体",
}

# 一些简单的中文/英文停用词（主要为了减少无信息词对多样性的干扰）
STOPWORDS = {
    # 中文常见虚词
    "的", "了", "在", "是", "和", "与", "也", "就", "而且", "但是", "因为", "所以", "如果",
    "对", "于", "中", "一个", "自己", "他们", "它", "她", "他",
    # 英文
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on",
    "for", "with", "this", "that", "it", "is", "are", "was", "were",
}

# ===================== 工具函数 =====================

def compute_mean_std(values):
    """
    计算一组数的均值和标准差（总体标准差）：
    - 列表为空 -> (0, 0)
    - 仅一个元素 -> 标准差视为 0

    用途：
    - 计算与人设分数的相关系数；
    - 如有需要，可在后续扩展更多标准化逻辑。
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean_v = sum(values) / float(n)
    if n == 1:
        return mean_v, 0.0
    var = sum((v - mean_v) ** 2 for v in values) / float(n)
    std = sqrt(var)
    return mean_v, std


def normalize_min_max(values, default_mid=False):
    """
    对一组数做 [0,1] 的 min-max 归一化，返回与输入等长的列表。

    参数：
        values: List[float]
        default_mid: 若所有值相同或列表为空：
            - True  -> 全部返回 0.5
            - False -> 全部返回 0.0

    设计说明：
    - 在本脚本中，多处需要在“课程内 / 全局内”把某个指标归一化到 [0,1]，便于后续组合。
    - 若没有差异（例如所有人反思次数都一样），则该指标在该上下文中不区分人，
      此时默认返回常数（0.0 或 0.5），以避免除零错误。
    """
    if not values:
        return []

    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) < 1e-8:
        fill = 0.5 if default_mid else 0.0
        return [fill for _ in values]

    span = v_max - v_min
    return [(v - v_min) / span for v in values]


def parse_iso_timestamp(ts_str):
    """
    解析 ISO8601 时间戳字符串为 datetime 对象。
    - xAPI_interaction_profile.py 中生成的 timestamp 使用 ISO 格式（含时区或不含时区）。
    - 若格式不合法或为空，返回 None。
    """
    if not ts_str:
        return None
    try:
        # 处理可能存在的 'Z' 结尾
        if ts_str.endswith("Z"):
            ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None


def tokenize_text(text):
    """
    对反思文本进行分词，返回 token 列表。

    设计说明（结合 LEARNER-C 的预处理思路）：
    --------------------------------------------------
    - Hsu et al. 2023 在构建词–文档矩阵前进行了文本预处理（分词、去停用词等），
      本函数在部署约束下实现一个简化版本：
        1）若环境安装了 jieba，则使用 jieba.lcut 对中英文混合文本分词；
        2）否则：
            - 将文本统一转为小写；
            - 用正则将非“字母/数字/汉字”的字符替换为空格；
            - 按空格切分得到 token。
    - 为避免过多无信息词稀释“价值关键词密度”和“多样性”，
      在分词后会过滤一小部分中英停用词。
    """
    if text is None:
        return []
    text = str(text).strip()
    if not text:
        return []

    if jieba is not None:
        raw_tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
    else:
        # 保留字母、数字和汉字，其余全部视为分隔符
        text_norm = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
        raw_tokens = text_norm.split()

    tokens = []
    for t in raw_tokens:
        t_norm = t.strip().lower()
        if not t_norm:
            continue
        if t_norm in STOPWORDS:
            continue
        tokens.append(t_norm)
    return tokens


def analyze_reflection_text(text):
    """
    针对单条反思文本，计算若干“文本深度”和“价值语汇”相关统计。

    返回：
        {
            "tokens": List[str],           # 分词结果
            "word_count": int,            # 词数
            "lexical_diversity": float,   # 词汇多样性（不同比例）
            "value_keyword_hits": int,    # 命中的价值关键词数量（去重计数）
            "value_keyword_density": float, # 价值关键词密度（命中数 / 词数）
            "value_keywords_used": set[str], # 命中的具体价值关键词集合
        }

    设计说明与论文依据：
    --------------------------------------------------
    - 文本长度（word_count）：
        * 在 LEARNER-C 场景中，更丰富的反思往往包含更多具体描述与概念，
          在没有自动摘要与主题模型的情况下，词数可以作为最基础的“展开程度”代理。
    - 词汇多样性（lexical_diversity）：
        * type-token ratio 越高，说明使用了更多不同词汇，通常意味着更丰富的表达与思考路径。
    - 价值关键词密度与覆盖度：
        * 参考 LEARNER-C 中围绕 “metaverse / value / experience / us” 等词的共现网络分析，
          本脚本将这些词及其中文对应和扩展词表入 VALUE_KEYWORDS，
          用“命中数量/密度 + 不同关键词覆盖度”作为“价值相关概念卷入程度”的代理。
    """
    tokens = tokenize_text(text)
    word_count = len(tokens)
    if word_count == 0:
        return {
            "tokens": [],
            "word_count": 0,
            "lexical_diversity": 0.0,
            "value_keyword_hits": 0,
            "value_keyword_density": 0.0,
            "value_keywords_used": set(),
        }

    unique_tokens = set(tokens)
    lexical_diversity = len(unique_tokens) / float(word_count)

    # 直接在原始字符串上匹配关键词（避免分词影响）
    text_lower = str(text).lower()
    value_keywords_used = set()
    for kw in VALUE_KEYWORDS:
        if kw.lower() in text_lower:
            value_keywords_used.add(kw.lower())

    value_keyword_hits = len(value_keywords_used)
    value_keyword_density = value_keyword_hits / float(word_count) if word_count > 0 else 0.0

    return {
        "tokens": tokens,
        "word_count": word_count,
        "lexical_diversity": lexical_diversity,
        "value_keyword_hits": value_keyword_hits,
        "value_keyword_density": value_keyword_density,
        "value_keywords_used": value_keywords_used,
    }


# ===================== 数据库存储接口（不在 main 中调用） =====================

def save_reflection_analysis_to_db(db, reflection_results):
    """
    把“反思深度与价值观演变”分析结果写入 MongoDB 的接口函数（默认不在 main 中调用）。

    设计目的：
    --------------------------------------------------
    - 应对你的需求：“最终结果不需要写回数据库，但需要预留写回接口”。
    - 如果将来你希望把分析结果持久化到 MLS.ReflectionValueAnalysis 集合，
      可以在 main() 末尾手动调用本函数。

    写入字段设计说明：
    --------------------------------------------------
    1）字段 learner_uid / course_uid：
       - 与 MLS.Interaction 集合中的 _lrn_uid / _course_uid 保持一一对应，
         方便后续按课程或学习者维度聚合。

    2）reflection_count / freq_norm：
       - reflection_count：该学习者在该课程中的反思条数；
       - freq_norm：在（学习者, 课程）范围内对频率做 [0,1] 归一化后的结果。

    3）depth_score_avg：
       - 该 (学习者, 课程) 的平均“反思深度”得分，
         来自所有反思文本的长度、多样性、价值关键词密度的综合评分。
       - 这一设计对应 LEARNER-C 中“反思文本复杂度和概念卷入程度”的总体思路。

    4）value_evolution_score / value_growth_norm：
       - value_evolution_score：后期反思的价值分数减去早期反思的价值分数；
       - value_growth_norm：只对正向变化部分做 [0,1] 归一化后的得分，
         用于表示“价值相关语汇是否显著增强”。

    5）reflection_to_action_rate / action_norm：
       - reflection_to_action_rate：有多少比例的反思在 30 分钟内伴随“explored-extension”事件；
       - action_norm：在所有记录中对此比值做 [0,1] 归一化。

    6）reflection_index / reflection_index_norm / reflection_label / reflection_level：
       - reflection_index：由 freq_norm, depth_norm, value_growth_norm, action_norm
         线性组合而成的综合指数；
       - reflection_index_norm：对其再做一次 [0,1] 归一化；
       - reflection_label：基于 reflection_index_norm 与 value_evolution_score 的中文标签；
       - reflection_level：数值等级（0=低反思/不稳定, 1=中等, 2=成长型）。

    注意：
    --------------------------------------------------
    - 本函数不会在 main() 中自动调用。
    - 如果你希望实际写回数据库，请在 main() 中手动解除注释：
        save_reflection_analysis_to_db(db, reflection_results)
    """
    col = db[REFLECTION_COLLECTION]

    # 为方便重复实验，这里先清空集合（若你不想清空，可以改为 update 或 upsert）
    db.drop_collection(REFLECTION_COLLECTION)
    col = db[REFLECTION_COLLECTION]

    docs_to_insert = []
    for (lrn_uid, crs_uid), res in reflection_results.items():
        doc = {
            "learner_uid": lrn_uid,
            "course_uid": crs_uid,
            "reflection_count": res.get("reflection_count", 0),
            "freq_norm": res.get("freq_norm"),
            "depth_score_avg": res.get("depth_score_avg"),
            "value_evolution_score": res.get("value_evolution_score"),
            "value_growth_norm": res.get("value_growth_norm"),
            "reflection_to_action_rate": res.get("reflection_to_action_rate"),
            "action_norm": res.get("action_norm"),
            "reflection_index": res.get("reflection_index"),
            "reflection_index_norm": res.get("reflection_index_norm"),
            "reflection_label": res.get("reflection_label"),
            "reflection_level": res.get("reflection_level"),
            "created_at": datetime.utcnow(),
        }
        docs_to_insert.append(doc)

    if docs_to_insert:
        col.insert_many(docs_to_insert, ordered=False)
        col.create_index(
            [("learner_uid", 1), ("course_uid", 1)],
            name="idx_learner_course"
        )
        print(f"[接口调用] 已写入 ReflectionValueAnalysis 文档数：{len(docs_to_insert)}")
    else:
        print("[接口调用] 没有可写入 ReflectionValueAnalysis 的文档。")


# ===================== 主分析逻辑 =====================

def main():
    # ---------- 1. 连接 MongoDB ----------
    client = MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[DB_NAME]
    xapi_col = db[XAPI_COLLECTION]
    profile_col = db[PROFILE_COLLECTION]

    print("连接 MongoDB 成功。")

    # ---------- 2. 读取 LearnerProfile 中“反思深度”人设 ----------
    print("读取 LearnerProfile 中的反思深度人设信息（global_profile.reflection_depth.score）...")
    persona_scores = {}  # lrn_uid -> persona_reflection_score

    cursor_profiles = profile_col.find(
        {},
        {"learner_uid": 1, "global_profile": 1}
    )

    for doc in cursor_profiles:
        lrn_uid = doc.get("learner_uid")
        if not lrn_uid:
            continue
        g_profile = doc.get("global_profile") or {}
        ref = g_profile.get("reflection_depth") or {}
        score = ref.get("score")
        if score is not None:
            # 画像方案中 reflection_depth.score 设计在 [0,1] 区间，这里直接读取。
            persona_scores[lrn_uid] = float(score)

    all_learners_with_persona = list(persona_scores.keys())
    print(f"具备反思深度人设的学习者数量：{len(all_learners_with_persona)}")

    if not all_learners_with_persona:
        print("没有任何学习者具备反思深度人设，无法进行对比分析。")
        return

    # ---------- 3. 随机采样学习者 ----------
    if N_SAMPLE > 0 and N_SAMPLE < len(all_learners_with_persona):
        sampled_learners = random.sample(all_learners_with_persona, N_SAMPLE)
    else:
        sampled_learners = all_learners_with_persona
    sampled_set = set(sampled_learners)

    print(f"随机选取用于分析的学习者数量：{len(sampled_learners)} (N_SAMPLE = {N_SAMPLE})")

    # ---------- 4. 一次性加载采样学习者的反思与拓展事件 ----------
    """
    事件筛选策略与论文依据：
    --------------------------------------------------
    1）使用的 verb：
       - reflected-on-activity：
         * 对应元宇宙学习活动后的“提交反思文本”事件，
           result.response 中包含反思内容；
         * 这是本脚本分析“反思深度与价值观演变”的核心数据源。
       - explored-extension：
         * 对应在课程中进行“拓展探索”的行为（例如访问额外资源、可选活动），
           与画像设计中的“反思之后是否去做拓展”直接对应。

    2）查询条件：
       - 仅针对采样学习者（_lrn_uid in sampled_learners），避免处理全部数据；
       - 只保留上述两个 verb.id 的事件。
    """
    query = {
        "_lrn_uid": {"$in": sampled_learners},
        "verb.id": {"$in": [VERBS["reflected_on_activity"], VERBS["explored_extension"]]}
    }

    print("统计待加载的反思与拓展事件数量（count_documents）...")
    total_events = xapi_col.count_documents(query)
    print(f"与采样学习者相关的反思/拓展事件总数：{total_events}")

    print("开始一次性加载所有相关事件到内存（list）...")
    events = list(xapi_col.find(
        query,
        {
            "verb.id": 1,
            "result": 1,
            "context": 1,
            "timestamp": 1,
            "_lrn_uid": 1,
            "_course_uid": 1,
        }
    ))
    print(f"已从 MongoDB 读取事件条数：{len(events)}")

    if not events:
        print("没有任何反思或拓展事件，无法进行本维度分析。")
        return

    # ---------- 5. 将事件按 (学习者, 课程) 聚合为“反思列表”与“拓展时间列表” ----------
    """
    聚合逻辑说明：
    --------------------------------------------------
    - 粒度：以 (学习者, 课程) 为聚合单位，与其他画像分析脚本保持一致。
    - 对每个 (学习者, 课程)：
        * reflections_by_lc[(lrn_uid, crs_uid)] = [ {timestamp, text, format, ...}, ... ]
        * extensions_by_lc[(lrn_uid, crs_uid)] = [ ts1, ts2, ... ]  # 升序排序后用于匹配窗口内拓展行为。
    """
    reflections_by_lc = defaultdict(list)   # (lrn_uid, course_uid) -> List[dict]
    extensions_by_lc = defaultdict(list)    # (lrn_uid, course_uid) -> List[datetime]

    for doc in tqdm(events, desc="整理反思/拓展事件", unit="event"):
        lrn_uid = doc.get("_lrn_uid")
        crs_uid = doc.get("_course_uid")
        if not lrn_uid or not crs_uid:
            continue

        verb_id = (doc.get("verb") or {}).get("id")
        ts = parse_iso_timestamp(doc.get("timestamp"))
        if ts is None:
            # 没有时间戳无法做时间切片，直接跳过
            continue

        key = (lrn_uid, crs_uid)

        if verb_id == VERBS["reflected_on_activity"]:
            result = doc.get("result") or {}
            context = doc.get("context") or {}
            ctx_ext = (context.get("extensions") or {})
            reflection_format = ctx_ext.get("https://legend-meta.com/xapi/ext/reflection-format", None)
            text = result.get("response") or ""
            reflections_by_lc[key].append({
                "timestamp": ts,
                "text": text,
                "format": reflection_format,
            })
        elif verb_id == VERBS["explored_extension"]:
            extensions_by_lc[key].append(ts)

    # 按时间排序，便于后续早期/后期切片与窗口匹配
    for key in reflections_by_lc:
        reflections_by_lc[key].sort(key=lambda r: r["timestamp"])
    for key in extensions_by_lc:
        extensions_by_lc[key].sort()

    if not reflections_by_lc:
        print("聚合后没有任何反思事件，结束分析。")
        return

    # ---------- 6. 对所有反思文本进行逐条文本分析 ----------
    """
    此步骤对应 LEARNER-C 中的“文本预处理与特征提取”阶段：
    --------------------------------------------------
    - 对每条反思文本计算：
        * word_count：文本长度；
        * lexical_diversity：词汇多样性；
        * value_keyword_density：价值关键词密度。
    - 这些特征将用于构建“反思深度”与“价值概念卷入程度”的基础指标。
    """
    all_word_counts = []
    all_lex_divs = []
    all_value_dens = []

    # 先对每条反思附加文本特征
    for key, reflist in reflections_by_lc.items():
        for r in reflist:
            analysis = analyze_reflection_text(r.get("text"))
            r["tokens"] = analysis["tokens"]
            r["word_count"] = analysis["word_count"]
            r["lexical_diversity"] = analysis["lexical_diversity"]
            r["value_keyword_hits"] = analysis["value_keyword_hits"]
            r["value_keyword_density"] = analysis["value_keyword_density"]
            r["value_keywords_used"] = analysis["value_keywords_used"]

            all_word_counts.append(analysis["word_count"])
            all_lex_divs.append(analysis["lexical_diversity"])
            all_value_dens.append(analysis["value_keyword_density"])

    if not all_word_counts:
        print("所有反思文本均为空或无法解析，结束分析。")
        return

    # 对三个基础特征做全局 min-max 归一化，用于构建“反思深度得分”
    wc_norms = normalize_min_max(all_word_counts, default_mid=True)
    lv_norms = normalize_min_max(all_lex_divs, default_mid=True)
    vd_norms = normalize_min_max(all_value_dens, default_mid=False)

    # 将归一化结果回填到每条反思记录，并计算单条反思的“深度得分”和“价值得分”
    """
    深度得分与价值得分的计算思路：
    --------------------------------------------------
    - 单条反思的深度得分 depth_score_ref：
        * 参考 LEARNER-C 中“反思文本复杂度 + 概念卷入”的综合视角，
          这里采用三个归一化特征的加权平均：
            depth_score_ref = 0.3 * wc_norm + 0.3 * lv_norm + 0.4 * vd_norm
          其中价值关键词密度权重略高，突出“围绕元宇宙/价值的概念讨论”。

    - 单条反思的价值得分 value_score_ref：
        * 更侧重“价值语汇的稀疏/密集程度与不同关键词覆盖度”，综合：
            - value_keyword_density（密度）
            - 使用到的不同价值关键词数量 / 总关键词表大小（覆盖度）
    """
    total_reflections = sum(len(v) for v in reflections_by_lc.values())
    idx = 0
    for key, reflist in reflections_by_lc.items():
        for r in reflist:
            wc_n = wc_norms[idx]
            lv_n = lv_norms[idx]
            vd_n = vd_norms[idx]
            idx += 1

            depth_score = 0.3 * wc_n + 0.3 * lv_n + 0.4 * vd_n

            coverage = 0.0
            if VALUE_KEYWORDS:
                coverage = len(r["value_keywords_used"]) / float(len(VALUE_KEYWORDS))

            value_score = 0.5 * vd_n + 0.5 * coverage

            r["wc_norm"] = wc_n
            r["lex_div_norm"] = lv_n
            r["value_density_norm"] = vd_n
            r["depth_score"] = depth_score
            r["value_score"] = value_score

    print(f"已完成 {total_reflections} 条反思文本的深度与价值特征计算。")

    # ---------- 7. 针对每个 (学习者, 课程) 计算聚合指标 ----------
    """
    聚合指标设计：
    --------------------------------------------------
    对每个 (学习者, 课程)，计算：
    - reflection_count：反思总次数；
    - depth_score_avg：所有反思的 depth_score 平均值；
    - value_early / value_late：
        * 按时间排序后的反思序列：
          若有 N 条反思：
            - N >= 4：前 N//2 条视为“早期”，后 N-N//2 条视为“后期”；
            - N = 2 或 3：前 1 条为“早期”，其余为“后期”；
            - N = 1：无法切分，记为早期/后期均为该条 value_score（后续会特殊处理）。
        * value_early = 早期反思 value_score 的平均；
        * value_late  = 后期反思 value_score 的平均；
    - value_evolution_score = value_late - value_early：
        * 若 > 0：后期更频繁/全面地使用元宇宙与价值语汇，视作“价值观/概念理解有正向演变”；
        * 若 ≈ 0：变化不明显；
        * 若 < 0：可能转向其他主题或价值语汇减少。
    - reflection_to_action_rate：
        * 对每条反思，检查在窗口 REFLECTION_ACTION_WINDOW_SECONDS 内，
          是否存在同一 (学习者, 课程) 下的 explored-extension 事件；
        * 有至少一个拓展事件则认为该反思“触发了拓展行动”；
        * 统计比例 = 触发拓展的反思数 / 总反思数。
    """
    reflection_metrics = {}  # (lrn_uid, crs_uid) -> dict of metrics

    for key, reflist in reflections_by_lc.items():
        lrn_uid, crs_uid = key
        if not reflist:
            continue

        n = len(reflist)
        reflection_count = n

        depth_scores = [r["depth_score"] for r in reflist]
        depth_score_avg = sum(depth_scores) / float(len(depth_scores)) if depth_scores else 0.0

        # 按反思时间切分早期/后期
        reflist_sorted = sorted(reflist, key=lambda r: r["timestamp"])
        if n >= 4:
            split_idx = n // 2
        elif n >= 2:
            split_idx = 1
        else:
            split_idx = 1  # N = 1 的情况

        early_refs = reflist_sorted[:split_idx]
        late_refs = reflist_sorted[split_idx:] if split_idx < n else reflist_sorted

        if early_refs:
            value_early = sum(r["value_score"] for r in early_refs) / float(len(early_refs))
        else:
            value_early = 0.0

        if late_refs:
            value_late = sum(r["value_score"] for r in late_refs) / float(len(late_refs))
        else:
            value_late = 0.0

        value_evolution_score = value_late - value_early

        # 计算“反思驱动拓展”的比例
        ext_times = extensions_by_lc.get(key, [])
        ext_count = len(ext_times)
        reflection_with_action = 0

        if ext_times:
            # 双指针扫描：对每条反思，查找窗口内的最近拓展事件
            j = 0
            for r in reflist_sorted:
                ts_ref = r["timestamp"]
                # 将指针移动到第一个 >= ts_ref 的拓展事件
                while j < ext_count and ext_times[j] < ts_ref:
                    j += 1
                has_action = False
                k = j
                while k < ext_count:
                    delta = (ext_times[k] - ts_ref).total_seconds()
                    if delta < 0:
                        k += 1
                        continue
                    if delta <= REFLECTION_ACTION_WINDOW_SECONDS:
                        has_action = True
                        break
                    else:
                        # 已超过窗口，不再继续
                        break
                if has_action:
                    reflection_with_action += 1

        if reflection_count > 0:
            reflection_to_action_rate = reflection_with_action / float(reflection_count)
        else:
            reflection_to_action_rate = 0.0

        reflection_metrics[key] = {
            "reflection_count": reflection_count,
            "depth_score_avg": depth_score_avg,
            "value_early": value_early,
            "value_late": value_late,
            "value_evolution_score": value_evolution_score,
            "reflection_to_action_rate": reflection_to_action_rate,
        }

    print(f"完成 {len(reflection_metrics)} 个 (学习者, 课程) 的反思聚合指标计算。")

    if not reflection_metrics:
        print("没有可用的反思聚合指标，结束分析。")
        return

    # ---------- 8. 对聚合指标做归一化并构造综合 reflection_index ----------
    """
    综合指数设计：
    --------------------------------------------------
    对每个 (学习者, 课程) 计算：
    - freq_norm：在所有 (lrn, course) 中对 reflection_count 做 [0,1] 归一化；
    - depth_norm：对 depth_score_avg 做 [0,1] 归一化；
    - value_growth_norm：
        * 将 value_evolution_score 中的正值取出（负值视为 0，代表没有明显正向演变），
          对这些正值做 [0,1] 归一化，得到 value_growth_norm；
    - action_norm：对 reflection_to_action_rate 做 [0,1] 归一化。

    综合指数：
        reflection_index_raw =
            0.3 * freq_norm +
            0.4 * depth_norm +
            0.3 * (0.5 * value_growth_norm + 0.5 * action_norm)

    再对 reflection_index_raw 做 [0,1] 的 min-max 归一化，得到 reflection_index_norm。

    权重解释：
    - 深度（depth_norm）权重最高（0.4）：
        * 对应你对“反思深度”的核心关注；
    - 频率（freq_norm）权重 0.3：
        * 体现“是否形成持续反思习惯”；
    - 价值演变与反思驱动行动（剩余 0.3）：
        * 结合 LEARNER-C 中“价值/概念理解的演变”和画像方案中“反思后去做拓展”的行为，
          共同代表“反思是否真正指向价值澄清与学习路径调整”。
    """
    freq_vals = []
    depth_vals = []
    value_growth_vals = []
    action_vals = []

    for key, m in reflection_metrics.items():
        freq_vals.append(m["reflection_count"])
        depth_vals.append(m["depth_score_avg"])
        # 只取正向变化作为“成长”部分
        value_growth_vals.append(max(m["value_evolution_score"], 0.0))
        action_vals.append(m["reflection_to_action_rate"])

    freq_norm_list = normalize_min_max(freq_vals, default_mid=True)
    depth_norm_list = normalize_min_max(depth_vals, default_mid=True)
    value_growth_norm_list = normalize_min_max(value_growth_vals, default_mid=False)
    action_norm_list = normalize_min_max(action_vals, default_mid=False)

    reflection_index_raw_list = []
    keys_list = list(reflection_metrics.keys())

    for i, key in enumerate(keys_list):
        freq_norm = freq_norm_list[i]
        depth_norm = depth_norm_list[i]
        value_growth_norm = value_growth_norm_list[i]
        action_norm = action_norm_list[i]

        # 组合成 raw index
        value_part = 0.5 * value_growth_norm + 0.5 * action_norm
        reflection_index_raw = 0.3 * freq_norm + 0.4 * depth_norm + 0.3 * value_part
        reflection_index_raw_list.append(reflection_index_raw)

        # 先临时保存，后面还要归一化 reflection_index_raw
        m = reflection_metrics[key]
        m["freq_norm"] = freq_norm
        m["depth_norm"] = depth_norm
        m["value_growth_norm"] = value_growth_norm
        m["action_norm"] = action_norm
        m["reflection_index_raw"] = reflection_index_raw

    reflection_index_norm_list = normalize_min_max(reflection_index_raw_list, default_mid=True)

    for i, key in enumerate(keys_list):
        reflection_metrics[key]["reflection_index"] = reflection_metrics[key]["reflection_index_raw"]
        reflection_metrics[key]["reflection_index_norm"] = reflection_index_norm_list[i]

    print("已完成反思综合指数（reflection_index）与归一化（reflection_index_norm）计算。")

    # ---------- 9. 按 reflection_index_norm 与 value_evolution_score 生成分类标签 ----------
    """
    分类规则说明：
    --------------------------------------------------
    - 对每个 (学习者, 课程)，已知：
        * reflection_index_norm ∈ [0,1]：整体“反思深度与价值观演变”水平；
        * value_evolution_score：价值相关语汇“后期 - 早期”的变化。
    - 结合上述两个量，定义三档标签：

      1）成长型价值反思者（reflection_level = 2）：
         - 条件：
             reflection_index_norm >= 0.66 且 value_evolution_score > 0
         - 解释：
             * 反思频率与文本深度整体较高；
             * 后期反思中围绕元宇宙/学习价值的语汇更为丰富，
               符合 LEARNER-C 中所说“核心概念节点在共现网络中变得更中心/更紧密”。

      2）稳定深度反思者（reflection_level = 1）：
         - 条件：
             reflection_index_norm >= 0.5 且 value_evolution_score >= -0.05
         - 解释：
             * 反思频率与深度不低，但价值语汇变化不大或略有波动；
             * 可能说明其对元宇宙与学习价值的看法较为稳定。

      3）浅层或不稳定反思者（reflection_level = 0）：
         - 其他情况（包括反思样本很少或指数较低）。
         - 解释：
             * 反思条数较少、文本较短或多为描述性内容，
               价值相关语汇使用有限或变化方向不明确。

    - 特殊情况：
         若 reflection_count < 2：
             统一标记为“反思样本不足，暂难判断价值观演变”，level = 0。
    """
    label_counts = defaultdict(int)

    for key, m in reflection_metrics.items():
        reflection_count = m["reflection_count"]
        idx_norm = m["reflection_index_norm"]
        ve = m["value_evolution_score"]

        if reflection_count < 2:
            label = "反思样本不足型学习者（仅有零星反思，暂难判断深度与价值观演变）"
            level = 0
        else:
            if idx_norm >= 0.66 and ve > 0.0:
                label = "成长型价值反思者（反思频率与深度较高，且围绕元宇宙/学习价值的语汇在后期明显增强）"
                level = 2
            elif idx_norm >= 0.5 and ve >= -0.05:
                label = "稳定深度反思者（反思深度较高，但价值相关语汇变化有限或方向较稳定）"
                level = 1
            else:
                label = "浅层或不稳定反思者（反思频率较低或文本多为描述性，价值相关语汇使用有限/变化不稳定）"
                level = 0

        m["reflection_label"] = label
        m["reflection_level"] = level
        label_counts[label] += 1

    print("反思深度与价值观演变标签分布（按学习者-课程对统计）：")
    for label, cnt in label_counts.items():
        print(f"- {label}: {cnt} 条记录")

    # ---------- 10. （可选）写回数据库接口——默认不调用 ----------
    """
    如你在需求中所述：
    - 当前版本脚本只需完成“读取细粒度 xAPI → 计算反思深度与价值观演变 → 输出结果与人设对比”，
      不需要真正把结果写回数据库。
    - 上面定义的 save_reflection_analysis_to_db(db, reflection_results) 即为“写回接口”；
      若未来需要，可手动解除下面的注释。

    示例（默认注释掉）：
        save_reflection_analysis_to_db(db, reflection_metrics)
    """
    # 若需要写回数据库，请取消下一行注释：
    # save_reflection_analysis_to_db(db, reflection_metrics)

    # ---------- 11. 按学习者汇总 global_reflection_index 并与人设对比 ----------
    """
    验证思路：
    --------------------------------------------------
    1）行为侧指标：
       - 对每个学习者，把其在所有课程上的 reflection_index_norm 取平均，
         得到 global_reflection_index ∈ [0,1]，
         代表“基于细粒度反思行为推断的整体反思深度与价值演变水平”。

    2）人设侧指标：
       - LearnerProfile.global_profile.reflection_depth.score
         是在粗粒度统计基础上，通过画像推断得到的“反思深度”分数。

    3）对比目的：
       - 通过皮尔逊相关系数，检验“基于细粒度 xAPI 的反思分析”和
         “基于粗粒度统计的人设反思维度”在总体趋势上是否一致。
       - 若相关为正且具有一定强度，说明本脚本的细粒度分析在方向上与既有人设设计一致。

    与文献的关系：
    --------------------------------------------------
    - 行为侧指标的构造遵循 LEARNER-C 中“多次反思文本 + 核心概念共现 + 时间比较”的思想，
      但以价值关键词密度和覆盖度作为网络指标的轻量替代。
    - 与人设对比的步骤本身不引入新的理论假设，更偏向系统工程上的一致性检验。
    """
    learner_to_ref_vals = defaultdict(list)
    for (lrn_uid, crs_uid), m in reflection_metrics.items():
        learner_to_ref_vals[lrn_uid].append(m["reflection_index_norm"])

    learner_global_reflection = {}
    for lrn_uid, vals in learner_to_ref_vals.items():
        if vals:
            learner_global_reflection[lrn_uid] = sum(vals) / float(len(vals))

    xs = []  # 人设中的 reflection_depth.score
    ys = []  # 行为分析得到的 global_reflection_index

    for lrn_uid in sampled_learners:
        persona_score = persona_scores.get(lrn_uid)
        analyzed_ref = learner_global_reflection.get(lrn_uid)
        if persona_score is not None and analyzed_ref is not None:
            xs.append(float(persona_score))
            ys.append(float(analyzed_ref))

    if len(xs) >= 2:
        mean_x, std_x = compute_mean_std(xs)
        mean_y, std_y = compute_mean_std(ys)
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / float(len(xs))
        if std_x > 1e-6 and std_y > 1e-6:
            corr = cov / (std_x * std_y)
        else:
            corr = 0.0

        avg_global_ref = sum(ys) / float(len(ys))
        avg_persona_score = sum(xs) / float(len(xs))

        print("=========================================================")
        print("【反思深度与价值观演变维度：人设 vs 行为分析 全局对比】")
        print(f"- 采样学习者数量（具备人设）：{len(sampled_learners)}")
        print(f"- 实际参与对比的学习者数量：{len(xs)}")
        print(f"- 行为分析 global_reflection_index 平均值：{avg_global_ref:.3f}")
        print(f"- 人设 reflection_depth.score 平均值：{avg_persona_score:.3f}")
        print(f"- 皮尔逊相关系数：{corr:.3f}")
        print("  （相关系数用于粗略验证：细粒度 xAPI 反思分析是否与人设反思维度方向一致。）")
        print("=========================================================")
    else:
        print("参与对比的学习者样本太少，无法计算相关系数。")

    print("反思深度与价值观演变维度分析完成。")


if __name__ == "__main__":
    main()
