"""
Fragments 文档向量化处理脚本

设计思路：
1. 多维度信息融合
   - 概念语义维度：基于"涉及概念"字段，使用概念在知识图谱中的位置信息增强
   - 内容语义维度：基于"具体内容"和"内容类型"字段
   - 元数据维度：基于Type、位置等信息
   - 结构维度：基于学习单元所属的课程信息

2. 分层向量计算
   - 概念增强向量：专门用于概念匹配，考虑知识点关系和主题层级
   - 内容语义向量：用于内容语义匹配
   - 综合向量：前两者的加权组合，用于通用检索

3. 知识图谱增强策略
   - 扩展概念：基于数据库中的知识点关系，为每个概念找到相关概念
   - 主题层级：利用topic-concept关系，为概念添加主题上下文
   - 课程上下文：利用course-unit关系，为资源添加课程领域信息

4. 嵌入模型选择：使用 all-MiniLM-L6-v2
   - 该模型提供384维向量，在质量和效率之间取得良好平衡
   - 与系统其他部分保持一致，确保向量空间兼容性
   - 支持中英文混合文本，适合MOOC内容

5. 实用性考虑
   - 支持后续的多种检索场景：概念精确匹配、语义相似匹配、资源类型匹配
   - 向量可解释性：保留原始文本信息用于调试
   - 性能优化：批量处理、缓存机制、模型单例

具体计算步骤：
1. 为每个Fragment文档构建增强文本描述
2. 使用all-MiniLM-L6-v2模型编码为384维向量
3. 计算多种类型的向量以适应不同匹配需求
4. 存储原始数据和新计算的向量到新集合
"""

import pymongo
import pymysql
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from tqdm import tqdm
import time
import threading
from functools import lru_cache

# 尝试导入sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers 库未安装，将使用模拟嵌入")
    print("请安装: pip install sentence-transformers")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingModel:
    """嵌入模型封装类，使用 all-MiniLM-L6-v2"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmbeddingModel, cls).__new__(cls)
                cls._instance._initialize_model()
            return cls._instance
    
    def _initialize_model(self):
        """初始化模型"""
        self.model_name = 'all-MiniLM-L6-v2'
        self.dimension = 384  # all-MiniLM-L6-v2的维度
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"正在加载嵌入模型: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                # 测试模型
                test_embedding = self.model.encode("test")
                logger.info(f"模型加载成功，维度: {len(test_embedding)}")
                self._dummy_mode = False
            except Exception as e:
                logger.error(f"加载模型失败: {e}，使用模拟模式")
                self._dummy_mode = True
                self.model = None
        else:
            logger.warning("sentence-transformers不可用，使用模拟模式")
            self._dummy_mode = True
            self.model = None
    
    def encode(self, text: str) -> List[float]:
        """编码文本为向量"""
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        if self._dummy_mode:
            return self._dummy_encode(text)
        
        try:
            # 使用模型编码
            embedding = self.model.encode(text)
            # 转换为列表并归一化
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"编码失败: {e}，使用模拟编码")
            return self._dummy_encode(text)
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本"""
        if not texts:
            return []
        
        if self._dummy_mode:
            return [self._dummy_encode(text) for text in texts]
        
        try:
            embeddings = self.model.encode(texts)
            # 归一化每个向量
            embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        except Exception as e:
            logger.error(f"批量编码失败: {e}")
            return [self._dummy_encode(text) for text in texts]
    
    def _dummy_encode(self, text: str) -> List[float]:
        """模拟编码（备用）"""
        # 基于文本内容的确定性伪随机向量
        np.random.seed(abs(hash(text)) % (2**32))
        vector = np.random.randn(self.dimension)
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


class FragmentEmbeddingGenerator:
    def __init__(self, 
                 mongo_uri: str = "mongodb://localhost:27017/",
                 mysql_config: Dict = None):
        """
        初始化向量生成器
        
        Args:
            mongo_uri: MongoDB连接URI
            mysql_config: MySQL连接配置
        """
        # MongoDB连接
        self.mongo_client = pymongo.MongoClient(mongo_uri)
        self.mongo_db = self.mongo_client["MLS"]
        self.fragments_col = self.mongo_db["Fragments"]
        self.embedding_col = self.mongo_db["Fragment_Embedding"]
        
        # MySQL连接
        if mysql_config is None:
            mysql_config = {
                'host': 'localhost',
                'user': 'root',
                'password': '123456',
                'database': 'mls',
                'charset': 'utf8mb4'
            }
        self.mysql_config = mysql_config
        self.mysql_conn = pymysql.connect(**mysql_config)
        
        # 初始化嵌入模型
        self.embedding_model = EmbeddingModel()
        self.embedding_dim = self.embedding_model.dimension
        
        # 缓存数据结构
        self.concept_cache = {}  # 知识点缓存
        self.topic_cache = {}    # 主题缓存
        self.unit_cache = {}     # 学习单元缓存
        self.course_cache = {}   # 课程缓存
        self.concept_relations = {}  # 知识点关系缓存
        
        # 加载缓存数据
        self._load_cache_data()
        
        logger.info(f"FragmentEmbeddingGenerator初始化完成，使用模型: {self.embedding_model.model_name}")
        logger.info(f"嵌入维度: {self.embedding_dim}")
    
    def _load_cache_data(self):
        """从MySQL加载缓存数据以提高性能"""
        logger.info("开始加载MySQL缓存数据...")
        
        try:
            # 加载知识点
            with self.mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT uid, name, explanation FROM Concepts")
                concepts = cursor.fetchall()
                for concept in concepts:
                    self.concept_cache[concept['uid']] = concept
                    # 同时按名称缓存，用于名称到UID的映射
                    self.concept_cache[concept['name']] = concept['uid']
            
            # 加载主题
            with self.mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT uid, name, explanation FROM Topics")
                topics = cursor.fetchall()
                for topic in topics:
                    self.topic_cache[topic['uid']] = topic
            
            # 加载学习单元
            with self.mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT uid, oid, name, type FROM Units")
                units = cursor.fetchall()
                for unit in units:
                    self.unit_cache[unit['oid']] = unit  # 使用OID作为key
            
            # 加载课程
            with self.mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT uid, oid, name, about FROM Courses")
                courses = cursor.fetchall()
                for course in courses:
                    self.course_cache[course['oid']] = course
            
            # 加载知识点-主题关系
            with self.mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT tpc_uid, cpt_uid FROM Topic_Concept")
                relations = cursor.fetchall()
                for rel in relations:
                    cpt_uid = rel['cpt_uid']
                    if cpt_uid not in self.concept_relations:
                        self.concept_relations[cpt_uid] = {'topics': [], 'prerequisites': [], 'successors': []}
                    self.concept_relations[cpt_uid]['topics'].append(rel['tpc_uid'])
            
            # 加载主题层级关系（用于扩展主题上下文）
            with self.mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT pnt_uid, son_uid FROM Topic_Topic")
                topic_relations = cursor.fetchall()
                self.topic_hierarchy = {}
                for rel in topic_relations:
                    if rel['son_uid'] not in self.topic_hierarchy:
                        self.topic_hierarchy[rel['son_uid']] = []
                    self.topic_hierarchy[rel['son_uid']].append(rel['pnt_uid'])
            
            # 加载课程-单元关系
            with self.mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("SELECT crs_uid, unt_uid FROM Course_Unit")
                course_unit_relations = cursor.fetchall()
                self.course_unit_map = {}
                for rel in course_unit_relations:
                    unit_uid = rel['unt_uid']
                    if unit_uid not in self.course_unit_map:
                        self.course_unit_map[unit_uid] = []
                    self.course_unit_map[unit_uid].append(rel['crs_uid'])
            
            logger.info(f"缓存加载完成: {len(self.concept_cache)//2}个知识点, {len(self.topic_cache)}个主题, "
                       f"{len(self.unit_cache)}个学习单元, {len(self.course_cache)}个课程")
        
        except Exception as e:
            logger.error(f"加载缓存数据失败: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def _get_concept_uid_by_name(self, concept_name: str) -> Optional[str]:
        """根据知识点名称获取UID（带缓存）"""
        return self.concept_cache.get(concept_name)
    
    @lru_cache(maxsize=1000)
    def _get_topic_hierarchy(self, topic_uid: str, max_depth: int = 3) -> tuple:
        """获取主题的层级路径（从当前主题到根主题）（带缓存）"""
        hierarchy = [topic_uid]
        current = topic_uid
        depth = 0
        
        while depth < max_depth:
            parents = self.topic_hierarchy.get(current, [])
            if not parents:
                break
            # 取第一个父主题（假设单父层级）
            parent = parents[0]
            if parent in hierarchy:  # 防止循环
                break
            hierarchy.append(parent)
            current = parent
            depth += 1
        
        return tuple(hierarchy)
    
    def _expand_concept_with_context(self, concept_name: str) -> str:
        """
        为概念扩展上下文信息
        
        包括：
        1. 概念的解释
        2. 所属主题
        3. 主题层级
        4. 相关概念（通过主题关联）
        """
        concept_uid = self._get_concept_uid_by_name(concept_name)
        if not concept_uid:
            return concept_name  # 无法找到对应的概念
        
        concept_info = self.concept_cache.get(concept_uid, {})
        explanation = concept_info.get('explanation', '')
        
        # 获取主题信息
        topics_text = ""
        if concept_uid in self.concept_relations:
            topic_uids = self.concept_relations[concept_uid].get('topics', [])
            for topic_uid in topic_uids[:2]:  # 取前两个主题
                topic_info = self.topic_cache.get(topic_uid, {})
                if topic_info:
                    topics_text += f"，属于{topic_info.get('name', '')}领域"
                    
                    # 添加主题层级
                    hierarchy = self._get_topic_hierarchy(topic_uid)
                    if len(hierarchy) > 1:
                        parent_topics = []
                        for uid in hierarchy[1:]:
                            if uid in self.topic_cache:
                                parent_info = self.topic_cache[uid]
                                parent_topics.append(parent_info.get('name', ''))
                        if parent_topics:
                            topics_text += f"（属于{'→'.join(parent_topics)}）"
        
        # 构建增强文本（限制总长度）
        enhanced_text = f"{concept_name}"
        
        # 添加解释（限制长度）
        if explanation:
            # 截取前100个字符，确保不截断中文字符
            truncated_explanation = explanation[:100]
            if len(explanation) > 100:
                truncated_explanation += "..."
            enhanced_text += f"，解释：{truncated_explanation}"
        
        # 添加主题信息
        if topics_text:
            enhanced_text += topics_text
        
        return enhanced_text
    
    def _get_unit_context(self, unit_oid: str) -> Dict[str, Any]:
        """获取学习单元的上下文信息（所属课程等）"""
        unit_info = self.unit_cache.get(unit_oid, {})
        if not unit_info:
            return {}
        
        context = {
            'unit_name': unit_info.get('name', ''),
            'unit_type': unit_info.get('type', ''),
            'courses': []
        }
        
        # 获取所属课程
        unit_uid = unit_info.get('uid')
        if unit_uid and unit_uid in self.course_unit_map:
            course_uids = self.course_unit_map[unit_uid]
            for course_uid in course_uids[:2]:  # 最多取两个课程
                # 需要从OID反向查找课程
                for course in self.course_cache.values():
                    if course.get('uid') == course_uid:
                        course_info = {
                            'name': course.get('name', ''),
                            'about': course.get('about', '')[:200] if course.get('about') else ''
                        }
                        if course_info['name']:  # 只添加有名称的课程
                            context['courses'].append(course_info)
                        break
        
        return context
    
    def _generate_concept_enhanced_text(self, fragment: Dict) -> str:
        """
        生成概念增强文本
        
        用于创建概念增强向量，重点关注知识点及其关系
        """
        concepts = fragment.get('涉及概念', [])
        if not concepts:
            # 如果没有概念，使用内容类型作为基础
            content_type = fragment.get('内容类型', '')
            specific_content = fragment.get('具体内容', '')[:100]
            return f"教学片段：{specific_content}，领域：{content_type}"
        
        enhanced_concepts = []
        for concept in concepts[:5]:  # 限制最多5个概念
            enhanced = self._expand_concept_with_context(concept)
            enhanced_concepts.append(enhanced)
        
        # 获取学习单元上下文
        unit_oid = fragment.get('OID', '')
        unit_context = self._get_unit_context(unit_oid)
        
        # 构建概念增强文本
        text_parts = []
        
        # 1. 核心概念部分
        text_parts.append(f"核心知识点：{'；'.join(enhanced_concepts)}")
        
        # 2. 内容类型和具体内容
        content_type = fragment.get('内容类型', '')
        specific_content = fragment.get('具体内容', '')[:150]  # 限制长度
        if specific_content:
            text_parts.append(f"内容片段：{specific_content}")
        elif content_type:
            text_parts.append(f"教学领域：{content_type}")
        
        # 3. 资源类型信息
        resource_type = fragment.get('Type', '')
        if resource_type:
            type_mapping = {
                'video': '视频教学资源',
                'ar': '增强现实教学资源',
                'vr': '虚拟现实教学资源',
                'interact': '互动教学资源',
                'cooperate': '协作学习资源'
            }
            text_parts.append(f"教学形式：{type_mapping.get(resource_type, resource_type)}")
        
        # 4. 课程上下文
        if unit_context.get('courses'):
            course_names = [c['name'] for c in unit_context['courses'] if c.get('name')]
            if course_names:
                text_parts.append(f"所属课程：{'、'.join(course_names[:2])}")  # 最多2个课程
        
        return "。".join(text_parts)
    
    def _generate_content_semantic_text(self, fragment: Dict) -> str:
        """
        生成内容语义文本
        
        用于创建内容语义向量，重点关注具体内容和上下文
        """
        text_parts = []
        
        # 1. 具体内容（主要部分）
        specific_content = fragment.get('具体内容', '')
        if specific_content:
            text_parts.append(f"教学内容：{specific_content}")
        
        # 2. 内容类型
        content_type = fragment.get('内容类型', '')
        if content_type:
            text_parts.append(f"学科领域：{content_type}")
        
        # 3. 涉及概念（简化版）
        concepts = fragment.get('涉及概念', [])
        if concepts:
            text_parts.append(f"知识点：{'、'.join(concepts[:3])}")
        
        # 4. 资源类型
        resource_type = fragment.get('Type', '')
        if resource_type:
            text_parts.append(f"资源格式：{resource_type}")
        
        # 5. 位置信息（可选）
        position = fragment.get('位置', '')
        if position and '秒' in position:
            text_parts.append(f"时间位置：{position}")
        
        return "。".join(text_parts)
    
    def _generate_metadata_text(self, fragment: Dict) -> str:
        """
        生成元数据文本
        
        用于创建元数据向量，重点关注资源属性和结构信息
        """
        text_parts = []
        
        # 资源类型
        resource_type = fragment.get('Type', '')
        if resource_type:
            type_details = {
                'video': '视频教学，适合讲解和演示',
                'ar': '增强现实教学，适合沉浸式学习体验',
                'vr': '虚拟现实教学，适合实践操作训练',
                'interact': '互动教学，适合练习和测试',
                'cooperate': '协作学习，适合小组讨论和合作'
            }
            text_parts.append(f"教学形式：{type_details.get(resource_type, resource_type)}")
        
        # 内容类型
        content_type = fragment.get('内容类型', '')
        if content_type:
            text_parts.append(f"学科分类：{content_type}")
        
        # 学习单元信息
        unit_oid = fragment.get('OID', '')
        unit_context = self._get_unit_context(unit_oid)
        if unit_context.get('unit_type'):
            text_parts.append(f"单元类型：{unit_context['unit_type']}")
        
        # 时长信息
        position = fragment.get('位置', '')
        if position:
            # 尝试解析时长
            try:
                if '秒' in position and '-' in position:
                    start_end = position.replace('秒', '').split('-')
                    if len(start_end) == 2:
                        start = float(start_end[0])
                        end = float(start_end[1])
                        if end > start:
                            duration = end - start
                            if duration <= 10:
                                text_parts.append("短片段，适合微学习")
                            elif duration <= 60:
                                text_parts.append("中等片段，适合知识点讲解")
                            else:
                                text_parts.append("长片段，适合深度讲解")
            except:
                pass
        
        # 概念数量
        concepts = fragment.get('涉及概念', [])
        if concepts:
            concept_count = len(concepts)
            if concept_count == 1:
                text_parts.append("聚焦单个知识点")
            elif concept_count <= 3:
                text_parts.append("覆盖少量相关知识点")
            else:
                text_parts.append("覆盖多个知识点")
        
        return "。".join(text_parts)
    
    def generate_embeddings(self, fragment: Dict) -> Dict[str, Any]:
        """
        为单个Fragment文档生成多种embedding
        
        Returns:
            包含三种向量和生成文本的字典
        """
        try:
            # 生成三种增强文本
            concept_text = self._generate_concept_enhanced_text(fragment)
            content_text = self._generate_content_semantic_text(fragment)
            metadata_text = self._generate_metadata_text(fragment)
            
            # 记录生成的文本（用于调试）
            logger.debug(f"概念文本长度: {len(concept_text)}")
            logger.debug(f"内容文本长度: {len(content_text)}")
            logger.debug(f"元数据文本长度: {len(metadata_text)}")
            
            # 批量编码以提高效率
            texts_to_encode = [concept_text, content_text, metadata_text]
            embeddings_list = self.embedding_model.encode_batch(texts_to_encode)
            
            # 确保获取到三个向量
            if len(embeddings_list) == 3:
                concept_vector, content_vector, metadata_vector = embeddings_list
            else:
                # 备用方案：分别编码
                concept_vector = self.embedding_model.encode(concept_text)
                content_vector = self.embedding_model.encode(content_text)
                metadata_vector = self.embedding_model.encode(metadata_text)
            
            # 生成综合向量（加权组合）
            # 权重可以根据需求调整，这里给概念更高的权重
            if concept_vector and content_vector:
                concept_array = np.array(concept_vector)
                content_array = np.array(content_vector)
                
                # 加权组合：概念60%，内容40%
                combined_array = concept_array * 0.6 + content_array * 0.4
                
                # 归一化
                norm = np.linalg.norm(combined_array)
                if norm > 0:
                    combined_array = combined_array / norm
                
                combined_vector = combined_array.tolist()
            else:
                combined_vector = concept_vector or content_vector or metadata_vector
            
            embeddings = {
                'concept_enhanced_embedding': concept_vector,
                'content_semantic_embedding': content_vector,
                'metadata_embedding': metadata_vector,
                'combined_embedding': combined_vector,
                'generated_texts': {
                    'concept_enhanced': concept_text,
                    'content_semantic': content_text,
                    'metadata': metadata_text
                }
            }
            
            return embeddings
        
        except Exception as e:
            logger.error(f"生成embedding失败: {e}")
            # 返回标准零向量
            empty_vector = [0.0] * self.embedding_dim
            return {
                'concept_enhanced_embedding': empty_vector,
                'content_semantic_embedding': empty_vector,
                'metadata_embedding': empty_vector,
                'combined_embedding': empty_vector,
                'generated_texts': {
                    'concept_enhanced': '',
                    'content_semantic': '',
                    'metadata': ''
                }
            }
    
    def process_single_fragment(self, fragment: Dict) -> Dict[str, Any]:
        """
        处理单个Fragment文档
        
        保留原始数据（去除错误的嵌入字段），添加新的嵌入向量
        """
        try:
            # 复制原始数据（排除错误的嵌入字段）
            processed_doc = {}
            for k, v in fragment.items():
                if k != '嵌入表达':  # 排除错误的嵌入字段
                    processed_doc[k] = v
            
            # 添加系统字段
            processed_doc['processed_at'] = datetime.now().isoformat()
            processed_doc['embedding_version'] = '2.0'  # 版本号，区别于之前的错误版本
            processed_doc['embedding_model'] = self.embedding_model.model_name
            processed_doc['embedding_dimension'] = self.embedding_dim
            
            # 生成嵌入向量
            embeddings = self.generate_embeddings(fragment)
            processed_doc.update(embeddings)
            
            return processed_doc
        
        except Exception as e:
            logger.error(f"处理文档失败: {e}")
            # 返回一个基本的处理文档
            basic_doc = {
                '_id': fragment.get('_id'),
                'UID': fragment.get('UID'),
                'OID': fragment.get('OID'),
                'Type': fragment.get('Type'),
                'processed_at': datetime.now().isoformat(),
                'error': str(e)
            }
            return basic_doc
    
    def batch_process(self, batch_size: int = 100, skip_existing: bool = True, 
                      max_workers: int = 1) -> Dict[str, int]:
        """
        批量处理所有Fragment文档
        
        Args:
            batch_size: 批处理大小
            skip_existing: 是否跳过已处理的文档（根据OID判断）
            max_workers: 最大工作线程数（目前为1，可扩展）
            
        Returns:
            处理统计信息
        """
        try:
            # 获取所有Fragment文档
            total_count = self.fragments_col.count_documents({})
            logger.info(f"开始处理 {total_count} 个Fragment文档")
            
            # 如果跳过已处理的，先获取已处理的OID
            processed_oids = set()
            if skip_existing:
                try:
                    processed_oids = set(self.embedding_col.distinct('OID'))
                    logger.info(f"已找到 {len(processed_oids)} 个已处理文档")
                except Exception as e:
                    logger.warning(f"获取已处理文档失败: {e}，将处理所有文档")
            
            # 分批处理
            cursor = self.fragments_col.find({})
            processed = 0
            skipped = 0
            errors = 0
            
            with tqdm(total=total_count, desc="处理进度") as pbar:
                batch_docs = []
                
                for fragment in cursor:
                    pbar.update(1)
                    
                    # 检查是否已处理
                    fragment_oid = fragment.get('OID')
                    if skip_existing and fragment_oid in processed_oids:
                        skipped += 1
                        continue
                    
                    try:
                        processed_doc = self.process_single_fragment(fragment)
                        
                        # 检查是否有错误字段
                        if 'error' not in processed_doc:
                            batch_docs.append(processed_doc)
                            processed += 1
                        else:
                            errors += 1
                            logger.warning(f"文档处理异常 (OID: {fragment_oid}): {processed_doc.get('error')}")
                        
                        # 批量插入
                        if len(batch_docs) >= batch_size:
                            self._insert_batch(batch_docs)
                            batch_docs = []
                            pbar.set_postfix({
                                '已处理': processed, 
                                '跳过': skipped, 
                                '错误': errors,
                                '批大小': batch_size
                            })
                    
                    except Exception as e:
                        errors += 1
                        logger.error(f"处理文档失败 (OID: {fragment_oid}): {e}")
                        continue
                
                # 插入最后一批
                if batch_docs:
                    self._insert_batch(batch_docs)
            
            logger.info(f"处理完成: 成功 {processed}, 跳过 {skipped}, 错误 {errors}")
            
            # 创建索引
            self._create_indexes()
            
            return {
                'total': total_count,
                'processed': processed,
                'skipped': skipped,
                'errors': errors
            }
        
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            raise
    
    def _insert_batch(self, batch_docs: List[Dict]):
        """批量插入文档"""
        if not batch_docs:
            return
        
        try:
            # 使用insert_many提高性能
            result = self.embedding_col.insert_many(batch_docs, ordered=False)
            logger.debug(f"成功插入 {len(result.inserted_ids)} 个文档")
        except pymongo.errors.BulkWriteError as e:
            # 处理批量写入错误
            logger.warning(f"批量插入部分失败: {e.details}")
            # 尝试单个插入剩余文档
            for doc in batch_docs:
                try:
                    self.embedding_col.insert_one(doc)
                except Exception as e2:
                    logger.error(f"单个插入失败: {e2}")
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            # 备用：逐个插入
            success_count = 0
            for doc in batch_docs:
                try:
                    self.embedding_col.insert_one(doc)
                    success_count += 1
                except Exception as e2:
                    logger.error(f"单个插入失败: {e2}")
            logger.info(f"备用插入完成: {success_count}/{len(batch_docs)}")
    
    def _create_indexes(self):
        """创建集合索引"""
        try:
            # 创建复合索引
            indexes_info = []
            
            # 在OID上创建索引（最重要的查询字段）
            self.embedding_col.create_index([("OID", pymongo.ASCENDING)], name="oid_idx")
            indexes_info.append("OID索引")
            
            # 在Type上创建索引
            self.embedding_col.create_index([("Type", pymongo.ASCENDING)], name="type_idx")
            indexes_info.append("Type索引")
            
            # 在内容类型上创建索引
            self.embedding_col.create_index([("内容类型", pymongo.ASCENDING)], name="content_type_idx")
            indexes_info.append("内容类型索引")
            
            # 在UID上创建索引
            self.embedding_col.create_index([("UID", pymongo.ASCENDING)], name="uid_idx")
            indexes_info.append("UID索引")
            
            logger.info(f"索引创建完成: {', '.join(indexes_info)}")
            
            # 获取索引信息
            indexes = list(self.embedding_col.list_indexes())
            logger.debug(f"集合现有索引数量: {len(indexes)}")
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
    
    def verify_embeddings(self, sample_size: int = 5) -> bool:
        """
        验证嵌入生成结果
        
        Args:
            sample_size: 抽样检查的文档数量
            
        Returns:
            验证是否通过
        """
        logger.info("开始验证嵌入生成结果...")
        
        try:
            # 随机抽取几个文档检查
            pipeline = [{'$sample': {'size': sample_size}}]
            sample_docs = list(self.embedding_col.aggregate(pipeline))
            
            if not sample_docs:
                logger.warning("没有找到文档进行验证")
                return False
            
            all_valid = True
            for i, doc in enumerate(sample_docs):
                logger.info(f"\n--- 验证样本 {i+1} ---")
                logger.info(f"OID: {doc.get('OID')}")
                logger.info(f"Type: {doc.get('Type')}")
                logger.info(f"内容类型: {doc.get('内容类型')}")
                
                # 检查必要的字段
                required_fields = ['concept_enhanced_embedding', 'content_semantic_embedding', 
                                  'combined_embedding', 'processed_at']
                
                for field in required_fields:
                    if field not in doc:
                        logger.error(f"缺失字段: {field}")
                        all_valid = False
                    else:
                        if 'embedding' in field:
                            vector = doc[field]
                            if not isinstance(vector, list):
                                logger.error(f"字段 {field} 不是列表类型")
                                all_valid = False
                            elif len(vector) != self.embedding_dim:
                                logger.error(f"字段 {field} 维度错误: {len(vector)} != {self.embedding_dim}")
                                all_valid = False
                
                # 显示向量信息
                if 'concept_enhanced_embedding' in doc:
                    vector = doc['concept_enhanced_embedding']
                    if isinstance(vector, list) and vector:
                        norm = np.linalg.norm(vector)
                        logger.info(f"概念向量范数: {norm:.4f}")
                        logger.info(f"概念向量前5维: {vector[:5]}")
            
            if all_valid:
                logger.info("验证通过！")
            else:
                logger.error("验证失败！")
            
            return all_valid
        
        except Exception as e:
            logger.error(f"验证过程中出错: {e}")
            return False
    
    def close(self):
        """关闭连接"""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
            logger.debug("MongoDB连接已关闭")
        
        if hasattr(self, 'mysql_conn'):
            self.mysql_conn.close()
            logger.debug("MySQL连接已关闭")
        
        logger.info("所有连接已关闭")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("Fragment文档向量化处理开始")
    logger.info("=" * 60)
    
    generator = None
    
    try:
        # 初始化生成器
        start_init = time.time()
        generator = FragmentEmbeddingGenerator()
        init_time = time.time() - start_init
        logger.info(f"初始化完成，耗时: {init_time:.2f}秒")
        
        # 执行批量处理
        start_process = time.time()
        logger.info("开始批量处理文档...")
        
        # 可以根据数据量调整batch_size
        # 较小的batch_size适合调试，较大的batch_size适合生产环境
        batch_size = 50  # all-MiniLM-L6-v2 处理速度较快，可以适当增大
        
        results = generator.batch_process(
            batch_size=batch_size, 
            skip_existing=True,
            max_workers=1
        )
        
        process_time = time.time() - start_process
        total_time = time.time() - start_init
        
        # 输出统计信息
        logger.info("=" * 60)
        logger.info("处理统计信息:")
        logger.info(f"  总文档数: {results['total']}")
        logger.info(f"  成功处理: {results['processed']}")
        logger.info(f"  跳过已处理: {results['skipped']}")
        logger.info(f"  处理错误: {results['errors']}")
        logger.info(f"  初始化耗时: {init_time:.2f}秒")
        logger.info(f"  处理耗时: {process_time:.2f}秒")
        logger.info(f"  总耗时: {total_time:.2f}秒")
        
        if results['processed'] > 0:
            docs_per_second = results['processed'] / process_time
            logger.info(f"  处理速度: {docs_per_second:.2f} 文档/秒")
        
        logger.info("=" * 60)
        
        # 验证结果
        logger.info("开始验证处理结果...")
        verification_passed = generator.verify_embeddings(sample_size=3)
        
        if verification_passed:
            logger.info("✓ 处理结果验证通过")
        else:
            logger.warning("⚠ 处理结果验证有问题，请检查日志")
        
        # 输出集合信息
        total_docs = generator.embedding_col.count_documents({})
        logger.info(f"Fragment_Embedding集合现有文档数: {total_docs}")
        
        # 示例：查看一个处理后的文档结构
        sample_doc = generator.embedding_col.find_one({})
        if sample_doc:
            logger.info("\n示例文档字段:")
            for key in sorted(sample_doc.keys()):
                if 'embedding' in key:
                    vector = sample_doc[key]
                    if isinstance(vector, list):
                        logger.info(f"  {key}: 向量[{len(vector)}维]")
                    else:
                        logger.info(f"  {key}: {type(vector)}")
                elif key not in ['_id', 'generated_texts']:
                    value = sample_doc[key]
                    if isinstance(value, str) and len(value) > 50:
                        logger.info(f"  {key}: {value[:50]}...")
                    else:
                        logger.info(f"  {key}: {value}")
        
        logger.info("=" * 60)
        logger.info("处理完成！")
        
    except KeyboardInterrupt:
        logger.info("\n用户中断处理")
    except Exception as e:
        logger.error(f"处理过程出错: {e}", exc_info=True)
    finally:
        if generator:
            generator.close()


if __name__ == "__main__":
    main()