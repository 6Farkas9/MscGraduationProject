import torch
from sentence_transformers import SentenceTransformer

# 检查是否有GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 加载模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

# 示例文本
texts = [
    "The weather is beautiful today",
    "It's so sunny outside",
    "The dog is playing in the garden",
    "I love machine learning",
    "让我来试试中文"
]

# 生成嵌入向量
embeddings = model.encode(texts, convert_to_tensor=True)  # 返回PyTorch Tensor
print(f"嵌入向量形状: {embeddings.shape}")  # torch.Size([4, 384])

# 计算余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(embeddings.cpu().numpy())
print("余弦相似度矩阵:")
print(cosine_sim)