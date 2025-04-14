import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embed_model = SentenceTransformer(r"/Users/wangweijun/LLM/models/text2vec-base-chinese")

vec = embed_model.encode("这是一个测试句子。")
print(f"句子嵌入的纬度: {vec.shape}")

# 定义两个中文句子
sentence1 = "这是一个测试句子。"
sentence2 = "这是一个用于测试的句子。"

# 生成句子嵌入
embedding1 = embed_model.encode(sentence1)
embedding2 = embed_model.encode(sentence2)

# 计算余弦相似度
similarity = cosine_similarity([embedding1], [embedding2])[0][0]
print(f"余弦相似度: {similarity}")

# 或者直接计算点积（更快）
similarity = np.dot(embedding1, embedding2)
print(f"计算点积  : {similarity}")
