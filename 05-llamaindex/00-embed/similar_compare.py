import torch
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# 加载 BGE 中文嵌入模型
# model_name = "/Users/wangweijun/LLM/models/paraphrase-multilingual-MiniLM-L12-v2"
model_name = "/Users/wangweijun/LLM/models/text2vec-base-chinese-sentence"
embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    device=str(device),
    normalize=True,  # 归一化向量，方便计算余弦相似度
)

# 嵌入文档
documents = ["忘记密码如何处理？", "用户账号被锁定"]
doc_embeddings = [embed_model.get_text_embedding(doc) for doc in documents]

# 嵌入查询并计算相似度
query = "密码重置流程"
query_embedding = embed_model.get_text_embedding(query)

# 计算余弦相似度（因为 normalize=True，点积就是余弦相似度）
similarity = np.dot(query_embedding, doc_embeddings[1])

# paraphrase-multilingual-MiniLM-L12-v2：忘记密码如何处理？：0.6902，用户账号被锁定：0.5717
# text2vec-base-chinese-sentence：       忘记密码如何处理？：0.8986，用户账号被锁定：0.8464
print(f"相似度：{similarity:.4f}")
