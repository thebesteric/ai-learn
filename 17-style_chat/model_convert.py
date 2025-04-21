import numpy as np
from sentence_transformers import SentenceTransformer, models

"""
如何判断 sentence 是否做了 normalize 归一化处理？
1. 检查模型结构：查看模型的最后一层是否是 Normalize 层。
[
  {
    "idx": 0,
    "name": "0",
    "path": "",
    "type": "sentence_transformers.models.Transformer"
  },
  {
    "idx": 1,
    "name": "1",
    "path": "1_Pooling",
    "type": "sentence_transformers.models.Pooling"
  },
  {
    "idx": 2,
    "name": "2",
    "path": "2_Normalize",
    "type": "sentence_transformers.models.Normalize"
  }
]
2. 检查模型权重：检查模型的权重是否在合理范围内。
text = "测试文本"
vec = model.encode(text)
print("修正后模长:", np.linalg.norm(vec)) # 应输出 ≈ 1.0
"""


def convert_sentence_with_pooling_and_normalize(model_path, new_path):
    # 原始模型
    bert = models.Transformer(model_path)

    # 添加 pooling 池化层
    # 主要用于对输入的特征图进行下采样（降低特征图的尺寸），同时保留重要的特征信息
    pooling = models.Pooling(bert.get_word_embedding_dimension(), pooling_mode='mean')

    # 添加缺失的 normalize 归一化层
    # 归一化层，用于对输入数据进行归一化处理，使数据具有特定的均值和标准差。归一化可以加速模型的训练过程，避免某些特征因数值范围过大而对模型产生过大影响，同时有助于缓解梯度消失或梯度爆炸问题
    normalize = models.Normalize()

    # 组合完整模型
    full_model = SentenceTransformer(modules=[bert, pooling, normalize])
    print(full_model)
    # 保存模型
    full_model.save(new_path)

    print(f"完整的 sentence 模型转换完成: {new_path}")


model_path = r"/Users/wangweijun/LLM/models/paraphrase-multilingual-MiniLM-L12-v2"
# 添加 normalize 归一化层
convert_sentence_with_pooling_and_normalize(model_path, model_path)

# 加载修复后的模型
model = SentenceTransformer(r"/Users/wangweijun/LLM/models/paraphrase-multilingual-MiniLM-L12-v2")

# 验证向量归一化
text = "测试文本"
vec = model.encode(text)
print("修正后模长:", np.linalg.norm(vec))  # 应输出 ≈ 1.0
