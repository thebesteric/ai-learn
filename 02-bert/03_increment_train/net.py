import torch
from transformers import BertModel

"""
增量微调设置
"""


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", DEVICE)

# 加载预训练模型，模型位置一定要指定到含有 config.json 的目录
cache_dir = r"/Users/wangweijun/LLM/models/bert-base-chinese"
model_path = f"{cache_dir}/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
bert_pretrained_model = BertModel.from_pretrained(model_path).to(DEVICE)
print(bert_pretrained_model)
"""
(embeddings): BertEmbeddings(
    # word_embeddings：表示词变成向量的模型
    # 21128：表示字典的大小
    # 768：表示纬度有 768 个纬度，也就是一个词会被编码成 768 个数字
    (word_embeddings): Embedding(21128, 768, padding_idx=0)
    # position_embeddings： 将词的位置进行编码，解决全连接神经网络无法处理序列数据的问题
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
)

(pooler): BertPooler(
    # 模型会输出 768 个纬度的向量，但是我们只需要 2 个纬度的向量，所以需要将 768 个纬度的向量转换为 2 个纬度的向量
    # 所以我们就要设计一个下游任务，将 768 个纬度的向量转换为 2 个纬度的向量
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
)
"""

# 定义下游任务（增量模型）
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个全连接层（Fully Connected Layer），实现二分类任务，将 768 个纬度的向量转换为 2 个纬度的向量
        # 这里可以根据数据集的纬度，自己修改
        self.fc_layer = torch.nn.Linear(768, 2)

    # 使用模型处理数据，执行前向计算
    def forward(self, input_ids, attention_mask, token_type_ids):
        # 冻结 Bert 模型的参数，让其不参与训练
        with torch.no_grad():
            # 目前为止 Transformer 模型都是沿用了 RNN 数据的模式，数据是 NSV 模式，N 表示批次，S 表示序列长度，V 表示数据特征
            out = bert_pretrained_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 增量模型参与训练，取 NSV 的 V，即数据特征
        out = self.fc_layer(out.last_hidden_state[:, 0])
        return out