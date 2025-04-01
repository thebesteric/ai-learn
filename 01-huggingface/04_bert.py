from transformers import BertTokenizer, BertForSequenceClassification, pipeline, AutoModelForCausalLM, AutoTokenizer, \
    AutoModelForMaskedLM

# 加载模型和分词器（会先访问 huggingface，在访问模型路径，如果不存在，则下载，否则使用本地模型）
cache_dir = r"/Users/wangweijun/LLM/models/bert-base-chinese"
# model = BertForSequenceClassification.from_pretrained("bert-base-chinese", cache_dir=cache_dir)
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=cache_dir)

# 加载本地模型
model_path = f"{cache_dir}/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# 创建分类 pipeline
# 因为是 bert 文本分类模型，所以 task 使用 text-classification，表示是文本分类任务
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cpu")

# 进行文本分类
response = classifier(
    "今天天气不错"
)

# [{'label': 'LABEL_0', 'score': 0.6622371077537537}]
print(response)
"""
BertForSequenceClassification(
  (bert): BertModel(
    # 向量模型：文本转向量
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    # 编码模型：提取特征，将词向量的特征做理解和提取
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    # 池化层：保持模型的适用性
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  # 防止模型过拟合
  (dropout): Dropout(p=0.1, inplace=False)
  # 分类层，out_features 表示 2 分类
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
"""
print(model)