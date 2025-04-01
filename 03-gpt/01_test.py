import torch
from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", DEVICE)

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained(r"/Users/wangweijun/LLM/models/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")
tokenizer = BertTokenizer.from_pretrained(r"/Users/wangweijun/LLM/models/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")
print("model loaded", model)

# 使用 pipeline 调用模型
text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=DEVICE)

for i in range(3):
    # do_sample：表示是否进行随机采样，为 True 表示随机采样，为 False 时，每次都会生成相同的结果
    print(text_generator("这是很久以前的事情了，", max_length=100, do_sample=True, top_k=50, top_p=0.9))