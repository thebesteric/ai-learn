import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# 加载预训练的分词器，用于文本编码
tokenizer = AutoTokenizer.from_pretrained(
    r"/Users/wangweijun/LLM/models/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")
# 加载预训练的模型，用于语言模型任务
model = AutoModelForCausalLM.from_pretrained(
    r"/Users/wangweijun/LLM/models/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")


# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 加载训练的权重（中文古诗词）
model.load_state_dict(torch.load("./params/net.pt", map_location=device))

# 将模型加载到设备
model = model.to(device)

# 使用系统自带的 pipeline 工具生成内容
# 检查 CUDA 是否可用
device_index = 0 if torch.cuda.is_available() else -1
pipeline = TextGenerationPipeline(model, tokenizer, device=device_index)

# 生成内容
print(pipeline("天高", max_length=24, do_sample=True))