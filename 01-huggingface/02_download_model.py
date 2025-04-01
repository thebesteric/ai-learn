from transformers import AutoTokenizer, AutoModelForCausalLM

"""
将模型下载到本地调用
"""

# GPT2 模型
model_name = "uer/gpt2-chinese-cluecorpussmall"
cache_dir = "/Users/wangweijun/LLM/models/uer/gpt2-chinese-cluecorpussmall"

# BERT 模型
# model_name = "bret-base-chinese"
# cache_dir = "/Users/wangweijun/LLM/models/bret-base-chinese"

# 下载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir
)
print("模型下载完成")

# 下载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)
print("模型分词器下载完成")
