from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

"""
本地调用模型
包含 config.json 的目录就是模型具体位置的目录
"""

# 模型路径
model_dir = r"/Users/wangweijun/LLM/models/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"

# 加载模型和分词器（纯离线加载）
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 使用 pipeline 调用模型
# gpt-2 是一个续写模型（文本生成模型）
# task：text-generation 表示是文本生成任务，因为 GPT-2 就是一个续写模型
# device：cup 表示使用 CPU，cuda 表示使用 GPU
generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device="cpu")

# llm = HuggingFacePipeline(pipeline=generator)
# response = llm.invoke("这是很久以前的事情了")

# 生成文本
response = generator(
    # 生成后续文本的种子文本，会根据种子文本生成后续文本
    "这是很久以前的事情了，",
    # 表示最大长度
    max_length=50,
    # 表示返回几段独立的续写文本
    num_return_sequences=2,
    # 表示是否切断文本，以适应文本最大的输入长度。对应 input 的最大长度
    # 可以在 tokenizer_config.json 中的 model_max_length 查看最大输入长度
    truncation=True,
    # 控制模型在每一步生成时，仅从概率最高的 k 个词中选择下一个词
    # 50 表示从 21128 个词中，概率最高的前 50 个词中选择下一个词，当词的范围圈定下来后，temperature 从这 50 个词中选择下一词
    # top_k 越大，结果就会变差
    top_k=50,
    # 称为核采样数，进一步控制模型生成词的选择范围，它会从 top_k 筛选出来的词，进行概率的累积，当概率达到 p 的词汇，该模型会从这组词汇中进行采样。
    # 0.9 表示模型会在累积概率达到 90% 的词汇中取选择下一个词，进一步增加生成文本的质量
    # 这里建议 top_p 值高一点
    top_p=0.9,
    # 温度系数：控制模型生成文本的随机性。
    # 就是从 vocab.txt 里的 21128 个词中随机去取，值越小，生成的文本越保守（选择概率较高的值），值越大，生成的文本越多样（倾向选择更多不同的词）。
    # 当设置很低很低的时候，可能生成的文本几乎都是一样的。
    # 0.7 是一个较为常见的值，
    temperature=0.7,
    # 控制生成的文本是否清理分词时引入的空格
    clean_up_tokenization_spaces=True,
)
print(response)
