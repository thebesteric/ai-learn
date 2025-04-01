import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
使用 transformers 加载 Qwen 模型
利用 AutoTokenizer, AutoModelForCausalLM
"""

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("DEVICE: ", DEVICE)

# 定义模型路径
model_dir = "/Users/wangweijun/LLM/models/Qwen/Qwen2___5-0___5B-Instruct"

# torch_dtype 设置为 auto，模型会自动选择合适的数据类型，否则需要参考 config.json 的 "torch_dtype": "bfloat16",
# trust_remote_code=True 是否信任从远程仓库下载并执行的自定义代码
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype="auto", trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)

# 调用模型
# 定义提示词
prompt = "你好，请介绍下你自己"
# 将提示词封装为 message
message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 使用分词器的 apply_chat_template 将消息进行转换
# tokenize=False 表示此时不进行令牌化
text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

# 将转换后的文本进行令牌化，并返回模型的输入张量格式
model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

# 将数据输入模型，得到输出
response = model.generate(model_inputs.input_ids, max_new_tokens=512)
print(response)

"""
DEVICE:  mps
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198, 108386,  37945, 100157,  16872,
         107828, 151645,    198, 151644,  77091,    198, 111308,   3837, 104198,
         101919, 102661,  99718, 104197, 100176, 102064, 104949,   1773,  35946,
          99882,  31935,  64559,  99320,  56007,   1773,  35946,  99250,  70500,
         100751,  99553, 100168,   5373,  99795,  33108, 110205,   9370, 105051,
         101904,   1773, 106870, 110117,  86119,  57191,  85106, 100364,  37945,
         102422, 106525,   3837, 105351, 110121, 113445, 100143,   1773, 151645]],
       device='mps:0')
"""

# 对输出的内容进行解码还原
# skip_special_tokens=True 表示跳过特殊标记
output_text = tokenizer.batch_decode(response, skip_special_tokens=True)
print(output_text)

"""
['system\nYou are a helpful assistant.\nuser\n你好，请介绍下你自己\nassistant\n您好！我是来自阿里云的大规模语言模型，我叫通义千问。我可以回答各种问题、创作文字和图片内容，并且能够进行对话交流。如果您有任何问题或需要帮助，请随时告诉我！']
"""