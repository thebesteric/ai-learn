from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface import HuggingFaceLLM

"""
https://docs.llamaindex.ai/en/stable/examples/llm/huggingface/
pip install llama-index-llms-huggingface
"""

model_name = "/Users/wangweijun/LLM/models/Qwen/Qwen2.5-0.5B-Instruct"

# 使用 HuggingFaceLLM 加载本地大模型
llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer_name=model_name,
    model_kwargs={"trust_remote_code": True},
    tokenizer_kwargs={"trust_remote_code": True}
)

# 调用大模型
# response = llm.chat(messages=[ChatMessage(role="user", content="你叫什么名字？")], stream=True)
# for chunk in response:
#     print(chunk)

response = llm.chat(messages=[ChatMessage(role="user", content="xtuner时什么？")])
print(response)