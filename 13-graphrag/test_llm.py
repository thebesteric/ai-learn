from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="123456")

# 单轮对话
chat_completion = client.chat.completions.create(
    # 模型为 ollama 模型名称
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "你好"},
    ],
)
print(chat_completion.choices[0])

# 查看可以调用的模型
models = client.models.list()
print(models)