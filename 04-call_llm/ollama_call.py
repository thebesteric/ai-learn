from openai import OpenAI

"""
使用 openai 调用 Ollama
启动 Ollama 服务：nohup ollama serve > ollama.log 2>&1 &
停止 Ollama 服务：ps -ef | grep ollama
"""

client = OpenAI(base_url="http://localhost:11434/v1", api_key="123456")

# 单轮对话
chat_completion = client.chat.completions.create(
    # 模型为 ollama 模型名称
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "你好，请介绍下你自己"},
    ],
)
print(chat_completion.choices[0])
"""
Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='您好！我是Qwen，由阿里云开发的AI助手。我能够提供各种信息查询、问答交互、创意写作等多种语言服务。虽然我没有真实的物理形态，但我可以24小时为您解答问题、提供建议或进行闲聊等，希望能成为您的得力助手。请问您有什么需要帮助的吗？', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None))
"""

# 多轮对话

# 记录历史信息
chat_history = []
print("欢迎使用多轮对话，\"/q\" 退出，\"/new\" 新建聊天")
while True:
    user_input = input("请输入您的问题：")
    if user_input == "/q":
        print("退出对话")
        break
    if user_input == "/new":
        chat_history = []
        print("历史信息已清空")
        continue
    # 增加历史信息
    chat_history.append({"role": "user", "content": user_input})
    try:
        chat_completion = client.chat.completions.create(
            model="qwen2.5:7b",
            messages=chat_history,
        )
        # 获取模型回复
        model_response = chat_completion.choices[0]
        print("AI:", model_response.message.content)
        # 增加历史信息
        chat_history.append({"role": "assistant", "content": chat_completion.choices[0].message.content})
    except Exception as e:
        print("Error:", e)
        break


