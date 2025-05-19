# 多轮对话
from openai import OpenAI


# 定义多轮对话方法
def run_chat_session():
    # 初始化客户端
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="123456")
    # client = OpenAI(base_url="http://192.168.1.193:8000/v1", api_key="123456")
    # 初始化对话历史
    chat_history = []
    # 启动对话循环
    while True:
        # 获取用户输入
        user_input = input("用户：")
        if user_input.lower() == "exit":
            print("退出对话。")
            break
        # 更新对话历史(添加用户输入)
        chat_history.append({"role": "user", "content": user_input})
        # 支持所有推理
        chat_history.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})
        # 调用模型回答
        try:
            chat_complition = client.chat.completions.create(messages=chat_history,
                                                             model="qwen3:8b",
                                                             # 这种方式不支持 ollama 推理
                                                             extra_body={
                                                                 "chat_template_kwargs": {"enable_thinking": False},
                                                                 "use_flash_attention_2": True,
                                                             })
            # chat_complition = client.chat.completions.create(messages=chat_history,
            #                                                  model="/home/ubuntu/llm/models/Qwen/Qwen3-8B",
            #                                                  # 这种方式不支持 ollama 推理
            #                                                  extra_body={
            #                                                      "chat_template_kwargs": {"enable_thinking": False}
            #                                                  })

            # 获取最新回答
            model_response = chat_complition.choices[0]
            print("AI:", model_response.message.content)
            # 更新对话历史（添加 AI 模型的回复）
            chat_history.append({"role": "assistant", "content": model_response.message.content})
        except Exception as e:
            print("发生错误：", e)
            break


if __name__ == '__main__':
    run_chat_session()
