from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

# client = OpenAI()
# model = "gpt-4"
client = OpenAI(api_key='sk-31bb7a65dd4047aba9b14a95c08be52c', base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
model = "qwen-plus"

# 告诉大模型这个函数的存在
def get_completion(messages, model):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        tools=[
            {  # 用 JSON 描述函数。可以定义多个。由大模型决定调用谁
                "type": "function",
                "function": {
                    "name": "sum_number",
                    "description": "计算一组数的和",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "numbers": {
                                "type": "array",
                                "items": {
                                    "type": "number"
                                }
                            }
                        }
                    }
                }
            },
        ]
    )
    return response.choices[0].message


# 给一个需要使用该函数的Prompt
prompt = "我今年18岁，我的舅舅今年38岁，我的爷爷今年72岁，我和舅舅一共多少岁了？"
# prompt = "30*8等多少？"

messages = [
    {"role": "system", "content": "你是一个数学家，你可以计算任何算式。"},
    {"role": "user", "content": prompt}
]
response = get_completion(messages, model)
messages.append(response)  # 注意这一句，必须加入到上下文中，否则报错
print("=====模型回复=====")
print(response)

# 解析函数名称和参数
# 当大模型返回了需要调用的名称和参数之后，我们可以通过本地代码解析出来，然后再去调用相应函数。
if (response.tool_calls is not None):
    # 是否要调用 sum
    tool_call = response.tool_calls[0]
    if (tool_call.function.name == "sum_number"):
        # 调用 sum
        args = json.loads(tool_call.function.arguments)
        # sum([18, 38])
        result = sum(args["numbers"])
        print("=====函数返回=====")
        print(result)

        # 再次调用大模型获取最终结果
        # 本地函数执行完得到结果后，再将这个结果给大模型，让大模型用自然语言组织起最终答案。
        # 这里需要怎么给大模型呢？需要将函数调用结果，tool_call_id，role，name等一起加入到prompt中
        messages.append(
            {
                "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
                "role": "tool",
                "name": "sum_number",
                "content": str(result)  # 数值result 必须转成字符串
            }
        )
        # 经测试，tool_call_id和role是必须参数，name可以不要，但最好也加上。
        print("=====messages=====")
        print(messages)
        # 再次调用大模型
        print("=====最终回复=====")
        print(get_completion(messages,  model).content)
