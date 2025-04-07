import json

import requests
from openai import OpenAI

client = OpenAI(api_key="xxx", base_url="http://127.0.0.1:11434/v1")

def get_weather(city: str):
    """
    查询即时天气函数
    :param city: 必要参数，字符串类型，用于表示查询天气的具体城市名称
    注意：中国的城市需要用对应的城市的英文名称代替，例如如果需要查询北京的天气，则 city 参数需要输入 "Beijing"
    :return: OpenWeather API 查询即时天气的结果，返回结果对象类型为解析后的 JSON 格式对象，并用字符串形式表示，其中包含了全部中要的天气信息
    """
    # Open Weather API 配置
    open_weather_base_url = "https://api.openweathermap.org/data/2.5/weather"
    open_weather_api_key = "57ae333b23774c9bb9b82273213d7d47"
    params = {
        "q": city,
        "appid": open_weather_api_key,
        "lang": "zh_cn",
        "units": "metric",
    }
    response = requests.get(open_weather_base_url, params=params)
    data = response.json()
    return json.dumps(data, ensure_ascii=False)


available_tools = {
    "get_weather": get_weather
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询即时天气函数，输入城市名称，返回该城市的天气信息。一次只能查询一个城市的天气信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，注意：中国的城市需要用对应的城市的英文名称代替，例如如果需要查询北京的天气，则 city 参数需要输入 'Beijing'"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# 用户提问信息
messages =[
    {"role": "user", "content": "北京天气如何？"}
]

response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=messages,
    tools=tools,
)

# tool_calls，表示模型需要调用工具
print(response.choices[0].finish_reason)

# ChatCompletionMessage(content='', refusal=None, role='assistant', annotations=None, audio=None, function_call=None,
# tool_calls=[ChatCompletionMessageToolCall(id='call_jxv8rp81', function=Function(arguments='{"city":"Beijing"}', name='get_weather'), type='function', index=0)])
print(response.choices[0].message)
# [ChatCompletionMessageToolCall(id='call_jxv8rp81', function=Function(arguments='{"city":"Beijing"}', name='get_weather'), type='function', index=0)]
print(response.choices[0].message.tool_calls)
# ChatCompletionMessageToolCall(id='call_jxv8rp81', function=Function(arguments='{"city":"Beijing"}', name='get_weather'), type='function', index=0)
tool_call = response.choices[0].message.tool_calls[0]

# get_weather
tool_name = tool_call.function.name
# {"city":"Beijing"}
tool_args = json.loads(tool_call.function.arguments)

print(f"工具名: {tool_name}")
print(f"参数: {tool_args}")

# 根据工具名，获取工具函数
function_to_call = available_tools[tool_name]
print(f"function_to_call = {function_to_call}")

# 调用工具，获取结果
function_response = function_to_call(**tool_args)
print(function_response)

# 封装模型调用结果信息
function_response_message = {
    "role": "tool",
    "tool_call_id": tool_call.id,
    "name": tool_name,
    "content": function_response
}

# 将工具调用信息封装为字典对象
tool_call_response = response.choices[0].message.model_dump()
print(f"tool_call_response = {tool_call_response}")

# 将工具响应添加到消息列表中
messages.append(tool_call_response)
messages.append(function_response_message)
# 调用模型生成最终响应
second_response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=messages,
    tools=tools,
)
# 最终响应
print(second_response.choices[0].message)

# stop，表示此次对话结束
print(second_response.choices[0].finish_reason)