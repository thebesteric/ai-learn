import json

import requests
from openai import OpenAI

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

def write_file(content):
    """
    将指定内容写入本地文件
    :param content: 必要参数，字符串类型，用于表示需要写入文件的具体内容
    :return: 是否写入成功
    """
    return "已经成功写入本地文件"


available_tools = {
    "get_weather": get_weather,
    "write_file": write_file
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
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将指定内容写入本地文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "用于表示需要写入文件的具体内容"
                    }
                },
                "required": ["content"]
            }
        }
    }
]

client = OpenAI(api_key="xxx", base_url="http://127.0.0.1:11434/v1")

def create_function_response_message(messages: list, response):
    """
    一次性构建完整的回复
    :param messages: 消息列表，用于存储对话历史
    :param response: 模型的响应对象，包含工具调用信息
    :return:
    """
    function_call_messages = response.choices[0].message.tool_calls
    messages.append(response.choices[0].message.model_dump())
    for function_call_message in function_call_messages:
        tool_name = function_call_message.function.name
        tool_args = json.loads(function_call_message.function.arguments)

        # 运行外部函数
        function_to_call = available_tools[tool_name]
        try:
            print(f"===> 调用 {tool_name} 函数，{tool_args}")
            function_response = function_to_call(**tool_args)
        except Exception as e:
            function_response = f"函数运行报错：{str(e)}"

        # 拼接消息队列
        messages.append(
            {
                "role": "tool",
                "content": function_response,
                "tool_call_id": function_call_message.id,
            }
        )
    # 返回消息队列
    return messages


def chat_base(messages):
    response = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=messages,
        tools=tools,
    )

    if response.choices[0].finish_reason == "tool_calls":
        while True:
            messages = create_function_response_message(messages, response)
            response = client.chat.completions.create(
                model="qwen2.5:7b",
                messages=messages,
                tools=tools,
            )
            if response.choices[0].finish_reason != "tool_calls":
                break
    return response

# 用户提问信息
messages = [
    {"role": "user", "content": "北京天气如何？等天气查询出结果后，将天气信息写入本地文档"}
]
response = chat_base(messages)

# ChatCompletionMessage(content='北京的天气是晴朗，当前气温为12.94℃（感觉温度为10.81℃），最低和最高气温均为12.94℃。湿度较低，仅为20%，能见度良好，风速较小，约为2.11米/秒。天气信息已成功写入本地文件。\n\n以下是北京的具体天气信息：\n- 天气状况: 晴\n- 温度: 12.94℃\n- 感觉温度: 10.81℃\n- 最低气温: 12.94℃\n- 最高气温: 12.94℃\n\n请在本地查看该文件获取更多信息。', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=None)
print(response.choices[0].message)
# stop
print(response.choices[0].finish_reason)
