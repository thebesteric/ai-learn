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
    {"role": "user", "content": "北京和合肥的天气如何？"}
]

response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=messages,
    tools=tools,
)

# tool_calls，表示模型需要调用工具
print(response.choices[0].finish_reason)

# ChatCompletionMessage(content='', refusal=None, role='assistant', annotations=None, audio=None, function_call=None,
# tool_calls=[
# ChatCompletionMessageToolCall(id='call_00u8c8ku', function=Function(arguments='{"city":"Beijing"}', name='get_weather'), type='function', index=0),
# ChatCompletionMessageToolCall(id='call_9ogaho24', function=Function(arguments='{"city":"Hefei"}', name='get_weather'), type='function', index=0)
# ])
print(response.choices[0].message)

# [
# ChatCompletionMessageToolCall(id='call_00u8c8ku', function=Function(arguments='{"city":"Beijing"}', name='get_weather'), type='function', index=0),
# ChatCompletionMessageToolCall(id='call_9ogaho24', function=Function(arguments='{"city":"Hefei"}', name='get_weather'), type='function', index=0)
# ]
print(response.choices[0].message.tool_calls)

# {'content': '', 'refusal': None, 'role': 'assistant', 'annotations': None, 'audio': None, 'function_call': None,
# 'tool_calls': [
# {'id': 'call_b5bruaj6', 'function': {'arguments': '{"city":"Beijing"}', 'name': 'get_weather'}, 'type': 'function', 'index': 0},
# {'id': 'call_m6165ke0', 'function': {'arguments': '{"city":"Hefei"}', 'name': 'get_weather'}, 'type': 'function', 'index': 0}
# ]}
print(response.choices[0].message.model_dump())

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

# 添加工具调用信息
second_messages = create_function_response_message(messages, response)

# [
#   {'role': 'user', 'content': '北京和合肥的天气如何？'},
#   {'content': '', 'refusal': None, 'role': 'assistant', 'annotations': None, 'audio': None, 'function_call': None,
#       'tool_calls': [
#           {'id': 'call_6u93t4j9', 'function': {'arguments': '{"city":"Beijing"}', 'name': 'get_weather'}, 'type': 'function', 'index': 0},
#           {'id': 'call_w6ol92u7', 'function': {'arguments': '{"city":"Hefei"}', 'name': 'get_weather'}, 'type': 'function', 'index': 0}
#       ]
#   },
#   {'role': 'tool', 'content': '{"coord": {"lon": 116.3972, "lat": 39.9075}, "weather": [{"id": 800, "main": "Clear", "description": "晴", "icon": "01n"}], "base": "stations", "main": {"temp": 19.94, "feels_like": 18.36, "temp_min": 19.94, "temp_max": 19.94, "pressure": 1009, "humidity": 14, "sea_level": 1009, "grnd_level": 1004}, "visibility": 10000, "wind": {"speed": 2.58, "deg": 150, "gust": 6.11}, "clouds": {"all": 7}, "dt": 1743938124, "sys": {"type": 1, "id": 9609, "country": "CN", "sunrise": 1743889847, "sunset": 1743936156}, "timezone": 28800, "id": 1816670, "name": "Beijing", "cod": 200}', 'tool_call_id': 'call_r5s3r74k'},
#   {'role': 'tool', 'content': '{"coord": {"lon": 117.2808, "lat": 31.8639}, "weather": [{"id": 800, "main": "Clear", "description": "晴", "icon": "01n"}], "base": "stations", "main": {"temp": 21.01, "feels_like": 19.79, "temp_min": 21.01, "temp_max": 21.01, "pressure": 1014, "humidity": 24, "sea_level": 1014, "grnd_level": 1011}, "visibility": 10000, "wind": {"speed": 4, "deg": 140}, "clouds": {"all": 0}, "dt": 1743938519, "sys": {"type": 1, "id": 9661, "country": "CN", "sunrise": 1743889993, "sunset": 1743935585}, "timezone": 28800, "id": 1808722, "name": "Hefei", "cod": 200}', 'tool_call_id': 'call_co0fmrw1'}
# ]
print(f"second_messages = {second_messages}")
# 调用模型生成最终响应
second_response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=second_messages,
    tools=tools,
)
# 北京的当前天气为晴朗，温度约为19.94℃，体感温度为18.36℃。风速大约是2.58米/秒，在东南方向；能见度为10000米。\n\n合肥的当前天气同样也是晴朗，温度约为21.01℃，体感温度为19.79℃。风速在4米/秒左右，并且来自东偏南的方向；能见度也为10000米。
print(second_response.choices[0].message)
# stop
print(second_response.choices[0].finish_reason)