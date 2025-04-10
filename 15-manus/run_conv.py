import json
import os

from openai import OpenAI
from functions import get_weather, python_inter, tools_dict


def run_conv(messages: list[dict],
             api_key,
             model="deepseek-chat",
             base_url="https://api.deepseek.com/v1",
             tools=None,
             function_list=None) -> str:
    """
    能够自动执行外部函数调用的 Chat 对话模板
    :param messages: 必要参数，消息列表，用于存储对话历史
    :param api_key: 必要参数，调用模型的 API Key
    :param model: 可选参数，模型名称，默认为 "deepseek-chat"
    :param base_url: 必要参数，调用模型的 Base URL
    :param tools: 可选参数，工具列表，用于指定模型可以调用的工具列表，默认为 None
    :param function_list: 可选参数，函数列表，用于指定模型可以调用的外部函数列表，默认为 None
    :return: 返回模型的响应对象
    """
    user_messages = messages

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 如果没有外部函数库，则执行普通的对话任务
    if tools is None or function_list is None:
        response = client.chat.completions.create(
            model=model,
            messages=user_messages,
        )
        final_response = response.choices[0].message.content
    # 如果存在外部函数库，则需要灵活选取外部函数进行回答
    else:
        # 创建外部函数库字典
        available_tools = {func.__name__: func for func in function_list}

        # 第一次调用模型
        first_response = client.chat.completions.create(
            model=model,
            messages=user_messages,
            tools=tools,
            tool_choice="auto",
        )
        # 如果模型没有调用工具，则直接返回模型的响应
        if len(first_response.choices[0].message.tool_calls) == 0:
            final_response = first_response.choices[0].message.content
            return f"Assistant: {final_response}"

        # 模型调用了工具
        response_message = first_response.choices[0].message

        # 获取函数名
        function_name = response_message.tool_calls[0].function.name
        # 获取函数对象
        function_to_call = available_tools[function_name]
        # 获取函数参数
        function_args = json.loads(response_message.tool_calls[0].function.arguments)

        # 将函数参数输入到函数中，获取函数返回值
        function_response = function_to_call(**function_args)

        # messages 中追加 first response 消息，此时的消息是一个 tool_calls 消息
        user_messages.append(response_message.model_dump())

        # messages 中追加 tool response 消息
        user_messages.append(
            {
                "role": "tool",
                "content": function_response,
                "tool_call_id": response_message.tool_calls[0].id,
            }
        )

        # 第二次调用模型，此时 user_messages 中包含了两个消息，一个是 tool_calls 消息，一个是 tool response 消息，让模型进行总结回答
        second_response = client.chat.completions.create(
            model=model,
            messages=user_messages,
            tools=tools,
        )

        # 获取最终结果
        final_response = second_response.choices[0].message.content

    return f"Assistant: {final_response}"


available_tools = [get_weather, python_inter]

tools_schema = [tools_dict["get_weather"], tools_dict["python_inter"]]

if __name__ == "__main__":
    api_key = os.getenv("DEEPSEEK_API_KEY") or input("请输入 API Key：")

    messages = [
        {"role": "user", "content": "合肥的天气如何？"}
    ]
    response = run_conv(messages, api_key=api_key, tools=tools, function_list=available_tools)
    print(response)

    # messages = [
    #     {"role": "user", "content": "帮我用 python 代码模拟一组数据，生成一个股市 K 线图"}
    # ]
    # response = run_conv(messages, api_key=api_key, tools=tools_schema, function_list=available_tools)
    # print(response)