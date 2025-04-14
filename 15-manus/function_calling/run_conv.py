import json
import os

from openai import OpenAI
from functions import get_weather, python_inter, tools_dict, sql_inter, write_file


def call_tools(client, model, tool_call_messages, user_messages, tools, available_tools):
    # 多工具调用
    for tool_call_message in tool_call_messages:
        # 获取函数名
        function_name = tool_call_message.function.name
        # 获取函数参数
        function_args = json.loads(tool_call_message.function.arguments)
        # 获取函数对象
        function_to_call = available_tools[function_name]

        # 模型调用了工具
        print("\n>>>>>>>>>> 🔧 工具调用 🔧 >>>>>>>>>>")
        print(f"Function name：{tool_call_message.function.name}")
        print(f"Function args：{tool_call_message.function.arguments}")
        print("<<<<<<<<<< 🔧 工具调用 🔧 <<<<<<<<<<\n")

        # 将函数参数输入到函数中，获取函数返回值
        try:
            function_response = function_to_call(**function_args)
        except Exception as e:
            function_response = f"函数运行报错：{str(e)}"

        # messages 中追加 tool response 消息
        user_messages.append(
            {
                "role": "tool",
                "content": function_response,
                "tool_call_id": tool_call_message.id,
            }
        )

    # 工具调用结束，此时再次调用模型，获取下一次对话的响应
    next_response = client.chat.completions.create(
        model=model,
        messages=user_messages,
        tools=tools,
    )
    next_message = next_response.choices[0].message
    # 将下一次对话的响应添加到 messages 中
    user_messages.append(next_message.model_dump())

    # 返回模型的响应
    return next_response


def run_conv(messages: list[dict],
             api_key,
             model="deepseek-chat",
             base_url="https://api.deepseek.com/v1",
             tools=None,
             function_list=None) -> str:
    """
    能够自动执行外部函数调用的 Chat 对话模板，同时支持工具并行和串行的调用
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
        current_response = client.chat.completions.create(
            model=model,
            messages=user_messages,
            tools=tools,
            tool_choice="auto",
        )
        # 如果模型没有调用工具，则直接返回模型的响应
        tool_calls = current_response.choices[0].message.tool_calls
        if len(tool_calls) == 0:
            final_response = current_response.choices[0].message.content
            return f"Assistant: {final_response}"

        # 此时的消息是一个 tool_calls 消息
        response_message = current_response.choices[0].message
        # messages 中追加 tool_calls 消息，
        user_messages.append(response_message.model_dump())

        # 最终的响应结果
        final_response = None
        while True:
            next_response = call_tools(client, model, tool_calls, user_messages, tools, available_tools)
            # 模型依然需要调用工具（工具串行的情况）
            if next_response.choices[0].finish_reason == "tool_calls":
                tool_calls = next_response.choices[0].message.tool_calls
            # 如果模型不需要再调用工具，则直接返回模型的响应
            else:
                # 最终响应
                final_response = next_response.choices[0].message.content
                break

    return f"Assistant: {final_response}"


available_tools = [get_weather, python_inter, sql_inter, write_file]

tools_schema = [tools_dict["get_weather"], tools_dict["python_inter"], tools_dict["sql_inter"], tools_dict["write_file"]]

if __name__ == "__main__":
    api_key = os.getenv("DEEPSEEK_API_KEY") or input("请输入 API Key：")

    # messages = [
    #     {"role": "user", "content": "合肥的天气如何？"}
    # ]
    # response = run_conv(messages, api_key=api_key, tools=tools_schema, function_list=available_tools)
    # print(response)

    # messages = [
    #     {"role": "user", "content": "帮我用 python 代码模拟一组数据，生成一个股市 K 线图"}
    # ]
    # response = run_conv(messages, api_key=api_key, tools=tools_schema, function_list=available_tools)
    # print(response)

    # messages = [
    #     {"role": "user", "content": "查询一下当前数据库有哪些表？"}
    # ]
    # response = run_conv(messages, api_key=api_key, tools=tools_schema, function_list=available_tools)
    # print(response)

    # messages = [
    #     {"role": "user", "content": "合肥的天气如何，并查询一下当前数据库有哪些表？"}
    # ]
    # response = run_conv(messages, api_key=api_key, tools=tools_schema, function_list=available_tools)
    # print(response)

    messages = [
        {"role": "user", "content": "查询合肥的天气，并写入到本地文件？"}
    ]
    # response = run_conv(messages, api_key=api_key, model="qwen2.5:7b", base_url="http://127.0.0.1:11434/v1", tools=tools_schema, function_list=available_tools)
    response = run_conv(messages, api_key=api_key, tools=tools_schema, function_list=available_tools)
    print(response)