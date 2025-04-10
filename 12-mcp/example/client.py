import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Dict

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, stdio_client
from openai import OpenAI

load_dotenv()


class MultiServerMCPClient:
    def __init__(self):
        """初始化：管理多个 MCP 服务器的客户端"""
        self.exit_stack = AsyncExitStack()
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.llm_base_url = os.getenv("LLM_BASE_URL")
        self.llm_model_name = os.getenv("LLM_MODEL_NAME")
        if not self.llm_api_key:
            raise ValueError("❌ 未找到 LLM_API_KEY，请在 .env 文件中设置")

        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)
        # 存储 {server_name -> MCP ClientSession} 映射关系
        self.sessions: Dict[str, ClientSession] = {}
        # 存储工具信息
        self.tools_by_session: Dict[str, list] = {}
        # 所有工具的列表
        self.all_tools = []

    async def connect_to_server(self, servers: dict):
        """
        同时连接到多个 MCP 服务器，并列出可用工具
        :param servers: 如：{“weather": "weather_server.py", "news": "news_server.py"}
        :return:
        """
        print("✅ MCP 客户端已经初始化，连接到 MCP 服务器")

        for server_name, script_path in servers.items():
            session = await self._start_one_server(script_path)
            self.sessions[server_name] = session
            # 列出当前服务器上的工具
            resp = await session.list_tools()
            # 将工具添加到 self.tools_by_session 中
            self.tools_by_session[server_name] = resp.tools
            for tool in resp.tools:
                # 工具的完整名称
                func_name = f"{server_name}_{tool.name}"
                self.all_tools.append({
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                })

        # 转化 function calling 格式
        self.all_tools = await self.transform_json(self.all_tools)

        print("\n✅ MCP 客户端已连接到下列服务器")
        for name in servers:
            print(f" - {name}: {servers[name]}")

        print(f"\n✅ 汇总的工具:")
        for tool in self.all_tools:
            print(f" - {tool['function']['name']}: {tool['function']['description']}")

    async def transform_json(self, json2_data) -> list:
        """
        将类似 json2 的格式转换为类型 json1 的格式，多余字段会被直接删除，主要用于 Claude Function calling 参数格式转换为 OpenAI Function calling 参数格式
        :param json2_data: 一个可被解释为列表的 python 对象，或已解析的 JSON 数据
        :return: 转换后的新列表
        """
        result = []
        for item in json2_data:
            # 确保有 "type" 和 "function" 两个字段
            if not isinstance(item, dict) or "type" not in item or "function" not in item:
                continue
            old_func = item["function"]
            # 确保 function 下有我们需要的关键字段
            if not isinstance(old_func, dict) or "name" not in old_func or "description" not in old_func:
                continue
            # 处理 function 字段
            new_func = {
                "name": old_func["name"],
                "description": old_func["description"],
                "parameters": {}
            }
            # 读取 input_schema 字段，并转成 parameters 字段
            if "input_schema" in old_func and isinstance(old_func["input_schema"], dict):
                old_schema = old_func["input_schema"]
                # 新的 parameters 保留 type, properties, required 这三个字段
                new_func["parameters"]["type"] = old_schema.get("type", "object")
                new_func["parameters"]["properties"] = old_schema.get("properties", {})
                new_func["parameters"]["required"] = old_schema.get("required", [])

            new_item = {
                "type": item["type"],
                "function": new_func
            }

            result.append(new_item)

        return result

    async def _start_one_server(self, script_path: str) -> ClientSession:
        """启动一个 MCP 服务器，并返回 ClientSession 对象"""
        # 服务器脚本类型
        is_python = script_path.endswith(".py")
        is_js = script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("服务器版本仅支持 Python 和 JavaScript 脚本")
        # 启动 MCP 服务器
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[script_path],
            env=None,
        )

        # 启动 MCP 服务器并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        return session

    async def chat_base(self, messages: list):
        response = self.client.chat.completions.create(
            model=self.llm_model_name,
            messages=messages,
            tools=self.all_tools,
        )
        # 处理工具调用
        if response.choices[0].finish_reason == "tool_calls":
            while True:
                messages = await self.create_function_response_message(messages, response)
                response = self.client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=messages,
                    tools=self.all_tools,
                )
                if response.choices[0].finish_reason != "tool_calls":
                    break
        # 返回响应
        return response

    async def create_function_response_message(self, messages: list, response):
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
            print(f"===> 调用 {tool_name} 函数，{tool_args}")
            function_response = await self._call_mcp_tool(tool_name, tool_args)

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

    async def process_query(self, user_query: str) -> str:
        """
        OpenAI 最新的 Function calling 逻辑
        1. 发送用户消息 + tools 信息
        2. 若模型 finish_reason 为 tool_calls，则解析 toolCalls 并执行相应的 MCP 工具
        3. 把调用结果返回给 OpenAI，让模型生成最终回答
        :param user_query: 用户的输入
        :return:
        """
        messages = [{"role": "user", "content": user_query}]

        # 第一次请求
        response = self.client.chat.completions.create(
            model=self.llm_model_name,
            messages=messages,
            tools=self.all_tools,
        )
        content = response.choices[0]
        print(f"response = {content}")
        print(f"all_tools = {self.all_tools}")

        # 如果模型调用了 MCP 工具
        if content.finish_reason == "tool_calls":
            # 解析 tool_calls
            tool_calls = content.message.tool_calls[0]
            tool_name = tool_calls.function.name
            tool_args = json.loads(tool_calls.function.arguments)

            print(f"===> 调用 {tool_name} 函数，{tool_args}")

            # 执行 MCP 工具
            result = await self._call_mcp_tool(tool_name, tool_args)

            # 把工具调用历史追加到 messages
            messages.append(content.message.model_dump())
            messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_calls.id,
                }
            )

            # 第二次请求，让模型真和工具结果，生成最终回答
            response = self.client.chat.completions.create(
                model=self.llm_model_name,
                messages=messages,
                tools=self.all_tools,
            )
            return response.choices[0].message.content

        # 如果模型没有调用工具，直接返回模型的回答
        return content.message.content

    async def _call_mcp_tool(self, tool_full_name: str, tool_args: dict) -> str:
        """
        根据 tool_full_name 调用 MCP 工具
        :param tool_full_name: 工具名称，tool_full_name = {server_name}_{tool.name}
        :param tool_args: 工具参数
        :return:
        """
        parts = tool_full_name.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"无效的工具名称: {tool_full_name}")
        server_name, tool_name = parts
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"未找到对应的服务器: {server_name}")

        # 执行 MCP 工具
        resp = await session.call_tool(tool_name, tool_args)
        print(f"===> 调用 {tool_name} 函数，{tool_args}，结果为 {resp}")
        return resp.content if resp.content else "工具执行无输出"

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP 客户端已启动，输入 ‘\\q' 退出")
        messages = []
        while True:
            query = input("你：").strip()
            if query.lower() == "\\q":
                break
            try:
                messages.append({"role": "user", "content": query})
                messages = messages[-20:]
                response = await self.chat_base(messages)
                messages.append(response.choices[0].message.model_dump())
                result = response.choices[0].message.content
                print(f"\nAI：{result}")
            except Exception as e:
                print(f"⚠️ 调用过程发生错误：{str(e)}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main():
    # 服务器脚本路径
    servers = {
        # "write": "write_server.py",
        # "weather": "weather_server.py",
        "SQLServer": "sql_server.py",
        "PythonServer": "python_server.py",
    }

    client = MultiServerMCPClient()
    try:
        await client.connect_to_server(servers)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
