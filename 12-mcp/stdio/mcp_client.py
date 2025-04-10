import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from mcp import StdioServerParameters, stdio_client, ClientSession
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()


class MCPClient:
    def __init__(self):
        """初始化 MCP 客户端"""
        self.session = Optional[ClientSession]
        self.exit_stack = AsyncExitStack()
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model_name = os.getenv("MODEL_NAME")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # self.client = ChatOpenAI(api_key=SecretStr(self.api_key), base_url=self.base_url, model=self.model_name)

    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 服务器，并列出可用工具"""
        print("✅ MCP 客户端已经初始化，连接到 MCP 服务器")

        # 服务器脚本类型
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("服务器版本仅支持 Python 和 JavaScript 脚本")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )

        # 启动 MCP 服务器并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        print(f"✅ MCP 客户端已经列出可用工具：{[tool.name for tool in tools]}")

    async def process_query(self, query: str) -> str:
        """
        使用大模型处理查询，并调用可用的 MCP 工具（Function Calling）
        """
        messages = [
            {"role": "system", "content": "你是一个智能助手，帮助回答用户的问题。"},
            {"role": "user", "content": query},
        ]

        # 可用的工具列表
        list_tools_response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
        } for tool in list_tools_response.tools]
        # print(f"✅ MCP 客户端已经列出可用工具：{available_tools}")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=available_tools,
                temperature=0.3,
            )
            content = response.choices[0]
            if content.finish_reason == "tool_calls":
                tool_call = content.message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)

                # 执行工具
                result = await self.session.call_tool(tool_name, tool_args)
                print(f"\n\n✅ MCP 客户端已经执行工具：{tool_name}，结果：{result}\n\n")

                # 将模型返回调用哪个工具数据和工具返回结果都存入 messages 中
                messages.append(content.message.model_dump())
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result.content[0].text,
                })
                # 将上面的结果再返回给大模型用于生产最终的结果
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                )
                return response.choices[0].message.content

            # 非工具调用的响应
            return content.message.content
            # return response.content
        except Exception as e:
            return f"⚠️ 调用 OpenAI API 发生错误：{str(e)}"

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP 客户端已启动，输入 ‘\\q' 退出")

        while True:
            try:
                query = input("用户：").strip()
                if query == "\\q":
                    break
                response = await self.process_query(query)
                print(f"OpenAI：{response}")
            except Exception as e:
                print(f"⚠️发生错误：{e}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()
        print("✅ MCP 客户端已关闭")


async def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    # python mcp_client.py mcp_server.py
    # uv run mcp_client.py mcp_server.py
    asyncio.run(main())
