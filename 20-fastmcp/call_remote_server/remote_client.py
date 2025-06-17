import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv, find_dotenv
from fastmcp import Client
from fastmcp.utilities.mcp_config import MCPConfig
from openai import OpenAI


class Configuration:
    def __init__(self) -> None:
        # 加载环境变量
        load_dotenv(find_dotenv(), override=True)

    @staticmethod
    def resolve_env_vars(data: Any) -> Any:
        """
        递归处理字典中的字符串，替换形如 ${VAR_NAME} 的环境变量占位符。
        """
        if isinstance(data, dict):
            return {key: Configuration.resolve_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [Configuration.resolve_env_vars(item) for item in data]
        elif isinstance(data, str):
            # 将字符串中的 $VAR 或 ${VAR} 格式的变量替换为实际值
            return os.path.expandvars(data)
        return data

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        # 打开指定路径的文件，以只读模式读取
        # 使用 json.load 将文件内容解析为 Python 字典并返回
        with open(file_path, 'r') as f:
            config = json.load(f)
            return Configuration.resolve_env_vars(config)


class MCPClient:
    def __init__(self):
        """初始化 MCP 客户端"""
        # self.api_key = "empty"
        # self.base_url = "http://localhost:11434/v1"
        # self.model_name = "qwen2.5:7b"
        self.base_url = os.getenv("DASHSCOPE_BASE_URL")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.model_name = "qwen-max"
        self.llm = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.client = None
        self.tools = None
        self.messages = [{"role": "system", "content": "你是一个智能助手，帮助回答用户的问题。"}]

    async def connect_to_server(self, servers_config: Dict[str, Any]):
        self.client = Client(transport=servers_config)
        async with self.client:
            # Make MCP calls within the context
            await self.client.ping()
            print(f"✅ MCP 客户端已经初始化，连接到 MCP 服务器。\n连接状态：{self.client.is_connected()}")
            self.tools = await self.client.list_tools()
            print(f"✅ MCP 客户端已经列出可用工具：")
            for tool in self.tools:
                print(f"{tool}")

    async def execute_tool(self, tool_name: str,
                           tool_args: Dict[str, Any],
                           retries: int = 2,
                           delay: float = 1.0) -> Any:
        # 检查会话是否已初始化
        if not self.client:
            raise RuntimeError(f"❌ Server not initialized")

        # 循环尝试执行工具
        # 如果成功，返回执行结果
        attempt = 0
        while attempt < retries:
            try:
                # 执行工具
                async with self.client:
                    result = await self.client.call_tool(tool_name, tool_args)
                    print(f"✅ MCP 客户端已经执行工具：{tool_name}，参数：{tool_args}，结果：{result}")
                    break
            except Exception as e:
                attempt += 1
                logging.warning(f"⚠️ Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"⚠️ Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("❌ Max retries reached. Failing.")
                    raise
        # 返回工具执行的结果
        return result

    async def process_query(self, query: str) -> str:
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
        } for tool in self.tools]

        self.messages.append({"role": "user", "content": query})

        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
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
                async with self.client:
                    result = await self.client.call_tool(tool_name, tool_args)
                    # result = await self.execute_tool(tool_name, tool_args)
                    print(f"✅ MCP 客户端已经执行工具：{tool_name}，参数：{tool_args}，结果：{result}")

                # 将模型返回调用哪个工具数据和工具返回结果都存入 messages 中
                self.messages.append(content.message.model_dump())
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result[0].text,
                })
                # 将上面的结果再返回给大模型用于生产最终的结果
                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    temperature=0.3,
                )
                self.messages.append(response.choices[0].message.model_dump())
                return response.choices[0].message.content

            # 非工具调用的响应
            self.messages.append(content.message.model_dump())
            return content.message.content
            # return response.content
        except Exception as e:
            return f"⚠️ 调用 LLM 发生错误：{str(e)}"

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("✅ MCP 客户端已启动，输入 ‘\\q' 退出")
        while True:
            try:
                query = input("用户：").strip()
                if query == "\\q":
                    break
                response = await self.process_query(query)
                print(f"LLM：{response}")
            except Exception as e:
                print(f"⚠️发生错误：{e}")

    async def cleanup(self):
        """清理资源"""
        await self.client.close()
        print(f"✅ MCP 客户端已关闭。\n连接状态: {self.client.is_connected()}")


async def main():
    client = MCPClient()
    try:
        config = Configuration()
        servers_config = config.load_config('servers_config.json')
        await client.connect_to_server(servers_config)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
