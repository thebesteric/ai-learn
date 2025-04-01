import asyncio
import os
from contextlib import AsyncExitStack

from openai import OpenAI


class MCPClient:
    def __init__(self):
        """初始化 MCP 客户端"""
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_mock_server(self):
        """模拟 MCP 服务器连接"""
        print("✅MCP 客户端已经初始化，模拟连接到 MCP 服务器")

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP 客户端已启动，输入 ‘\\q' 退出")

        while True:
            try:
                query = input("用户：").strip()
                if query == "\\q":
                    break
                print(f"[MCP Mock Response] 你说的是：{query}")
            except Exception as e:
                print(f"⚠️发生错误：{e}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()
        print("✅MCP 客户端已关闭")


async def main():
    """主函数"""
    client = MCPClient()
    try:
        await client.connect_to_mock_server()
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
