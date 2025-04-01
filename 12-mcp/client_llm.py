import asyncio
import os
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession
from openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

class MCPClient:
    def __init__(self):
        """初始化 MCP 客户端"""
        self.api_key = os.getenv("API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model_name = os.getenv("MODEL_NAME")
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # self.client = ChatOpenAI(api_key=SecretStr(self.api_key), base_url=self.base_url, model=self.model_name)

    async def process_query(self, query: str) -> str:
        """处理用户查询"""
        messages = [
            {"role": "system", "content": "你是一个智能助手，帮助回答用户的问题。"},
            {"role": "user", "content": query},
        ]
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                # lambda: self.client.invoke(input=messages)
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                )
            )
            return response.choices[0].message.content
            # return response.content
        except Exception as e:
            return f"⚠️调用 OpenAI API 发生错误：{str(e)}"

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
        print("✅MCP 客户端已关闭")


async def main():
    """主函数"""
    client = MCPClient()
    try:
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
