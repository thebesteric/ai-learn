import asyncio
import json
from pathlib import Path
from typing import Any
import pandas as pd
import graphrag.api as api

import httpx
from graphrag.config.load_config import load_config
from mcp.server import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("RAG_Query")
USER_AGENT = "RAG_Query-app/1.0"


@mcp.tool()
async def rag_query(query: str) -> str:
    """
    用于查询机器学习决策树的相关信息
    :param query: 用户提出的问题
    :return: 最终获取到的答案
    """
    print(f"🔧 RAG 被调用，用户提出问题：{query}")
    PROJECT_DIRECTORY_PATH = Path("../graphrag_test")
    graphrag_config = load_config(PROJECT_DIRECTORY_PATH)

    # 获取关键表
    entities = pd.read_parquet(f"{PROJECT_DIRECTORY_PATH}/output/entities.parquet")
    communities = pd.read_parquet(f"{PROJECT_DIRECTORY_PATH}/output/communities.parquet")
    community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY_PATH}/output/community_reports.parquet")

    # 进行全局搜索
    response, context = await api.global_search(
        config=graphrag_config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=2,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=query
    )

    return response


if __name__ == '__main__':
    # response = asyncio.run(rag_query("什么是决策树"))
    # print(response)

    # 以标准 I/O 模式运行 MCP 服务器
    mcp.run(transport="stdio")
