import asyncio
from pathlib import Path
from pprint import pprint
import pandas as pd
import graphrag.api as api
from graphrag.config.load_config import load_config
from graphrag.index.typing.pipeline_run_result import PipelineRunResult
from markdown_it.common.entities import entities
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='qwen2.5:7b', temperature=0.3, base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))

# 项目目录
PROJECT_DIRECTORY_PATH = Path("./graphrag_test")
# 生产 GraphRagConfig 对象
graphrag_config = load_config(PROJECT_DIRECTORY_PATH)
# print(graphrag_config)

# 创建索引
async def build_index():
    index_result: list[PipelineRunResult] = await api.build_index(config=graphrag_config)
    for result in index_result:
        status = f"ERROR\n{result.errors}" if result.errors else "SUCCEED"
        print(f"Workflow Name: {result.workflow}\tStatus: {status}")


# c
async def global_query(query: str, community_level: int = 2):
    # 获取关键表
    entities = pd.read_parquet(f"{PROJECT_DIRECTORY_PATH}/output/entities.parquet")
    communities = pd.read_parquet(f"{PROJECT_DIRECTORY_PATH}/output/communities.parquet")
    community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY_PATH}/output/community_reports.parquet")
    # 执行查询
    response, context = await api.global_search(
        config=graphrag_config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        community_level=community_level,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query=query
    )

    return (response, context)


if __name__ == '__main__':
    # 执行并打印返回值
    response, context = asyncio.run(global_query("请帮我对比ID3和C4.5决策树算法的优劣势"))
    print(response)
    pprint(context)

