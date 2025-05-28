import calendar
import dateutil.parser as parser
from datetime import date
from langchain.tools import Tool, tool
from langchain.agents import load_tools
from langchain import hub
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from pydantic import SecretStr
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor

load_dotenv()
# tools = load_tools(["serpapi"])
tools = [TavilySearchResults(max_results=1),]


# 自定义工具
@tool("date_to_weekday")
def date_to_weekday(date_str: str) -> str:
    """Convert date to weekday name"""
    d = parser.parse(date_str)
    return calendar.day_name[d.weekday()]


tools += [date_to_weekday]  # 将自定义的tool添加到tools数组中
from langchain_openai import ChatOpenAI

# 获取要使用的提示
llm = ChatOpenAI(
    temperature=0.95,
    model="qwen-plus",
    api_key=SecretStr('sk-31bb7a65dd4047aba9b14a95c08be52c'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

import json

# 下载一个现有的 Prompt 模板
prompt = hub.pull("hwchase17/react")
print(prompt)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "周杰伦生日是那天？是星期几？"})
