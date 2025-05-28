from langchain import hub
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

# 将初始化工具，让它提供答案而不是文档
tools = [TavilyAnswer(name="Intermediate Answer", description="Answer Search")]

# 初始化大模型
# llm = ChatOpenAI(temperature=0, model="gpt-4")
llm = ChatOpenAI(
    temperature=0.95,
    model="qwen-plus",
    api_key=SecretStr('sk-31bb7a65dd4047aba9b14a95c08be52c'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 获取使用提示 可以修改此提示
prompt = hub.pull("hwchase17/self-ask-with-search")

# 使用搜索代理构建自助询问
agent = create_self_ask_with_search_agent(llm, tools, prompt)

# 通过传入代理和工具创建代理执行程序
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 运行代理
agent_executor.invoke({"input": "中国有哪些省份呢?用中文回复"})