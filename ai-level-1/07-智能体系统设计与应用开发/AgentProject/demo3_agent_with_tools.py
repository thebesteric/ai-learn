from langchain.tools.retriever import create_retriever_tool

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

# 初始化大模型
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

llm = ChatOpenAI(
    temperature=0.95,
    model="qwen-plus",
    api_key=SecretStr('sk-31bb7a65dd4047aba9b14a95c08be52c'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 加载HTML内容为一个文档对象
loader = WebBaseLoader("https://new.qq.com/rain/a/20240920A07Y5Y00")
docs = loader.load()
# print(docs)

# 分割文档
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# 向量化
embeddings = HuggingFaceEmbeddings(model_name="/Users/wangweijun/llm/models/bge-large-zh-v1.5", model_kwargs = {'device': 'cpu'}, encode_kwargs = {'normalize_embeddings': False})
vector = FAISS.from_documents(documents, embeddings)

# 创建检索器
retriever = vector.as_retriever()
# 创建一个工具来检索文档
retriever_tool = create_retriever_tool(
    retriever,
    "iPhone_price_search",
    "搜索有关 iPhone 16 的价格信息。对于iPhone 16的任何问题，您必须使用此工具！",
)

# 加载所需的库
from langchain_community.tools.tavily_search import TavilySearchResults

# 查询 Tavily 搜索 API 并返回 json 的工具
search = TavilySearchResults()

# 创建将在下游使用的工具列表
tools = [search, retriever_tool]

from langchain import hub

# 获取要使用的提示
prompt = hub.pull("hwchase17/openai-functions-agent")
# 打印Prompt
print(prompt)

# 使用OpenAI functions代理
from langchain.agents import create_openai_functions_agent

# 构建OpenAI函数代理：使用 LLM、提示模板和工具来初始化代理
agent = create_openai_functions_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor

# 将代理与AgentExecutor工具结合起来
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行代理
# agent_executor.invoke({"input": "目前市场上苹果手机16的各个型号的售价是多少？"})
agent_executor.invoke({"input": "美国2024年谁胜出了总统的选举?"})
