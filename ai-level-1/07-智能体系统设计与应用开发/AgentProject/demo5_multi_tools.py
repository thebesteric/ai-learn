from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()


# 定义查询订单状态的函数
def query_order_status(order_id):
    if order_id == "1024":
        return "订单 1024 的状态是：已发货，预计送达时间是 3-5 个工作日。"
    else:
        return f"未找到订单 {order_id} 的信息，请检查订单号是否正确。"


# 定义退款政策说明函数
def refund_policy(keyword):
    print("keyword = ", keyword)
    return "我们的退款政策是：在购买后7天内可以申请全额退款，需提供购买凭证。"


# 初始化工具
tools = [
    Tool(
        name="queryOrderStatus",
        func=query_order_status,
        description="根据订单ID查询订单状态",
        args={"order_id": "订单的ID"}
    ),
    Tool(
        name="refundPolicy",
        func=refund_policy,
        description="查询退款政策内容",
        args={"keyword": "查询退款政策内容"}
    ),
    TavilySearchResults(max_results=1)
]

# 获取使用的提示
prompt = hub.pull("hwchase17/openai-functions-agent")

# 选择将驱动代理的LLM
llm = ChatOpenAI(
    temperature=0.95,
    model="qwen-plus",
    api_key=SecretStr('sk-31bb7a65dd4047aba9b14a95c08be52c'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 构建OpenAI函数代理
# agent = create_openai_functions_agent(llm, tools, prompt)
agent = create_tool_calling_agent(llm, tools, prompt)

# 通过传入代理和工具创建代理执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 定义一些测试询问
queries = [
    "请问订单1024的状态是什么？",
    "退款政策内容？",
    "目前市场上苹果手机16的各个型号的售价是多少？"
]

# 运行代理并输出结果
for input in queries:
    response = agent_executor.invoke({"input": input})
    print(f"客户提问：{input}")
    print(f"代理回答：{response}\n")
