import requests
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
import requests
from dotenv import load_dotenv
from pydantic import SecretStr
from pypinyin import lazy_pinyin, Style

load_dotenv()
import os


# 获取天气信息的函数
def get_weather(city):
    # 1213-根据城市查询天气 - 代码参考（根据实际业务情况修改）

    # 接口请求入参配置
    # api_host = 'http://v.juhe.cn/weather/index'  # 接口请求URL
    # api_key = os.getenv("WEATHER_API_KEY")  # 在个人中心->我的数据,接口名称上方查看
    # request_params = {
    #     'key': api_key,
    #     'cityname': city,
    # }

    app_id = os.getenv("OPEN_WEATHER_APP_ID")
    api_host = os.getenv("OPEN_WEATHER_API_HOST")
    request_params = {
        'appid': app_id,
        'q': ''.join(lazy_pinyin(city, style=Style.NORMAL)),
        'lang': 'zh_cn',
    }

    # 发起接口网络请求
    response = requests.get(api_host, params=request_params)

    # 解析响应结果
    if response.status_code == 200:
        responseResult = response.json()
        return responseResult
        # 网络请求成功。可依据业务逻辑和接口文档说明自行处理。
        # print(responseResult)
    else:
        # 网络异常等因素，解析结果异常。可依据业务逻辑自行处理。
        print('请求异常')


# 初始化工具
tools = [
    TavilySearchResults(max_results=1),
    Tool(
        name="getWeather",
        func=get_weather,
        description="根据城市名称查询天气情况",
        args={"city": "查询城市名"}
    )
]

# 获取要使用的提示
llm = ChatOpenAI(
    temperature=0.95,
    model="qwen-plus",
    api_key=SecretStr('sk-31bb7a65dd4047aba9b14a95c08be52c'),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 获取使用的提示
prompt = hub.pull("hwchase17/openai-functions-agent")

# 构建 OpenAI 工具代理
agent = create_openai_tools_agent(llm, tools, prompt)

# 通过传入代理和工具创建代理执行程序
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 定义一些测试询问
queries = [
    "成都今天天气情况？",
    "目前市场上黄金的平均售价是多少？"
]

# 运行代理并输出结果
for input in queries:
    response = agent_executor.invoke({"input": input})
    print(f"提问：{input}")
    print(f"回答：{response}\n")
