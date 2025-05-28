# 加载所需的库
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()

# 查询 Tavily 搜索 API 并返回 json 的工具
search = TavilySearchResults()
# 执行查询
res = search.invoke("目前市场上苹果手机16的售价是多少？")
print(res)