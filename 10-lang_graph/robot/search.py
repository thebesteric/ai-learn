import getpass
import os

from langchain_community.tools import TavilySearchResults

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

tool = TavilySearchResults(max_results=2)
tools = [tool]

if __name__ == '__main__':
    responses = tool.invoke("What's a 'node' in LangGraph?")
    # [{'title': 'xxx', 'url': 'xxx', 'content': 'xxx', 'score': 0.9008183}, {'title': 'xxx', 'url': 'xxx', 'content': 'xxx', 'score': 0.8991304}]
    print(responses)