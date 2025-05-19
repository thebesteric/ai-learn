from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import SecretStr


@tool(description="天气搜索工具")
def search_weather(city: str):
    return f"{city} 气温23摄氏度，风力5级，白天有小到中雨"


tools = [search_weather]
tool_node = ToolNode(tools=tools)
model = ChatOpenAI(model='qwen2.5:7b', base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama")).bind_tools(tools)

graph = create_react_agent(model, tools=tools, checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": [{"role": "user", "content": "合肥天气如何？"}]}

for chunk in graph.stream(inputs, config=config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
