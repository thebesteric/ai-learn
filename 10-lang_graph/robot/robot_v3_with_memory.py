import getpass
import json
import os
from typing import Annotated, TypedDict, Literal

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr


# if not os.environ.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

# tool = TavilySearchResults(max_results=2)
# tools = [tool]

@tool(description="天气搜索工具")
def weather_search_tool(query: str):
    """天气搜索工具"""
    print(f"===> call tools: weather_search_tool, query: {query}")
    if "上海" in query:
        return "天气很热，37度"
    if "北京" in query:
        return "天气很冷，零下10度"
    if "合肥" in query:
        return "天气凉爽，23度"
    return "未知城市"


# 定义一个位置搜素工具
@tool(description="地理位置搜索工具")
def position_search_tool(query: str):
    """地理位置搜索工具"""
    print(f"===> call tools: position_search_tool, query: {query}")
    if "上海" in query:
        return "上海位于中国华东"
    if "北京" in query:
        return "北京位于中国华北"
    if "合肥" in query:
        return "合肥位于中国东北"
    return "未知城市"


tools = [weather_search_tool, position_search_tool]
tool_node = ToolNode(tools=tools)

llm = ChatOpenAI(model='qwen2.5:7b', temperature=0.3, base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))
# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def chatbot(state: ChatState):
    message = llm_with_tools.invoke(state["messages"])
    return {"messages": [message]}


def route_tools(state: ChatState):
    """判断是否需要继续（路由函数）"""
    messages = state['messages']
    # 获取最后一条消息
    last_message = messages[-1]
    # 如果 llm 调用了工具，表示大模型需要使用工具，则转到 tools 节点
    if last_message.tool_calls:
        # 返回节点的名字
        return "tools"
    # 否则，停止（回复用户）
    return END


graph_builder = StateGraph(ChatState)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("chatbot")

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    source="chatbot",
    path=route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"\
    path_map={"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")

# 创建内存检查点
memory = MemorySaver()
# 添加检查点
graph = graph_builder.compile(checkpointer=memory)

try:
    graph_png = graph.get_graph().draw_mermaid_png()
    with open("robot_v3.png", "wb") as f:
        f.write(graph_png)
except Exception as e:
    # This requires some extra dependencies and is optional
    print(e)


def stream_graph_updates(user_input: str):
    for event in graph.stream(
            input={"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "1"}},
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # final_state = graph.invoke(
        #     input={"messages": [HumanMessage(content=user_input)]},
        #     config={"configurable": {"thread_id": "1"}}
        # )
        # response = final_state["messages"][-1].content
        # print(response)

        stream_graph_updates(user_input)
    except Exception as e:
        # fallback if input() is not available
        print(e)
        break
