from typing import Annotated
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt


class State(TypedDict):
    messages: Annotated[list, add_messages]


# 用于提取信息的工具函数，这里简单示例，实际需更复杂逻辑
@tool(description="信息提取工具")
def extract_info(query: str) -> str:
    # 简单模拟提取，实际需结合NLP技术
    result = "时间:无, 地点:无, 人物:无"
    if "昨天" in query:
        result = result.replace("时间:无", "时间:昨天")
    if "北京" in query:
        result = result.replace("地点:无", "地点:北京")
    if "小明" in query:
        result = result.replace("人物:无", "人物:小明")
    if "时间:无, 地点:无, 人物:无" == result:
        human_response = interrupt({"query": f"请提取这句话中的时间、地点、人物: {query}"})
        result = human_response["data"]
    return result + "，如果缺少信息，请询问用户补充对应信息。"


tools = [extract_info]
llm = ChatOpenAI(model='qwen2.5:7b', temperature=0.3, base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # 禁用并行工具调用，避免恢复时重复调用
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, {"configurable": {"thread_id": "1"}},
                              stream_mode="values"):
        for value in event.values():
            if "messages" in event:
                event["messages"][-1].pretty_print()


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print(e)
        break