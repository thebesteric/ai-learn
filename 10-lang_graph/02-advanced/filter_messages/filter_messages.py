from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

@tool(description="天气搜索工具")
def search_weather(query: str):
    return ["今天是个好天气"]


tools = [search_weather]
tool_node = ToolNode(tools=tools)
model = ChatOpenAI(model='qwen2.5:7b', base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama")).bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    return "tools"


def filter_messages(messages: list):
    # 仅保留最后一条消息
    return messages[-1:]


def call_model(state: MessagesState):
    messages = filter_messages(state["messages"])
    response = model.invoke(input=messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")


app = workflow.compile(checkpointer=MemorySaver())
# messages_filter_png = app.get_graph().draw_mermaid_png()
# with open("messages_filter.png", "wb") as f:
#     f.write(messages_filter_png)

config = {"configurable": {"thread_id": "1"}}

input_message = HumanMessage(content="你好，我是 jack")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()


input_message = HumanMessage(content="我叫什么名字")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()