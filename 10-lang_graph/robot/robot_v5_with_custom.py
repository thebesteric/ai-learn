import getpass
import json
import os
from typing import Annotated, TypedDict, Literal

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from pydantic import SecretStr


# if not os.environ.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

# tool = TavilySearchResults(max_results=2)
# tools = [tool]

@tool(description="搜索工具")
def web_search_tool(query: str):
    """搜索工具"""
    print(f"===> call tools: weather_search_tool, query: {query}")
    if "LangGraph" in query:
        return "LangGraph 发布于 2025年1月1日"
    return "未知城市"

@tool(description="请求来自人类的帮助")
def human_assistance_tool(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)


tools = [web_search_tool, human_assistance_tool]
tool_node = ToolNode(tools=tools)

llm = ChatOpenAI(model='qwen2.5:7b', temperature=0.3, base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))
# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


def chatbot(state: ChatState):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
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
    path=tools_condition,
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
    with open("robot_v5.png", "wb") as f:
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
            value["messages"][-1].pretty_print()

        # if "messages" in event:
        #     event["messages"][-1].pretty_print()


user_input = (
    "你能查一下 LangGraph 是什么时候发布的吗？"
    "当你有了答案，使用 human_assistance_tool 工具进行复核。"
)
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "2024年1月17日",
    },
)

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


graph.update_state(config, {"name": "LangGraph (library)"})
snapshot = graph.get_state(config)
print({k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})