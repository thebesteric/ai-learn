from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from pydantic import SecretStr

llm = ChatOpenAI(model='qwen2.5:7b', base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))


class ChatState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(ChatState)


def chatbot(state: ChatState):
    message = llm.invoke(state["messages"])
    return {"messages": [message]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

# 接下来，添加一个 entry 点。这告诉我们的图**每次运行时从何处开始工作**
graph_builder.add_edge(START, "chatbot")

# 同样，设置一个 finish 点。这指示图表 “每次运行此节点时，都可以退出。”
graph_builder.add_edge("chatbot", END)

# 最后，我们需要能够运行我们的图。为此，调用图构建器上的"compile()"方法。这将创建一个"CompiledGraph"，我们可以用它来调用我们的状态。
graph = graph_builder.compile()

try:
    graph_png = graph.get_graph().draw_mermaid_png()
    with open("robot_v1.png", "wb") as f:
        f.write(graph_png)
except Exception as e:
    # This requires some extra dependencies and is optional
    print(e)


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except Exception as e:
        # fallback if input() is not available
        print(e)
        break
