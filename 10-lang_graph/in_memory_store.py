import os
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph
from langgraph.store.base import IndexConfig, BaseStore
from langgraph.store.memory import InMemoryStore
from langchain_community.embeddings import DashScopeEmbeddings

in_memory_store = InMemoryStore(
    index=IndexConfig(
        embed=DashScopeEmbeddings(model="text-embedding-v1"),
        dims=1536,
    )
)

model = init_chat_model("deepseek-chat", model_provider="deepseek")


def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    # namespace 是由两部分构成 memories + user_id
    namespace = ("memories", user_id)

    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])

    # Store new memories if the user asks the model to remember
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        # memory = "User name is Eric"
        store.put(namespace, str(uuid.uuid4()), {"data": last_message.content})

    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"
    response = model.invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=MemorySaver(), store=in_memory_store)

# 这是是 thread_id = 1 的线程
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
input_message = HumanMessage(content="Hi! I am Eric, please remember me.")
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# 我们先改变一下 config，使用一个新的线程，用户保持不变
config = {"configurable": {"thread_id": "2", "user_id": "1"}}
input_message = HumanMessage(content="what is my name?")
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

print("=========================================================")
namespace = ("memories", "1")
for memory in in_memory_store.search(namespace):
    print(f"memory.value: {memory.value}")
