from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import MessageGraph

"""
### 了解 MessageGraph 消息图
MessageGraph 主要用于聊天机器人的场景，每个节点都可以返回消息列表
"""

# 创建一个 Message 图的实例
build = MessageGraph()

# 给图添加一个名为 "chatbot" 的节点，该节点接收 state 状态，并返回 AIMessage 类型的消息列表
# AI 消息列表中包含一个 ToolMessage，该 ToolMessage 包含一个名为 "search" 的工具，参数为 {"query": "X"}
build.add_node(
    "chatbot",
    lambda state: [
        AIMessage(
            content="Hello!",
            tool_calls=[{"id": "123", "name": "search", "args": {"query": "X"}}]
        )
    ]
)

# 给图添加一个名为 "search" 的节点，该节点接收 state 状态，并返回 ToolMessage 类型的消息列表
# 该工具的消息内容为 "Searching..."，调用的工具 ID 是 "123"
build.add_node(
    "search",
    lambda state: [
        ToolMessage(content="Searching...", tool_call_id="123")
    ]
)

# 设置入口节点为 "chatbot"
build.set_entry_point("chatbot")

# 添加一条从 "chatbot" 到 "search" 的边
build.add_edge("chatbot", "search")

# 设置结束节点为 "search"
build.set_finish_point("search")

graph = build.compile()
result = graph.invoke([HumanMessage(content="Hi there! Can you help me search for X?")])
print(result)
