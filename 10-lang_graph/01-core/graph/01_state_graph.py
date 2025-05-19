from langgraph.constants import START
from langgraph.graph import StateGraph


"""
### 了解 StateGraph 状态图
StateGraph 可以给节点传递状态信息，节点可以返回新的状态信息
> state 后面会和 checkpointer 关联，checkpointer 可以保存 state 状态信息
"""


# 定义一个节点，接收状态和配置，返回新的状态
def my_mode(state, config):
    return {
        "x": state["x"] + 1,
        "y": state["y"] + 2,
    }

# 创建一个状态图构建的 builder，使用字典作为状态类型
build = StateGraph(dict)

# 添加一个节点，节点名为 "my_mode"，接收状态和配置，返回新的状态
build.add_node("my_mode", my_mode)
# 添加一条从起始节点到 "my_mode" 的边
build.add_edge(START, "my_mode")

# 编译状态图，生成可运行图对象
graph = build.compile()
print(graph)

# 调用图对象，传入初始状态，返回新的状态
print(graph.invoke({"x": 0, "y": 0}))
