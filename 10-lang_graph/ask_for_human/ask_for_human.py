from pathlib import Path
from typing import TypedDict, Optional, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import StateGraph


# 定义状态结构
class TripState(TypedDict):
    user_input: Optional[str]
    user: Optional[str]
    place: Optional[str]
    date: Optional[str]
    next_question: Optional[str]
    complete: bool


# 实体提取节点
def extract_entities(state: TripState) -> TripState:
    user_input = state.get("user_input", "")

    # 简单规则提取（实际应用可替换为LLM或NER模型）
    if "我" in user_input or "本人" in user_input:
        state["user"] = "我"

    # 地点提取（示例简化）
    place_keywords = ["去", "到", "在"]
    for kw in place_keywords:
        if kw in user_input:
            start_idx = user_input.index(kw) + len(kw)
            state["place"] = user_input[start_idx:].split()[0].strip("，。！？")
            break

    # 时间提取（示例简化）
    time_keywords = ["周末", "下周", "明天", "后天"]
    for kw in time_keywords:
        if kw in user_input:
            state["date"] = kw
            break

    return state


# 完整性检查节点
def check_completeness(state: TripState) -> TripState:
    missing_fields = []

    if not state.get("user"):
        missing_fields.append("请问怎么称呼您？")
    if not state.get("place"):
        missing_fields.append("你想去哪里玩呢？")
    if not state.get("date"):
        missing_fields.append("打算什么时候出发？")

    if not missing_fields:
        state["complete"] = True
        state["next_question"] = None
    else:
        state["next_question"] = missing_fields[0]
        state["complete"] = False

    return state


# 构建工作流
workflow = StateGraph(TripState)

# 添加节点
workflow.add_node("extract", extract_entities)
workflow.add_node("check", check_completeness)

# 设置边连接
workflow.add_edge("extract", "check")


# 条件分支
def route_decision(state: TripState) -> Literal["end", "ask_question"]:
    return "end" if state["complete"] else "ask_question"

# 添加终止节点
def final_response(state: TripState) -> TripState:
    return state

workflow.add_node("final_response", final_response)
workflow.add_node("ask_question", lambda x: x)  # 仅传递状态


workflow.add_conditional_edges(
    "check",
    route_decision,
    {
        "ask_question": "ask_question",  # 继续循环
        "end": "final_response"  # 生成最终响应
    }
)

workflow.add_edge("final_response", END)
workflow.add_edge("ask_question", END)  # 中断返回用户


# 设置入口点
workflow.set_entry_point("extract")

# 编译执行
app = workflow.compile(checkpointer=MemorySaver())

if not Path("./ask_for_human.png").exists():
    graph = app.get_graph().draw_mermaid_png()
    with open("ask_for_human.png", "wb") as f:
        f.write(graph)

if __name__ == '__main__':
    response = app.invoke(input={"user_input": "我想去北京玩"}, config={"configurable": {"thread_id": 1}, "recursion_limit": 10})
    print(response)

    response = app.invoke(input={"user_input": "明天出发"}, config={"configurable": {"thread_id": 1}, "recursion_limit": 10})
    print(response)