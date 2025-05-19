from typing import TypedDict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph


class State(TypedDict):
    input: str

def step_1(state: State):
    print("===> step_1")


def step_2(state: State):
    print("===> step_2")


def step_3(state: State):
    print(f"===> step_3")


build = StateGraph(State)
build.add_node("step_1", step_1)
build.add_node("step_2", step_2)
build.add_node("step_3", step_3)
build.add_edge(START, "step_1")
build.add_edge("step_1", "step_2")
build.add_edge("step_2", "step_3")
build.add_edge("step_3", END)

# 静态断点
static_interrupt_app = build.compile(checkpointer=MemorySaver(), interrupt_before=["step_3"])
# static_interrupt_app_png = static_interrupt_app.get_graph().draw_mermaid_png()
# with open("static_interrupt_app.png", "wb") as f:
#     f.write(static_interrupt_app_png)

config = {"configurable": {"thread_id": "1"}}
for event in static_interrupt_app.stream({"input": "hello"}, config, stream_mode="values"):
   print(event)


user_approval = input("是否继续批准执行? [y/n]：")
if user_approval == "y":
    # 注意：如果使用 interrupt_before，此时不能在传入 input 了，否则图会重新执行
    for event in static_interrupt_app.stream(None, config, stream_mode="values"):
        print(event)
else:
    print("终止执行")