import operator
from typing import TypedDict, Annotated, Optional

from langgraph.constants import END, START
from langgraph.graph import StateGraph


class NumberState(TypedDict):
    nums: Annotated[list, operator.add]


def step_1(state):
    return {"nums": [1]}


def step_2(state):
    return {"nums": [2]}


def step_3(state):
    return {"nums": [3]}


workflow = StateGraph(NumberState)

workflow.add_node("step_1", step_1)
workflow.add_node("step_2", step_2)
workflow.add_node("step_3", step_3)

workflow.add_edge(START, "step_1")
workflow.add_edge("step_1", "step_2")
workflow.add_edge("step_2", "step_3")
workflow.add_edge("step_3", END)

app = workflow.compile()

for chunk in app.stream({"nums": []}, stream_mode="values"):
    print(chunk)

# {'nums': []}
# {'nums': [1]}
# {'nums': [1, 2]}
# {'nums': [1, 2, 3]}