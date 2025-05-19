import operator
from typing import TypedDict, List, Annotated

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send


class OverallState(TypedDict):
    subjects: List[str]
    jokes: Annotated[List[str], operator.add]


def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']] is not None


build = StateGraph(OverallState)

build.add_node("generate_joke", lambda state: {"jokes": [f"Joke about {state['subjects']}"]})

build.add_conditional_edges(START, continue_to_jokes, {True: "generate_joke", False: END})
build.add_edge("generate_joke", END)

graph = build.compile()

result = graph.invoke({"subjects": ["cats", "dogs"]})
print(result)

graph_png = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_png)
