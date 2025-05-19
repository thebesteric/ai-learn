from typing import List, Literal

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessageGraph
from pydantic import SecretStr, BaseModel

prompt = """
你的工作是从用户那里获取他们想要创建那种类型的提示词模板的信息。

你应该从用户那里获取以下信息：

- 提示的目的是什么？
- 将向提示词模板传递哪些变量？
- 输出需要有哪些限制？
- 输出的结果必须要遵守的要求？

如果你无法辨别这些信息，请用户进行澄清，不要试图猜测。
在你能够识别到以上信息后，调用相关工具。
"""


def get_messages_info(messages):
    return [SystemMessage(content=prompt)] + messages


@tool(description="提示词模板生成工具")
class PromptInstructions(BaseModel):
    # 目标
    objective: str
    # 变量
    variables: List[str]
    # 约束
    constraints: List[str]
    # 输出要求
    output_requirements: List[str]


tools = [PromptInstructions]
model = ChatOpenAI(model='qwen2.5:7b', base_url="http://127.0.0.1:11434/v1", api_key=SecretStr("ollama"))
model_with_tools = model.bind_tools(tools)

chain = get_messages_info | model_with_tools

prompt_system = """根据需求，生成一个提示词模板：{reqs}"""


def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]['args']
            print(f"==> tool_call = {tool_call}")
        elif isinstance(m, ToolMessage):
            print(f"==> tool_message = {ToolMessage}")
            continue
        elif tool_call is None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs


prompt_gen_chain = get_prompt_messages | model


def get_state(messages) -> Literal["add_tool_message", "info", "__end__"]:
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"


workflow = MessageGraph()
workflow.add_node("info", chain)
workflow.add_node("prompt", prompt_gen_chain)


@workflow.add_node
def add_tool_message(state):
    return ToolMessage(content="Prompt generated!", tool_call_id=state[-1].tool_calls[0]["id"])


workflow.add_edge(START, "info")
workflow.add_conditional_edges("info", get_state)
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

graph_png = graph.get_graph().draw_mermaid_png()
with open("user_prompt_gen.png", "wb") as f:
    f.write(graph_png)

config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("User（输入 /q 退出）：")
    if user_input == "/q":
        break
    output = None
    for output in graph.stream([HumanMessage(content=user_input)], config=config, stream_mode="updates"):
        last_message = next(iter(output.values()))
        last_message.pretty_print()

    if output and "prompt" in output:
        print("Done!")

# 我需要制作一个收集客户满意度反馈表
# 目的就是收集客户满意度反馈信息
# 变量至少包含是客户的姓名、评分等级（1-5）、互动日期、客户的评论、提供的服务项目
# 输出的限制：不能包含客户的敏感信息
# 输出的要求：需要有良好的表格形式，包含上述每个变量的信息
