from langchain_community.tools.tavily_search import TavilySearchResults

# 创建TavilySearchResults工具，设置最大结果数为1
tools = [TavilySearchResults(max_results=1)]

from langchain import hub
from langchain_openai import ChatOpenAI
import asyncio
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, SecretStr

# 从 LangChain 的 Hub 中获取 prompt 模板，可以进行修改
# prompt = hub.pull("wfh/react-agent-executor")
# prompt.pretty_print()

prompt = """
### System:
You are a helpful assistant.

### Placeholder:
{messages}
"""

# 选择驱动代理的LLM，使用OpenAI的ChatGPT-4o模型
# llm = ChatOpenAI(model="gpt-4o")
llm = ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("ollama"), base_url="http://localhost:11434/v1")

# 创建一个REACT代理执行器，使用指定的LLM和工具，并应用从Hub中获取的prompt
agent_executor = create_react_agent(llm, tools, prompt=prompt)

# 调用代理执行器，询问“谁是美国公开赛的冠军”
#agent_executor.invoke({"messages": [("user", "谁是美国公开赛的获胜者")]})

import operator
from typing import Annotated, List, Tuple, TypedDict


# 定义一个 TypedDict 类 PlanExecute，用于存储输入、计划、过去的步骤和响应
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str # 最终结果


# 定义一个Plan模型类，用于描述未来要执行的计划
class Plan(BaseModel):
    """未来要执行的计划"""

    steps: List[str] = Field(
        description="需要执行的不同步骤，应该按顺序排列"
    )


from langchain_core.prompts import ChatPromptTemplate

# 创建一个计划生成的提示模板
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。不要添加任何多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息 - 不要跳过步骤。""",
        ),
        ("placeholder", "{messages}"),
    ]
)
# 使用指定的提示模板创建一个计划生成器，使用OpenAI的ChatGPT-4o模型
llm = ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("ollama"), base_url="http://localhost:11434/v1")
planner = planner_prompt | llm.with_structured_output(Plan)

# 调用计划生成器，询问“当前澳大利亚公开赛冠军的家乡是哪里？”
#planner.invoke({"messages": [("user", "现任澳网冠军的家乡是哪里?")]})

from typing import Union


# 定义一个响应模型类，用于描述用户的响应
class Response(BaseModel):
    """用户响应"""

    response: str


# 定义一个行为模型类，用于描述要执行的行为。该类继承自 BaseModel。
# 类中有一个属性 action，类型为 Union[Response, Plan]，表示可以是 Response 或 Plan 类型。
# action 属性的描述为：要执行的行为。如果要回应用户，使用 Response；如果需要进一步使用工具获取答案，使用 Plan。
class Act(BaseModel):
    """要执行的行为"""
    action: Union[Response, Plan] = Field(
        description="要执行的行为。如果要回应用户，使用Response。如果需要进一步使用工具获取答案，使用Plan。"
    )


# 创建一个重新计划的提示模板
replanner_prompt = ChatPromptTemplate.from_template(
"""
对于给定的目标，提出一个简单的逐步计划。这个计划应该包含独立的任务，如果正确执行将得出正确的答案。不要添加任何多余的步骤。最后一步的结果应该是最终答案。确保每一步都有所有必要的信息 - 不要跳过步骤。

你的目标是：
{input}

你的原计划是：
{plan}

你目前已完成的步骤是：
{past_steps}

相应地更新你的计划。
如果不需要更多步骤并且可以返回给用户，那么就这样直接返回给用户。如果需要，填写计划。只添加仍然需要完成的步骤。不要返回已完成的步骤作为计划的一部分。
当已完成的步骤中包含最终答案时，不要添加任何额外的步骤，直接返回给用户
"""
)

# 使用指定的提示模板创建一个重新计划生成器
replanner = replanner_prompt | ChatOpenAI(model="qwen2.5:7b", temperature=0.0, api_key=SecretStr("ollama"), base_url="http://localhost:11434/v1").with_structured_output(Act)

from typing import Literal


# 定义一个异步主函数
async def main():
    # 定义一个异步函数，用于生成计划步骤
    async def plan_step(state: PlanExecute):
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}

    # 定义一个异步函数，用于执行步骤
    async def execute_step(state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""对于以下计划：
{plan_str}\n\n你的任务是执行第{1}步，{task}。"""
        agent_response = await agent_executor.ainvoke(
            {"messages": [("user", task_formatted)]}
        )
        return {
            "past_steps": state["past_steps"] + [(task, agent_response["messages"][-1].content)],
        }

    # 定义一个异步函数，用于重新计划步骤
    async def replan_step(state: PlanExecute):
        output = await replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    # 定义一个函数，用于判断是否结束
    def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"

    from langgraph.graph import StateGraph, START

    # 创建一个状态图，初始化PlanExecute
    workflow = StateGraph(PlanExecute)

    # 添加计划节点
    workflow.add_node("planner", plan_step)

    # 添加执行步骤节点
    workflow.add_node("agent", execute_step)

    # 添加重新计划节点
    workflow.add_node("replan", replan_step)

    # 设置从开始到计划节点的边
    workflow.add_edge(START, "planner")

    # 设置从计划到代理节点的边
    workflow.add_edge("planner", "agent")

    # 设置从代理到重新计划节点的边
    workflow.add_edge("agent", "replan")

    # 添加条件边，用于判断下一步操作
    workflow.add_conditional_edges(
        "replan",
        # 传入判断函数，确定下一个节点
        should_end,
    )

    # 编译状态图，生成LangChain可运行对象
    app = workflow.compile()

    # 将生成的图片保存到文件
    # graph_png = app.get_graph().draw_mermaid_png()
    # with open("plan_execute.png", "wb") as f:
    #     f.write(graph_png)

    # 设置配置，递归限制为50
    config = {"recursion_limit": 50}
    # 输入数据
    inputs = {"input": "2024年巴黎奥运会100米自由泳决赛冠军的家乡是哪里?请用中文答复"}
    # 异步执行状态图，输出结果
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


# 运行异步函数
asyncio.run(main())
