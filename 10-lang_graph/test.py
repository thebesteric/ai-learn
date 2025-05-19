import re

from typing import Literal, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from pydantic import SecretStr, BaseModel, Field


# ================== 强化状态类 ==================
class AgentState(BaseModel):
    messages: Annotated[list, Field(default_factory=list)]
    next_agent: Literal["researcher", "calculator", "writer", "done"] = "researcher"
    step_count: int = 0
    is_finalized: bool = False
    search_data: dict = Field(default_factory=dict)  # 新增搜索数据存储

    @classmethod
    def from_raw(cls, raw):
        """统一状态转换方法"""
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            return cls(
                messages=raw.get('messages', []),
                next_agent=raw.get('next_agent', 'researcher'),
                step_count=raw.get('step_count', 0),
                is_finalized=raw.get('is_finalized', False),
                search_data=raw.get('search_data', {})
            )
        raise ValueError(f"无效状态类型: {type(raw)}")

    def to_safe_dict(self):
        """安全转换为字典"""
        return {
            "messages": [msg.model_dump() for msg in self.messages],
            "next_agent": self.next_agent,
            "step_count": self.step_count,
            "is_finalized": self.is_finalized,
            "search_data": self.search_data
        }


# ================== 模型配置 ==================
# model = ChatOpenAI(
#     model='qwen2.5:7b',
#     base_url="http://127.0.0.1:11434/v1",
#     api_key=SecretStr("ollama")
# )
model = ChatOpenAI(
    model='glm-4-plus',
    base_url="https://open.bigmodel.cn/api/paas/v4",
    api_key=SecretStr("8550a02b44cd4a3badec476eb3323971.ENsprGJVt1RlYBkk")
)


# ================== 核心节点定义 ==================
def create_researcher():
    def agent(raw_state):
        state = AgentState.from_raw(raw_state)
        if state.is_finalized:
            return state

        try:
            # 获取搜索查询
            search_query = next(
                msg.content for msg in reversed(state.messages)
                if isinstance(msg, HumanMessage)
            )

            messages = convert_messages(state.messages)

            response = model.invoke(messages)
            print(f"\n[研究员输出] {response.content[:300]}...")

            # 解析下一步建议
            next_agent = "supervisor"
            if match := re.search(r'建议下一步\s*[:：]\s*(\w+)', response.content):
                suggestion = match.group(1).lower()
                if suggestion in {"researcher", "calculator", "writer", "done"}:
                    next_agent = suggestion

            return AgentState(
                messages=state.messages + [response],
                next_agent=next_agent,
                step_count=state.step_count + 1,
                is_finalized=False
            )
        except Exception as e:
            print(f"[研究员错误] {str(e)}")
            return AgentState(
                messages=state.messages,
                next_agent="supervisor",
                step_count=state.step_count + 1,
                is_finalized=True
            )

    return agent


def create_calculator():
    def agent(raw_state):
        state = AgentState.from_raw(raw_state)
        if state.is_finalized:
            return state

        try:
            messages = convert_messages(state.messages)
            messages.append(SystemMessage(
                content="请执行GDP增长率计算，需包含完整计算过程和公式，最后用'建议下一步：选项'格式结尾"
            ))

            response = model.invoke(messages)
            print(f"\n[计算器输出] {response.content[:200]}...")

            return AgentState(
                messages=state.messages + [response],
                next_agent=re.search(r'建议下一步\s*[:：]\s*(\w+)', response.content).group(1).lower(),
                step_count=state.step_count + 1,
                is_finalized=False
            )
        except Exception as e:
            print(f"[计算器错误] {str(e)}")
            return AgentState(
                messages=state.messages,
                next_agent="supervisor",
                step_count=state.step_count + 1,
                is_finalized=True
            )

    return agent


def create_writer():
    def agent(raw_state):
        state = AgentState.from_raw(raw_state)
        if state.is_finalized:
            return state

        try:
            messages = convert_messages(state.messages)
            messages.append(SystemMessage(
                content="请生成包含[END]标识的最终报告，要求：\n"
                        "1. 包含数据来源说明\n"
                        "2. 使用Markdown格式\n"
                        "3. 最后必须添加[END]标识\n"
                        "4. 必须引用搜索数据中的具体内容"
            ))

            response = model.invoke(messages)
            if "[END]" not in response.content:
                response.content += "\n[END]"

            return AgentState(
                messages=state.messages + [response],
                next_agent="done",
                step_count=state.step_count + 1,
                is_finalized=True
            )
        except Exception as e:
            print(f"[作家错误] {str(e)}")
            return AgentState(
                messages=state.messages,
                next_agent="supervisor",
                step_count=state.step_count + 1,
                is_finalized=True
            )

    return agent


# ================== 辅助函数 ==================
def convert_messages(raw_messages):
    converted = []
    for msg in raw_messages:
        if isinstance(msg, dict):
            try:
                msg_type = {
                    "user": HumanMessage,
                    "system": SystemMessage,
                    "assistant": AIMessage
                }[msg["role"]]
                converted.append(msg_type(content=msg.get("content", "")))
            except KeyError:
                converted.append(AIMessage(content=str(msg)))
        elif hasattr(msg, 'content'):
            converted.append(msg)
        else:
            converted.append(AIMessage(content=str(msg)))
    return converted


# ================== Supervisor节点 ==================
def supervisor(raw_state):
    state = AgentState.from_raw(raw_state)
    if state.is_finalized:
        return state

    try:
        print(f"\n[Supervisor] 当前步数: {state.step_count}")

        # 智能流程控制
        if state.step_count >= 8:
            print("[强制终止] 达到最大步数限制")
            return AgentState(
                messages=state.messages,
                next_agent="writer",
                step_count=state.step_count + 1,
                is_finalized=False
            )

        messages = convert_messages(state.messages)
        control_prompt = '''请严格选择下一步：
1. 需要补充数据 → researcher
2. 需要计算 → calculator 
3. 生成报告 → writer
4. 完成 → done

注意：必须优先选择需要完成的最关键步骤'''

        response = model.invoke(messages + [SystemMessage(content=control_prompt)])
        decision = re.search(r'\b(researcher|calculator|writer|done)\b', response.content.lower())

        new_agent = decision.group(1).lower() if decision else "done"
        # 强制writer前必须存在搜索数据
        if new_agent == "writer" and not state.search_data:
            new_agent = "researcher"
        # 强制done前必须生成报告
        if new_agent == "done" and not any("[END]" in msg.content for msg in messages):
            new_agent = "writer"

        return AgentState(
            messages=state.messages,
            next_agent=new_agent,
            step_count=state.step_count + 1,
            is_finalized=False
        )
    except Exception as e:
        print(f"[Supervisor错误] {str(e)}")
        return AgentState(
            messages=state.messages,
            next_agent="writer",
            step_count=state.step_count + 1,
            is_finalized=False
        )


# ================== 工作流配置 ==================
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", create_researcher())
builder.add_node("calculator", create_calculator())
builder.add_node("writer", create_writer())


# 定义条件边
def route_supervisor(state):
    return state.next_agent


builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "calculator": "calculator",
        "writer": "writer",
        "done": END
    }
)

# 添加节点到supervisor
for agent in ["researcher", "calculator", "writer"]:
    builder.add_edge(agent, "supervisor")

builder.set_entry_point("supervisor")
workflow = builder.compile()

# ================== 执行入口 ==================
if __name__ == "__main__":
    try:
        initial_state = AgentState(
            messages=[HumanMessage(content="成都市2025年GDP增长率分析")]
        )

        final_report = None
        for step in workflow.stream(initial_state):
            node_name, raw_state = step.popitem()
            current_state = AgentState.from_raw(raw_state)

            print(f"\n[系统状态] 当前节点: {node_name}")
            print(f"下一步: {current_state.next_agent}")
            print(f"步数: {current_state.step_count}")
            print(f"完成状态: {current_state.is_finalized}")

            # 捕获最终报告
            if node_name == "writer":
                final_report = next(
                    (msg.content for msg in reversed(current_state.messages)
                     if "[END]" in msg.content),
                    None
                )

            if current_state.is_finalized or node_name == "__end__":
                print("\n====== 流程完成 =====")
                if final_report:
                    print(final_report)
                else:
                    print("警告：未检测到完整报告")
                break

    except Exception as e:
        print(f"\n!!! 流程异常终止: {str(e)}")
        if 'current_state' in locals():
            print("最后状态:", current_state.to_safe_dict())
