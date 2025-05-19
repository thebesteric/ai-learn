import operator
from typing import TypedDict, Optional, List, Annotated, Dict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


# 定义日志结构
class Logs(TypedDict):
    id: str  # 日志 ID
    question: str  # 问题文本
    docs: Optional[List]  # 文档列表
    answer: str  # 回答文本
    grade: Optional[int]  # 评分
    grader: Optional[str]  # 评分人
    feedback: Optional[str]  # 反馈信息


# =====================================
# ========== 创建故障分析状态图 ==========
# =====================================
class FailureAnalysisState(TypedDict):
    docs: List[Logs]  # 日志列表
    failures: List[Logs]  # 失败的日志列表
    fa_summary: str  # 故障分析总结


def get_failures(state):
    docs = state["docs"]
    # 模拟获取失败的日志
    failures = [doc for doc in docs if "grade" in doc]
    return {"failures": failures}


def generate_fa_summary(state):
    failures = state["failures"]
    # 模拟生成故障分析总结
    fa_summary = "有 {} 个错误，RAG 文档检索质量差.".format(len(failures))
    return {"fa_summary": fa_summary}


fa_builder = StateGraph(FailureAnalysisState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_fa_summary)
fa_builder.add_edge(START, "get_failures")
fa_builder.add_edge("get_failures", "generate_summary")
fa_builder.add_edge("generate_summary", END)


# fa_app = fa_builder.compile()
# fa_graph_png = fa_app.get_graph().draw_mermaid_png()
# with open("fa_graph.png", "wb") as f:
#     f.write(fa_graph_png)


# =====================================
# ========== 问题分析状态图 ==========
# =====================================
class QuestionSummarizationState(TypedDict):
    docs: List[Logs]  # 日志列表
    qa_summary: str  # 问题总结
    report: str  # 报告


def generate_qs_summary(state):
    docs = state["docs"]
    # 模拟生成问题分析总结
    qa_summary = "Questions is focused on AI."
    return {"qa_summary": qa_summary}


def send_to_slack(state):
    qa_summary = state["qa_summary"]
    # 模拟发送报告到 Slack
    report = "QA Summary: {}".format(qa_summary)
    return {"report": report}


def format_report_for_slack(state):
    report = state["report"]
    # 模拟格式化报告
    formatted_report = "*最终报告*:\n{}".format(report)
    return {"report": formatted_report}


qs_builder = StateGraph(QuestionSummarizationState)
qs_builder.add_node("generate_summary", generate_qs_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_node("format_report_for_slack", format_report_for_slack)
qs_builder.add_edge(START, "generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", "format_report_for_slack")
qs_builder.add_edge("format_report_for_slack", END)


# qs_app = qs_builder.compile()
# qs_graph_png = qs_app.get_graph().draw_mermaid_png()
# with open("qs_graph.png", "wb") as f:
#     f.write(qs_graph_png)


class EntryGraphState(TypedDict):
    raw_logs: Annotated[List[Dict], operator.add]
    docs: Annotated[List[Logs], operator.add]
    fa_summary: str
    report: str


def convert_logs_to_docs(state):
    raw_logs = state["raw_logs"]
    docs = [Logs(**log) for log in raw_logs]
    return {"docs": docs}


entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("convert_logs_to_docs", convert_logs_to_docs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())
entry_builder.add_edge(START, "convert_logs_to_docs")
entry_builder.add_edge("convert_logs_to_docs", "failure_analysis")
entry_builder.add_edge("convert_logs_to_docs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

app_graph = entry_builder.compile()
app_graph_png = app_graph.get_graph().draw_mermaid_png()
with open("app_graph.png", "wb") as f:
    f.write(app_graph_png)

raw_logs = [
    {
        "id": "1",
        "question": "如何导入 ChatOpenAI?",
        "answer": "要导入 ChatOpenAI，可以使用：'from langchain_openai import ChatOpenAI'"
    },
    {
        "id": "2",
        "question": "如何使用 Chroma 向量存储?",
        "answer": "要使用 Chroma 向量存储，可以使用：'from langchain.vectorstores import Chroma'",
        "grade": 0,
        "grader": "文档相似性回顾",
        "feedback": "检索到的文档只讨论了向量存储概念，没有专门提到 Chroma 向量存储方式。"
    }
]

result = app_graph.invoke({"raw_logs": raw_logs}, debug=True)
print(result)
