import json
import os

from IPython.core.display import Markdown
from IPython.core.display_functions import display
from dotenv import load_dotenv
from openai import OpenAI

from tools.python_tools import python_inter, fig_inter, python_inter_tool, fig_inter_tool
from tools.mysql_tools import sql_inter, extract_data, sql_inter_tool, extract_data_tool
from tools.network_tools import get_answer, get_answer_github, get_answer_github_tool, get_answer_tool

tools = [python_inter_tool, fig_inter_tool, sql_inter_tool, extract_data_tool, get_answer_tool, get_answer_github_tool]


def print_code_if_exists(function_args):
    """
    如果存在代码片段，则打印代码
    :param function_args:
    :return:
    """

    def convert_to_markdown(language, code):
        return f"```{language}\n{code}\n```"

    if function_args.get("sql_query"):
        code = function_args["sql_query"]
        markdown_code = convert_to_markdown("sql", code)
        print("即将执行以下代码")
        display(markdown_code)

    elif function_args.get("py_code"):
        code = function_args["py_code"]
        markdown_code = convert_to_markdown("python", code)
        print("即将执行以下代码")
        display(markdown_code)


def create_function_response_message(messages: list, response) -> list:
    """
    调用外部工具，并更新消息列表
    :param messages: 原始消息列表
    :param response: 模型的响应对象，包含工具调用信息
    :return: messages，追加了外部工具运行结果后的消息列表
    """

    available_functions = {
        "python_inter": python_inter,
        "fig_inter": fig_inter,
        "sql_inter": sql_inter,
        "extract_data": extract_data,
        "get_answer": get_answer,
        "get_answer_github": get_answer_github
    }

    # 提取 function call messages
    function_call_messages = response.choices[0].message.tool_calls

    # 将 function call messages 追加到 messages 中
    messages.append(response.choices[0].message.model_dump())

    # 提取本次外部函数调用的每个任务请求
    for function_call_message in function_call_messages:
        # 提取工具名称
        tool_name = function_call_message.function.name
        # 提取工具参数
        tool_args = json.loads(function_call_message.function.arguments)
        # 获取函数对象
        function_to_call = available_functions[tool_name]
        # 打印代码
        print_code_if_exists(function_args=tool_args)

        # 调用函数
        try:
            tool_args["g"] = globals()
            # 调用外部函数
            function_response = function_to_call(**tool_args)
        except Exception as e:
            function_response = f"❌ 函数调用出错：{str(e)}"

        # 构建函数调用结果的消息
        messages.append(
            {
                "role": "tool",
                "content": function_response,
                "tool_call_id": function_call_message.id
            }
        )

        return messages


def chat_base(messages: list, client, model):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
    except Exception as e:
        print(f"❌ 调用模型出错：{str(e)}")
        return None

    if response.choices[0].finish_reason == "tool_calls":
        while True:
            messages = create_function_response_message(messages, response)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )
            if response.choices[0].finish_reason != "tool_calls":
                break

    return response


def save_markdown_to_file(content: str, filename_hint: str, directory: str = "research_task"):
    # 在当前项目目录下创建文件夹
    save_dir = os.path.join(os.getcwd(), directory)
    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 创建文件名
    filename = f"{filename_hint[:8]}.md"
    # 完整的文件路径
    file_path = os.path.join(save_dir, filename)
    # 将内容写入文件
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)
    print(f"文件已保存到: {file_path}")


def test():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("MODEL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    # response = chat_base([
    #     {"role": "user", "content": "帮我模拟一份数据，并绘制核密度分布图"}
    # ], client, model)
    # print(response.choices[0].message.content)

    # response = chat_base([
    #     {"role": "user", "content": "什么是大语言模型"}
    # ], client, model)
    # print(response.choices[0].message.content)

    response = chat_base([
        {"role": "user", "content": "我想了解一下 GitHub 上 Spring AI 这个项目"}
    ], client, model)
    print(response.choices[0].message.content)


class MiniManus:
    def __init__(self, api_key=None, base_url=None, model=None, messages=None):
        load_dotenv(override=True )
        self.api_key = api_key if api_key is not None else os.getenv("API_KEY")
        self.model = model if model is not None else os.getenv("MODEL")
        self.base_url = base_url if base_url is not None else os.getenv("BASE_URL")
        self.messages = messages if messages is not None else [
            {"role": "system", "content": "You are MiniManus a helpful assistant."}
        ]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            models = self.client.models.list()
            if models and self.model in [model.id for model in models.data]:
                print(f"✅ MiniManus 初始化完成，欢迎使用！")
            else:
                print(f"⚠️ MiniManus 初始化失败，请检查 API_KEY 和 BASE_URL 是否正确设置！")
        except Exception as e:
            print(f"❌ MiniManus 初始化失败，详细信息：{str(e)}")


    def chat(self):
        print("你好，我是 MiniManus，有什么需要帮助的？")
        while True:
            question = input("请输入你的问题（/q 退出）: ")
            if question.lower() in ["/q"]:
                print("再见！")
                break
            self.messages.append({"role": "user", "content": question})
            # 保留最近 20 条消息
            self.messages = self.messages[-20:]

            response = chat_base(self.messages, self.client, self.model)
            content = response.choices[0].message.content
            display(Markdown(content))
            print(f"MiniManus: {content}")
            self.messages.append(response.choices[0].message.model_dump())

    def research_task(self, question):
        prompt_style1 = """
        你是一名专业且细致的助手，你的任务是在用户提出问题之后，通过友好且有引导性的追问，更深入的理解用户真正的需求背景。这样你才能提供更加精准和更加有效的帮助。
        当用户提出一个宽泛或者不够明确的问题时，你应到积极主动的提出后续问题，应引导用户提供更多的背景和细节，以帮助你更加准确的回应。
        示例应到问题：
        用户提问示例：
        最近，在大模型技术领域，有一项非常热门的技术，叫 MCP，model context protocol，调用并深度总结，这项技术与 Function calling 之间的区别。
        
        你应该给出引导式回应示例：
        在比较 MCP 和 Function calling 之间的区别时，我可以通过以下几个方面来比较：
        - 定义和基本概念：MCP 和 Function calling 的基本原理和目标。
        - 工作机制：它们是如何处理用户的输入和模型的输出。
        - 应用场景：它们分别适用于哪些具体的应用场景。
        - 技术优势和局限性：它们在技术层面上的优势和局限性。
        - 生态和兼容性：它们是否能和现有的大模型和应用集成。
        - 未来发展趋势：它们在技术发展中的潜在趋势和未来的发展方向。
        请问你是否希望我特别关注某个方面，或者有特定的技术细节需要深入分析？
        
        再比如用户提出问题：
        请帮我详细整理，华为 910B 鲲鹏 920，如何部署 DeepSeek 模型？
        
        你应该给出的引导式回应示例：
        请提供以下详细信息，以便我能为您整理完整的部署指南：
        1. 你希望部署的 DeepSeek 模型具体是哪一个？（例如：DeepSeek-chat、DeepSeek-reasoner）
        2. 您的目标是在什么环境中部署 DeepSeek 模型？（例如：Linux 系统、Windows 系统、MacOS 系统）
        3. 是否有特定的深度学习框架要求？（例如：PyTorch、TensorFlow 等）
        4. 是否需要优化部署？（如使用昇腾 NPU 加速）
        5. 期望的使用场景？（如：推理、训练、微调等）
        请提供这些信息后，我将为您整理完整的部署指南。

        记住，保持友好而专业的态度，主动帮助用户明确需求，而不是直接给出不够精准的回答。
        
        begin!!
        
        现在用户提出的问题是：{}，请按照要求进行回复。
        """

        prompt_style2 = """
        你是一位知识渊博，擅长利用多种外部工具的资深研究员，当用户已明确提出具体需求：{}，现在你的任务是：
        首先明确用户的核心问题以及相关细节。
        尽可能调用可用的外部工具（例如：联网搜索工具 get_answer、GitHub 搜索工具 get_answer_github、本地代码运行工具 python_inter 以及其他工具），围绕用户给出的原始问题和补充细节，进行广泛而深入的进行信息收集。
        综合利用你从各种工具中获取的信息，提供详细、全面、专业且具有深度的解答。你的回答应尽量达到 2000 字以上，内容严谨准确且富有洞察力。
        
        示例流程：
        用户明确需求示例：
        我目前正在学习 ModelContextProtocol（MCP），主要关注它在 AI 模型开发领域中的具体应用场景、技术细节和一些业界最新的进展。
        
        你的回应流程示例：
        首先重述并确认用户的具体需求。
        明确你将调用哪些外部工具，例如：
        使用联网搜索工具查询官方或权威文档对 MCP 在 AI 模型开发领域的具体应用说明；
        调用 GitHub 搜索工具，寻找业界针对 MCP 技术项目；
        整理并分析通过工具获取的信息，形成一篇逻辑清晰、结构合理的深度报告。
        
        再比如用户需要编写数据分析报告示例：
        我想针对某电信公司过去一年的用户数据，编写一份详细的用户流失预测数据分析报告，报告需要包括用户流失趋势分析、流失用户特征分析、影响用户流失的关键因素分析，并给出未来减少用户流失的策略建议。
        你的回应流程示例： 
        明确并确认用户需求，指出分析内容包括用户流失趋势、流失用户特征、关键影响因素
        明确你将调用哪些外部工具，例如：
        使用数据分析工具对提供的用户数据进行流失趋势分析，生成趋势图表；
        使用代码执行环境（如调用 python_inter 工具）对流失用户进行特征分析，确定典型特征；
        通过统计分析工具识别影响用户流失的关键因素（如服务质量、价格敏感度、竞争对手促销），同时借助绘图工具（fig_inter）进行重要信息可视化展示；
        使用互联网检索工具检索行业内最新的客户保留策略与实践，提出有效的策略建议。
        
        记住，回答务必详细完整，字数至少在 2000 字以上，清晰展示你是如何运用各种外部工具进行深入研究并形成专业结论的。
        """

        response = self.client.chat.completions.create(model=self.model,
                                                       messages=[{"role": "user", "content": prompt_style1.format(question)}])
        content = response.choices[0].message.content
        display(Markdown(content))
        print(f"MiniManus: {content}")
        new_messages = [
            {"role": "user", "content": question},
            response.choices[0].message.model_dump()
        ]

        # 明确用户需求
        new_question = input("请输入您的补充说明（/q 退出）：")
        if new_question == "/q":
            print("再见！")
            return None
        else:
            new_messages.append({"role": "user", "content": prompt_style2.format(new_question)})
            second_response = chat_base(messages=new_messages, client=self.client,
                                        model=self.model)
            content = second_response.choices[0].message.content
            display(Markdown(content))
            print(f"MiniManus: {content}")
            # 保存到本地
            save_markdown_to_file(content, filename_hint=question)

    def clear_messages(self):
        self.messages = []

if __name__ == '__main__':
    mini_manus = MiniManus()

    # mini_manus.chat()

    mini_manus.research_task("我想了解一下大模型中 MCP 技术")
