import re
from typing import Tuple, Optional

from langchain.output_parsers import OutputFixingParser
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, render_text_description
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from pydantic import ValidationError

from action import Action
from ..utils.callback_handler import ColoredPrintHandler
from ..utils.print_utils import THOUGHT_COLOR

class ReActAgent:
    def __init__(self,
                 llm: BaseChatModel,
                 tools: list[BaseTool],
                 work_dir: str,
                 main_prompt_file: str,
                 max_thought_steps: int = 10):
        self.llm = llm
        self.tools = tools
        self.work_dir = work_dir
        self.main_prompt_file = main_prompt_file
        self.max_thought_steps = max_thought_steps

        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        # OutputFixingParser: 如果输出格式不正确，尝试修复
        self.robust_parser = OutputFixingParser.from_l1m(arser=self.output_parser, llm=llm)

        # 初始化 Prompt Template
        self.__init_prompt_templates()
        self.__init_chains()

        self.verbose_handler = ColoredPrintHandler(color=THOUGHT_COLOR)

    def __init_prompt_templates(self):
        with open(self.main_prompt_file, 'r', encoding='utf-8') as f:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template(f.read()),
                ]
            ).partiall(
                work_dir=self.work_dir,
                tools=render_text_description(self.tools),
                tool_names=','.join([tool.name for tool in self.tools]),
                format_instructions=self.output_parser.get_format_instructions(),
            )

    def __init_chains(self):
        self.main_chain = self.prompt | self.llm | StrOutputParser()

    def run(self, task: str, chat_history: ChatMessageHistory, verbose=False) -> str:
        """
        运行智能体
        :param task: 用户任务
        :param chat_history: 对话上下文信息（长时记忆）
        :param verbose: 是否显示详细信息
        :return:
        """
        # 初始化短时记忆：记录推理过程
        short_term_memory = []
        # 思考步数
        thought_step_count = 0
        # 最终回复
        reply = ""

        # 开始逐步思考
        while thought_step_count < self.max_thought_steps:
            if verbose:
                self.verbose_handler.on_thought_start(thought_step_count)

            # 执行一步思考
            action, response = self.__step(
                task=task,
                short_term_memory=short_term_memory,
                chat_history=chat_history,
                verbose=verbose,
            )

            # 如果是结束指令，执行最后一步
            if action.name == "FINISH":
                reply = self.__exec_action(action)
                break

            # 执行动作
            observation = self.__exec_action(action)

            if verbose:
                self.verbose_handler.on_tool_end(observation)

            # 更新短时记忆
            short_term_memory.append(
                self.__format_thought_observation(response, action, observation)
            )

            # 累加思考步数
            thought_step_count += 1

        # 如果思考步骤数达到上限，返回错误信息
        if thought_step_count >= self.max_thought_steps:
            reply = "抱歉，我没能完成您的任务。"

        # 更新长时记忆
        chat_history.add_user_message(task)
        chat_history.add_ai_message(reply)
        return reply

    def __step(self, task, short_term_memory, chat_history, verbose=False) -> Tuple[Action, str]:
        """
        执行一步思考
        :param task: 用户任务
        :param short_term_memory: 短时记忆
        :param chat_history: 对话上下文信息（长时记忆）
        :param verbose: 是否显示详细信息
        :return:
        """
        inputs = {
            "input": task,
            "agent_scratchpad": "\n".join(short_term_memory),
            "chat_history": chat_history.messages,
        }
        config = {
            "callbacks": [self.verbose_handler] if verbose else [],
        }

        response = ""
        for s in self.main_chain.stream(inputs, config=RunnableConfig(**config)):
            response += s

        # 提取 JSON 代码块
        json_action = self.__extract_json_action(response)

        # 带有容错的解析
        action = self.robust_parser.parse(
            json_action if json_action else response
        )
        return action, response

    def __exec_action(self, action):
        """
        执行工具
        :param action:
        :return:
        """
        tool = self.__find_tool(action.name)
        if tool is None:
            observation = (f"Error: 找不到工具或指令 `{action.name}`"
                           f"请从提供的工具/指令中选择，请确保按对象格式输出。")
        else:
            try:
                # 执行工具
                observation = tool.run(action.args)
            except ValidationError as e:
                # 如果参数校验出错
                observation = f"Validation Error in args: {str(e)}, args: {action.args}"
            except Exception as e:
                # 如果执行代码时出错
                observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"

        return observation

    @staticmethod
    def __extract_json_action(text: str) -> Optional[str]:
        """
        从回复中抽取 JSON 格式
        :param text: 模型响应
        :return: JSON 格式字符串
        """
        json_pattern = re.compile(r"```json(.*?)```", re.DOTALL)
        matches = json_pattern.findall(text)
        return matches[-1] if matches else None

    def __find_tool(self, tool_name):
        """
        根据名称查找工具
        :param tool_name: 工具名称
        :return: 工具对象
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    @staticmethod
    def __format_thought_observation(thought: str, action: Action, observation: str) -> str:
        ret = re.sub(r"```json(.*?)```", "", thought, flags=re.DOTALL)
        ret += "\n" + str(action) + "\n返回结果\n" + observation
        return ret
