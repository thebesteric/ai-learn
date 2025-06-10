import re
from typing import Union

from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_experimental.utilities import PythonREPL

from ..utils.callback_handler import ColoredPrintHandler
from ..utils.print_utils import CODE_COLOR
from excel_tools import get_first_n_rows


class PythonCodeParser(BaseOutputParser):
    """从 OpenAI 返回的文本中提取 Python 代码。"""

    @staticmethod
    def __remove_marked_lines(input_str: str) -> str:
        lines = input_str.strip().split('\n')
        if lines and lines[0].strip().startswith('```'):
            del lines[0]
        if lines and lines[-1].strip().startswith('```'):
            del lines[-1]
        ans = '\n'.join(lines)
        return ans

    def parse(self, text: str) -> str:
        # 使用正则表达式找到所有的Python代码块
        python_code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)

        # 从返回结果提取出 Python 代码文本
        python_code = None
        if len(python_code_blocks) > 0:
            python_code = python_code_blocks[0]
        python_code = self.__remove_marked_lines(python_code)
        return python_code


class ExcelAnalyser:
    """
    从ExceL文件中提取信息或分析数据（基于Python 代码实现）。
    输人中必须包含文件的完整路经和具体分析方式和分析依据，阈值常量等。
    """

    def __init__(self, llm: Union[BaseLanguageModel, BaseChatModel], prompt_file="./prompts/tools/excel_analyser.txt", verbose=False):
        self.llm = llm
        self.prompt = PromptTemplate.from_file(prompt_file)
        self.verbose = verbose
        self.verbose_handler = ColoredPrintHandler(CODE_COLOR)

    def analyse(self, query, filename):
        """分析一个 Excel 文件的内容"""
        inspections = get_first_n_rows(filename, 3)

        code_parser = PythonCodeParser()
        chain = self.prompt | self.llm | StrOutputParser

        # 调用模型生成 Python 代码
        response = ""
        for c in chain.stream({
            "query": query,
            "filename": filename,
            "inspections": inspections
        }, config=RunnableConfig(callbacks=[self.verbose_handler] if self.verbose else [])):
            response += c

        code = code_parser.parse(response)

        if code:
            ans = query + "\n" + PythonREPL().run(code)
            return ans

        return "没有找到可执行的 Python 代码"

    def as_tool(self):
        return StructuredTool.from_function(
            func=self.analyse,
            name="AnalyseExcel",
            description=self.__class__.__doc__.replace("\n", "")
        )
