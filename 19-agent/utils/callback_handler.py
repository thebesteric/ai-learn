from typing import Optional, Union, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk, LLMResult
from print_utils import color_print, RETURN_COLOR, OBSERVATION_COLOR, ROUND_COLOR


class ColoredPrintHandler(BaseCallbackHandler):
    def __init__(self, color: str):
        BaseCallbackHandler.__init__(self)
        self._color = color

    def on_llm_new_token(self,
                         token: str,
                         *,
                         chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
                         run_id: UUID,
                         parent_run_id: Optional[UUID] = None,
                         **kwargs: Any) -> Any:
        """流式调用时打印"""
        color_print(token, self._color, end="")
        return token

    def on_llm_end(self,
                  response: LLMResult,
                  **kwargs: Any) -> Any:
        """模型调用结束"""
        color_print("\n", self._color, end="")
        return response

    def on_tool_end(self,
                  output: Any,
                  **kwargs: Any) -> Any:
        """工具调用结束"""
        print()
        color_print("\n[Tool Return]", RETURN_COLOR)
        color_print(output, OBSERVATION_COLOR)
        return output

    @staticmethod
    def on_thought_start(index: int, **kwargs: Any):
        """自定义事件，当开始思考时打印"""
        color_print(f"\n[Thought {index}]", ROUND_COLOR)
        return index
