import os

from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings
from torch._functorch._aot_autograd.logging_utils import model_name

_ = load_dotenv(find_dotenv())


class ChatModelFactory:
    model_params = {
        "temperature": 0,
        "seed": 42,
    }

    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        if "gpt" in model_name:
            if not use_azure:
                return ChatOpenAI(model=model_name, **cls.model_params)
            else:
                return AzureChatOpenAI(
                    azure_deployment=model_name,
                    api_version="2024-05-01-preview",
                    **cls.model_params
                )

        elif model_name == "deepseek":
            # 换成开源模型试试
            # https://siliconflow.cn/
            # 一个 Model-as-a-Service 平台
            # 可以通过与 OpenAI API 兼容的方式调用各种开源语言模型。
            return ChatOpenAI(
                model="deepseek-chat",
                # 模型名称
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL"),
                **cls.model_params,
            )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

    @classmethod
    def get_default_model(cls):
        return cls.get_model("deepseek")


class EmbeddingModelFactory:
    @classmethod
    def get_model(cls, model_name: str, use_azure: bool = False):
        if model_name.startswith("text-embedding"):
            if not use_azure:
                return OpenAIEmbeddings(model=model_name)
            else:
                return AzureOpenAIEmbeddings(
                    azure_deployment=model_name,
                    api_version="2024-05-01-preview",
                )
        elif model_name == "dashscope":
            # return DashScopeEmbeddings(
            #     model="text-embedding-v3",
            #     dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            # )
            return OpenAIEmbeddings(
                model="text-embedding-v3",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=os.getenv("DASHSCOPE_BASE_URL")
            )
        else:
            raise NotImplementedError(f"Model {model_name} not implemented.")

    @classmethod
    def get_default_model(cls):
        return cls.get_model("dashscope")
