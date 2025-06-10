from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ..models.factory import ChatModelFactory


def write(query: str, verbose=False):
    """按用户要求撰写文档"""
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("你是专业的文档写手。你根据客户的要求，写一份文档。输出中文。"),
            HumanMessagePromptTemplate.from_template("{query}"),
        ]
    )
    llm = ChatModelFactory.get_default_model()
    chain = {"query": RunnablePassthrough()} | template | llm | StrOutputParser()
    return chain.invoke(query)
