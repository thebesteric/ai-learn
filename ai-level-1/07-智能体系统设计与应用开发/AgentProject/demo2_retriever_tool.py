import os
import uuid

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# 加载HTML内容为一个文档对象
loader = WebBaseLoader("https://new.qq.com/rain/a/20240920A07Y5Y00")
docs = loader.load()
# print(docs)

# 分割文档
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
for doc in documents:
    if not doc.id:
        doc.id = str(uuid.uuid4())

# 向量化
# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v1", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
# )

# embeddings = OpenAIEmbeddings()

embeddings = HuggingFaceEmbeddings(model_name="/Users/wangweijun/llm/models/bge-large-zh-v1.5", model_kwargs = {'device': 'cpu'}, encode_kwargs = {'normalize_embeddings': False})
vector = FAISS.from_documents(documents, embeddings)

# 创建检索器
retriever = vector.as_retriever()

# 测试检索结果
print(retriever.get_relevant_documents("目前市场上苹果手机16的售价是多少？"))