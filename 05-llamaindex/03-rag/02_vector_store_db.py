from chromadb.errors import InvalidCollectionException
import chromadb
from llama_index.core import ServiceContext, service_context, set_global_service_context
from llama_index.vector_stores.chroma import ChromaVectorStore

"""
https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/chroma/
pip install llama-index-vector-stores-chroma
"""

# 创建客户端
chroma_client = chromadb.PersistentClient()

try:
    # 获取已经存在的向量数据库
    chroma_collection = chroma_client.get_collection("quickstart")
    print("集合获取完毕", chroma_collection)
except InvalidCollectionException:
    # 如果集合不存在，则创建新的集合
    chroma_collection = chroma_client.create_collection("quickstart")
    print("集合创建完毕", chroma_collection)

# 定义向量存储
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)