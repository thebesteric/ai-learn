import chromadb
from chromadb.errors import InvalidCollectionException
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

from llama_index.vector_stores.chroma import ChromaVectorStore

"""
https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/
pip install llama-index-embeddings-huggingface

https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/chroma/
pip install llama-index-vector-stores-chroma
"""

# 1、初始化 HuggingFaceEmbedding 对象，用于将文本转换为向量
embed_model_name = "/Users/wangweijun/LLM/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

"""
将创建的嵌入模型赋值给全局设置的 Settings.embed_model 属性
这样在后续的索引构建和查询中就可以使用该模型进行嵌入计算
"""
Settings.embed_model = embed_model

# 2、使用 HuggingFaceLLM 加载本地大模型
model_name = "/Users/wangweijun/LLM/models/Qwen/Qwen2___5-0___5B-Instruct"
llm = HuggingFaceLLM(
    model_name=model_name,
    tokenizer_name=model_name,
    model_kwargs={"trust_remote_code": True},
    tokenizer_kwargs={"trust_remote_code": True}
)

"""
设置全局的 Settings.llm 属性
这样在后续的查询中就可以使用该模型进行大模型调用
"""
Settings.llm = llm

# 创建客户端
chroma_client = chromadb.PersistentClient(path="./chroma")

try:
    # 获取已经存在的向量数据库
    chroma_collection = chroma_client.get_collection("quickstart")
    print("集合获取完毕", chroma_collection)
except InvalidCollectionException:
    # 如果集合不存在，则创建新的集合，默认会在同级目录下创建一个 chroma 目录，来存储向量
    chroma_collection = chroma_client.create_collection("quickstart")
    print("集合创建完毕", chroma_collection)

# 定义向量存储
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# 创建存储上下文，使用 Chroma 向量存储
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 检查集合中是否有文档
if chroma_collection.count() == 0:
    # 若存储目录不存在，从指定目录读取文档，将数据加载到内存
    documents = SimpleDirectoryReader(input_dir="../data", required_exts=[".md"]).load_data()

    # 4、创建一个 VectorStoreIndex 对象，并使用文档数据来构建向量索引
    # 此索引将文档转换为向量，并存储向量，以便快速检索
    # 向量存储在 chroma 中
    index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, storage_context=storage_context)
    # 将索引持久化存储到本地的向量数据库
    index.storage_context.persist()
    print("索引创建完毕")

# 尝试从向量存储中加载索引
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)

# 5、创建一个查询引擎，这个索引可以接收查询，并返回相关文档的响应
"""
默认会在同级目录下创建一个 storage 目录，来存储向量
"""
query_engine = index.as_query_engine()

rep = query_engine.query("xtuner是什么？")
print(rep)
