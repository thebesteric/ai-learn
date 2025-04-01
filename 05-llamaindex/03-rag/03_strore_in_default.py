from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

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

# 检查存储目录是否存在
storage_dir = "./storage"
if os.path.exists(storage_dir):
    # 若存储目录存在，从存储中加载索引
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
else:
    # 若存储目录不存在，从指定目录读取文档，将数据加载到内存
    documents = SimpleDirectoryReader(input_dir="../data", required_exts=[".md"]).load_data()
    # 4、创建一个 VectorStoreIndex 对象，并使用文档数据来构建向量索引
    # 此索引将文档转换为向量，并存储向量，以便快速检索
    # 向量存储在内存中
    index = VectorStoreIndex.from_documents(documents)
    # 将索引持久化存储到本地的向量数据库
    index.storage_context.persist()

# 5、创建一个查询引擎，这个索引可以接收查询，并返回相关文档的响应
"""
默认会在同级目录下创建一个 storage 目录，来存储向量
"""
query_engine = index.as_query_engine()

rep = query_engine.query("xtuner是什么？")
print(rep)