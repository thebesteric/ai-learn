import streamlit as st
import chromadb
from chromadb.errors import InvalidCollectionException
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

from llama_index.vector_stores.chroma import ChromaVectorStore

st.set_page_config(page_title="LLamaIndex RAG Demo", page_icon="🦙", layout="wide")
st.title("LLamaIndex RAG Demo")

@st.cache_resource
def init_model():
    """
    初始化模型
    """
    # 初始化 HuggingFaceEmbedding 对象，用于将文本转换为向量
    embed_model_name = "/Users/wangweijun/LLM/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    Settings.embed_model = embed_model

    # 使用 HuggingFaceLLM 加载本地大模型
    model_name = "/Users/wangweijun/LLM/models/Qwen/Qwen2___5-0___5B-Instruct"
    llm = HuggingFaceLLM(
        model_name=model_name,
        tokenizer_name=model_name,
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True}
    )

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
    # todo 可以考虑使用 chroma_client.get_or_create_collection
    if chroma_collection.count() == 0:
        # 若存储目录不存在，从指定目录读取文档，将数据加载到内存
        documents = SimpleDirectoryReader(input_dir="../data", required_exts=[".md"]).load_data()

        # 创建节点解析器
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        # 将文档解析为节点
        base_node = node_parser.get_nodes_from_documents(documents=documents)

        # 创建一个 VectorStoreIndex 对象，并使用文档数据来构建向量索引
        # 此索引将文档转换为向量，并存储向量，以便快速检索
        # 向量存储在 chroma 中
        # index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, storage_context=storage_context)
        # 根据 node 节点，创建 index
        index = VectorStoreIndex(nodes=base_node, vector_store=vector_store, storage_context=storage_context)

        # 将索引持久化存储到本地的向量数据库
        index.storage_context.persist()
        print("索引创建完毕")

    # 尝试从向量存储中加载索引
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )

    # 5、创建一个查询引擎，这个索引可以接收查询，并返回相关文档的响应
    query_engine = index.as_query_engine()

    return query_engine

# 检查是否需要初始化模型
if "query_engine" not in st.session_state:
    st.session_state["query_engine"] = init_model()

def greet(question):
    query_engine = st.session_state["query_engine"]
    response = query_engine.query(question)
    return response

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，我是你的智能助理，请问有什么可以帮到你的？"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "你好，我是你的智能助理，请问有什么可以帮到你的？"}]

st.sidebar.button("清除聊天记录", on_click=clear_chat_history)

def generate_llama_index_response(prompt_input):
    return greet(prompt_input)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            response = generate_llama_index_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    messages = {"role": "assistant", "content": response}
    st.session_state.messages.append(messages)

# 环境搭建：
# pip install streamlit
# pip install watchdog

# 注意：
# st.set_page_config(page_title="LLamaIndex RAG Demo", page_icon="🦙", layout="wide")
# st.title("LLamaIndex RAG Demo")
# 必须在最前面，前面也不能包含任何文档级别的注释
# 所有文档级别的注释，都会被首页加载

# 运行命令：streamlit run 06_streamlit_app.py
