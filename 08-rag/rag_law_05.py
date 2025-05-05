import json
import re
import time
from pathlib import Path
import streamlit as st
import chromadb
import numpy as np
from llama_index.core import Settings, VectorStoreIndex, StorageContext, PromptTemplate, get_response_synthesizer
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.vllm import Vllm
from llama_index.vector_stores.chroma import ChromaVectorStore
from modelscope.utils.plugins import storage
from llama_index.core.postprocessor import SentenceTransformerRerank  # 新增重排序组件

# ================== Streamlit 页面配置 ==================
st.set_page_config(
    page_title="智能劳动法咨询助手",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)


def disable_streamlit_watcher():
    """Patch Streamlit to disable file watcher"""

    def _on_script_changed(_):
        return

    from streamlit import runtime
    runtime.get_instance()._on_script_changed = _on_script_changed


class OpenAIConfig:
    # 使用 VLLM 推理
    # API_BASE = "http://localhost:8000/v1"
    # LLM_MODEL_PATH = "/home/ubuntu/llm/models/Qwen/Qwen2.5-7B-Instruct-identity"

    # 使用 Ollama 推理
    API_BASE = "http://localhost:11434/v1"
    LLM_MODEL_PATH = "qwen2.5:7b"
    API_KEY = "no-key-required"
    TIME_OUT = 60


# 使用 OpenAI 推理
def init_open_ai_llm():
    return OpenAILike(
        model=OpenAIConfig.LLM_MODEL_PATH,
        api_base=OpenAIConfig.API_BASE,
        api_key=OpenAIConfig.API_KEY,
        temperature=0.3,
        max_tokens=4096,
        timeout=OpenAIConfig.TIME_OUT,
        is_chat_model=True,
        additional_kwargs={"stop": ["<|im_end|>"]}
    )


class Config:
    EMBEDDING_MODEL_PATH = "/Users/wangweijun/LLM/models/text2vec-base-chinese-sentence"
    # EMBEDDING_MODEL_PATH = "/Users/wangweijun/LLM/models/bge-large-zh-v1.5"
    LLM_MODEL_PATH = "/Users/wangweijun/LLM/models/Qwen/Qwen2.5-7B-Instruct"
    RERANK_MODEL_PATH = r"/Users/wangweijun/LLM/models/bge-reranker-large"  # 新增重排序模型路径

    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./chroma_db"
    PERSIST_DIR = "./storage"
    COLLECTION_NAME = "chinese_labor_law_collection"

    TOP_K = 15
    RERANK_TOP_K = 5  # 重排序后保留数量
    MIN_RERANK_SCORE = 0.4  # 重排序后保留的最小分数

    # RESPONSE_TEMPLATE = PromptTemplate((
    #     "<|im_start|>system\n"
    #     "您是中国劳动法领域专业助手，必须严格遵循以下规则：\n"
    #     "1.仅使用提供的法律条文回答问题\n"
    #     "2.若问题与劳动法无关或超出知识库范围，明确告知无法回答\n"
    #     "3.引用条文时标注出处\n\n"
    #     "可用法律条文（共{context_count}条）：\n{context_str}\n<|im_end|>\n"
    #     "<|im_start|>user\n问题：{query_str}<|im_end|>\n"
    #     "<|im_start|>assistant\n"
    # ))


@st.cache_resource(show_spinner="初始化模型中...")
def init_models():
    """ 初始化模型 """
    # 使用 HuggingFaceEmbedding 加载 embedding 模型
    embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBEDDING_MODEL_PATH,
    )

    # 使用 HuggingFaceLLM 加载 llm 模型
    llm = HuggingFaceLLM(
        model_name=Config.LLM_MODEL_PATH,
        tokenizer_name=Config.LLM_MODEL_PATH,
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True},
        generate_kwargs={"temperature": 0.3}
    )

    # 使用 OpenAI 加载 llm 模型
    # llm = init_open_ai_llm()

    # 初始化重排序器（新增）
    reranker = SentenceTransformerRerank(
        model=Config.RERANK_MODEL_PATH,
        top_n=Config.RERANK_TOP_K
    )

    # 全局配置
    Settings.embed_model = embed_model
    Settings.llm = llm

    # 验证 embedding 模型
    test_embedding = embed_model.get_text_embedding("测试文本")
    print(f"Embedding 纬度验证：{len(test_embedding)}")

    # 返回 embedding 模型，llm 模型, reranker 模型
    return embed_model, llm, reranker


def load_and_validate_json_files(data_dir: str) -> list[dict]:
    # 由于每条条款就是一个 document，我们不需要进行拆分为多个 node，所以就不使用 SimpleDirectoryReader 了
    # json_files = list(Path(data_dir).glob("*.json"))
    json_files = list(Path(data_dir).glob("train_lobor_law.json"))
    assert json_files, f"未找到 {data_dir} 目录下的 json 文件"

    all_data = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"{json_file.name} 文件，应该为列表结构")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"{json_file.name} 文件的列表中每个元素应该为字典结构")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"{json_file.name} 文件的字典中 \"{k}\" 的的值应该为字符串")
                    all_data.append(
                        {
                            "content": item,
                            "metadata": {
                                "source": json_file.name
                            }
                        }
                    )
            except Exception as e:
                raise RuntimeError(f"加载 {json_file.name} 失败：{str(e)}")

    print(f"成功加载 {len(all_data)} 个法律文件")
    return all_data


def create_nodes(raw_data: list[dict]) -> list[TextNode]:
    nodes = []
    for entry in raw_data:
        # 字典，key 为条款名称，value 为条款内容
        law_dict = entry["content"]
        # 条款来源
        source_file = entry["metadata"]["source"]

        for title, content in law_dict.items():
            # 使用文件来源和条款名称作为 ID（唯一标识符)
            node_id = f"{source_file}::{title}"
            # 切割 title，获取法律名称和条款名称
            parts = title.split(" ", 1)  # 1 表示做多分割一次，也就是无论多个个空格，只会产生一个长度为 2 的列表
            # 法律名称
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            # 条款名称
            law_clause = parts[1] if len(parts) > 1 else "未知条款"
            # 创建节点
            node = TextNode(
                id_=node_id,
                text=content,
                metadata={
                    "law_name": law_name,
                    "law_clause": law_clause,
                    "title": title,
                    "source_file": source_file,
                    "content_type": "legal_clause"
                }
            )
            nodes.append(node)
    print(f"成功创建 {len(nodes)} 个文本节点")
    return nodes


@st.cache_resource(show_spinner="加载知识库中...")
def init_vector_db(nodes: list[TextNode]) -> VectorStoreIndex:
    # 创建 chroma 客户端，使用持久化存储
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    # 创建或者获取 chroma 集合
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        # 使用 cosine 余弦相似度算法
        metadata={"hnsw:space": "cosine"}
    )
    # 确保上下文正确初始化
    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    # 判断是否需要创建索引
    if chroma_collection.count() == 0 and nodes is not None:
        print("创建索引...")
        # 将节点存储到 storage 中，store_text 表示是否存储文档内容
        storage_context.docstore.add_documents(nodes, store_text=True)
        # 将文档存储中的数据保存到指定目录
        storage_context.persist(Config.PERSIST_DIR)
        # 创建索引
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,  # 将索引的数据就会和之前存储的文档数据关联起来，并且可以利用相同的存储机制进行持久化和管理
            show_progress=True
        )
        # 将索引生成的数据持久化到指定目录
        index.storage_context.persist(Config.PERSIST_DIR)
    else:
        print("加载索引...")
        storage_context = StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection),
            persist_dir=Config.PERSIST_DIR
        )
        # 加载索引
        index = VectorStoreIndex.from_vector_store(
            vector_store=storage_context.vector_store,
            embed_model=Settings.embed_model,
            storage_context=storage_context,
        )

    doc_count = len(storage_context.docstore.docs)
    print(f"存储的文档数量：{doc_count}")

    if doc_count > 0:
        sample_key = next(iter(storage_context.docstore.docs.keys()))
        print(f"示例节点 ID：{sample_key}")
    else:
        print("警告，文档存储为空，请核实")

    # 返回索引
    return index


# ================== 界面组件 ==================
def init_chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg.get("cleaned", msg["content"])  # 优先使用清理后的内容

        with st.chat_message(role):
            st.markdown(content)

            # 如果是助手消息且包含 COT 思维链
            if role == "assistant" and msg.get("think"):
                with st.expander("📝 模型思考过程（历史对话）"):
                    for think_content in msg["think"]:
                        st.markdown(f'<span style="color: #808080">{think_content.strip()}</span>',
                                    unsafe_allow_html=True)

            # 如果是助手消息且有参考依据（需要保持原有参考依据逻辑）
            if role == "assistant" and "reference_nodes" in msg:
                show_reference_details(msg["reference_nodes"])


def show_reference_details(nodes):
    with st.expander("查看支持依据"):
        for idx, node in enumerate(nodes, 1):
            meta = node.node.metadata
            st.markdown(f"**[{idx}] {meta['title']}**")
            st.caption(f"来源文件：{meta['source_file']} | 法律名称：{meta['law_name']}")
            st.markdown(f"相关度：`{node.score:.4f}`")
            # st.info(f"{node.node.text[:300]}...")
            st.info(f"{node.node.text}")


# 关键字过滤函数
def is_legal_question(text: str) -> bool:
    """判断问题是否属于法律咨询"""
    legal_keywords = ["劳动法", "合同", "工资", "工伤", "解除", "赔偿"]
    return any(keyword in text for keyword in legal_keywords)


def main():
    # 禁用 Streamlit 文件热重载
    disable_streamlit_watcher()
    st.title("⚖️ 智能劳动法咨询助手")
    st.markdown("欢迎使用劳动法智能咨询系统，请输入您的问题，我们将基于最新劳动法律法规为您解答。")

    # 初始化会话状态
    if "history" not in st.session_state:
        st.session_state.history = []

    embed_model, llm, reranker = init_models()

    # 不存在向量数据库，菜进行初始化数据
    if not Path(Config.VECTOR_DB_DIR).exists():
        print("初始化数据...")
        raw_data = load_and_validate_json_files(Config.DATA_DIR)
        nodes = create_nodes(raw_data)
    else:
        # 已经存在数据，不需要加载
        nodes = None

    print("\n初始化向量存储...")
    start_time = time.time()
    index = init_vector_db(nodes)
    print(f"索引加载完成，耗时 {time.time() - start_time:.2f} 秒")

    # 创建检索器和响应合成器（修改部分）
    retriever = index.as_retriever(
        similarity_top_k=Config.TOP_K  # 扩大初始检索数量
    )
    response_synthesizer = get_response_synthesizer(
        # text_qa_template=Config.RESPONSE_TEMPLATE,
        verbose=True
    )

    # 初始化聊天界面
    init_chat_interface()

    if prompt := st.chat_input("请输入劳动法相关问题"):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 处理查询
        with st.spinner("正在分析问题..."):
            start_time = time.time()

            # 检索流程
            initial_nodes = retriever.retrieve(prompt)
            reranked_nodes = reranker.postprocess_nodes(initial_nodes, query_str=prompt)

            # 过滤节点
            filtered_nodes = [node for node in reranked_nodes if node.score > Config.MIN_RERANK_SCORE]

            if not filtered_nodes:
                response_text = "⚠️ 未找到相关法律条文，请尝试调整问题描述或咨询专业律师。"
            else:
                # 生成回答
                response = response_synthesizer.synthesize(prompt, nodes=filtered_nodes)
                response_text = response.response

            # 显示回答
            with st.chat_message("assistant"):
                # 提取思维链内容并清理响应文本
                think_contents = re.findall(r'<think>(.*?)</think>', response_text, re.DOTALL)
                cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

                # 显示清理后的回答
                st.markdown(cleaned_response)

                # 如果有思维链内容则显示
                if think_contents:
                    with st.expander("📝 模型思考过程（点击展开）"):
                        for content in think_contents:
                            st.markdown(f'<span style="color: #808080">{content.strip()}</span>',
                                        unsafe_allow_html=True)

                # 显示参考依据（保持原有逻辑）
                # show_reference_details(filtered_nodes[:3])
                show_reference_details(filtered_nodes)

            # 添加助手消息到历史（需要存储原始响应）
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,  # 保留原始响应
                "cleaned": cleaned_response,  # 存储清理后的文本
                "think": think_contents  # 存储思维链内容
            })


if __name__ == "__main__":
    # streamlit run rag_law_05.py --server.address=127.0.0.1 --server.fileWatcherType=none
    main()
