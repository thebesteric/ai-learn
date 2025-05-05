import json
import time
from pathlib import Path

import chromadb
from llama_index.core import Settings, VectorStoreIndex, StorageContext, PromptTemplate, get_response_synthesizer
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.vllm import Vllm
from llama_index.vector_stores.chroma import ChromaVectorStore
from modelscope.utils.plugins import storage
from llama_index.core.postprocessor import SentenceTransformerRerank  # 新增重排序组件

"""
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-chroma
pip install llama-index-llms-openai-like
"""


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
        max_tokens=1024,
        timeout=OpenAIConfig.TIME_OUT,
        is_chat_model=True,
        additional_kwargs={"stop": ["<|im_end|>"]}
    )


"""
0、配置区
"""


class Config:
    EMBEDDING_MODEL_PATH = "/Users/wangweijun/LLM/models/text2vec-base-chinese-sentence"
    # EMBEDDING_MODEL_PATH = "/Users/wangweijun/LLM/models/bge-large-zh-v1.5"
    LLM_MODEL_PATH = "/Users/wangweijun/LLM/models/Qwen/Qwen2.5-7B-Instruct"
    RERANK_MODEL_PATH = r"/Users/wangweijun/LLM/models/bge-reranker-large"  # 新增重排序模型路径

    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./chroma_db"
    PERSIST_DIR = "./storage"
    COLLECTION_NAME = "chinese_labor_law_collection"

    TOP_K = 10
    RERANK_TOP_K = 3  # 重排序后保留数量

    RESPONSE_TEMPLATE = PromptTemplate((
        "<|im_start|>system\n"
        "您是中国劳动法领域专业助手，必须严格遵循以下规则：\n"
        "1.仅使用提供的法律条文回答问题\n"
        "2.若问题与劳动法无关或超出知识库范围，明确告知无法回答\n"
        "3.引用条文时标注出处\n\n"
        "可用法律条文（共{context_count}条）：\n{context_str}\n<|im_end|>\n"
        "<|im_start|>user\n问题：{query_str}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ))


"""
1、初始化模型
"""


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


"""
2、数据处理

思考：该项目是 RAG 法律条文助手，对检索的本地数据有以下几点要求
1. 法律依据要清晰，每个法律条款来单独说明
2. 每条法律都有唯一的来源
因为，该项目构建的本地知识库，应该按照每个法律条款进行划分
"""


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


"""
3、创建 node 节点
"""


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


"""
4、向量存储
"""


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


"""
5、主程序
"""


def main():
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
        text_qa_template=Config.RESPONSE_TEMPLATE,
        verbose=True
    )

    while True:
        query = input("请输入劳动法的相关问题：")
        if query == "/q":
            break

        # 执行查询
        print(f"查询中，请稍后...\n")

        # 执行检索-重排序-回答流程（新增重排序步骤）
        start_time = time.time()

        # 1. 初始检索
        initial_nodes = retriever.retrieve(query)
        retrieval_time = time.time() - start_time

        # 保存初始分数到元数据
        for node in initial_nodes:
            node.node.metadata['initial_score'] = node.score

        # 2. 重排序
        reranked_nodes = reranker.postprocess_nodes(
            initial_nodes,
            query_str=query
        )
        rerank_time = time.time() - start_time - retrieval_time

        # 3. 合成答案
        result = response_synthesizer.synthesize(
            query,
            nodes=reranked_nodes
        )
        synthesis_time = time.time() - start_time - retrieval_time - rerank_time

        # 显示结果
        print(f"智能助手：\n{result.response}\n")
        print(f"支持依据：")
        for idx, node in enumerate(result.source_nodes, 1):
            metadata = node.metadata
            print(f"{idx}. {metadata['title']}")
            print(f"    法律名称：{metadata['law_name']}")
            print(f"    来源文件：{metadata['source_file']}")
            print(f"    条款内容：{node.text}")
            print(f"    初始相关度：{node.node.metadata['initial_score']:.4f}")
            print(f"    重排序得分：{node.score:.4f}")
            print("-" * 60)

        print(f"\n[性能分析] 检索: {retrieval_time:.2f}s | 重排序: {rerank_time:.2f}s | 合成: {synthesis_time:.2f}s")


if __name__ == "__main__":
    main()
