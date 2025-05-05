import json
import time
from pathlib import Path

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
        max_tokens=4096,
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


# 关键字过滤函数
def is_legal_question(text: str) -> bool:
    """判断问题是否属于法律咨询"""
    legal_keywords = ["劳动法", "合同", "工资", "工伤", "解除", "赔偿"]
    return any(keyword in text for keyword in legal_keywords)


# ================== 新增评估类 ==================
class RecallEvaluator:
    def __init__(self, retriever, reranker):
        self.retriever = retriever
        self.reranker = reranker

    def calculate_recall(self, retrieved_nodes, relevant_ids):
        retrieved_ids = [n.node.metadata["title"] for n in retrieved_nodes]
        hit = len(set(retrieved_ids) & set(relevant_ids))
        return hit / len(relevant_ids) if relevant_ids else 0.0

    def evaluate(self, benchmark):
        results = []
        for case in benchmark:
            # 初始检索
            initial_nodes = self.retriever.retrieve(case["question"])
            # 重排序
            reranked_nodes = self.reranker.postprocess_nodes(
                initial_nodes,
                query_str=case["question"]
            )
            # 计算召回率
            recall = self.calculate_recall(reranked_nodes, case["relevant_ids"])
            results.append(recall)

            print(f"问题：{case['question']}")
            print(f"初始检索结果：{[n.node.metadata['title'] for n in initial_nodes]}")
            print(f"重排序后结果：{[n.node.metadata['title'] for n in reranked_nodes]}")
            print(f"召回条款：{[n.node.metadata['title'] for n in reranked_nodes[:3]]}")
            print(f"目标条款：{case['relevant_ids']}")
            print(f"召回率：{recall:.1%}\n")

        avg_recall = np.mean(results)
        print("\n=== 召回率评估报告 ===")
        print(f"平均召回率：{avg_recall:.1%}")
        return avg_recall


class E2EEvaluator:
    def __init__(self, query_engine):
        self.query_engine = query_engine

    def evaluate_case(self, response, standard):
        try:
            # 获取实际命中的条款
            retrieved_clauses = [node.node.metadata["title"] for node in response.source_nodes]

            # 获取标准答案要求的条款
            required_clauses = standard["standard_answer"]["条款"]

            # 计算命中情况
            hit_clauses = list(set(retrieved_clauses) & set(required_clauses))
            missed_clauses = list(set(required_clauses) - set(retrieved_clauses))

            # 计算命中率
            clause_hit = len(hit_clauses) / len(required_clauses) if required_clauses else 0.0

            return {
                "clause_score": clause_hit,
                "hit_clauses": hit_clauses,
                "missed_clauses": missed_clauses
            }
        except Exception as e:
            print(f"评估失败：{str(e)}")
            return None

    def evaluate(self, benchmark):
        results = []
        for case in benchmark:
            try:
                response = self.query_engine.query(case["question"])
                case_result = self.evaluate_case(response, case)

                if case_result:
                    print(f"\n问题：{case['question']}")
                    print(f"命中条款：{case_result['hit_clauses']}")
                    print(f"缺失条款：{case_result['missed_clauses']}")
                    print(f"条款命中率：{case_result['clause_score']:.1%}")
                    results.append(case_result)
                else:
                    results.append(None)
            except Exception as e:
                print(f"查询失败：{str(e)}")
                results.append(None)

        # 计算统计数据
        valid_results = [r for r in results if r is not None]
        avg_hit = np.mean([r["clause_score"] for r in valid_results]) if valid_results else 0

        print("\n=== 端到端评估报告 ===")
        print(f"有效评估案例：{len(valid_results)}/{len(benchmark)}")
        print(f"平均条款命中率：{avg_hit:.1%}")

        # 输出详细错误分析
        for i, result in enumerate(results):
            if result is None:
                print(f"案例{i + 1}：{benchmark[i]['question']} 评估失败")

        return results


# ================== 新增测试数据集 ==================
# 召回率评估用例
RETRIEVAL_BENCHMARK = [
    # 劳动合同解除类
    {
        "question": "劳动者可以立即解除劳动合同的情形有哪些？",
        "relevant_ids": ["中华人民共和国劳动合同法 第三十八条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第三十九条", "中华人民共和国劳动法 第三十二条"]
    },
    {
        "question": "用人单位单方解除劳动合同需要提前多久通知？",
        "relevant_ids": ["中华人民共和国劳动合同法 第四十条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第三十七条", "中华人民共和国劳动法 第二十六条"]
    },

    # 工资与补偿类
    {
        "question": "经济补偿金的计算标准是什么？",
        "relevant_ids": ["中华人民共和国劳动合同法 第四十七条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第八十七条", "中华人民共和国劳动法 第二十八条"]
    },
    {
        "question": "试用期工资最低标准是多少？",
        "relevant_ids": ["中华人民共和国劳动合同法 第二十条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第十九条", "中华人民共和国劳动法 第四十八条"]
    },

    # 工伤与福利类
    {
        "question": "工伤认定需要哪些材料？",
        "relevant_ids": ["中华人民共和国劳动合同法 第三十条"],
        "confusing_ids": ["中华人民共和国劳动法 第七十三条", "中华人民共和国劳动合同法 第十七条"]
    },
    {
        "question": "女职工产假有多少天？",
        "relevant_ids": ["中华人民共和国劳动法 第六十二条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第四十二条", "中华人民共和国劳动法 第六十一条"]
    },

    # 劳动合同订立类
    {
        "question": "无固定期限劳动合同的订立条件是什么？",
        "relevant_ids": ["中华人民共和国劳动合同法 第十四条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第十三条", "中华人民共和国劳动法 第二十条"]
    },
    {
        "question": "劳动合同必须包含哪些条款？",
        "relevant_ids": ["中华人民共和国劳动合同法 第十七条"],
        "confusing_ids": ["中华人民共和国劳动法 第十九条", "中华人民共和国劳动合同法 第十条"]
    },

    # 特殊用工类
    {
        "question": "劳务派遣岗位的限制条件是什么？",
        "relevant_ids": ["中华人民共和国劳动合同法 第六十六条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第五十八条", "中华人民共和国劳动法 第二十条"]
    },
    {
        "question": "非全日制用工的每日工作时间上限？",
        "relevant_ids": ["中华人民共和国劳动合同法 第六十八条"],
        "confusing_ids": ["中华人民共和国劳动法 第三十六条", "中华人民共和国劳动合同法 第三十八条"]
    },
    # 劳动合同终止类
    {
        "question": "劳动合同终止的法定情形有哪些？",
        "relevant_ids": ["中华人民共和国劳动合同法 第四十四条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第四十六条", "中华人民共和国劳动法 第二十三条"]
    },
    {
        "question": "劳动合同期满后必须续签的情形？",
        "relevant_ids": ["中华人民共和国劳动合同法 第四十五条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第十四条", "中华人民共和国劳动法 第二十条"]
    },

    # 劳动保护类
    {
        "question": "女职工哺乳期工作时间限制？",
        "relevant_ids": ["中华人民共和国劳动法 第六十三条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第四十二条", "中华人民共和国劳动法 第六十一条"]
    },
    {
        "question": "未成年工禁止从事的劳动类型？",
        "relevant_ids": ["中华人民共和国劳动法 第六十四条"],
        "confusing_ids": ["中华人民共和国劳动法 第五十九条", "中华人民共和国劳动合同法 第六十六条"]
    },

    {
        "question": "工伤保险待遇包含哪些项目？",
        "relevant_ids": ["中华人民共和国劳动法 第七十三条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第三十条", "中华人民共和国劳动法 第四十四条"]
    },

    # 劳动争议类
    {
        "question": "劳动争议仲裁时效是多久？",
        "relevant_ids": ["中华人民共和国劳动法 第八十二条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第六十条", "中华人民共和国劳动法 第七十九条"]
    },
    {
        "question": "集体合同的法律效力如何？",
        "relevant_ids": ["中华人民共和国劳动法 第三十五条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第五十五条", "中华人民共和国劳动法 第三十三条"]
    },

    # 特殊条款类
    {
        "question": "服务期违约金的上限规定？",
        "relevant_ids": ["中华人民共和国劳动合同法 第二十二条"],
        "confusing_ids": ["中华人民共和国劳动合同法 第二十三条", "中华人民共和国劳动法 第一百零二条"]
    },
    {
        "question": "无效劳动合同的认定标准？",
        "relevant_ids": ["中华人民共和国劳动合同法 第二十六条"],
        "confusing_ids": ["中华人民共和国劳动法 第十八条", "中华人民共和国劳动合同法 第三十九条"]
    }
]

# 端到端评估用例
E2E_BENCHMARK = [
    # 案例1：劳动合同解除
    {
        "question": "用人单位在哪些情况下不得解除劳动合同？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第四十二条"],
            "标准答案": "根据《劳动合同法》第四十二条，用人单位不得解除劳动合同的情形包括：\n1. 从事接触职业病危害作业的劳动者未进行离岗前职业健康检查\n2. 在本单位患职业病或者因工负伤并被确认丧失/部分丧失劳动能力\n3. 患病或非因工负伤在规定的医疗期内\n4. 女职工在孕期、产期、哺乳期\n5. 连续工作满15年且距退休不足5年\n6. 法律、行政法规规定的其他情形\n违法解除需按第八十七条支付二倍经济补偿金",
            "必备条件": ["职业病危害作业未检查", "孕期女职工", "连续工作满15年"]
        }
    },

    # 案例2：工资支付
    {
        "question": "拖欠工资劳动者可以采取哪些措施？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第三十条", "中华人民共和国劳动法 第五十条"],
            "标准答案": "劳动者可采取以下救济措施：\n1. 根据劳动合同法第三十条向法院申请支付令\n2. 依据劳动合同法第三十八条解除合同并要求经济补偿\n3. 向劳动行政部门投诉\n逾期未支付的，用人单位需按应付金额50%-100%加付赔偿金（劳动合同法第八十五条）",
            "必备条件": ["支付令申请", "解除劳动合同", "行政投诉"]
        }
    },

    # 案例3：竞业限制
    {
        "question": "竞业限制的最长期限是多久？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第二十四条"],
            "标准答案": "根据劳动合同法第二十四条：\n- 竞业限制期限不得超过二年\n- 适用人员限于高管/高级技术人员/其他保密人员\n- 需按月支付经济补偿\n注意区分服务期约定（第二十二条）",
            "限制条件": ["期限≤2年", "按月支付补偿"]
        }
    },

    # 案例4：劳务派遣
    {
        "question": "劳务派遣用工的比例限制是多少？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第六十六条"],
            "标准答案": "劳务派遣用工限制：\n- 临时性岗位不超过6个月\n- 辅助性岗位≤用工总量10%\n- 违法派遣按每人1000-5000元罚款（第九十二条）\n派遣协议需包含岗位/期限/报酬等条款（第五十九条）",
            "限制条件": ["临时性≤6月", "辅助性≤10%"]
        }
    },

    # 案例5：非全日制用工
    {
        "question": "非全日制用工的工资支付周期要求？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第七十二条"],
            "标准答案": "非全日制用工支付规则：\n- 工资结算周期≤15日\n- 小时工资≥当地最低标准\n- 终止用工不支付经济补偿（第七十一条）\n区别于全日制月薪制（第三十条）",
            "支付规则": ["周期≤15天", "小时工资≥最低标准"]
        }
    },

    # 案例6：劳动合同无效
    {
        "question": "劳动合同被确认无效后的工资支付标准？",
        "standard_answer": {
            "条款": ["中华人民共和国劳动合同法 第二十八条"],
            "标准答案": "无效劳动合同的工资支付：\n1. 参照本单位相同岗位工资支付\n2. 无相同岗位的按市场价\n3. 已付报酬不足的需补差\n过错方需承担赔偿责任（第八十六条）",
            "支付规则": ["参照同岗位", "市场价补差"]
        }
    }
]

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
    query_engine = index.as_query_engine(
        similarity_top_k=Config.TOP_K,
        node_postprocessors=[reranker]
    )
    response_synthesizer = get_response_synthesizer(
        # text_qa_template=Config.RESPONSE_TEMPLATE,
        verbose=True
    )

    # 新增评估模式
    run_mode = input("请选择模式：1-问答模式 2-评估模式\n请输入选项：")

    if run_mode == "2":
        print("\n=== 开始评估 ===")

        # 召回率评估
        recall_evaluator = RecallEvaluator(retriever, reranker)
        recall_result = recall_evaluator.evaluate(RETRIEVAL_BENCHMARK)

        # 端到端评估
        e2e_evaluator = E2EEvaluator(query_engine)
        e2e_results = e2e_evaluator.evaluate(E2E_BENCHMARK)

        # 生成报告
        print("\n=== 最终评估报告 ===")
        print(f"重排序召回率：{recall_result:.1%}")
        print(f"端到端条款命中率：{np.mean([r['clause_score'] for r in e2e_results]):.1%}")
        return

    while True:
        query = input("请输入劳动法的相关问题：")
        if query == "/q":
            break

        # 使用关键词判断，判断提问类型是否符合法律问题
        # if not is_legal_question(question):  # 新增判断函数
        #     print("\n您好！我是劳动法咨询助手，专注解答《劳动法》《劳动合同法》等相关问题。")
        #     continue

        # 执行查询
        print(f"查询中，请稍后...\n")

        # 执行检索-重排序-回答流程（新增重排序步骤）
        start_time = time.time()

        # 1. 初始检索，返回 TOP_K 个节点
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

        # ★★★★★ 添加过滤逻辑在此处，从重排序后的结果中过滤 ★★★★★
        filtered_nodes = [
            node for node in reranked_nodes
            if node.score > Config.MIN_RERANK_SCORE
        ]
        # for node in reranked_nodes:
        #     print(node.score)

        # 一般对模型的回复做限制就从 filtered_nodes 的返回值下手
        print("原始节点分数：", [node.score for node in initial_nodes])
        print("重排序后节点分数：", [reranked_node.score for reranked_node in reranked_nodes])
        print("重排序过滤后的结果：", [filtered_node.score for filtered_node in filtered_nodes])
        # 空结果处理
        if not filtered_nodes:
            print("你的问题未匹配到相关资料！")
            continue

        # 3. 使用 filtered_nodes 合成答案
        result = response_synthesizer.synthesize(
            query,
            nodes=filtered_nodes
        )
        synthesis_time = time.time() - start_time - retrieval_time - rerank_time

        # 显示结果
        print(f"\n智能助手：\n{result.response}\n")
        print(f"支持依据：")
        for idx, node in enumerate(result.source_nodes, 1):
            metadata = node.metadata
            print(f"[{idx}]. {metadata['title']}")
            print(f"    法律名称：{metadata['law_name']}")
            print(f"    来源文件：{metadata['source_file']}")
            print(f"    条款内容：{node.text}")
            print(f"    初始相关度：{node.node.metadata['initial_score']:.4f}")
            print(f"    重排序得分：{node.score:.4f}")
            print("-" * 60)

        print(f"\n[性能分析] 检索: {retrieval_time:.2f}s | 重排序: {rerank_time:.2f}s | 合成: {synthesis_time:.2f}s")


if __name__ == "__main__":
    main()
