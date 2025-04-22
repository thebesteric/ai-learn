from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
import os

# 1. 加载文档
documents = SimpleDirectoryReader(input_files=["../data/ai.txt"]).load_data()

# 2. 初始化模型和解析器
embed_model = HuggingFaceEmbedding(
    # 指定了一个预训练的sentence-transformer模型的路径
    model_name="/Users/wangweijun/LLM/models/paraphrase-multilingual-MiniLM-L12-v2"
)

semantic_parser = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=90,
    embed_model=embed_model
)

# 3. 执行语义分割
semantic_nodes = semantic_parser.get_nodes_from_documents(documents)

# 4. 打印结果
print(f"语义分割节点数: {len(semantic_nodes)}")
for i, node in enumerate(semantic_nodes[:2]):  # 只打印前两个节点
    print(f"\n节点{i + 1}:\n{node.text}")
    print("-" * 50)
