from llama_index.core import SimpleDirectoryReader

# 加载所有文档
documents = SimpleDirectoryReader(input_files=["../data/ai.txt"]).load_data()

# 使用固定节点切分
from llama_index.core.node_parser import TokenTextSplitter

fixed_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=20)
fixed_nodes = fixed_splitter.get_nodes_from_documents(documents)
print("固定分块示例：", [len(n.text) for n in fixed_nodes[:3]])  # 输出：[200, 200, 200]
print(print("首个节点内容:\n", fixed_nodes[0].text))
print(print("第二个节点内容:\n", fixed_nodes[1].text))

print("\n==============================\n")

# 使用句子分割器
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=256)
nodes = splitter.get_nodes_from_documents(documents)

# 查看结果
print(f"生成节点数: {len(nodes)}")
print("首个节点内容:\n", nodes[0].text)
print("第二个节点内容:\n", nodes[1].text)
