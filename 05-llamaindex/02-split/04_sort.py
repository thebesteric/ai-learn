results = [
    "模型正则化方法描述",  # 相关度：0.7
    "硬件加速技术进展",  # 相关度：0.65
    "过拟合解决方案详解",  # 相关度：0.92
    "数据清洗方法"  # 相关度：0.85
]

# 应用重排列
from llama_index.postprocessor.cohere_rerank import CohereRerank

# 模型名称              速度 精度 硬件要求 适用场景
# BM25                 快  中   低      关键词匹配
# Cross-Encoder        慢  高   高      小规模精准排序
# ColBERT              中  高   中      平衡速度与精度

reranker = CohereRerank(api_key="COHERE_API_KEY", top_n=2)
reranked_results = reranker.postprocess_nodes(results, query_str="如何防止模型过拟合？")

print("重排列后的结果：", [res.node.text for res in reranked_results])

# 原始排序：
# 1. 模型正则化方法简述（相关度0.7）
# 2. 硬件加速技术进展（相关度0.65）
# 3. 过拟合解决方案详解（相关度0.92）← 正确答案
# 4. 数据集清洗方法
# 重排序后：
# 1. 过拟合解决方案详解（评分0.95）← 正确答案
# 2. 模型正则化方法简述（评分0.88）

