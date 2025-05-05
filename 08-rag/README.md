### RAG-法律条文助手

#### [rag_law_01.py](rag_law_01.py)

- 基础版

#### [rag_law_02.py](rag_law_02.py)

- 增加了重排序模型，对检索结果进行重新排序

#### [rag_law_03.py](rag_law_03.py)

- 取消提示词，因为提示词对模型的影响较小，不一定会对结果产生影响
- 增加了对重排序结果进行过滤，如果排序结果中没有相关的内容，则提示：无法处理

#### [rag_law_04.py](rag_law_04.py)

- 增加召回率评估
- 增加端到端评估

#### [rag_law_05.py](rag_law_05.py)

- 嵌入 Streamlit 界面

```bash
streamlit run rag_law_05.py --server.address=127.0.0.1 --server.fileWatcherType=none
```