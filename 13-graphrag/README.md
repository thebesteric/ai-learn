### 创建文件夹
`mkdir -p ./graphrag_test/input`
> 注意 input 文件夹的名称不能改变

### 添加外部文件
将外部文件放到`./graphrag_test/input`中

### 初始化
`graphrag init --root ./graphrag_test`

### 配置环境
#### env
```text
GRAPHRAG_MODEL=xxx
GRAPHRAG_API_KEY=xxx
GRAPHRAG_API_BASE=xxx

GRAPHRAG_EMBEDDING_MODEL=xxx
GRAPHRAG_EMBEDDING_API_KEY=xxx
GRAPHRAG_EMBEDDING_API_BASE=xxx
```
#### setting.yaml
注意要放开`encoding_model: cl100k_base`，并合理设置`chunks.size`
```yaml
models:
  default_chat_model:
    type: openai_chat # or azure_openai_chat
    api_base: ${GRAPHRAG_API_BASE}
    # api_version: 2024-05-01-preview
    auth_type: api_key # or azure_managed_identity
    api_key: ${GRAPHRAG_API_KEY} # set this in the generated .env1 file
    # audience: "https://cognitiveservices.azure.com/.default"
    # organization: <organization_id>
    model: ${GRAPHRAG_MODEL}
    # deployment_name: <azure_model_deployment_name>
    encoding_model: cl100k_base # automatically set by tiktoken if left undefined
    model_supports_json: true # recommended if this is available for your model.
    concurrent_requests: 25 # max number of simultaneous LLM requests allowed
    async_mode: threaded # or asyncio
    retry_strategy: native
    max_retries: -1                   # set to -1 for dynamic retry logic (most optimal setting based on server response)
    tokens_per_minute: 0              # set to 0 to disable rate limiting
    requests_per_minute: 0            # set to 0 to disable rate limiting
  default_embedding_model:
    type: openai_embedding # or azure_openai_embedding
    api_base: ${GRAPHRAG_EMBEDDING_API_BASE}
    # api_version: 2024-05-01-preview
    auth_type: api_key # or azure_managed_identity
    api_key: ${GRAPHRAG_EMBEDDING_API_KEY}
    # audience: "https://cognitiveservices.azure.com/.default"
    # organization: <organization_id>
    model: ${GRAPHRAG_EMBEDDING_MODEL}
    # deployment_name: <azure_model_deployment_name>
    encoding_model: cl100k_base # automatically set by tiktoken if left undefined
    model_supports_json: true # recommended if this is available for your model.
    concurrent_requests: 25 # max number of simultaneous LLM requests allowed
    async_mode: threaded # or asyncio
    retry_strategy: native
    max_retries: -1                   # set to -1 for dynamic retry logic (most optimal setting based on server response)
    tokens_per_minute: 0              # set to 0 to disable rate limiting
    requests_per_minute: 0            # set to 0 to disable rate limiting
    
chunks:
  size: 200
  overlap: 50
  group_by_columns: [id]
```


### indexing
`graphrag index --root ./graphrag_test`

### 查询
- 全局查询，跨文档综合分析  
`graphrag query --root ./graphrag_test --method global --query "请介绍下ID3算法"`
- 局部查询，单文档精准检索  
`graphrag query --root ./graphrag_test --method local --query "请介绍下ID3算法"`
- DRIFT 查询，动态漂移分析  
`graphrag query --root ./graphrag_test --method drift --query "请介绍下ID3算法"`
- 基础查询，传统RAG检索  
`graphrag query --root ./graphrag_test --method basic --query "请介绍下ID3算法"`