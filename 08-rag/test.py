from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="123",
    api_base="http://localhost:8000/v1",
    api_key="123",
    temperature=0.3,
    max_tokens=1024,
    timeout=60,
    is_chat_model=True,
    additional_kwargs={"stop": ["<|im_end|>"]}
)

response = llm.complete("你好")
print(response)