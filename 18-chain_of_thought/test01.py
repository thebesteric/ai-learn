from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="123456")


def chain_of_thought(question):
    prompt = f"""
			Q: {question}
			A: 让我们一步一步地思考这个问题。
    	"""
    response = client.chat.completions.create(
        model="qwen2.5:7B",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content


# 测试数学问题
result = chain_of_thought("如果一辆车在2小时内行驶了120公里，那么它的平均速度是多少公里/小时？")
print(result)
