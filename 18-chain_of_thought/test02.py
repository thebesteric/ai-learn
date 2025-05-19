from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="123456")

few_shot_prompt = """接收到用户的问题后，按照如下模板进行回复：
Q: 动物园有 15 只狮子和 20 只老虎。如果将 3 只狮子转移到另一个动物园，还剩下多少只大型猫科动物？
A: 最初有 15 只狮子 + 20 只老虎 = 35 只大型猫科动物。转移 3 只狮子后，35 - 3 = 32。因此，答案是 32。
"""


def few_shot_cot(question):
    full_prompt = few_shot_prompt + f"Q: {question}\nA: Let's think step by step."

    response = client.chat.completions.create(
        model="qwen2.5:7B",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content


result = few_shot_cot("动物园有 15 只狮子和 18 只老虎。如果将 3 只狮子转移到另一个动物园，还剩下多少只大型猫科动物？")
print(result)
