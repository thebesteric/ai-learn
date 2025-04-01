import os

import requests

# GPT-2 是一个续写模型

ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
API_HOST = os.getenv("HUGGING_FACE_API_HOST")
API_URL = f"{API_HOST}uer/gpt2-chinese-cluecorpussmall"
print(API_URL)


headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

response = requests.post(API_URL, headers=headers, json={
	"inputs": "这是很久之前的事情了",
})
print(response.json())