import logging
import sys

from llama_index.core import SimpleDirectoryReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

"""
pip install llama-index
"""

# 通过目录，加载本地文档进行解析
documents = SimpleDirectoryReader(input_dir="../data").load_data()
for document in documents:
    print(document, "\n")

print("=" * 50, "\n")

# 通过后缀名，加载本地文档进行解析
documents = SimpleDirectoryReader(input_dir="../data", required_exts=[".txt"]).load_data()
for document in documents:
    print(document, "\n")

print("=" * 50, "\n")

# 通过文件，加载本地文档进行解析
documents = SimpleDirectoryReader(input_files=["../data/pdf内容研报.pdf"]).load_data()
for document in documents:
    print(document.text, "\n")

