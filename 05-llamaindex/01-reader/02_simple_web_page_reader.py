import logging
import sys

from llama_index.readers.web import SimpleWebPageReader

"""
pip install llama-index llama-index-readers-web
"""

# 通过目录，加载本地文档进行解析
documents = SimpleWebPageReader(html_to_text=True).load_data(
    urls=[
        "http://paulgraham.com/worked.html"
    ]
)
print(documents)
