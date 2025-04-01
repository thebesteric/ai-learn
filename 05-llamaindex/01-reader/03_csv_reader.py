import logging
import sys

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import CSVReader
from llama_index.readers.web import SimpleWebPageReader

"""
pip install llama-index-readers-file
"""

# 通过目录，加载本地文档进行解析
csv_parser = CSVReader()
file_extractor = {".csv": csv_parser}
docutemts = SimpleDirectoryReader(input_dir="../data",
                                  required_exts=[".csv"],
                                  file_extractor=file_extractor).load_data()
print(docutemts)