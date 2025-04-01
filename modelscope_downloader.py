import os
import sys
from modelscope import snapshot_download

# SDK 下载
def download_model_from_modelscope(model_id, cache_dir="/Users/wangweijun/LLM/models"):
    """
    从 ModelScope 下载指定模型并返回其本地目录。

    :param model_id: 要下载的模型的唯一标识符。
    :param cache_dir: 模型下载后的缓存目录，默认为 "model_cache"。
    :return:
    """
    # 调用 snapshot_download 函数从 ModelScope 下载模型到指定的缓存目录
    # 如果 cache_dir 目录存在，则不会进行下载
    model_dir = snapshot_download(model_id=model_id, cache_dir=cache_dir)
    # 打印下载后模型所在的本地目录
    print(f"model_dir: {model_dir}")

if __name__ == '__main__':
    download_model_from_modelscope("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")