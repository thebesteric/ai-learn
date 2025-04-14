import base64
import json
import os

import requests
import tiktoken
from dotenv import load_dotenv
from lxml import etree


def google_search(query, num_results=10, site_url=None):
    """
    使用 Google Custom Search API 进行搜索
    :param query: 搜索关键词
    :param num_results: 要返回的结果数量
    :param site_url: 要搜索的网站 URL
    :return: 搜索结果列表
    """
    print("正在调用 google_search 搜索数据...")

    load_dotenv(override=True)

    # 从环境变量中获取 API 密钥和 CSE ID
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    url = "https://www.googleapis.com/customsearch/v1"

    # API 请求参数
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results,
    }
    if site_url is not None:
        params["siteSearch"] = site_url

    # 发送请求
    response = requests.get(url, params=params)
    response.raise_for_status()

    # 解析响应
    search_results = response.json().get("items", [])

    # 提取所需要的信息
    results = [{
        "title": item["title"],
        "link": item["link"],
        "snippet": item["snippet"]
    } for item in search_results]

    return results


def windows_compatible_name(s, max_length=255):
    """
    将字符串转换为符合 Windows 文件名规则的字符串
    :param s: 输入的字符串
    :param max_length: 输出字符串的最大长度，默认为 255
    :return: 一个可以安全用作 Windows 文件名的字符串
    """
    # Windows 文件中不允许的字符列表
    forbidden_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
    for char in forbidden_chars:
        # 将非法字符替换为下划线
        s = s.replace(char, "_")

    # 去除尾部的空格或点
    s = s.rstrip(" .")

    # 检查是否存在以下不允许被用于文档名称的关键词，如果存在，则替换为下划线
    reserved_names = ["CON", "PRN", "AUX", "NUL",
                      "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                      "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"]
    for name in reserved_names:
        if name.lower() in s.lower():
            s = s.replace(name, "_")

    # 如果字符串的长度超过最大长度，则截取并添加省略号
    if len(s) > max_length:
        s = s[:max_length - 3] + "..."

    return s


def get_search_text(q, url):
    load_dotenv(override=True)
    cookie = os.getenv("SEARCH_COOKIE")
    user_agent = os.getenv("SEARCH_USER_AGENT")

    headers = {
        'authority': 'www.zhihu.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'cache-control': 'no-cache',
        'cookie': cookie,
        "upgrade-insecure-requests": "1",
        'user-agent': user_agent,
    }

    code_ = False
    title = None

    # 普通回答地址
    if "zhihu.com/question" in url:
        res = requests.get(url, headers=headers).text
        res_xpath = etree.HTML(res)
        title = res_xpath.xpath('//div/div[1]/div/h1/text()')[0]
        text_d = res_xpath.xpath('//div/div/div/div[2]/div/div/div[2]/span[1]/div/div/span/p/text()')
    # 知乎专栏地址
    elif 'zhuanlan' in url:
        headers['authority'] = 'zhuanlan.zhihu.com'
        res = requests.get(url, headers=headers).text
        res_xpath = etree.HTML(res)
        title = res_xpath.xpath('//div[1]/div/main/div/article/header/h1/text()')[0]
        text_d = res_xpath.xpath('//div/main/div/article/div[1]/div/div/div/p/text()')
        code_ = res_xpath.xpath('//div/main/div/article/div[1]/div/div/div/pre/code/text()')
    # 特定回答的问答网址
    elif 'answer' in url:
        res = requests.get(url, headers=headers).text
        res_xpath = etree.HTML(res)
        title = res_xpath.xpath('//div/div[1]/div/h1/text()')[0]
        text_d = res_xpath.xpath('//div[1]/div/div[3]/div/div/div/div[2]/span[1]/div/div/span/p/text()')

    if title is None:
        return None
    else:
        title = windows_compatible_name(title)
        text = ''
        for t in text_d:
            txt = str(t).replace('\n', ' ')
            text += txt

        if code_:
            for c in code_:
                co = str(c).replace('\n', ' ')
                text += co

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        json_data = [
            {
                "link": url,
                "title": title,
                "content": text,
                "tokens": len(encoding.encode(text))
            }
        ]

        # 自动创建目录
        dir_path = f"./auto_search/{q}"
        os.makedirs(dir_path, exist_ok=True)

        with open(f"{dir_path}/{title}.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        return title


def get_search_result(q):
    """
    当你无法回答某个问题时，调用该函数，能够获取答案
    :param q: 必须参数，询问的问题，字符串类型
    :return: 某个问题的答案，以字符串形式返回
    """
    # 调用 google_search 函数获取搜索结果
    results = google_search(q, num_results=5, site_url="https://www.zhihu.com")

    # 创建对应问题的子文件夹
    folder_path = f"./auto_search/{q}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 单独提取 links 放在一个 list 中
    num_tokens = 0
    content = ''
    for item in results:
        url = item["link"]
        title = get_search_text(q, url)
        with open(f"{folder_path}/{title}.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)
        num_tokens += json_data[0]["tokens"]
        # 最多到 12k tokens，这里是为了防止 token 过多，导致模型无法处理
        if num_tokens <= 12000:
            content += json_data[0]["content"]
        else:
            break

    return content


def get_github_readme(dic):
    """
    用于查询某个 github 仓库的 readme 文件
    :param dic: 包含 owner 和 repo 的字典，例如：{"owner": "thebesteric", "repo": "agile"}
    :return:
    """
    load_dotenv(override=True)
    github_token = os.getenv("GITHUB_TOKEN")
    user_agent = os.getenv("SEARCH_USER_AGENT")

    owner = dic["owner"]
    repo = dic["repo"]

    headers = {
        "Authorization": github_token,
        "User-Agent": user_agent,
    }

    response = requests.get(f"https://api.github.com/repos/{owner}/{repo}/readme", headers=headers)

    readme_data = response.json()
    encoded_content = readme_data.get("content", "")
    decoded_content = base64.b64decode(encoded_content).decode("utf-8")

    return decoded_content


def extract_github_repos(search_results):
    """
    从搜索结果中提取 GitHub 仓库的 owner 和 repo
    :param search_results: 搜索结果列表，每个结果包含 "link" 键
    :return: 包含 owner 和 repo 的列表
    """
    # 筛选出项目主页链接
    repo_links = [result["link"] for result in search_results
                  if "/issues/" not in result["link"]
                  and "/blob/" not in result["link"]
                  and "github.com" in result["link"]
                  and len(result["link"].split("/")) == 5]
    # 从筛选后的链接中提取 owner 和 repo
    repos_info = [{"owner": link.split("/")[3], "repo": link.split("/")[4]} for link in repo_links]

    return repos_info


def get_search_text_github(q, dic):
    # 创建标题
    title = dic["owner"] + "_" + dic["repo"]
    title = windows_compatible_name(title)
    # 创建问题答案正文
    text = get_github_readme(dic)

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    json_data = [
        {
            "title": title,
            "content": text,
            "tokens": len(encoding.encode(text))
        }
    ]

    # 自动创建目录
    dir_path = f"./auto_search/{q}"
    os.makedirs(dir_path, exist_ok=True)
    with open(f"{dir_path}/{title}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    return title


def get_answer_github(q, g="globals()"):
    """
    当你无法回答某个问题时，调用该函数，能够获取答案
    :param q: 必须参数，询问的问题，字符串类型
    :param g: 字符串形式变量，表示环境变量，无须设置，保持默认参数即可
    :return: 某个问题的答案，以字符串形式返回
    """
    print("正在调用 get_answer_github 函数")

    # 调用 google_search 函数获取搜索结果
    search_results = google_search(q, num_results=5, site_url="https://github.com")
    results = extract_github_repos(search_results)

    # 创建对应问题的子文件夹
    folder_path = f"./auto_search/{q}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    num_tokens = 0
    content = ""
    print("正在读取相关项目的 README 文件...")
    for dic in results:
        title = get_search_text_github(q, dic)
        with open(f"{folder_path}/{title}.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)
        num_tokens += json_data[0]["tokens"]
        # 最多到 12k tokens，这里是为了防止 token 过多，导致模型无法处理
        if num_tokens <= 12000:
            content += json_data[0]["content"]
        else:
            break

    print("正在进行最后的整理...")
    return content

get_answer_github_tool = {
    "type": "function",
    "function": {
        "name": "get_answer_github",
        "description": "GitHub 联网搜索工具，当用户提出的问题超出了你的知识范围，或者该问题你不知道答案时，请调用该函数来获取问题的答案。\n"
                       "该函数会自动从 GitHub 上搜索得到项目的 README 文件，然后你可以根据 README 文件进行总结，并回答用户的问题。\n\n"
                       "⚠️ 特别注意：当用户点名需要想了解 GitHub 上的项目的时候，请调用此函数，其他情况请调用 get_answer 外部函数进行查询。",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "一个满足 GitHub 搜索格式的问题，往往是需要从用户问题中提取出一个合适的搜索的项目关键词，进行查询，用字符串类型",
                    "example": "DeepSeek-R1"
                },
                "g": {
                    "type": "string",
                    "description": "字符串形式变量，表示环境变量，无须设置，保持默认参数即可",
                    "default": "globals()"
                }
            },
            "required": ["q"]
        }
    }
}


def get_answer(q, g="globals()"):
    """
    当你无法回答某个问题时，调用该函数，能够获取答案
    :param q: 必须参数，询问的问题，字符串类型
    :param g: 字符串形式变量，表示环境变量，无须设置，保持默认参数即可
    :return: 某个问题的答案，以字符串形式返回
    """
    print("正在调用 get_answer 函数")

    # 调用 google_search 函数获取搜索结果
    results = google_search(query=q, num_results=5, site_url="https://www.zhihu.com")

    # 创建对应问题的子文件夹
    folder_path = f"./auto_search/{q}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 单独提取 links 放在一个 list 中
    num_tokens = 0
    content = ''
    for item in results:
        url = item["link"]
        print(f"正在检索: {url}")
        title = get_search_text(q, url)
        if title is None:
            continue
        with open(f"{folder_path}/{title}.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)
        num_tokens += json_data[0]["tokens"]
        # 最多到 12k tokens，这里是为了防止 token 过多，导致模型无法处理
        if num_tokens <= 12000:
            content += json_data[0]["content"]
        else:
            break

    print("正在进行最后的整理...")
    return content


get_answer_tool = {
    "type": "function",
    "function": {
        "name": "get_answer",
        "description": "联网搜索工具，当用户提出的问题超出了你的知识范围，或者该问题你不知道答案时，请调用该函数来获取问题的答案。\n"
                       "该函数会自动从知乎上搜索得到问题的相关文本内容，然后你可以根据文本内容进行总结，并回答用户的问题。\n\n"
                       "⚠️ 特别注意：当用户点名需要想了解 GitHub 上的项目的时候，请调用 get_answer_github 函数，而不是该函数。",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "一个满足知乎搜索格式的问题，用字符串类型",
                    "example": "什么是 MCP？"
                },
                "g": {
                    "type": "string",
                    "description": "字符串形式变量，表示环境变量，无须设置，保持默认参数即可",
                    "default": "globals()"
                }
            },
            "required": ["q"]
        }
    }
}

if __name__ == '__main__':
    # results = google_search("什么是 MCP", num_results=5, site_url="https://www.zhihu.com")
    # print(results)

    # url = "https://www.zhihu.com/question/7762420288"
    # q = "什么是 MCP"
    # result = get_search_text(q, url)
    # print(result)

    # result = get_answer_github("DeepSeek-R1", g=globals())
    # print(result)

    result = get_answer("什么是 MCP", g=globals())
    print(result)
