import json
import re
import requests
from bs4 import BeautifulSoup


def fetch_and_parse(url):
    # 请求网页
    response = requests.get(url)
    # 设置网页编码格式
    response.encoding = 'utf-8'
    # 解析网页内容
    soup = BeautifulSoup(response.text, 'html.parser')
    # 提取正文内容
    content = soup.find_all('p')
    # 初始化存储数据
    data = []
    # 提取文本并格式化
    for para in content:
        text = para.get_text(strip=True)
        # 只处理非空文本
        if text:
            # 根据需求格式化内容
            data.append(text)
    # 将 data 列表转换为字符串
    data_str = '\n'.join(data)
    return data_str


def extract_law_articles(title, url):
    data_str = fetch_and_parse(url)
    # 正则表达式，匹配每个条款号及其内容
    pattern = re.compile(r'第([一二三四五六七八九十零百]+)条.*?(?=\n第|$)', re.DOTALL)
    # 初始化字典来存储条款号和内容
    law_articles = {}
    # 搜索所有匹配项
    for match in pattern.finditer(data_str):
        article_number = match.group(1)
        article_content = match.group(0).replace('第' + article_number + '条', '').strip()
        law_articles[f"{title} 第{article_number}条"] = article_content
    # 转换字典为 JSON 字符串
    json_str = json.dumps(law_articles, ensure_ascii=False, indent=4)
    return json_str


if __name__ == '__main__':
    # 中华人民共和国劳动法
    json_str = extract_law_articles("中华人民共和国劳动法", "https://www.gov.cn/banshi/2005-05/25/content_905.htm")
    print(json_str)
    # 中华人民共和国劳动合同法
    json_str = extract_law_articles("中华人民共和国劳动合同法", "https://www.gov.cn/jrzg/2007-06/29/content_667720.htm")
    print(json_str)
