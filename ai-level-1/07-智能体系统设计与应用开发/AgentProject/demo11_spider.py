from openai import OpenAI
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
import pandas as pd
from datetime import datetime

# 爬虫工具库：https://spidertools.cn/#/curl2Request


client = OpenAI()


def check_ticket(date, start, end):
    city_mapping = {
        "北京": "BJP",
        "上海": "SHH",
        "合肥": "HFH",
    }
    start = city_mapping[start]
    end = city_mapping[end]

    url = 'https://kyfw.12306.cn/otn/leftTicket/queryG?leftTicketDTO.train_date={}&leftTicketDTO.from_station={}&leftTicketDTO.to_station={}&purpose_codes=ADULT'.format(
        date, start, end)
    headers = {
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "If-Modified-Since": "0",
        "Pragma": "no-cache",
        "Referer": "https://kyfw.12306.cn/otn/leftTicket/init?linktypeid=dc",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "sec-ch-ua": "\"Chromium\";v=\"128\", \"Not;A=Brand\";v=\"24\", \"Google Chrome\";v=\"128\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"macOS\""
    }
    cookies = {
        "_uab_collina": "",
        "JSESSIONID": "",
        "_jc_save_wfdc_flag": "dc",
        "_jc_save_fromStation": "%%u6C99%",
        "guidesStatus": "off",
        "highContrastMode": "",
        "cursorStatus": "off",
        "BIGipServerotn": "..0000",
        "BIGipServerpassport": "..0000",
        "route": "",
        "_jc_save_toStation": "%u4E0Au6D77%2CSHH",
        "_jc_save_fromDate": "",
        "_jc_save_toDate": ""
    }

    session = requests.session()
    res = session.get(url, headers=headers, cookies=cookies)

    data = res.json()

    # 这是一个列表
    result = data["data"]["result"]

    lis = []
    for index in result:
        index_list = index.replace('有', 'Yes').replace('无', 'No').split('|')
        # print(index_list)
        train_number = index_list[3]  # 车次

        if 'G' in train_number:
            time_1 = index_list[8]  # 出发时间
            time_2 = index_list[9]  # 到达时间
            prince_seat = index_list[25]  # 特等座
            first_class_seat = index_list[31]  # 一等座
            second_class = index_list[30]  # 二等座
            dit = {
                '车次': train_number,
                '出发时间': time_1,
                '到站时间': time_2,
                "是否可以预定": index_list[11],

            }
            lis.append(dit)
        else:
            # print(index_list)
            time_1 = index_list[8]  # 出发时间
            time_2 = index_list[9]  # 到达时间

            dit = {
                '车次': train_number,
                '出发时间': time_1,
                '到站时间': time_2,
                "是否可以预定": index_list[11],

            }
            lis.append(dit)
    # print(lis)
    content = pd.DataFrame(lis)
    # print(content)
    return content


def check_date():
    today = datetime.now().date()
    return today


def get_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "check_ticket",
                    "description": "给定日期查询有没有票",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "日期",
                            },
                            "start": {
                                "type": "string",
                                "description": "出发站",
                            },
                            "end": {
                                "type": "string",
                                "description": "终点站",
                            }

                        },

                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_date",
                    "description": "返回当前的日期",
                    "parameters": {
                        "type": "object",
                        "properties": {

                        }
                    }
                }
            }
        ]
    )
    return response.choices[0].message


prompt = "查询后天 北京到上海的票"

messages = [
    {"role": "system", "content": "你是一个有用的助手，可以解决任何问题"},
    {"role": "user", "content": prompt}
]
response = get_completion(messages)

if (response.content is None):  # 解决 OpenAI 的一个 400 bug
    response.content = ""
messages.append(response)  # 把大模型的回复加入到对话中
print("=====GPT回复=====")
print(response)

# 如果返回的是函数调用结果，则打印出来
while (response.tool_calls is not None):
    # 1106 版新模型支持一次返回多个函数调用请求
    for tool_call in response.tool_calls:
        args = json.loads(tool_call.function.arguments)
        print("参数：", args)

        if (tool_call.function.name == "check_ticket"):
            print("Call: check_ticket")
            result = check_ticket(**args)
        elif (tool_call.function.name == "check_date"):
            print("Call: check_date")
            result = check_date()

        print("=====函数返回=====")
        print(result)

        messages.append({
            "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
            "role": "tool",
            "name": tool_call.function.name,
            "content": str(result)  # 数值result 必须转成字符串
        })

    response = get_completion(messages)
    if (response.content is None):  # 解决 OpenAI 的一个 400 bug
        response.content = ""
    print("=====GPT回复2=====")
    print(response)
    messages.append(response)  # 把大模型的回复加入到对话中

print("=====最终回复=====")
print(response.content)