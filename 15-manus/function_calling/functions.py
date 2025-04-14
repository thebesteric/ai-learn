import json

import pymysql
import requests


def get_weather(city: str):
    """
    查询即时天气函数
    :param city: 必要参数，字符串类型，用于表示查询天气的具体城市名称
    注意：中国的城市需要用对应的城市的英文名称代替，例如如果需要查询北京的天气，则 city 参数需要输入 "Beijing"
    :return: OpenWeather API 查询即时天气的结果，返回结果对象类型为解析后的 JSON 格式对象，并用字符串形式表示，其中包含了全部中要的天气信息
    """
    # Open Weather API 配置
    open_weather_base_url = "https://api.openweathermap.org/data/2.5/weather"
    open_weather_api_key = "57ae333b23774c9bb9b82273213d7d47"
    params = {
        "q": city,
        "appid": open_weather_api_key,
        "lang": "zh_cn",
        "units": "metric",
    }
    response = requests.get(open_weather_base_url, params=params)
    data = response.json()
    return json.dumps(data, ensure_ascii=False)


def python_inter(python_code):
    """
    运行用户提供的 Python 代码，并返回执行结果
    :param python_code: 字符串形式的 Python 代码
    :return: 代码运行的最终结果
    """
    g = globals()

    try:
        # 若是表达式，直接运行并返回结果，如"1+1"，"max([1,2,3])"
        result = eval(python_code, g)
        return json.dumps(str(result), ensure_ascii=False)
    except Exception:
        # 记录执行前的全局变量
        global_vars_before = set(g.keys())
        try:
            # 处理完整的 Python 代码，如"a=1;b=2;print(a+b)"
            exec(python_code, g)
        except Exception as e:
            return json.dumps(f"代码执行出错：{str(e)}", ensure_ascii=False)

        global_vars_after = set(g.keys())
        new_vars = global_vars_after - global_vars_before

        if new_vars:
            safe_result = {}
            for var in new_vars:
                try:
                    # 尝试序列化，确保可以转换为
                    json.dumps(g[var])
                    safe_result[var] = g[var]
                except (TypeError, OverflowError):
                    # 如果无法序列化，则转换为字符串
                    safe_result[var] = str(g[var])
            return json.dumps(safe_result, ensure_ascii=False)
        else:
            return json.dumps("代码执行成功", ensure_ascii=False)


def sql_inter(sql_query):
    """
    查询本地 MySQL 数据库，通过运行一段 SQL 代码来进行数据库查询
    :param sql_query: 字符串形式的 SQL 查询语句，用于执行 MySQL 中的 school 数据库中各张表进行查询，并获取表中的各类相关信息
    :return: sql_query 的执行结果
    """
    connection = pymysql.connect(host='127.0.0.1',
                                 user='root',
                                 password='root',
                                 database='school',
                                 charset='utf8mb4')
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            results = cursor.fetchall()
    finally:
        connection.close()

    return json.dumps(results, ensure_ascii=False)


def write_file(filename, content):
    """
    将指定内容写入本地文件
    :param filename: 必要参数，字符串类型，用于表示需要写入文件的文件名
    :param content: 必要参数，字符串类型，用于表示需要写入文件的具体内容
    :return: 是否写入成功
    """
    with open(filename, "w") as file:
        file.write(content)
    return "已经成功写入本地文件"


tools_dict = {
    get_weather.__name__: {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询即时天气函数，输入城市名称，返回该城市的天气信息。一次只能查询一个城市的天气信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，注意：中国的城市需要用对应的城市的英文名称代替，例如如果需要查询北京的天气，则 city 参数需要输入 'Beijing'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    python_inter.__name__: {
        "type": "function",
        "function": {
            "name": "python_inter",
            "description": "运行用户提供的 Python 代码，并返回执行结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": "用户提供的 Python 代码，可以是表达式，也可以是完整的 Python 代码。"
                    }
                },
                "required": ["python_code"]
            }
        }
    },
    sql_inter.__name__: {
        "type": "function",
        "function": {
            "name": "sql_inter",
            "description": "查询本地 MySQL 数据库，通过运行一段 SQL 代码来进行数据库查询",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "字符串形式的 SQL 查询语句，用于执行 MySQL 中的 school 数据库中各张表进行查询，并获取表中的各类相关信息"
                    }
                },
                "required": ["sql_query"]
            }
        }
    },
    write_file.__name__: {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将指定内容写入本地文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名"
                    },
                    "content": {
                        "type": "string",
                        "description": "需要写入文件的具体内容"
                    }
                },
                "required": ["content"]
            }
        }
    }
}
