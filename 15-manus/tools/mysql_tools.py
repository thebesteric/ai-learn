import csv
import json
import os

import pandas as pd
import pymysql
from dotenv import load_dotenv


def sql_inter(sql_query, g='globals()'):
    """
    用于执行一段 SQL 代码，并最终获取 SQL 代码的执行结果
    :param sql_query: 字符串形式的 SQL 查询语句，用于执行 MySQL 中的 school 数据库中各张表进行查询，并获取表中的各类相关信息
    :param g: 字符串形式变量，表示环境变量，无须设置，保持默认参数即可
    :return: sql_query 的执行结果
    """
    print("正在调用 sql_inter 查询数据库...")

    load_dotenv(override=True)
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    database = os.getenv("DB_NAME")

    connection = pymysql.connect(host=host,
                                 port=int(port),
                                 user=user,
                                 password=password,
                                 database=database,
                                 charset='utf8mb4')
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            print("✅ SQL 查询成功")
    finally:
        connection.close()

    print("正在进行最后的整理...")
    return json.dumps(results, ensure_ascii=False)

sql_inter_args_example = '{"sql_query": "SHOW TABLES;"}'
sql_inter_tool = {
    "type": "function",
    "function": {
        "name": "sql_inter",
        "description": "当用户需要使用 SQL 进行数据库查询任务时，请调用此函数。\n"
                       "该函数使用 pymsql 来连接 MySQL 数据库，用于在指定的 MySQL 服务器上运行一段 SQL 代码，完成数据库查询相关工作。\n"
                       "📌 注意：\n"
                       "- 本函数只负责运行 SQL 代码并进行查询，如果要进行数据提取，则使用 extract_data 函数；\n"
                       "- 同时需要注意，编写外部函数的参数消息时，必须是满足 json 格式的字符串。\n\n"
                       f"例如：合规的示例代码：{sql_inter_args_example}"
    },
    "parameters": {
        "type": "object",
        "properties": {
            "sql_query": {
                "type": "string",
                "description": "The SQL query to execute in MySQL database."
            },
            "g": {
                "type": "string",
                "description": "Global environment variables, default to globals().",
                "default": "globals()"
            }
        },
        "required": ["sql_query"]
    }
}

def extract_data(sql_query, df_name, g='globals()'):
    """
    借助 pymsql 将 MySQL 中的某张表读取，并保存到本地
    :param sql_query: 字符串形式的 SQL 查询语句，用于提取 MySQL 中的某张表
    :param df_name: 将 MySQL 数据库中提取的表进行本地保存时的变量名，以字符串形式表示
    :param g: 字符串形式变量，表示环境变量，无须设置，保持默认参数即可
    :return: 表格读取和保存结果
    """
    print("正在调用 extract_data 提取数据库数据...")

    load_dotenv(override=True)
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    database = os.getenv("DB_NAME")

    connection = pymysql.connect(host=host,
                                 port=int(port),
                                 user=user,
                                 password=password,
                                 database=database,
                                 charset='utf8mb4')

    g[df_name] = pd.read_sql(sql_query, connection)
    print("✅ 数据提取成功")
    return f"以成功创建 pandas 对象 {df_name}，该变量保存了同名表格的数据信息"


extract_data_args_example = '{"sql_query": "SELECT * FROM student;", "df_name": "student"}'
extract_data_tool = {
    "type": "function",
    "function": {
        "name": "extract_data",
        "description": "当用户需要从 MySQL 数据库中提取某张表的数据到当前 Python 到环境中，注意，本函数只负责数据表的提取。"
                       "并不负责数据查询，若需要在 MySQL 中进行数据查询，请使用 sql_inter 函数。\n"
                       "同时需要注意，编写外部函数的参数消息时，必须是满足 json 格式的字符串。\n"
                       f"例如：合规的示例代码：{extract_data_args_example}\n",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "The SQL query to extract a table from MySQL database."
                },
                "df_name": {
                    "type": "string",
                    "description": "The name of the variable to store the  extracted table in the local environment."
                },
                "g": {
                    "type": "string",
                    "description": "Global environment variables, default to globals().",
                    "default": "globals()"
                }
            },
            "required": ["sql_query", "df_name"]
        }
    }
}


def export_table_to_csv(table_name, output_file):
    """
    将 MySQL 数据库中的某个表导出为 CSV 文件
    :param table_name: 需要导出的表名
    :param output_file: 输出的 CSV 文件路径
    :return:
    """
    connection = pymysql.connect(host='127.0.0.1',
                                 user='root',
                                 password='root',
                                 database='school',
                                 charset='utf8mb4')
    try:
        with connection.cursor() as cursor:
            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)

            # 获取所有列名
            column_names = [desc[0] for desc in cursor.description]
            # 获取查询结果
            rows = cursor.fetchall()
            with open(output_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(column_names)
                # 写入数据
                writer.writerows(rows)
            print(f"数据表 {table_name} 已经成功导出到文件 {output_file}")
    finally:
        connection.close()


if __name__ == '__main__':
    result = sql_inter("SHOW TABLES;", g=globals())
    print(result)

    extract_data("SELECT * FROM student;", "student", g=globals())
    print(globals()["student"])