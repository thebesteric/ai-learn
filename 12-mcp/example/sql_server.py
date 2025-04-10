import asyncio
import csv
import json

import pymysql
from mcp.server import FastMCP

mcp = FastMCP("SQLServer")
USER_AGENT = "SQLServer-app/1.0"


@mcp.tool()
async def sql_inter(sql_query):
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


@mcp.tool()
async def export_table_to_csv(table_name, output_file):
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


async def test():
    print(await sql_inter("SELECT * FROM student_scores"))


if __name__ == '__main__':
    # asyncio.run(test())
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport="stdio")
