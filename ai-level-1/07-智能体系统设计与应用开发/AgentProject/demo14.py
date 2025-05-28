from openai import OpenAI
import os
import pymysql
import json
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

database_schema_string = """
-- 创建班级表
CREATE TABLE Classes (
    class_id INT PRIMARY KEY,
    class_name VARCHAR(100) NOT NULL
);

-- 创建学生表
CREATE TABLE Students (
    student_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    class_id INT,
    FOREIGN KEY (class_id) REFERENCES Classes(class_id)
);

-- 创建成绩表
CREATE TABLE Scores (
    score_id INT PRIMARY KEY,
    student_id INT,
    subject VARCHAR(100) NOT NULL,
    score FLOAT NOT NULL,
    FOREIGN KEY (student_id) REFERENCES Students(student_id)
);
"""

"""
-- 插入班级表数据
INSERT INTO Classes (class_id, class_name) VALUES
(1, '一班'),
(2, '二班'),
(3, '三班'),
(4, '四班'),
(5, '五班');

-- 插入学生表数据
INSERT INTO Students (student_id, NAME, class_id) VALUES
(1, '张三', 1),
(2, '李四', 1),
(3, '王五', 2),
(4, '赵六', 3),
(5, '钱七', 4);

-- 插入成绩表数据
INSERT INTO Scores (score_id, student_id, SUBJECT, score) VALUES
(1, 1, '数学', 85.5),
(2, 1, '英语', 90.0),
(3, 2, '数学', 78.0),
(4, 3, '英语', 88.5),
(5, 4, '数学', 92.0);
"""

connection = pymysql.connect(
    host='127.0.0.1',  # 例如 'localhost'
    port=3306,  # MySQL默认端口号
    user='root',  # MySQL用户名
    password='root',  # MySQL用户密码
    database='test'  # 要连接的数据库名
)
cursor = connection.cursor()

def get_sql_completion(messages, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
        tools=[{
            # 摘自 OpenAI 官方示例 https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": "使用这个函数来回答有关业务的用户问题。输出应该是一个完整的SQL查询",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                            SQL查询提取信息以回答用户的问题。
                            SQL应该使用这些表结构来编写:
                            {database_schema_string}
                            查询应以纯文本形式返回，而不是JSON格式。
                            查询应仅包含MySQL支持的语法.
                            """,
                        }
                    },
                    "required": ["query"],
                }
            }
        }],
    )
    return response.choices[0].message

def ask_database(query):
    cursor.execute(query)
    records = cursor.fetchall()
    return records

prompt = "查询一班的学生`数学`成绩是多少？"
messages = [
    {"role": "system", "content": "基于表回答用户问题"},
    {"role": "user", "content": prompt}
]
response = get_sql_completion(messages)
# print(response.tool_calls[0].function.arguments)
# if response.content is None:
#     response.content = ""
messages.append(response)
# print("====Function Calling====")
# print(response)

if response.tool_calls is not None:
    tool_call = response.tool_calls[0]
    if tool_call.function.name == "ask_database":
        arguments = tool_call.function.arguments
        args = json.loads(arguments)
        print("====SQL====")
        print(args["query"])
        result = ask_database(args["query"])
        print("====MySQL数据库查询结果====")
        print(result)
#
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": "ask_database",
            "content": str(result)
        })
        response = get_sql_completion(messages)
        print("====最终回复====")
        print(response)