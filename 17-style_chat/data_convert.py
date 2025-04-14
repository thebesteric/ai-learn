import json


def convert_format(source_data):
    target_data = []
    for item in source_data:
        # 构建新的对话格式
        new_convo = {
            "conversation": [
                {
                    "input": item["user"],
                    "output": f"{item['style']}\n{item['assistant']}"
                }
            ]
        }
        target_data.append(new_convo)
    return target_data


# 从文件读取源数据
with open("style_chat_data.json", "r", encoding="utf-8") as f:
    source_data = json.load(f)

# 执行转换
converted_data = convert_format(source_data)

# 写入目标文件
with open("xtuner_train_output.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)
