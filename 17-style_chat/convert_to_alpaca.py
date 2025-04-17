import json


def convert_json(input_json):
    result = []
    for input in input_json:
        for item in input["conversation"]:
            new_item = {
                "instruction": item["input"],
                "input": "",
                "output": item["output"]
            }
            result.append(new_item)
    return result


try:
    # 从文件中读取输入的 JSON 数据
    with open('xtuner_train_output.json', 'r', encoding='utf-8') as file:
        input_data = json.load(file)

    # 进行转换
    output_data = convert_json(input_data)

    # 将转换后的数据写入文件
    with open('alpaca_style_output.json', 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    print("转换完成，结果已保存到 alpaca_style_output.json 文件中。")
except FileNotFoundError:
    print("错误：未找到输入文件 'xtuner_train_output.json'。")
except json.JSONDecodeError:
    print("错误：输入文件不是有效的 JSON 格式。")
except Exception as e:
    print(f"发生未知错误：{e}")
