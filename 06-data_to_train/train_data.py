import json

# 读取原始 JSON 文件
input_file = "data/ruozhiba_qaswift.json"  # 你的 JSON 文件名
output_file = "data/ruozhiba_qaswift_train.json"  # 输出的 JSON 文件名

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换后的数据
converted_data = []

for item in data:
    converted_item = {
        "instruction": item["query"],
        "input": "",
        "output": item["response"]
    }
    converted_data.append(converted_item)

# 保存为 JSON 文件（最外层是列表）
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=4)

print(f"转换完成，数据已保存为 {output_file}")