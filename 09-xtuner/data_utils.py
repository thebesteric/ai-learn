import json

# 源数据文件路径
source_file = 'data/ruozhiba_qaswift.json'
# 目标数据文件路径
target_file = 'data/target_data.json'

# 读取源数据
with open(source_file, 'r', encoding='utf-8') as f:
    source_data = json.load(f)

# 转换数据
target_data = []
for item in source_data:
    conversation = {
        "conversation": [
            {
                "input": item["query"],
                "output": item["response"]
            }
        ]
    }
    target_data.append(conversation)

# 保存转换后的数据
with open(target_file, 'w', encoding='utf-8') as f:
    json.dump(target_data, f, ensure_ascii=False, indent=4)

print(f"数据已成功转换并保存到 {target_file}")
