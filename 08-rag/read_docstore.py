import json

with open("./storage/docstore.json", "r", encoding="utf-8") as file:
    data = json.load(file)

format_json = json.dumps(data, ensure_ascii=False, indent=4)
print(format_json)