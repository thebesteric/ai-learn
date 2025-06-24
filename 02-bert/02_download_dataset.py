from datasets import load_dataset, load_from_disk

# pip install datasets

dataset_path="/Users/wangweijun/llm/datasets/ChnSentiCorp"

# 下载数据集到本地，默认下载在：~/.cache/huggingface/datasets
# dataset = load_dataset(path="lansinuote/ChnSentiCorp", cache_dir="dataset_path")
# dataset.save_to_disk(dataset_path)

# 直接读取本地数据集
datasets = load_from_disk(dataset_path=dataset_path)
print(datasets)
print("=" * 50, "我是分隔符", "=" * 50)

# 扩展：转换为 CSV 格式
# csv_dir = "/Users/wangweijun/llm/datasets/ChnSentiCorp/csv_files"
# for split_name in datasets.keys():
#     csv_path = f"{csv_dir}/{split_name}.csv"
#     datasets[split_name].to_csv(csv_path)
#     print(f"{split_name} split saved to {csv_path}")
# print("=" * 50, "我是分隔符", "=" * 50)

# 扩展：加载 CSV 格式数据
dataset = load_dataset(path="csv", data_files=f"{dataset_path}/csv_files/train.csv")
print(dataset)
print("=" * 50, "我是分隔符", "=" * 50)

# 获取训练集数据集
train_datasets = datasets["train"]
for i, data in enumerate(train_datasets, 1):
    if i <= 10:
        print(i, data)
    else:
        break

