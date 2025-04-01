from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset

"""
自定义模型可以使用的数据集
"""


# 继承 torch 的 Dataset 类
class MyDataset(Dataset):
    # 初始化数据集
    def __init__(self, load_from="disk", dataset_path=None, split="train"):
        # 加载数据集
        if load_from == "csv":
            # 从 CSV 加载数据集
            self.dataset = load_dataset("csv", data_files=dataset_path + f"/{split}.csv", split="train")
        else:
            # 从磁盘加载数据集
            self.dataset = load_from_disk(dataset_path)
            if split == "train":
                self.dataset = self.dataset["train"]
            elif split == "test":
                self.dataset = self.dataset["test"]
            elif split == "validation":
                self.dataset = self.dataset["validation"]
            else:
                raise ValueError("split 参数错误，只能为 train、test、validation")

    # 返回数据集长度
    def __len__(self):
        return len(self.dataset)

    # 获取数据集的某一个样本
    def __getitem__(self, index):
        text = self.dataset[index]["text"]
        label = self.dataset[index]["label"]
        return text, label


if __name__ == '__main__':
    dataset_path = "/Users/wangweijun/LLM/datasets/lansinuote/ChnSentiCorp"
    datasets = MyDataset("disk", dataset_path, "test")

    # dataset_path = "/Users/wangweijun/LLM/datasets/lansinuote/ChnSentiCorp/csv_files"
    # datasets = MyDataset("csv", dataset_path, "test")

    # 1200
    print(datasets.__len__())
    # ('这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般', 1)
    print(datasets.__getitem__(0))

    for i in range(datasets.__len__()):
        if i == 10:
            break
        print(datasets.__getitem__(i))
