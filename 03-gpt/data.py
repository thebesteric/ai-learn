from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        self.lines = [line.strip() for line in lines]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        return self.lines[index]


if __name__ == '__main__':
    dataset = MyDataset("./data/chinese_poems.txt")
    for i, data in enumerate(dataset):
        print(data)
        if i > 10:
            break