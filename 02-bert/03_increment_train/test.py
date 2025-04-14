from datetime import datetime

import torch
from net import Model
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from MyData import MyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", DEVICE)

DATASET_PATH = r"/Users/wangweijun/LLM/datasets/ChnSentiCorp"
MODEL_PATH = r"/Users/wangweijun/LLM/models/bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


# 将传入的字符串进行编码
def collate_fn(data):
    sentence = [i[0] for i in data]
    label = [i[1] for i in data]
    # 编码
    data = tokenizer.batch_encode_plus(
        # 要编码的文本数据
        batch_text_or_text_pairs=sentence,
        # 是否加入特殊字符
        add_special_tokens=True,
        # 表示编码后的最大长度，它的上限是 tokenizer_config.json 中的 model_max_length 的值
        max_length=512,
        # 是否切断文本，以适应文本最大的输入长度，即：长了就截断
        truncation=True,
        # 一律补 0 到 max_length，即：短了就补 0
        padding="max_length",
        # 编码后返回的类型
        # 可选：tf、pt、np，None
        # tf：返回 TensorFlow 的张量 Tensor
        # pt：返回 PyTorch 的张量 torch.Tensor
        # np：返回 Numpy 的数组 ndarray
        # None：返回 Python 的列表 list
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=True,
        return_special_tokens_mask=True,
        # 返回编码后的序列长度
        return_length=True,
    )

    # 编码后的文本数据
    input_ids = data["input_ids"]
    # attention_mask：注意力掩码，标识哪些位置是有意义的，有意义的事 1，哪些位置是填充的，填充的是 0
    attention_mask = data["attention_mask"]
    # token_type_ids：第一个句子和特殊符号的位置是 0，第二个句子的位置是 1，只针对上下文的编码
    token_type_ids = data["token_type_ids"]
    # 标签，转换为张量
    label = torch.LongTensor(label)

    return input_ids, attention_mask, token_type_ids, label

# 创建数据集
test_dataset = MyDataset(DATASET_PATH, "test")
test_loader = DataLoader(
    # 指定数据集
    dataset=test_dataset,
    # 批次越大，显存占用越大，训练速度越快
    batch_size=200,
    # 打乱数据
    shuffle=True,
    # 舍弃最后一个批次，防止形状出错
    # 比如：数据总共有 1000 条，批次大小为 100，那么最后一个批次就只有 100 条，形状就不会出错
    # 比如：数据总共有 1003 条，批次大小为 100，那么最后一个批次就只有 3 条，形状就会出错
    # 因为数据是被打乱了，训练轮数也不止一轮，所以舍弃的数据，一定有概率会被学到
    drop_last=True,
    # 加载的数据进行编码
    collate_fn=collate_fn
)

if __name__ == '__main__':
    acc = 0.0
    total = 0

    # 开始测试
    print(DEVICE)
    model = Model().to(DEVICE)

    # 加载模型训练参数
    model.load_state_dict(torch.load("params/4.pth"))
    # 开启测试模型
    model.eval()

    # 开始测试

    for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(test_loader):
        # 将数据加载到 DEVICE 上
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        label = label.to(DEVICE)
        # 前向计算；将数据输入模型，得到输出
        out = model(input_ids, attention_mask, token_type_ids)

        # print(f"out: {out.argmax(dim=1)}")
        # print(f"label: {label}")

        # 将输出转换为张量
        out = out.argmax(dim=1)
        # 计算正确率的个数
        acc += (out == label).sum().item()
        # 计算总个数
        total += len(label)
        # 打印每批正确个数
        print(f"第 {i+1} 批次，正确个数: {(out == label).sum().item()}")


    print(f"最终平均精度为 acc: {acc / total}")