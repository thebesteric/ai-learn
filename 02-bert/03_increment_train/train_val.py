import time
from datetime import datetime

import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer, AdamW

"""
训练模型
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", DEVICE)

# 定义训练的轮次，轮次表示将整个数据集完整训练完一次为一轮

# 因为不知道什么时候能训练完全，先给大一点，因为中途需要监控，可以手动停止训练
EPOCHS = 30000

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


# 创建训练数据集
train_dataset = MyDataset(DATASET_PATH, "train")
train_loader = DataLoader(
    # 指定数据集
    dataset=train_dataset,
    # 批次越大，显存占用越大，训练速度越快
    batch_size=100,
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

# 创建验证数据集
val_dataset = MyDataset(DATASET_PATH, "validation")
val_loader = DataLoader(
    # 指定数据集
    dataset=train_dataset,
    # 批次越大，显存占用越大，训练速度越快
    batch_size=100,
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
    # 开始训练
    print(DEVICE)
    model = Model().to(DEVICE)
    # 定义优化器
    optimizer = AdamW(model.parameters())
    # 定义损失函数
    # CrossEntropyLoss 函数主要用于解决多分类问题，其核心作用是衡量模型预测结果与真实标签之间的差异。
    # 在训练深度学习模型时，目标是最小化这个损失函数的值，使得模型的预测结果尽可能接近真实标签。
    loss_fn = torch.nn.CrossEntropyLoss()

    # 初始化验证最佳准确率
    best_val_acc = 0.0

    # 开始训练
    for epoch in range(EPOCHS):
        for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(train_loader):
            # 将数据加载到 DEVICE 上
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            label = label.to(DEVICE)
            # 前向计算；将数据输入模型，得到输出
            out = model(input_ids, attention_mask, token_type_ids)
            # 根据输出，计算损失，就是计算两者的误差
            loss = loss_fn(out, label)
            # 根据误差优化参数
            # 所有参数的梯度清零
            optimizer.zero_grad()
            # 执行反向传播（求导），计算损失函数相对于模型所有可训练参数的梯度
            loss.backward()
            # 根据计算得到的梯度，使用优化算法来自动更新模型的参数
            optimizer.step()

            # 每隔 5 个批次，输出训练信息
            if i % 5 == 0:
                out = out.argmax(dim=1)
                # 计算训练精度
                # label: 0 0 1 0 1
                # out:   0 1 1 0 0
                # (out == label).sum().item() = 3
                # len(label) = 5
                # accuracy = 3 / 5 = 0.6
                accuracy = (out == label).sum().item() / len(label)

                datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"{datatime}: 训练集: epoch: {epoch}, i: {i}, loss: {loss}, acc: {accuracy}")

        # 验证模型，判断模型是否过拟合
        # 设置为评估模式
        model.eval()
        # 验证的时候，不需要模型参与训练，所以不需要梯度
        with torch.no_grad():
            val_acc = 0.0
            val_loss = 0.0
            for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(val_loader):
                # 将数据加载到 DEVICE 上
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                token_type_ids = token_type_ids.to(DEVICE)
                label = label.to(DEVICE)
                # 前向计算；将数据输入模型，得到输出
                out = model(input_ids, attention_mask, token_type_ids)
                # 根据输出，计算损失，就是计算两者的误差
                val_loss = loss_fn(out, label)
                # 根据输出计算验证的精度
                out = out.argmax(dim=1)
                val_acc += (out == label).sum().item() / len(label)

            # 计算验证的平均损失
            val_loss /= len(val_loader)
            # 计算验证的平均精度
            val_acc /= len(val_loader)

            datatime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{datatime}: 验证集: epoch: {epoch}, val_loss: {val_loss}, val_acc: {val_acc}")


            # 根据验证准确率，保存最优参数
            if val_acc > best_val_acc:
                # 把最优的参数保存下来，就是为了方式过拟合，因为一旦过拟合是无法回退的，如果没有保存，那么只有重新训练
                # 这就是为什么要保存最优参数的原因
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"params/best_bert.pth")
                print(f"epoch: {epoch}, 最优参数: acc = {best_val_acc}, 保存成功")

        # 保存最后一轮参数
        torch.save(model.state_dict(), f"params/last_train.pth")
        print(f"epoch: {epoch}, 最后一轮参数， 参数保存成功")
