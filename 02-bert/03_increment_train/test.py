import os
from datetime import datetime

import torch
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from net import Model
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from MyData import MyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = r"/Users/wangweijun/llm/datasets/ChnSentiCorp"
MODEL_PATH = r"/Users/wangweijun/llm/models/bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


# 将传入的字符串进行编码
def collate_fn(data):
    sentence = [item[0] for item in data]
    label = [item[1] for item in data]
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


def evaluate_model(model, test_loader, device):
    """
    评估模型在测试集上的性能
    :param model: 待评估模型
    :param test_loader: 测试数据加载器
    :param device: 计算设备
    :return: 评估指标字典
    """
    model.eval()
    all_preds, all_labels = [], []

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        # 将数据转移到设备
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(outputs, dim=1)

        # 收集预测结果
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro'),
        'recall_macro': recall_score(all_labels, all_preds, average='macro'),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision_weighted': precision_score(all_labels, all_preds, average='weighted'),
        'recall_weighted': recall_score(all_labels, all_preds, average='weighted'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'classification_report': classification_report(all_labels, all_preds, digits=4)
    }
    return metrics


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    绘制并保存混淆矩阵
    :param cm: 混淆矩阵
    :param class_names: 类别名称列表
    :param save_path: 保存路径（可选）
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    plt.show()


def save_metrics_to_file(metrics, save_path):
    """
    将评估指标保存到文本文件
    :param metrics: 评估指标字典
    :param save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("模型评估报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n\n")

        f.write("宏平均指标 (Macro-average):\n")
        f.write(f"  精确率 (Precision): {metrics['precision_macro']:.4f}\n")
        f.write(f"  召回率 (Recall): {metrics['recall_macro']:.4f}\n")
        f.write(f"  F1分数 (F1 Score): {metrics['f1_macro']:.4f}\n\n")

        f.write("加权平均指标 (Weighted-average):\n")
        f.write(f"  精确率 (Precision): {metrics['precision_weighted']:.4f}\n")
        f.write(f"  召回率 (Recall): {metrics['recall_weighted']:.4f}\n")
        f.write(f"  F1分数 (F1 Score): {metrics['f1_weighted']:.4f}\n\n")

        f.write("分类报告 (Classification Report):\n")
        f.write(metrics['classification_report'])

        f.write("\n\n混淆矩阵 (Confusion Matrix):\n")


# 创建数据集
test_dataset = MyDataset("disk", DATASET_PATH, "test")
print(f"数据集大小：{len(test_dataset)}")
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
    print("device: ", DEVICE)
    model = Model(768, 2).to(DEVICE)

    # 加载模型训练参数
    model_weight_path = "params/best_bert.pth"
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"模型权重文件不存在: {model_weight_path}")
    # 加载权重
    model.load_state_dict(torch.load(model_weight_path))

    # 评估模型
    metrics = evaluate_model(model, test_loader, DEVICE)

    # 打印评估结果
    print("\n" + "=" * 50)
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print("\n宏平均指标 (Macro-average):")
    print(f"  精确率 (Precision): {metrics['precision_macro']:.4f}")
    print(f"  召回率 (Recall): {metrics['recall_macro']:.4f}")
    print(f"  F1分数 (F1 Score): {metrics['f1_macro']:.4f}")

    print("\n加权平均指标 (Weighted-average):")
    print(f"  精确率 (Precision): {metrics['precision_weighted']:.4f}")
    print(f"  召回率 (Recall): {metrics['recall_weighted']:.4f}")
    print(f"  F1分数 (F1 Score): {metrics['f1_weighted']:.4f}")

    print("\n分类报告 (Classification Report):")
    print(metrics['classification_report'])

    # 可视化混淆矩阵
    # 注意：根据您的实际类别修改class_names
    class_names = ["类别0", "类别1"]  # 替换为您的实际类别名称
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, "confusion_matrix.png")

    # 保存评估结果
    save_metrics_to_file(metrics, "evaluation_report.txt")

    print("评估完成!")

    # # 开启测试模型
    # model.eval()
    # # 开始测试
    # for i, (input_ids, attention_mask, token_type_ids, label) in enumerate(test_loader):
    #     # 将数据加载到 DEVICE 上
    #     input_ids = input_ids.to(DEVICE)
    #     attention_mask = attention_mask.to(DEVICE)
    #     token_type_ids = token_type_ids.to(DEVICE)
    #     label = label.to(DEVICE)
    #     # 前向计算；将数据输入模型，得到输出
    #     out = model(input_ids, attention_mask, token_type_ids)
    #
    #     # print(f"out: {out.argmax(dim=1)}")
    #     # print(f"label: {label}")
    #
    #     # 将输出转换为张量
    #     out = out.argmax(dim=1)
    #     # 计算正确率的个数
    #     acc += (out == label).sum().item()
    #     # 计算总个数
    #     total += len(label)
    #     # 打印每批正确个数
    #     print(f"第 {i + 1} 批次，正确个数: {(out == label).sum().item()}")
    #
    # print(f"最终平均精度为 acc: {acc / total}")
