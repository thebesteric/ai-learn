import torch
from net import Model
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from MyData import MyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", DEVICE)

MODEL_PATH = r"/Users/wangweijun/LLM/models/bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

# 实例化模型
model = Model().to(DEVICE)
# 评价结果，要和模型输出的维度保持一致
results = ['负向评价', '正向评价']

# 将传入的字符串进行编码
def collate_fn(text):
    sentence = [text]
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

    return input_ids, attention_mask, token_type_ids


def test():
    # 加载模型训练参数
    model.load_state_dict(torch.load("params/4.pth"))
    # 开启评估模型
    model.eval()

    while True:
        # 输入文本
        text = input("请输入测试文本（输入'q'退出）：")
        if text == 'q':
            break
        input_ids, attention_mask, token_type_ids = collate_fn(text)
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE)

        # 前向计算；将数据输入模型，得到输出
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # dim = 1 是因为 out 的形状是 [1, 2], 如：[[0], [1]]，我们只需要，[0], [1]
            result = torch.argmax(out, dim=1)
            print("预测结果：", results[result])

if __name__ == '__main__':
    test()
