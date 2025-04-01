import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# 加载预训练的分词器，用于文本编码
tokenizer = AutoTokenizer.from_pretrained(
    r"/Users/wangweijun/LLM/models/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")
# 加载预训练的模型，用于语言模型任务
model = AutoModelForCausalLM.from_pretrained(
    r"/Users/wangweijun/LLM/models/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3")

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 加载训练的权重参数（中文古诗词），权重参数就是训练的结果
model.load_state_dict(torch.load("./params/net.pt", map_location=device))

# 将模型加载到设备
model = model.to(device)

# 定义一个函数，prompt 是提示词，row 是要生成的行数，col 是每行的字符数
def generate_text(prompt, row, col):
    # 定义一个递归函数，用于生成内容
    def generate_loop(data):
        # 禁用梯度计算
        with torch.no_grad():
            # 使用 data 字典中的数据作为模型的输入，获取输出
            out = model(**data)
        # 获取最后一个字（logits 表示未归一化的概率输出）
        out = out["logits"]
        # 选择每个序列最后一个 logits，对应于下一个词的预测
        out = out[:, -1]

        # 找到概率排名前 50 的值，并以此为分界线，小于该值的全部舍弃
        topk_value = torch.topk(out, 50).values
        # 获取每个输出序列中前 50 个最大的 logits，也就是第 50 名（为保持原纬度不变，需要对结果增加一个纬度，因为索引操作会降纬）
        topk_value = topk_value[:, -1].unsqueeze(dim=1)
        # 将所有小于第 50 位的 logits 设置为 -inf（负无穷大），减少低概率的选择，从而实现丢弃操作
        out = out.masked_fill(out < topk_value, -float("inf"))

        # 将特殊符号的 logits 的值设置为 -inf（负无穷大），防止模型选择这些符号
        for i in ",.()《》[]【】{}":
            out[:, tokenizer.get_vocab()[i]] = -float("inf")

        # 去除特殊符号
        out[:, tokenizer.get_vocab()["[UNK]"]] = -float("inf")
        out[:, tokenizer.get_vocab()["[SEP]"]] = -float("inf")
        out[:, tokenizer.get_vocab()["[PAD]"]] = -float("inf")

        # 根据概率进行采样，采用无放回采样，即从候选集中随机选择一个元素，而不重复选择
        out = out.softmax(dim=-1)
        # 从概率分布中采样，选择下一个词的 ID
        out = out.multinomial(num_samples=1)

        # 强制添加标点符号
        # 计算当前生成文本的长度与预期长度的倍数
        c = data["input_ids"].shape[1] / (col + 1)
        # 计算当前生成文本的长度是预期长度的整数倍，则添加标点符号
        if c % 1 == 0:
            if c % 2 == 0:
                # 偶数位是句号
                out[:, 0] = tokenizer.get_vocab()["。"]
            else:
                # 奇数位是逗号
                out[:, 0] = tokenizer.get_vocab()["，"]

        # 将生成新词 ID 添加到输入序列的末尾
        data["input_ids"] = torch.cat([data["input_ids"], out], dim=1)
        # 更新注意力掩码，标记所有有效位置
        data["attention_mask"] = torch.ones_like(data["input_ids"])
        # 更新 token 的 ID 类型，通常在 BERT 模型中使用，但是在 GPT 模型中是不用等
        data["token_type_ids"] = torch.ones_like(data["input_ids"])
        # 更新标签，这里将输入 ID 复制到标签中，在语言生成模型中通常用于预测下一个词
        data["labels"] = data["input_ids"].clone()

        # 检查生成的文本长度，是否超过或者达到指定的行数和列数
        if data["input_ids"].shape[1] >= row * col + row + 1:
            # 如果达到长度要求，则返回最终的 data 字典
            return data

        # 如果没有达到长度要求，则递归调用 generate_loop 函数，继续生成内容
        return generate_loop(data)

    # 生成 3 首古诗
    # 使用 tokenizer 对输入文本进行编码，并重复 3 次，以生成 3 首古诗
    data = tokenizer.batch_encode_plus([prompt] * 3, return_tensors="pt")
    # 移除编码后的序列中的最后一个 token（结束符号）
    data["input_ids"] = data["input_ids"][:, :-1]
    # 创建一个与 input_ids 相同形状的全 1 张量，作为注意力掩码
    data["attention_mask"] = torch.ones_like(data["input_ids"])
    # 创建一个与 input_ids 相同形状的全 0 张量，作为 token 的 ID 类型
    data["token_type_ids"] = torch.zeros_like(data["input_ids"])
    # 将输入 ID 复制到标签中，用于预测下一个词
    data["labels"] = data["input_ids"].clone()

    # 调用 generate_loop 函数，开始生成内容
    data = generate_loop(data)

    for i in range(3):
        # 将生成的 ID 转换为文本，并使用空格连接
        text = tokenizer.decode(data["input_ids"][i])
        # 输出生成的文本
        print(i, text)



if __name__ == '__main__':
    generate_text("白日", 4, 5)
