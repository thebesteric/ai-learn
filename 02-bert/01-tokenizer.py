from transformers import BertTokenizer

"""
AI 模型是如何处理字符数据的？
"""

# 加载字典和分词器
model_path = r"/Users/wangweijun/LLM/models/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
tokenizer = BertTokenizer.from_pretrained(model_path)
print(tokenizer)

"""
unk_token（[UNK]）：代表未知标记（Unknown token）。当模型碰到不在词汇表中的词时，就会用[UNK]来替代。
输入: "I love quantum physics"
如果 "quantum" 不在词汇表中，模型可能会将其替换为 "I love [UNK] physics"。

sep_token（[SEP]）：代表分隔标记（Separator token）。一般用于分隔不同的句子，像在问答任务或者句子对任务里。
输入: "How are you?[SEP]I am fine."

pad_token（[PAD]）：代表填充标记（Padding token）。在对输入序列进行填充时会用到，目的是让所有输入序列长度一致。
输入: "Hello world"
填充后: "Hello world[PAD][PAD][PAD]"

cls_token（[CLS]）：代表分类标记（Classification token）。一般放在输入序列的开头，用于分类任务，模型会利用这个标记的输出做分类决策。
输入: "[CLS]I love machine learning[SEP]"

mask_token（[MASK]）：代表掩码标记（Mask token）。在掩码语言模型（Masked Language Model, MLM）任务中使用，用来遮蔽部分输入词，让模型预测被遮蔽的词。
输入: "I love [MASK] learning"
模型预测: "I love machine learning"
"""

"""
理解 vocab.txt 文件
vocab_size: 21128，
如果是汉字的话，包含一组[单个汉字]和带[##的汉字]
单个汉字：表示一个字
带##的汉字：表示一个字的一部分，有上下文关系
通常都会使用单个汉字，方便扩展
所以其实模型可理解的汉字其实约为 21128 / 2 = 10564 个
例：
白日依山尽， -> 4635 3189 898 2255 2226 8024

为什么长了就截断，短了就补 0？
因为模型本身就是一个矩阵，所以输入的内容也要转换为一个矩阵，矩阵有个要求每个向量的长度要一致
"""

print("=" * 50, "我是分隔符", "=" * 50)

# 准备要编码的文本数据
sentence = ["白日依山尽，", "今天我的心情特别好"]

# 批量编码
encode = tokenizer.batch_encode_plus(
    # 要编码的文本数据
    batch_text_or_text_pairs=sentence,
    # 是否加入特殊字符
    add_special_tokens=True,
    # 表示编码后的最大长度，它的上限是 tokenizer_config.json 中的 model_max_length 的值
    max_length=12,
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
    return_tensors=None,
    return_attention_mask=True,
    return_token_type_ids=True,
    return_special_tokens_mask=True,
    # 返回编码后的序列长度
    return_length=True,
)
# input_ids：编码后的文本数据，0 表示 [PAD] 填充
# token_type_ids：第一个句子和特殊符号的位置是 0，第二个句子的位置是 1，只针对上下文的编码
# special_tokens_mask：特殊符号的位置是 1，其他位置是 0
# length：编码后的文本数据的长度
# attention_mask：注意力掩码，标识哪些位置是有意义的，有意义的事 1，哪些位置是填充的，填充的是 0
for k, v in encode.items():
    print(k, ":", v)

print("=" * 50, "我是分隔符", "=" * 50)

print(tokenizer.decode(encode.get("input_ids")[0]))
print(tokenizer.decode(encode.get("input_ids")[1]))
