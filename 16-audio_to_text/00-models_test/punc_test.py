from funasr import AutoModel

"""
标点恢复（Punctuation Restoration）

定义：在语音识别生成的文本中添加正确的标点符号。
目标：提高文本的可读性和语义完整性。
应用场景：自动生成带标点的会议记录、实时字幕。
"""

model = AutoModel(model="/Users/wangweijun/LLM/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")

res = model.generate(input="夏天有很多小朋友在沙滩上玩耍校园依山环虎风景如画冬天北风呼啸雪花飞舞")
# 夏天有很多小朋友在沙滩上玩耍，校园依山环虎，风景如画。冬天北风呼啸，雪花飞舞。
print(res)
