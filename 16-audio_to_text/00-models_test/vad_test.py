from funasr import AutoModel

"""
语音端点检测 （VAD，Voice Activity Detection）

定义：检测语音信号中的有效语音片段，去除静音或背景噪声。
目标：确定语音开始和结束的时间点。
应用场景：优化语音识别性能、降低计算资源消耗。
"""

model = AutoModel(model="/Users/wangweijun/LLM/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")

wav_file = '../sample_zh_8k.wav'
res = model.generate(input=wav_file)

# 夏天，有很多小朋友在沙滩上玩耍，校园依山环虎，风景如画，冬天北风呼啸，雪花飞舞。忽然一道闪电，把天空和大地都照亮了。古城门附近是一个休闲散心的好去处。
# 输出结果包含文件名和连续有效语音时间跨度时间戳，单位毫秒；
# [{'key': 'sample_zh_8k', 'value': [[440, 4330], [4800, 8210], [8650, 12260], [13000, 17290], [18100, 22430]]}]
print(res)
