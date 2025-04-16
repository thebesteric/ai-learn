from modelscope.pipelines import pipeline

"""
声纹识别（VoicePrint Recognition）

定义：通过比较说话人的语音特征来确定说话人身份。
目标：验证用户身份，通常用于安全认证、身份验证等场景。
应用场景：人脸识别、门禁系统、身份验证系统。
"""

sv_pipeline = pipeline(
    task='speaker-verification',
    model='/Users/wangweijun/LLM/models/iic/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)
speaker1_a_wav = '../speaker1_a_cn_16k.wav'
speaker1_b_wav = '../speaker1_b_cn_16k.wav'
speaker2_a_wav = '../speaker2_a_cn_16k.wav'

# 相同说话人语音
result = sv_pipeline([speaker1_a_wav, speaker1_b_wav])
# {'score': 0.6936, 'text': 'yes'}
print(result)

# 不同说话人语音
result = sv_pipeline([speaker1_a_wav, speaker2_a_wav])
# {'score': -0.08418, 'text': 'no'}
print(result)
