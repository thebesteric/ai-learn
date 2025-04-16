import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# https://modelscope.cn/models/iic/SenseVoiceSmall

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = "/Users/wangweijun/LLM/models/iic/SenseVoiceSmall"

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=model,
    model_revision="master",
    device=device)

rec_result = inference_pipeline('../sample_zh_8k.wav')
print(rec_result)

rec_result = inference_pipeline('../recorded_audio.mp3')
print(rec_result)
