from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

model = "/Users/wangweijun/LLM/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
vad = "/Users/wangweijun/LLM/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
punc = "/Users/wangweijun/LLM/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=model, model_revision="v2.0.4",
    vad_model=vad, vad_model_revision="v2.0.4",
    punc_model=punc, punc_model_revision="v2.0.4")

print(inference_pipeline)

rec_result = inference_pipeline(input='../sample_zh_8k.wav')
print(rec_result)


rec_result = inference_pipeline(input='../recorded_audio.mp3')
print(rec_result)