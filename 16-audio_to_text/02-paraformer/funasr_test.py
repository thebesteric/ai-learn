from funasr import AutoModel

# https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
# https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/paraformer/README_zh.md

model = "/Users/wangweijun/LLM/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
vad = "/Users/wangweijun/LLM/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
punc = "/Users/wangweijun/LLM/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

voice_model = AutoModel(model=model, model_revision="v2.0.4",
                        vad_model=vad, vad_model_revision="v2.0.4",
                        punc_model=punc, punc_model_revision="v2.0.4",
                        disable_update=True)

res = voice_model.generate(input="../sample_zh_8k.wav")
print(res)

rec_result = voice_model.generate(input='../recorded_audio.mp3')
print(rec_result[0]["text"])
