import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# https://modelscope.cn/models/iic/SenseVoiceSmall

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = "/Users/wangweijun/LLM/models/iic/SenseVoiceSmall"
vad_model = "/Users/wangweijun/LLM/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
punc_model = "/Users/wangweijun/LLM/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

model = AutoModel(
    model=model,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model=vad_model,
    vad_kwargs={"max_single_segment_time": 30000},
    # punc_model=punc_model,
    device=device,
    disable_update=True
)

# en
res = model.generate(
    input="../sample_zh_8k.wav",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
print(res)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
