import torch
from faster_whisper import WhisperModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# https://huggingface.co/Systran/faster-whisper-large-v3

model_path = "/Users/wangweijun/LLM/models/openai/faster-whisper-large-v3-turbo/snapshots/0c94664816ec82be77b20e824c8e8675995b0029"
model = WhisperModel(model_path, device=str(device), compute_type="float32")
print(model)

segments, info = model.transcribe("sample_zh_8k.wav", beam_size=1, initial_prompt="这是一段独白内容。")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))



