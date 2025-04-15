import torch
import whisper

# https://github.com/openai/whisper

# pip install -U openai-whisper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_fp16 = True if torch.cuda.is_available() else False
print(f"device: {device}, is_fp16: {is_fp16}")

model_path = "/Users/wangweijun/LLM/models/openai/whisper-large-v3-turbo"

model = whisper.load_model("large-v3-turbo", device=device, download_root=model_path)

result = model.transcribe(audio="sample_zh_8k.wav", fp16=is_fp16, language="zh", initial_prompt="这是一段独白内容。")
print(result)