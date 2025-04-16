import numpy as np
import torch
import whisper
import sounddevice as sd

# https://github.com/openai/whisper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
is_fp16 = True if torch.cuda.is_available() else False
print("is_fp16: ", is_fp16)

model_path = "/Users/wangweijun/LLM/models/openai/whisper-large-v3-turbo"

model = whisper.load_model("large-v3-turbo", device=device, download_root=model_path)

# 设置模型为评估模式
model.eval()


# 定义流式解码函数
def stream_decode(audio_buffer, sample_rate=16000):
    audio_tensor = torch.tensor(audio_buffer).float()
    result = model.transcribe(audio_tensor, fp16=is_fp16, language="zh", initial_prompt="这是一段独白内容。")
    return result['text']


# 音频缓冲区和其他参数
buffer_size = 16000  # 每个音频块的大小（1秒）
audio_buffer = np.zeros(buffer_size * 10, dtype=np.float32)  # 预留10秒缓冲区
buffer_offset = 0
silence_threshold = 0.01  # 声音门限


# 麦克风回调函数
def callback(indata, frames, time, status):
    global audio_buffer, buffer_offset

    if status:
        print(status, flush=True)

    # 计算当前音频块的音量
    volume_norm = np.linalg.norm(indata) * 10
    if volume_norm > silence_threshold:
        # 将新音频数据复制到缓冲区
        audio_buffer[buffer_offset:buffer_offset + frames] = indata[:, 0]
        buffer_offset += frames

        # 当缓冲区达到或超过设定的大小时进行处理
        if buffer_offset >= buffer_size:
            text = stream_decode(audio_buffer[:buffer_size])
            print(f"Transcription: {text}", flush=True)

            # 移动缓冲区的数据
            audio_buffer = np.roll(audio_buffer, -buffer_size)
            buffer_offset -= buffer_size
    else:
        # 如果检测到的音量低于门限，将缓冲区位置重置
        buffer_offset = 0


# 启动麦克风流
def start_streaming():
    stream = sd.InputStream(callback=callback, channels=1, samplerate=16000, blocksize=buffer_size)
    with stream:
        print("Listening...")
        while True:
            sd.sleep(1000)  # 继续监听


# 开始流式解码
start_streaming()
