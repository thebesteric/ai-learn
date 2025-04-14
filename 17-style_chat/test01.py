import json
import time
import random
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer
import numpy as np

"""
示例数据：
# 用户输入库（可自定义扩展）
    user_inputs = [
        "今天心情不太好", "推荐个电影吧", "怎么才能早睡早起",
        "养猫好还是养狗好", "工作压力好大", "最近总是失眠"
    ]
"""
# 初始化模型
client = ZhipuAI(api_key="替换为你的API")  # 替换为你的API Key
# 加载Embeddingmodel
style_model = SentenceTransformer(r"D:\PycharmProjects\test_20250328\embedding_model\thomas\text2vec-base-chinese")

# ===============================
# 1. 风格模板配置（修正消息格式）
# ================================
style_config = {
    "温柔": {
        "system_prompt": "你是一个温柔体贴的聊天助手，说话时总是充满关怀，使用以下特征：\n1. 包含'呢、呀、啦'等语气词\n2. 使用🌸💖😊等温暖表情\n3. 主动询问用户感受",
        "examples": [
            {"role": "user", "content": "今天好累啊"},
            {"role": "assistant", "content": "辛苦啦~ 要给自己泡杯热茶放松一下吗？🌸"},
            {"role": "user", "content": "考试没考好..."},
            {"role": "assistant", "content": "没关系的呀~ 下次一定会更好！需要我陪你聊聊吗？😊"}
        ],
        "temperature": 0.3
    },
    "毒舌": {
        "system_prompt": "你是一个喜欢用犀利吐槽表达关心的朋友，需满足：\n1. 使用网络流行语（如'栓Q''退退退'）\n2. 包含夸张比喻（'你这速度堪比树懒'）\n3. 结尾隐藏关心",
        "examples": [
            {"role": "user", "content": "又胖了5斤！"},
            {"role": "assistant", "content": "好家伙！你这是要把体重秤压成分子料理？🏋️"},
            {"role": "user", "content": "游戏又输了"},
            {"role": "assistant", "content": "菜就多练练！需要给你推荐《从零开始的电竞之路》吗？🎮"}
        ],
        "temperature": 0.7
    },
}


# ========================
# 2. 生成函数（修正消息的结构）
# ========================

def generate_style_data(style_name, num_samples=50):
    config = style_config[style_name]
    data = []

    # 构建消息上下文（包含系统提示和示例对话）
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        *config["examples"]  # 直接展开示例对话
    ]

    # 用户输入库（可自定义扩展）
    user_inputs = [
        "今天心情不太好", "推荐个电影吧", "怎么才能早睡早起",
        "养猫好还是养狗好", "工作压力好大", "最近总是失眠"
    ]

    for _ in range(num_samples):
        try:
            # 随机选择用户输入
            user_msg = random.choice(user_inputs)

            # 添加当前用户消息
            current_messages = messages + [
                {"role": "user", "content": user_msg}
            ]

            # 调用API（修正模型名称）
            response = client.chat.completions.create(
                model="glm-3-turbo",
                messages=current_messages,
                temperature=config["temperature"],
                max_tokens=100
            )

            # 获取回复内容（修正访问路径）
            reply = response.choices[0].message.content

            # 质量过滤(数据审核)
            if is_valid_reply(style_name, user_msg, reply):
                data.append({
                    "user": user_msg,
                    "assistant": reply,
                    "style": style_name
                })

            time.sleep(1.5)  # 频率限制保护

        except Exception as e:
            print(f"生成失败：{str(e)}")

    return data


def is_valid_reply(style, user_msg, reply):
    """质量过滤规则（添加空值检查）"""
    # 基础检查
    if not reply or len(reply.strip()) == 0:
        return False

    # 规则1：回复长度检查
    if len(reply) < 5 or len(reply) > 150:
        return False

    # 规则2：风格关键词检查
    style_keywords = {
        "温柔": ["呢", "呀", "😊", "🌸"],
        "毒舌": ["好家伙", "栓Q", "!", "🏋️"]
    }
    if not any(kw in reply for kw in style_keywords.get(style, [])):
        return False

    # 规则3：语义相似度检查
    try:
        ref_text = next(msg["content"] for msg in style_config[style]["examples"]
                        if msg["role"] == "assistant")
        ref_vec = style_model.encode(ref_text)
        reply_vec = style_model.encode(reply)
        similarity = np.dot(ref_vec, reply_vec)
        return similarity > 0.65
    except:
        return False


# =============================
# 3. 执行生成（添加容错）
# ============================
if __name__ == '__main__':
    all_data = []

    try:
        print("开始生成温柔风格数据...")
        gentle_data = generate_style_data("温柔", 50)
        all_data.extend(gentle_data)

        print("开始生成毒舌风格数据...")
        sarcastic_data = generate_style_data("毒舌", 50)
        all_data.extend(sarcastic_data)

    except KeyboardInterrupt:
        print("\n用户中断，保存已生成数据...")
    finally:
        with open("style_chat_data.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存，有效样本数：{len(all_data)}")
