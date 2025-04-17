import re
import json
from typing import Dict, Any


def universal_converter(original_template: Dict[str, Any]) -> Dict[str, Any]:
    """将多种风格的原始模板转换为lmdeploy官方格式"""

    # 字段映射关系（核心逻辑）
    field_mapping = {
        # 基础字段映射
        "SYSTEM": "system",
        "INSTRUCTION": ("user", "assistant"),  # 需要拆分处理
        "SUFFIX": "eoa",
        "SEP": "separator",
        "STOP_WORDS": "stop_words",

        # 特殊处理字段
        "SUFFIX_AS_EOS": None,  # 该字段在官方模板中不需要
    }

    # 初始化目标模板（包含必填字段默认值）
    converted = {
        "meta_instruction": "You are a helpful assistant.",  # 必填项
        "capability": "chat",  # 必填项
        "eosys": "<|im_end|>\n",  # 通常固定格式
        "eoh": "<|im_end|>\n",  # 通常固定格式
    }

    # 自动处理字段映射
    for src_key, dest_key in field_mapping.items():
        if src_key in original_template:
            value = original_template[src_key]

            # 处理需要拆分的字段（如INSTRUCTION）
            if isinstance(dest_key, tuple) and src_key == "INSTRUCTION":
                # 使用正则拆分user和assistant部分
                parts = re.split(r'(<\|im_start\|>assistant\n?)', value)
                converted["user"] = parts[0].strip()
                if len(parts) > 1:
                    converted["assistant"] = parts[1] + parts[2] if len(parts) > 2 else parts[1]

            # 处理直接映射字段
            elif dest_key and not isinstance(dest_key, tuple):
                converted[dest_key] = value

    # 特殊处理system字段的占位符
    if "system" in converted:
        converted["system"] = converted["system"].replace("{system}", "{{ system }}")

    # 处理用户输入占位符
    if "user" in converted:
        converted["user"] = converted["user"].replace("{input}", "{{ input }}")

    # 自动处理停止词（兼容列表和字符串）
    if "stop_words" in converted and isinstance(converted["stop_words"], str):
        converted["stop_words"] = [converted["stop_words"]]

    # 保留原始模板中的额外字段（带警告）
    for key in original_template:
        if key not in field_mapping:
            print(f"Warning: 发现未映射字段 [{key}]，已保留原样")
            converted[key] = original_template[key]

    return converted


# 示例用法
original_qwen_chat = dict(
    SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
    INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n"),
    SUFFIX="<|im_end|>",
    SUFFIX_AS_EOS=True,
    SEP="\n",
    STOP_WORDS=["<|im_end|>", "<|endoftext|>"]
)

# 执行转换
converted_template = universal_converter(original_qwen_chat)

# 生成JSON文件
with open('chat_template.json', 'w') as f:
    json.dump(converted_template, f,
              indent=2,
              ensure_ascii=False,
              separators=(',', ': '))