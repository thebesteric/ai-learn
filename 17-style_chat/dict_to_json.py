import re

# 原始对话模板配置
original_qwen_chat = dict(
    SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
    INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n"),
    SUFFIX="<|im_end|>",
    SUFFIX_AS_EOS=True,
    SEP="\n",
    STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
)

def convert_template(template):
    coverted = {}
    for key, value in template.items():
       if isinstance(value, str):
           coverted[key] = re.sub(r"\{(\w+)\}", r"{{ \1 }}", value)
       else:
           coverted[key] = value
    return coverted

# 执行转换
jinja2_qwen_chat = convert_template(original_qwen_chat)
# {'SYSTEM': '<|im_start|>system\n{{ system }}<|im_end|>\n', 'INSTRUCTION': '<|im_start|>user\n{{ input }}<|im_end|>\n<|im_start|>assistant\n', 'SUFFIX': '<|im_end|>', 'SUFFIX_AS_EOS': True, 'SEP': '\n', 'STOP_WORDS': ['<|im_end|>', '<|endoftext|>']}
print(jinja2_qwen_chat)
