"""from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

model_path = "./qa_ckpts/MERGED/llama38b-LoRD-VI-turthful_qa/"

# 修正计算精度为float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # 强制计算与输入类型一致
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,  # 替换旧版load_in_4bit参数
    device_map="auto",
    vocab_size=128256  # 显式指定词汇量
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    padding_side="left",  # 确保生成方向一致
    use_fast=False        # 关闭快速模式避免兼容性问题
)
# tokenizer.pad_token = tokenizer.eos_token  # 显式设置填充符 

# 计算需添加的 Token 数量
# target_vocab_size = 128256
# num_tokens_to_add = target_vocab_size - tokenizer.vocab_size
# print(num_tokens_to_add)

# # 生成占位符 Token（如 reserved_0 到 reserved_255）
# new_tokens = [f"reserved_{i}" for i in range(num_tokens_to_add)]

# # 添加至 Tokenizer
# tokenizer.add_tokens(new_tokens)
# print(f"扩展后词汇量: {len(tokenizer)}")  # 应为 128,256

# print(f"Tokenizer词汇量: {tokenizer.vocab_size}") 
# print(f"模型嵌入层大小: {model.get_input_embeddings().weight.shape[0]}")

# # 验证 Tokenizer 功能
# sample = "扩展后的测试文本"
# input_ids = tokenizer.encode(sample, add_special_tokens=False)
# print("最大索引值:", max(input_ids))  # 应 < 128,256

# # 检查特殊 Token 是否保留
# print("[PAD] ID:", tokenizer.pad_token_id)  # 应与原值一致

while True:
    message = input("👤User: ")
    if message == "q":
        break
    inputs = tokenizer(message, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.3,              # 降低随机性
        top_p=0.85,                   # 收紧采样范围
        repetition_penalty=2.0,        
        typical_p = 0.95,
        do_sample=True,               
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        # stopping_criteria=[StopOnTokens()]  # 添加终止条件（见下文）
    )
    print("🤖AI: ", end='')
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    torch.cuda.empty_cache()

print("Stop Chat.")"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

model_path = "./qa_ckpts/MERGED/llama38b-kd-ceval/"
# model_path = "./qa_ckpts/MERGED/llama38b-kd-truthful_qa/"
#model_path = "./qa_ckpts/MERGED/llama38b-vanilla-truthful_qa/"

# 配置4-bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    # padding_side="left",
    # use_fast=False
)

# 检查并扩展词汇表（如果需要）
target_vocab_size = 128256
current_vocab_size = len(tokenizer)

if current_vocab_size < target_vocab_size:
    print(f"当前词汇量: {current_vocab_size}, 目标词汇量: {target_vocab_size}")
    num_tokens_to_add = target_vocab_size - current_vocab_size
    new_tokens = [f"<extra_token_{i}>" for i in range(num_tokens_to_add)]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    print(f"已添加 {num_tokens_to_add} 个新token，词汇量现在为: {len(tokenizer)}")

# 确保特殊token设置正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '<s>'})

print("\n===== 模型加载完成 =====")
print(f"Tokenizer词汇量: {len(tokenizer)}")
print(f"模型嵌入层大小: {model.get_input_embeddings().weight.shape[0]}")
print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print("========================\n")

print(f"🤖AI: Hello! I'm an AI assistant. Ask me any question you have!")

import ftfy

def fix_mojibake(text: str) -> str:
    """
    修复因编码混淆导致的UTF-8乱码（如Latin-1误解码）
    支持多层编码错误修复（如 "ÃƒÂ" → "A"）
    """
    return ftfy.fix_text(text)

while True:
    try:
        message = input("👤User: ")
        if message.lower() == "q":
            break

        # ===== 新增约束指令 =====
        constraint_rules = [
            "【Mandatory Constraints】",  # Using "Mandatory" as in [3](@ref)'s "Mandatory clause"
            "1. Responses must not exceed 5 sentences",  # "Constraints" aligned with financial constraint terminology [1](@ref)
            "2. Output only core facts; explain the reasons in 3 sentences.",  # "Core facts" maintains precision requirement
            "3. For multi-step reasoning questions, provide the final conclusion directly",  # "Directly" corresponds to operative constraint principle [2](@ref)
            "4. Answer the question by Chinese"
        ]
        constraint_instruction = "\n".join(constraint_rules)
        # ======================
        
        # 使用对话格式（根据模型训练格式调整）
        formatted_input = f"### System: {constraint_instruction}\n### Human: 回答我的问题: {message}\n### Assistant:"
        
        # Tokenize输入
        inputs = tokenizer(
            formatted_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True
        ).to(model.device)
        
        # 生成响应
        outputs = model.generate(
            **inputs,  # 包含input_ids和attention_mask
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.3,
            repetition_penalty=1.3,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # 提取并解码新生成的部分
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        # 处理空响应
        if not response:
            response = "抱歉，我无法生成响应"

        # res = ''
        # for i in range(len(response)):
        #     if response[i] == "\\":
        #         byte_data = response[i:i+12].encode('latin1').decode('unicode_escape').encode('latin1')
        #         decoded_text = byte_data.decode('utf-8')
        #         res += decoded_text
        #         i += 8
        #     else:
        #         res += response[i]
        print(response)
        print(f"🤖AI: {fix_mojibake(response)}")
        
    except KeyboardInterrupt:
        print("\n退出对话...")
        break
    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        torch.cuda.empty_cache()

print("对话已结束")