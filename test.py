import re

def decode_escaped_text(text: str) -> str:
    # 提取所有十六进制字节码
    hex_codes = re.findall(r'[0-9a-fA-F]{2}', text)
    # 转换为字节并解码
    byte_data = bytes(int(code, 16) for code in hex_codes)
    return byte_data.decode('utf-8')

# 示例（纯转义序列文本）
pure_escape_text = r"\xe6\xa0\xb9\xe6\x8d\xae\xe4\xbd\xa0"
print(decode_escaped_text(pure_escape_text))  # 输出：根据你