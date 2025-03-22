import pandas as pd
from dateutil import parser

def debug_parse(time_str):
    """带调试输出的解析函数"""
    try:
        # 先打印原始字符串长度和内容
        print(f"原始字符串长度: {len(time_str)}, 内容: {repr(time_str)}")
        
        # 预处理：移除首尾空格/引号
        cleaned = time_str.strip().strip('"')
        print(f"清理后内容: {repr(cleaned)}")
        
        # 强制时区格式为+08:00
        if cleaned.count("+") == 1:
            parts = cleaned.rsplit("+", 1)
            cleaned = f"{parts[0]}+{parts[1].zfill(2)}:00"
            print(f"时区修正后: {cleaned}")
        
        # 执行解析并验证结果
        parsed = parser.parse(cleaned)
        print(f"解析成功 -> {parsed}\n")
        return parsed
    
    except Exception as e:
        print(f"解析失败！错误类型: {type(e).__name__}, 错误信息: {str(e)}\n")
        return pd.NaT

# 读取CSV文件（指定编码和分隔符）
input_path = r'Data\pre_processed_data_set\preShip_prePort_reProcessed_data.csv'
try:
    df = pd.read_csv(input_path, encoding='utf-8-sig', sep=',', quotechar='"')
except UnicodeDecodeError:
    df = pd.read_csv(input_path, encoding='gbk', sep=',', quotechar='"')

# 调试前5行数据
print("================ 开始调试解析 ================")
sample_data = df['leg_start_postime'].head().apply(debug_parse)

# 显示解析结果
print("\n解析结果：")
print(sample_data)