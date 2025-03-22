import pandas as pd

def process_data(input_path, output_path):
    # 读取数据（保持所有原始列）
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # 对ctry_code进行独热编码
    # 使用pd.get_dummies自动生成带前缀的列
    ctry_encoded = pd.get_dummies(df['port_code'], prefix='ctry')

    ctry_encoded = ctry_encoded.astype(int)
    
    # 合并原始数据 + 编码后的列（横向拼接）
    processed = pd.concat([df, ctry_encoded], axis=1)
    
    # 保存结果（保留所有原始列和新增的编码列）
    processed.to_csv(output_path, index=False, encoding='utf-8-sig')
    return processed

# 使用示例
if __name__ == "__main__":
    input_file = "Data\Ori_data_set\Port_static_data.csv"
    output_file = "Data\pre_processed_data_set3\processed_ports.csv"
    
    df_processed = process_data(input_file, output_file)
    print("处理后的数据列：\n", df_processed.columns.tolist())
    print("\n前3行示例：")
    print(df_processed.head(3))