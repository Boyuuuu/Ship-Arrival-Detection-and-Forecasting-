import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import StandardScaler

def process_data(input_path, output_path):
    # 读取数据
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # 1. port_code编码（1101类）
    port_encoder = BinaryEncoder(cols=['port_code'])
    port_encoded = port_encoder.fit_transform(df[['port_code']])
    
    # 2. ctry_code编码（127类）
    ctry_encoder = BinaryEncoder(cols=['ctry_code'])
    ctry_encoded = ctry_encoder.fit_transform(df[['ctry_code']])
    
    # 3. 数值特征标准化
    num_cols = ['lon', 'lat', 'timezone_offset']
    scaler = StandardScaler()
    scaled_nums = pd.DataFrame(scaler.fit_transform(df[num_cols]),
                              columns=[f'scaled_{c}' for c in num_cols])
    
    # 合并特征（按指定顺序）
    processed = pd.concat([
        df[['port_code', 'ctry_code']],  # 原始列
        port_encoded.add_prefix('port_'),
        ctry_encoded.add_prefix('ctry_'),
        scaled_nums
    ], axis=1)
    
    # 保存结果
    processed.to_csv(output_path, index=False, encoding='utf-8-sig')
    return processed



# 使用示例
if __name__ == "__main__":
    input_file = "Data\Ori_data_set\Port_static_data.csv"
    output_file = "Data\pre_processed_data_set\processed_ports.csv"
    
    df_processed = process_data(input_file, output_file)
    print("处理后的数据列：\n", df_processed.columns.tolist())
    print("\n前3行示例：")
    print(df_processed.head(3))