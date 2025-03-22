import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_ship_data(input_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(input_path)
    
    # 定义需要标准化的列
    columns_to_scale = [
        'build_year', 'deadweight', 'length', 
        'width', 'height', 'draught', 'max_speed'
    ]
    
    # 处理缺失值（用列均值填充）
    for col in columns_to_scale:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
    
    # 标准化处理
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[columns_to_scale])
    
    # 创建带后缀的新列名
    scaled_columns = [f"{col}_standardized" for col in columns_to_scale]
    
    # 合并数据
    df_scaled = pd.DataFrame(scaled_values, columns=scaled_columns)
    df_combined = pd.concat([df, df_scaled], axis=1)
    
    # 保存结果
    df_combined.to_csv(output_path, index=False)
    print(f"数据已处理并保存至：{output_path}")

# 使用示例
process_ship_data(
    input_path=r"Data\Ori_data_set\Ship_static_data.csv",  # 替换为实际输入路径
    output_path=r"Data\pre_processed_data_set\processed_ships.csv"  # 替换为实际保存路径
)