import pandas as pd

def calculate_statistics(file_path, column_names):
    """
    计算指定列的多种统计指标。

    参数:
        file_path (str): CSV 文件路径。
        column_names (list): 需要分析的列名列表。

    返回:
        DataFrame: 包含所有列的统计指标的 DataFrame。
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 检查列是否存在
    for column_name in column_names:
        if column_name not in df.columns:
            raise ValueError(f"列 '{column_name}' 不存在！")

    # 存储所有列的统计指标
    all_stats = []

    # 遍历每个列名，计算统计指标
    for column_name in column_names:
        data = df[column_name]

        # 计算统计指标
        statistics = {
            '列名 (Column Name)': column_name,
            '平均数 (Mean)': data.mean(),
            '中位数 (Median)': data.median(),
            '标准差 (Standard Deviation)': data.std(),
            '最小值 (Min)': data.min(),
            '最大值 (Max)': data.max(),
            '25% 分位数 (25th Percentile)': data.quantile(0.25),
            '50% 分位数 (50th Percentile)': data.quantile(0.50),
            '75% 分位数 (75th Percentile)': data.quantile(0.75),
            '偏度 (Skewness)': data.skew(),
            '峰度 (Kurtosis)': data.kurt(),
            '非空值数量 (Non-Null Count)': data.count(),
            '缺失值数量 (Null Count)': data.isnull().sum(),
            '唯一值数量 (Unique Count)': data.nunique(),
            '众数 (Mode)': data.mode().values[0] if not data.mode().empty else None
        }
        all_stats.append(statistics)

    # 将统计指标转换为 DataFrame
    stats_df = pd.DataFrame(all_stats)
    return stats_df

def save_statistics_to_csv(file_path, column_names, output_csv_path):
    """
    计算指定列的统计指标并保存到 CSV 文件。

    参数:
        file_path (str): CSV 文件路径。
        column_names (list): 需要分析的列名列表。
        output_csv_path (str): 输出 CSV 文件路径。
    """
    try:
        # 计算统计指标
        stats_df = calculate_statistics(file_path, column_names)

        # 保存到 CSV 文件
        stats_df.to_csv(output_csv_path, index=False)

        print(f"统计指标已保存到 '{output_csv_path}'")
    except ValueError as e:
        print(e)

# 示例用法
file_path = r'Data\split_datasets\One_Hot_dataset\all.csv'  # 替换为你的 CSV 文件路径
column_names = ['IP_VBY', 
                'IP_VDW',
                'IP_VL',
                'IP_VW',
                'IP_VH',
                'IP_VD',
                'IP_VMS',
                'IP_SPLO',
                'IP_SPLA',
                'IP_SPTO'
]  # 替换为你要分析的列名列表
output_csv_path = r'Data\Result\statistics.csv'  # 替换为你要保存的 CSV 文件路径

save_statistics_to_csv(file_path, column_names, output_csv_path)
