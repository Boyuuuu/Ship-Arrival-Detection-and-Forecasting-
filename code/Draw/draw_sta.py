import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置输出路径
output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

# 读取CSV文件
file_path = r''
df = pd.read_csv(file_path)

# 获取列名
columns = list(df.columns[1:8]) + list(df.columns[55:58])  # 2-8列 + 56-59列

# 遍历每一列并绘制直方图
for col in columns:
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图（使用密度归一化）
    hist_data = df[col].dropna()
    counts, bin_edges, _ = plt.hist(
        hist_data,
        bins='auto',
        color='black',
        alpha=0.7,
        label=col,
        align='mid',
        edgecolor='black',
        density=True  # 关键修改：启用密度归一化
    )
    
    # 计算均值和中位数
    mean_value = hist_data.mean()
    median_value = hist_data.median()
    
    # 绘制均值和中位数的垂直线
    plt.axvline(mean_value, color='orange', linestyle='dashed', linewidth=2.5, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2.5, label=f'Median: {median_value:.2f}')
    
    # 绘制趋势线（KDE，添加带宽调整）
    sns.kdeplot(
        hist_data,
        color='red',
        label='Trend Line',
        bw_adjust=1  # 关键修改：调整带宽参数
    )
    
    # 添加标题和标签
    plt.title(f'Distribution of {col}')
    plt.xlabel('Value')
    plt.ylabel('Density')  # 修改y轴标签
    plt.legend()
    
    # 保存图像
    output_path = os.path.join(output_dir, f'{col}_histogram.png')
    plt.savefig(output_path)
    
    # 显示图像
    plt.show()
    
    # 关闭图形
    plt.close()