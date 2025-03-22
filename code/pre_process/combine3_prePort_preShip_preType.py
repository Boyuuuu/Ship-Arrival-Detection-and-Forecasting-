import pandas as pd

# ====================== 文件路径配置 ======================
preType2_data_path = r"Data/pre_processed_data_set/preType2_data.csv"
preTime_preShip_prePort_path = r"Data/pre_processed_data_set/preTime_preShip_prePort.csv"
output_path = r"Data/pre_processed_data_set/preTime_preShip_prePort_preType2.csv"

# ====================== 数据读取 ======================
# 读取主数据文件
df_main = pd.read_csv(preTime_preShip_prePort_path)
# 读取包含类型编码的数据文件
df_type = pd.read_csv(preType2_data_path)

# ====================== 列合并操作 ======================
# 定义需要合并的列
merge_columns = [
    'typecode1', 'typecode2', 'typecode3',
    'typecode4', 'typecode5', 'typecode6',
    't2code1', 't2code2', 't2code3'
]

# 执行左连接合并
merged_df = df_main.merge(
    df_type[['ship_vessel_sub_type'] + merge_columns],  # 选择需要合并的列
    on='ship_vessel_sub_type',  # 使用共同列进行匹配
    how='left'  # 保留左表所有记录
)

# ====================== 结果保存 ======================
merged_df.to_csv(output_path, index=False)
print(f"数据合并完成，结果已保存至：{output_path}")