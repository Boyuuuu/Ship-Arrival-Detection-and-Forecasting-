import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'Data\Ori_data_set\Ship_dynamic_data_Training.csv')

# 读取模板文件
template_df = pd.read_csv(r'Data\pre_processed_data_set2\processed_ports.csv')

# 给模板文件的所有列名加上前缀
template_df1 = template_df.add_prefix('StartPort_')
template_df2 = template_df.add_prefix('EndPort_')

# 删除所有以 'code' 开头的列
df = df.loc[:, ~df.columns.str.startswith('start_port_code_')]
df = df.loc[:, ~df.columns.str.startswith('start_city_code_')]
df = df.loc[:, ~df.columns.str.startswith('end_port_code_')]


# 将源数据中的 'Source_ID' 列与模板文件中的 'Template_ID' 列进行匹配
# 使用 merge 函数进行左连接（left join），以保留源数据的所有行
merged_df = pd.merge(
    df,                    # 左 DataFrame（源数据）
    template_df1,                  # 右 DataFrame（模板数据）
    left_on='start_port_code',          # 左 DataFrame 的匹配列
    right_on='StartPort_port_code',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)

# 将源数据中的 'Source_ID' 列与模板文件中的 'Template_ID' 列进行匹配
# 使用 merge 函数进行左连接（left join），以保留源数据的所有行
merged_df2 = pd.merge(
    merged_df,                    # 左 DataFrame（源数据）
    template_df2,                  # 右 DataFrame（模板数据）
    left_on='end_port_code',          # 左 DataFrame 的匹配列
    right_on='EndPort_port_code',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)

merged_df2 = merged_df2.drop(columns=['StartPort_port_code',	'StartPort_ctry_code'	,'StartPort_name_en'	,'StartPort_name_cn','EndPort_port_code',	'EndPort_ctry_code',	'EndPort_name_en'	,'EndPort_name_cn'	,'EndPort_lon'	,'EndPort_lat'	,'EndPort_timezone_offset'])



# 去除 'Age' 和 'City' 列
#df = df.drop(columns=['Age', 'City'])

# 打印处理后的数据
print("\n去除 'Age' 和 'City' 列后的数据:")
print(merged_df2)
merged_df2.info()
# 将DataFrame保存为CSV文件
merged_df2.to_csv(r'Data\pre_processed_data_set2\Oridata_HotPort.csv', index=False)