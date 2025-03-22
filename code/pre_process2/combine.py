import pandas as pd

# 读取CSV文件
df_sub_type = pd.read_csv(r'Data\Ori_data_set\Ship_relationship_table2.csv')
df_type = pd.read_csv(r'Data\Ori_data_set\Ship_relationship_table3.csv')
df = pd.read_csv(r'Data\Ori_data_set\Ship_dynamic_data_Training.csv')
df_shipcode = pd.read_csv(r'Data\Ori_data_set\Ship_static_data.csv')
template_df = pd.read_csv(r'Data\pre_processed_data_set3\processed_ports.csv')


# 给模板文件的所有列名加上前缀
template_df_start = template_df.add_prefix('StartPort_')
template_df_end = template_df.add_prefix('EndPort_')



def process_data(df,code,prefixcode):

    # 对ctry_code进行独热编码
    # 使用pd.get_dummies自动生成带前缀的列
    ctry_encoded = pd.get_dummies(df[code], prefix=prefixcode)
    # 将布尔值转换为 0 和 1
    ctry_encoded = ctry_encoded.astype(int)
    # 合并原始数据 + 编码后的列（横向拼接）
    processed = pd.concat([df, ctry_encoded], axis=1)
    
    return processed


processed_df_sub_type = process_data(df_sub_type,'ship_vessel_sub_type','subtypecode')
processed_df_sub_type = processed_df_sub_type.drop(columns=["ship_vessel_type","name_en","name_cn"])


processed_df_type = process_data(df_type,'vessel_type','typecode')
processed_df_type = processed_df_type.drop(columns=["parent_code","name_en","name_cn","dict_type"])



# 将源数据中的 'Source_ID' 列与模板文件中的 'Template_ID' 列进行匹配
# 使用 merge 函数进行左连接（left join），以保留源数据的所有行
df_shipcode = pd.merge(
    df_shipcode,                    # 左 DataFrame（源数据）
    processed_df_sub_type,                  # 右 DataFrame（模板数据）
    left_on='vessel_sub_type',          # 左 DataFrame 的匹配列
    right_on='ship_vessel_sub_type',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)

df_shipcode = df_shipcode.drop(columns=["ship_vessel_sub_type"])
template_df_shipcode = pd.merge(
    df_shipcode,                    # 左 DataFrame（源数据）
    processed_df_type,                  # 右 DataFrame（模板数据）
    left_on='vessel_type',          # 左 DataFrame 的匹配列
    right_on='vessel_type',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)



df = pd.merge(
    df,                    # 左 DataFrame（源数据）
    template_df_shipcode,                  # 右 DataFrame（模板数据）
    left_on='ship_mmsi',          # 左 DataFrame 的匹配列
    right_on='ship_mmsi',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)
df = df.drop(columns=['vessel_type','vessel_sub_type'])




# 将源数据中的 'Source_ID' 列与模板文件中的 'Template_ID' 列进行匹配
# 使用 merge 函数进行左连接（left join），以保留源数据的所有行
merged_df = pd.merge(
    df,                    # 左 DataFrame（源数据）
    template_df_start,                  # 右 DataFrame（模板数据）
    left_on='start_port_code',          # 左 DataFrame 的匹配列
    right_on='StartPort_port_code',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)


# 将源数据中的 'Source_ID' 列与模板文件中的 'Template_ID' 列进行匹配
# 使用 merge 函数进行左连接（left join），以保留源数据的所有行
merged_df2 = pd.merge(
    merged_df,                    # 左 DataFrame（源数据）
    template_df_end,                  # 右 DataFrame（模板数据）
    left_on='end_port_code',          # 左 DataFrame 的匹配列
    right_on='EndPort_port_code',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)

merged_df2 = merged_df2.drop(columns=['StartPort_port_code',	'StartPort_ctry_code'	,'StartPort_name_en'	,'StartPort_name_cn','EndPort_port_code',	'EndPort_ctry_code',	'EndPort_name_en'	,'EndPort_name_cn'	,'EndPort_lon'	,'EndPort_lat'	,'EndPort_timezone_offset'])





# 将DataFrame保存为CSV文件
merged_df2.to_csv(r'Data\pre_processed_data_set3\result.csv', index=False)