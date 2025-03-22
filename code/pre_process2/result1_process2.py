import pandas as pd

# 读取源CSV文件
source_file_path = r'Data\split_datasets\One_Hot_dataset\pridict_port\all_with_predictions.csv'
source_df = pd.read_csv(source_file_path)

# 读取索引映射CSV文件
index_mapping_file_path = r'Result\Predict_city\index_mapping.csv'
index_mapping_df = pd.read_csv(index_mapping_file_path)

# 将索引映射转换为字典，方便快速查找
index_to_label = dict(zip(index_mapping_df['index'], index_mapping_df['ori']))

# 遍历源CSV文件的列名，进行替换
new_columns = []
for col in source_df.columns:
    if col.startswith('Class_') and col.endswith('_Probability'):
        # 提取索引值
        index = int(col.split('_')[1])
        # 获取对应的ori值（例如 OH_SPT_JP）
        ori_value = index_to_label.get(index, f'Unknown_{index}')
        # 提取标签部分（例如从 OH_SPT_JP 中提取 JP）
        if ori_value.startswith('OP_OH_EPT_'):
            label = ori_value[len('OP_OH_EPT_'):]  # 去掉前缀 OH_SPT_
        else:
            label = ori_value
            print(0)  # 如果不符合格式，直接使用原值
        # 构建新的列名
        new_col = f'IP_ECT_{label}'
        new_columns.append(new_col)
    else:
        new_columns.append(col)

# 更新列名
source_df.columns = new_columns




source_ORI_file_path = r'Data\Ori_data_set\Ship_dynamic_data_Training.csv'
df1 = pd.read_csv(source_ORI_file_path)
columns_to_merge = ['uuid', 'start_port_code', 'end_port_code']
df1_selected = df1[columns_to_merge]

merged_df = pd.merge(source_df, df1_selected, on='uuid', how='inner')


template_df_file_path  = r'Data\pre_processed_data_set3\processed_ports.csv'
template_df = pd.read_csv(template_df_file_path)


# 给模板文件的所有列名加上前缀
template_df_start = template_df.add_prefix('SP_')
template_df_end = template_df.add_prefix('EP_')

# 将源数据中的 'Source_ID' 列与模板文件中的 'Template_ID' 列进行匹配
# 使用 merge 函数进行左连接（left join），以保留源数据的所有行
merged_df = pd.merge(
    merged_df,                    # 左 DataFrame（源数据）
    template_df_start,                  # 右 DataFrame（模板数据）
    left_on='start_port_code',          # 左 DataFrame 的匹配列
    right_on='SP_port_code',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)


# 将源数据中的 'Source_ID' 列与模板文件中的 'Template_ID' 列进行匹配
# 使用 merge 函数进行左连接（left join），以保留源数据的所有行
merged_df2 = pd.merge(
    merged_df,                    # 左 DataFrame（源数据）
    template_df_end,                  # 右 DataFrame（模板数据）
    left_on='end_port_code',          # 左 DataFrame 的匹配列
    right_on='EP_port_code',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)

merged_df2 = merged_df2.drop(columns=['SP_port_code',	'SP_ctry_code'	,'SP_name_en'	,'SP_name_cn',
                                      'EP_port_code',	'EP_ctry_code',	'EP_name_en'	,'EP_name_cn'	,
                                      'EP_lon'	,'EP_lat'	,'EP_timezone_offset','Predicted_Class',
                                      'start_port_code','end_port_code','SP_lon',	'SP_lat',])


# 保存修改后的CSV文件
output_file_path = r'Data\split_datasets\One_Hot_dataset\pridict_port\all_with_predictions_updated.csv'
merged_df2.to_csv(output_file_path, index=False)

print(f"列名已更新，并保存到 {output_file_path}")