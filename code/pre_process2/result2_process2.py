import pandas as pd

# 读取源CSV文件
source_file_path = r'Data\split_datasets\One_Hot_dataset\pridict_city\splitdataset\pridict_port\all_withoutOP_pridicted.csv'
# 保存修改后的CSV文件
output_file_path = r'Data\split_datasets\One_Hot_dataset\pridict_city\splitdataset\pridict_port\all_with_predictions_updated.csv'
#初始CSV文件
Ori_path = r'Data\Ori_data_set\Ship_dynamic_data_Training.csv'
template_path = r'Result\Predict_port\index_mapping.csv'
port_template_path = r'Data\Ori_data_set\Port_static_data.csv'

ori_df = pd.read_csv(Ori_path)
df = pd.read_csv(source_file_path)
df = df.loc[:, ~df.columns.str.startswith('IP_ECT')]
df = df.loc[:, ~df.columns.str.startswith('IP_OH_SPT')]
df = df.loc[:, ~df.columns.str.startswith('IP_OH_SPPT')]
df = df.loc[:, ~df.columns.str.startswith('IP_OH_SCT')]


selected_columns = ['uuid', 'leg_start_postime', 'leg_end_postime']  # 必须包含匹配列 'uuid'
ori_df_selected = ori_df[selected_columns]

df = pd.merge(
    df,                    # 左 DataFrame（源数据）
    ori_df_selected,                  # 右 DataFrame（模板数据）
    left_on='uuid',          # 左 DataFrame 的匹配列
    right_on='uuid',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)



selected_columns = ['index', 'ori'] 
template_df = pd.read_csv(template_path)
template_df_selected = template_df[selected_columns]

df = pd.merge(
    df,                    # 左 DataFrame（源数据）
    template_df_selected,                  # 右 DataFrame（模板数据）
    left_on='IP_OH_EPPT',          # 左 DataFrame 的匹配列
    right_on='index',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)


# 删除前缀
df['end_port_code'] = df['ori'].str.replace(r'^OP_OH_EPPT_', '', regex=True)
port_template = pd.read_csv(port_template_path)
selected_columns = ['port_code', 'lon', 'lat','timezone_offset']  # 必须包含匹配列 'uuid'
port_template_selected = port_template[selected_columns]
df = pd.merge(
    df,                    # 左 DataFrame（源数据）
    port_template_selected,                  # 右 DataFrame（模板数据）
    left_on='end_port_code',          # 左 DataFrame 的匹配列
    right_on='port_code',       # 右 DataFrame 的匹配列
    how='left'                    # 连接方式
)
df['IP_EPLO'] = df['lon'] 
df['IP_EPLA'] = df['lat'] 
df['IP_EPTO'] = df['timezone_offset'] 
df = df.drop(columns=['lon','lat','timezone_offset','IP_OH_EPPT','index','ori','end_port_code','port_code'])




#IP_OH_EPPT,leg_start_postime,leg_end_postime,index,ori,end_port_code
df.to_csv(output_file_path, index=False)

print(f"列名已更新，并保存到 {output_file_path}")
'''
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


'''
