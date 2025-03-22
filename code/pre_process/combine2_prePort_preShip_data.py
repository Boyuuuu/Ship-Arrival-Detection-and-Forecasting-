import pandas as pd

# ====================== 文件路径配置 ======================
ship_preport_path = "Data/pre_processed_data_set/Ship_prePort_data.csv"    # 主数据文件
processed_ships_path = "Data/pre_processed_data_set/processed_ships.csv" # 被合并数据文件
output_path = "Data/pre_processed_data_set/preShip_prePort.csv"          # 输出路径

# ====================== 列名映射配置 ======================
# 原始列名 -> 新列名 的映射 (按需修改此处即可)
column_mapping = {
    # 原列名             新列名
    'vessel_type':         'ship_vessel_type',
    'vessel_sub_type':    'ship_vessel_sub_type',
    'build_year_standardized': 'ship_build_year',
    'deadweight_standardized': 'ship_deadweight',
    'length_standardized':     'ship_length',
    'width_standardized':      'ship_width',
    'height_standardized':     'ship_height',
    'draught_standardized':   'ship_draught',
    'max_speed_standardized':  'ship_max_speed'
}

# ====================== 数据预处理 ======================
# 读取主数据
ship_preport = pd.read_csv(ship_preport_path, dtype={'ship_mmsi': str})

# 读取被合并数据并去重
processed_ships = pd.read_csv(processed_ships_path, dtype={'ship_mmsi': str})
processed_ships = processed_ships.drop_duplicates(subset='ship_mmsi', keep='first')

# 从被合并数据中提取需要的列并重命名
columns_to_merge = list(column_mapping.keys()) + ['ship_mmsi']  # 必须包含合并键
processed_filtered = processed_ships[columns_to_merge].rename(columns=column_mapping)

# ====================== 数据合并 ======================
# 左连接合并数据
merged_data = pd.merge(
    ship_preport,
    processed_filtered,
    on='ship_mmsi',
    how='left'
)

# ====================== 结果保存 ======================
merged_data.to_csv(output_path, index=False)
print(f"合并完成！新增字段: {list(column_mapping.values())}")