import pandas as pd

def merge_port_features(source_df, template_df, port_type, need_ctry=True):
    """港口特征合并核心函数（精准控制列命名）"""
    port_column = f"{port_type}_port_code"
    print(f"\n▌ 正在合并 [{port_type.upper()}港口] 特征")

    # 验证必要列存在性
    if port_column not in source_df.columns:
        raise ValueError(f"源数据缺少关键列：{port_column}")
    if 'port_code' not in template_df.columns:
        raise ValueError("模板数据缺少port_code列")

    # 动态筛选特征列
    feature_prefix = ['port_port_code']
    explicit_features = []
    
    # 仅start港口需要空间特征
    if port_type == 'start':
        explicit_features = ['scaled_lon', 'scaled_lat', 'scaled_timezone_offset']
        if need_ctry:
            feature_prefix.append('ctry_ctry_code')
    
    # 生成待合并的特征列表
    port_features = [
        col for col in template_df.columns
        if any(col.startswith(p) for p in feature_prefix)  # 匹配前缀
        or col in explicit_features  # 显式包含空间特征
        and col not in ['port_code', 'ctry_code']  # 排除键列
    ]
    
    # 验证特征完整性
    required_features = [f'port_port_code_{i}' for i in range(11)] + explicit_features
    missing_feat = [f for f in required_features if f not in port_features]
    if missing_feat:
        raise ValueError(f"模板文件缺失必要特征：{missing_feat}")

    # 执行合并（阻止自动后缀）
    merged_df = pd.merge(
        left=source_df,
        right=template_df[['port_code'] + port_features],
        left_on=port_column,
        right_on='port_code',
        how='left',
        suffixes=('', '_TEMP'),  # 禁止自动生成_x/_y
        validate='m:1'
    ).drop(columns=['port_code', 'ctry_code'], errors='ignore')

    # 精准重命名（仅处理指定前缀）
    rename_rules = {
        'port_port_code': f"{port_type}_port_code",
        'ctry_ctry_code': f"{port_type}_city_code"
    }
    # 特殊处理start港口的空间特征
    if port_type == 'start':
        rename_rules.update({
            'scaled_lon': 'scaled_lon_startPort',
            'scaled_lat': 'scaled_lat_startPort',
            'scaled_timezone_offset': 'scaled_timezone_offset_startPort'
        })
    
    merged_df = merged_df.rename(columns=lambda x: next(
        (x.replace(k, v) for k, v in rename_rules.items() if x.startswith(k)), x
    ))

    # 清理残留列（如_TEMP后缀）
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('_TEMP')]
    
    return merged_df

def merge_features(template_path, source_path, output_path):
    # 读取数据
    template_df = pd.read_csv(template_path, encoding='utf-8-sig')
    source_df = pd.read_csv(source_path, encoding='utf-8-sig')

    # 分步合并
    merged_df = merge_port_features(source_df, template_df, 'start', need_ctry=True)
    merged_df = merge_port_features(merged_df, template_df, 'end', need_ctry=False)

    # 验证结果列
    expected_start_cols = ['scaled_lon_startPort', 'scaled_lat_startPort', 
                          'scaled_timezone_offset_startPort'] + \
                         [f'start_port_code_{i}' for i in range(11)]
    unexpected_end_cols = ['scaled_lon_endPort', 'scaled_lat_endPort', 
                          'scaled_timezone_offset_endPort']
    
    # 检查是否包含错误列
    if any(col in merged_df.columns for col in unexpected_end_cols):
        raise ValueError("错误：End港口包含不应存在的空间特征列")
    
    # 保存结果
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 合并完成！最终列示例：{merged_df.columns[-8:]}")
    return merged_df

# 测试用例
if __name__ == "__main__":
    template_file = r"Data\pre_processed_data_set\processed_ports.csv"
    source_file = r"Data\Ori_data_set\Ship_dynamic_data_Training.csv"
    output_file = r"Data\pre_processed_data_set\Ship_prePort_data.csv"

    try:
        merged_data = merge_features(template_file, source_file, output_file)
        print("\n[合并结果验证]")
        
        # 验证Start港口特征
        assert 'scaled_lon_startPort' in merged_data.columns, "缺失Start港口经度"
        assert 'start_port_code_10' in merged_data.columns, "缺失Start港口编码"
        
        # 验证End港口无空间特征
        assert 'scaled_lon_endPort' not in merged_data.columns, "错误包含End港口经度"
        assert 'end_port_code_10' in merged_data.columns, "缺失End港口编码"
        
        print("所有校验通过！")
        
    except Exception as e:
        print(f"❌ 执行失败：{str(e)}")