import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def smart_column_processor(file_path, column_rules):
    """
    智能列处理器（支持通配符和自动重命名）

    参数:
    file_path (str/pathlib.Path): 文件路径
    column_rules (dict): 列处理规则字典，格式为：
        {
            "保留列": ["uuid", "ship_*"],  # 支持通配符*
            "重命名映射": {
                "ship_vessel_type": "船舶类型",
                "start_port_code_*": "起始港口_{num}",  # 带编号的自动命名
                "end_port_code_*": "目的港口_{num}"
            },
            "排除列": ["temp_column"]  # 可选
        }

    返回:
    DataFrame: 处理后的数据
    """
    # 转换为Path对象
    file_path = Path(file_path)
    
    # 读取原始数据
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("仅支持CSV/Excel文件")

    # 处理列选择
    selected_columns = []
    
    # 处理保留列（支持通配符）
    for pattern in column_rules.get("保留列", []):
        if '*' in pattern:
            matched = [col for col in df.columns if col.startswith(pattern.replace('*', ''))]
            selected_columns.extend(matched)
        else:
            if pattern in df.columns:
                selected_columns.append(pattern)
    
    # 处理重命名映射
    rename_dict = {}
    final_columns = selected_columns.copy()
    
    for src_pattern, dst_pattern in column_rules.get("重命名映射", {}).items():
        # 处理通配符匹配
        if '*' in src_pattern:
            base = src_pattern.replace('*', '')
            matched_cols = [col for col in selected_columns if col.startswith(base)]
            
            for col in matched_cols:
                # 提取编号
                num = col.replace(base, '')
                # 生成新列名
                new_name = dst_pattern.format(num=num)
                rename_dict[col] = new_name
        else:
            if src_pattern in selected_columns:
                rename_dict[src_pattern] = dst_pattern
    
    # 处理排除列
    final_columns = [col for col in selected_columns if col not in column_rules.get("排除列", [])]
    
    # 筛选并重命名
    df = df[final_columns].rename(columns=rename_dict)
    
    return df

# 使用示例
if __name__ == "__main__":
    config = {
        "保留列": [
            "uuid",
            "ship_*",  # 自动选择所有以ship开头的列
            "typecode*",
            "t2code*",
            "scaled_lat_startPort",
            "scaled_lon_startPort",
            "scaled_timezone_offset_startPort",
            "start_port_code_*",
            "start_city_code_*",
            "end_port_code_*",
            "time_diff_seconds"
        ],
        "重命名映射": {
            #"ship_vessel_type": "PP_Type",
            #"ship_vessel_sub_type": "PP_subType",
            "ship_mmsi":"mmsi",
            "typecode*":"IP_T1{num}",
            "t2code*":"IP_T2{num}",
            "scaled_lat_startPort":"IP_Slat",
            "scaled_lon_startPort":"IP_Slon",
            "scaled_timezone_offset_startPort":"IP_Szone",
            "ship_build_year":"IP_VBY",
            "ship_deadweight":"IP_VDW",
            "ship_length":"IP_VL",
            "ship_width":"IP_VW",
            "ship_height":"IP_VH",
            "ship_draught":"IP_VD",
            "ship_max_speed":"IP_VMS",
            "start_port_code_*": "IP_SP_{num}",
            "start_city_code_*":"IP_SC_{num}",
            "end_port_code_*": "OP_EP_{num}",
            "time_diff_seconds": "OP_time"
        },
        "排除列":{
            "ship_vessel_type",
            "ship_vessel_sub_type"
        }
    }
    
   # 处理数据
    df = smart_column_processor(
        r'Data\pre_processed_data_set\preTime_preShip_prePort_preType2.csv',
        config
    )
    
    # 新增数据集拆分逻辑
    # 设置随机种子保证可复现
    SEED = 42
    
    # 第一次拆分：训练集70%
    train_df, remaining = train_test_split(df, 
                                         train_size=0.7, 
                                         random_state=SEED)
    
    # 第二次拆分：剩余30%按2:1分为测试集和验证集
    test_df, val_df = train_test_split(remaining, 
                                     test_size=1/3,  # 1/3 of 30% = 10% total
                                     random_state=SEED)
    
    # 创建保存路径
    save_dir = Path("Data/split_datasets")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据集（使用CSV格式）
    train_df.to_csv(save_dir/"train.csv", index=False)
    test_df.to_csv(save_dir/"test.csv", index=False)
    val_df.to_csv(save_dir/"val.csv", index=False)
    
    # 打印验证信息
    print("数据集拆分完成")
    print(f"总样本数: {len(df)}")
    print(f"训练集: {len(train_df)} ({len(train_df)/len(df):.1%})")
    print(f"测试集: {len(test_df)} ({len(test_df)/len(df):.1%})")
    print(f"验证集: {len(val_df)} ({len(val_df)/len(df):.1%})")
    print(f"数据集已保存至: {save_dir.absolute()}")