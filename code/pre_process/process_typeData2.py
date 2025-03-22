# 首先安装需要的库（如果尚未安装）
# pip install category_encoders pandas

import pandas as pd
from category_encoders import BinaryEncoder

# 1. 读取CSV文件
df = pd.read_csv(r"Data\pre_processed_data_set\preType_data.csv")

# 2. 指定需要编码的分类特征列
categorical_columns = ["ship_vessel_type"]

# 3. 复制原始分类列（用于后续恢复）
original_col = df[categorical_columns[0]].copy()

# 4. 初始化二进制编码器并转换
encoder = BinaryEncoder(cols=categorical_columns)
df_encoded = encoder.fit_transform(df)

# 5. 获取二进制编码后的列名列表
binary_cols = [col for col in df_encoded.columns 
               if col.startswith(f"{categorical_columns[0]}_")]

# 6. 重命名二进制列
df_encoded.rename(
    columns={col: f"t2code{i+1}" for i, col in enumerate(binary_cols)},
    inplace=True
)

# 7. 将原始分类列添加回数据框
df_encoded.insert(
    loc=df.columns.get_loc(categorical_columns[0]) + 1,  # 原始列位置的下一位
    column=categorical_columns[0],  # 原始列名
    value=original_col.values        # 原始数据
)

# 8. 将编码列移动到数据末尾
typecode_cols = [col for col in df_encoded if col.startswith("t2code")]
other_cols = [col for col in df_encoded if col not in typecode_cols]
df_final = df_encoded[other_cols + typecode_cols]

# 9. 验证结果
print("\n编码后的数据形状:", df_final.shape)
print("\n前3行示例数据（带原始列和编码列）:")
print(df_final[[categorical_columns[0]] + typecode_cols].head(3))

# 10. 保存结果
df_final.to_csv(r"Data\pre_processed_data_set\preType2_data.csv", index=False)