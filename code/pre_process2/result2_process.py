import pandas as pd

def print_csv_columns_pandas(filename):
    df = pd.read_csv(filename)
    print("CSV文件的列名为：")
    for column in df.columns:
        print(f"- {column}")
    print(len(df.columns))

# 使用示例
print_csv_columns_pandas(r'Data\split_datasets\One_Hot_dataset\all.csv')  # 替换成你的文件名


def remove_op_prefix_columns(filename, output_filename):
    # 读取CSV文件
    df = pd.read_csv(filename)
    
    # 找到所有列名以 "OP" 开头的列
    op_columns = [col for col in df.columns if col.startswith('OP')]
    
    # 删除这些列
    df_filtered = df.drop(columns=op_columns)
    


    # 保存到新的CSV文件
    df_filtered.to_csv(output_filename, index=False)
    print(f"已删除所有以 'OP' 开头的列，结果保存到 {output_filename}")

    
filename = r'Data\split_datasets\One_Hot_dataset\pridict_city\splitdataset\all.csv'
output_filename =  r'Data\split_datasets\One_Hot_dataset\pridict_city\splitdataset\pridict_port\all_withoutOP.csv'


remove_op_prefix_columns(filename,output_filename)
print_csv_columns_pandas(output_filename) 