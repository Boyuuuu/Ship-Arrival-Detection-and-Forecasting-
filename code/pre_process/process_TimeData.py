import csv
import re
from datetime import datetime

# 用户配置区
input_csv = r'Data\pre_processed_data_set\preShip_prePort.csv'
output_csv = r'Data\pre_processed_data_set\preTime_preShip_prePort.csv'
time_column1 = 'leg_start_postime'
time_column2 = 'leg_end_postime'
new_column_name = 'time_diff_seconds'

def parse_time(time_str):
    """解析含时区和毫秒的时间字符串（兼容有无毫秒的情况）"""
    # 标准化时区格式（将结尾的+08修正为+0800）
    adjusted_str = re.sub(r'\+08$', '+0800', time_str)
    
    # 双重解析尝试（带毫秒和不带毫秒）
    try:
        return datetime.strptime(adjusted_str, "%Y/%m/%d %H:%M:%S.%f%z")
    except ValueError:
        return datetime.strptime(adjusted_str, "%Y/%m/%d %H:%M:%S%z")

with open(input_csv, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    rows = list(reader)
    fieldnames = reader.fieldnames + [new_column_name]

for row in rows:
    try:
        start = parse_time(row[time_column1])
        end = parse_time(row[time_column2])
        row[new_column_name] = (end - start).total_seconds()
    except Exception as e:
        print(f"行 {reader.line_num} 错误: {str(e)}")
        row[new_column_name] = '计算错误'

with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'处理完成，结果已输出至：{output_csv}')