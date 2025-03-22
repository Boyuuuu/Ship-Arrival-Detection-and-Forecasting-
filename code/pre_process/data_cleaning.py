import csv
from dateutil.parser import parse as parse_date
import re

def validate_uuid(uuid_str):
    return re.match(r'^[0-9a-f]{32}$', uuid_str.lower()) is not None

def validate_mmsi(mmsi_str):
    return re.match(r'^[A-Z0-9]{9}$', mmsi_str) is not None  # 允许字母+数字组合

def validate_port_code(code):
    return re.match(r'^[A-Z0-9]{5,6}$', code) is not None

def validate_time(time_str):
    try:
        # 强制验证时区格式(如 +08/-05)
        if not re.search(r'\+\d{2}$|-\d{2}$', time_str):
            return False
        parse_date(time_str)  # 验证可解析的时间
        return True
    except:
        return False

def validate_route_line(route):
    return "LINESTRING" in route.upper()  # 核心修改：仅检查关键词存在性

def validate_distance(distance_str):
    try:
        return float(distance_str) >= 0
    except:
        return False

# 验证规则字典保持不变
validators = {
    'uuid': validate_uuid,
    'ship_mmsi': validate_mmsi,
    'start_port_code': validate_port_code,
    'end_port_code': validate_port_code,
    'leg_start_postime': validate_time,
    'leg_end_postime': validate_time,
    'route_line': validate_route_line,  # 使用新验证方法
    'distance': validate_distance
}

# 后续处理逻辑保持不变
with open(r'Data\Ori_data_set\Ship_dynamic_data_Training.csv', 'r') as f_in:
    reader = csv.DictReader(f_in)
    rows = list(reader)

valid_rows = []
invalid_rows = []

for row in rows:
    errors = []
    for field, validate in validators.items():
        if not validate(row[field]):
            errors.append(f"{field}格式错误")
    if errors:
        row['error_reason'] = '; '.join(errors)
        invalid_rows.append(row)
    else:
        valid_rows.append(row)

# 输出文件
with open('valid.csv', 'w', newline='') as f_valid:
    writer = csv.DictWriter(f_valid, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(valid_rows)

with open('invalid.csv', 'w', newline='') as f_invalid:
    fieldnames = reader.fieldnames + ['error_reason']
    writer = csv.DictWriter(f_invalid, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(invalid_rows)