import pandas as pd
import re
from datetime import datetime, timezone, timedelta


def datetime_to_seconds(year, month, day, hour=0, minute=0, second=0):
    """
    将日期时间转换为从公元1年1月1日00:00:00开始计算的总秒数
    包含闰年判断和月份天数处理
    """
    # 校验输入合法性
    if not (1 <= month <= 12):
        raise ValueError("月份必须在1-12之间")
    if year < 1:
        raise ValueError("年份必须大于等于1")

    # 闰年判断函数
    def is_leap(y):
        return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)

    # 每月天数表（非闰年）
    month_days = [31, 28, 31, 30, 31, 30, 
                  31, 31, 30, 31, 30, 31]
    
    # 调整闰年2月天数
    if is_leap(year):
        month_days[1] = 29

    # 校验日期有效性
    if day < 1 or day > month_days[month-1]:
        raise ValueError(f"{year}-{month}月最多有{month_days[month-1]}天")
    if not (0 <= hour <= 23):
        raise ValueError("小时必须在0-23之间")
    if not (0 <= minute <= 59):
        raise ValueError("分钟必须在0-59之间")
    if not (0 <= second <= 59):
        raise ValueError("秒数必须在0-59之间")

    # 计算总天数（使用数学公式代替循环）
    # 1. 计算完整年份的天数
    years = year - 1
    leap_count = years // 4 - years // 100 + years // 400
    total_days = years * 365 + leap_count

    # 2. 计算当年完整月份的天数
    for m in range(month-1):
        total_days += month_days[m]

    # 3. 添加当月已过天数
    total_days += day - 1

    # 转换为秒数
    return (total_days * 86400) + (hour * 3600) + (minute * 60) + second


def parse_to_cst(datetime_str):
    """
    将任意时区时间转换为中国标准时间（UTC+8）
    支持带毫秒的时间字符串
    输入示例：
    - "2021/7/13 10:34:31+08"
    - "2021/7/18 08:42:53.673+08"
    """
    try:
        # 解析原始时间和时区
        match = re.match(r"^(.*?)([+-]\d{1,2})$", datetime_str)
        if not match:
            raise ValueError("无效的时间格式")
        
        # 解析本地时间部分
        dt_part = match.group(1).strip()
        
        # 尝试解析带毫秒的时间
        try:
            naive_dt = datetime.strptime(dt_part, "%Y/%m/%d %H:%M:%S.%f")
        except ValueError:
            # 如果没有毫秒部分，尝试不包含毫秒的格式
            naive_dt = datetime.strptime(dt_part, "%Y/%m/%d %H:%M:%S")
        
        # 获取原始时区偏移
        tz_sign = -1 if match.group(2).startswith("-") else 1
        tz_hours = int(match.group(2)[1:])
        original_offset = tz_sign * tz_hours
        
        # 创建时区感知对象
        original_tz = timezone(timedelta(hours=original_offset))
        aware_dt = naive_dt.replace(tzinfo=original_tz)
        
        # 转换到UTC+8时区
        cst_tz = timezone(timedelta(hours=8))
        cst_dt = aware_dt.astimezone(cst_tz)
        
        return (cst_dt.year, cst_dt.month, cst_dt.day,
                cst_dt.hour, cst_dt.minute, cst_dt.second)
    except Exception as e:
        raise ValueError(f"时间解析失败: {datetime_str}, 错误: {e}")


def doTime(akey, bkey):
    """
    计算两个时间之间的秒数差值
    """
    try:
        y1, m1, d1, h1, min1, s1 = parse_to_cst(akey)
        y2, m2, d2, h2, min2, s2 = parse_to_cst(bkey)
        result1 = datetime_to_seconds(y1, m1, d1, h1, min1, s1)
        result2 = datetime_to_seconds(y2, m2, d2, h2, min2, s2)
        return result2 - result1
    except Exception as e:
        raise ValueError(f"时间差值计算失败: {akey} 和 {bkey}, 错误: {e}")


def process_dataframe(df):
    """
    处理DataFrame，计算时间差值并存储到新列
    """
    try:
        # 使用apply()高效计算时间差值
        df['time_diff_seconds'] = df.apply(
            lambda row: doTime(row['leg_start_postime'], row['leg_end_postime']),
            axis=1
        )
        return df
    except Exception as e:
        raise ValueError(f"DataFrame处理失败, 错误: {e}")


# 示例用法
if __name__ == "__main__":
    # 文件路径
    df_path = r'Data\split_datasets\One_Hot_dataset\pridict_city\splitdataset\pridict_port\all_with_predictions_updated.csv'
    output_path = r'Data\split_datasets\One_Hot_dataset\pridict_city\splitdataset\pridict_port\all_updated_with_Stime.csv'

    # 读取数据
    try:
        df = pd.read_csv(df_path)
        
        # 处理DataFrame
        df = process_dataframe(df)
        
        # 保存结果
        df.to_csv(output_path, index=False)
        print(f"处理完成，结果已保存到: {output_path}")
    except Exception as e:
        print(f"程序运行失败, 错误: {e}")

    # 测试
    test_str = "2021/7/13 10:34:31+08"
    test_str2 = "2021/7/18 08:42:53.673+08"
    print(f"时间差值: {doTime(test_str, test_str2)} 秒")