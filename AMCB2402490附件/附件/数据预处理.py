import requests
import json
import re
import time
import pandas as pd
import datetime

def date_pred(date, deltahour):
    """
    date: yyyymmddHHMM, string
    deltahour: hours, integer
    """
    time_obj = datetime.datetime.strptime(date, "%Y%m%d%H%M")
    new_date = (time_obj + datetime.timedelta(hours=deltahour)).strftime("%Y%m%d%H%M")
    return new_date

def get_current_tc_list(url, headers, min_num=2401, max_num=2421):
    """
    获取当前存在的台风列表，并筛选编号在min_num到max_num之间的台风。

    参数:
        url (str): 获取台风列表的URL
        headers (dict): HTTP请求头
        min_num (int): 台风编号下限
        max_num (int): 台风编号上限

    返回:
        list: 满足条件的台风信息列表
    """
    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        html_obj = response.text
        # 使用正则表达式提取JSON数据
        match = re.match(r".*?({.*}).*", html_obj, re.S)
        if not match:
            print("无法解析台风列表的JSON数据")
            return []
        data = json.loads(match.group(1)).get('typhoonList', [])
    except Exception as e:
        print(f"请求台风列表时发生错误: {e}")
        return []

    item_list = []
    for v in data:
        try:
            tc_num_str = v[4]  # 编号，假设为字符串
            # 处理可能的非数字字符，例如前导零或其他分隔符
            tc_num_clean = re.sub(r'\D', '', tc_num_str)  # 移除所有非数字字符
            if not tc_num_clean:
                print(f"台风编号 {tc_num_str} 无法转换为数字，跳过。")
                continue
            tc_num = int(tc_num_clean)
            if min_num <= tc_num <= max_num:
                item = {
                    'id': v[0],
                    'tc_num': tc_num_str,   # 编号
                    'name_cn': v[2],        # 中文名
                    'name_en': v[1],        # 英文名
                    'dec': v[6]             # 名字含义
                }
                item_list.append(item)
        except (ValueError, IndexError) as e:
            print(f"处理台风数据时发生错误: {e}")
            continue

    # 打印所有抓取到的台风编号以供调试
    print(f"抓取到的台风编号: {[item['tc_num'] for item in item_list]}")
    return item_list

def get_type(date_type):
    """
    将台风类型代码转换为中文描述。

    参数:
        date_type (str): 台风类型代码

    返回:
        str: 台风类型的中文描述
    """
    type_mapping = {
        'TC': '热带气旋',
        'TD': '热带低压',
        'TS': '热带风暴',
        'STS': '强热带风暴',
        'TY': '台风',
        'STY': '强台风',
        'SuperTY': '超强台风',
        '': '',
    }
    return type_mapping.get(date_type, '')

def get_tc_info(item, headers):
    """
    获取单个台风的详细信息，并筛选2024年的数据。

    参数:
        item (dict): 台风的基本信息
        headers (dict): HTTP请求头

    返回:
        pd.DataFrame: 台风详细信息的DataFrame
    """
    t = int(round(time.time() * 1000))  # 13位时间戳
    url = f'http://typhoon.nmc.cn/weatherservice/typhoon/jsons/view_{item["id"]}?t={t}&callback=typhoon_jsons_view_{item["id"]}'
    try:
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()
        html_obj = response.text
        # 使用正则表达式提取JSON数据
        match = re.match(r".*?({.*}).*", html_obj, re.S)
        if not match:
            print(f"无法解析台风 {item['tc_num']} 的JSON数据")
            return pd.DataFrame()
        data = json.loads(match.group(1)).get('typhoon', [])
    except Exception as e:
        print(f"请求台风 {item['tc_num']} 信息时发生错误: {e}")
        return pd.DataFrame()

    # 检查data是否为列表且长度足够
    if not isinstance(data, list) or len(data) <= 8:
        print(f"台风 {item['tc_num']} 的数据结构不符合预期")
        return pd.DataFrame()

    info_dicts = {
        'tc_num':[],       # 编号
        'name_cn':[],      # 中文名
        'name_en':[],      # 英文名
        'dateUTC':[],      # 日期 UTC
        'dateCST':[],      # 日期 CST
        'vmax':[],         # 最大风速 m/s
        'grade':[],        # 等级
        'latTC':[],        # 纬度 deg
        'lonTC':[],        # 经度 deg
        'mslp':[],         # 中心气压 hPa
        'attr':[]          # 属性, 预报 forecast，实况 analysis
    }

    # 先遍历实况
    for v in data[8]:
        try:
            date_utc = v[1]
            if not date_utc.startswith("2024"):
                continue  # 只保留2024年的数据
            info_dicts['tc_num'].append(item['tc_num'])
            info_dicts['name_cn'].append(item['name_cn'])
            info_dicts['name_en'].append(item['name_en'])
            info_dicts['dateUTC'].append(date_utc)
            info_dicts['dateCST'].append(date_pred(date_utc, 8))  # UTC to CST
            info_dicts['vmax'].append(v[7])
            info_dicts['grade'].append(get_type(v[3]))
            info_dicts['lonTC'].append(v[4])
            info_dicts['latTC'].append(v[5])
            info_dicts['mslp'].append(v[6])
            info_dicts['attr'].append('analysis')
        except IndexError as e:
            print(f"处理实况数据时发生错误: {e}")
            continue

    # 最新预报时刻
    if info_dicts['dateUTC']:
        dateUTC0 = info_dicts['dateUTC'][-1]
    else:
        dateUTC0 = None

    # 最新的一次预报
    if dateUTC0:
        try:
            # 确保 data[8][-1][11] 存在且不是 None
            forecast_data = data[8][-1][11]
            if forecast_data and isinstance(forecast_data, dict):
                BABJ_list = forecast_data.get('BABJ', [])
            else:
                print(f"台风 {item['tc_num']} 的预报数据为空或格式不正确")
                BABJ_list = []
        except (IndexError, KeyError, TypeError) as e:
            print(f"台风 {item['tc_num']} 的预报数据不完整或格式错误: {e}")
            BABJ_list = []

        for babj in BABJ_list:
            try:
                pred_hour = int(babj[0])  # 预报时效，hour
                dateUTC_pred = date_pred(dateUTC0, pred_hour)
                if not dateUTC_pred.startswith("2024"):
                    continue  # 只保留2024年的数据
                info_dicts['tc_num'].append(item['tc_num'])
                info_dicts['name_cn'].append(item['name_cn'])
                info_dicts['name_en'].append(item['name_en'])
                info_dicts['dateUTC'].append(dateUTC_pred)
                info_dicts['dateCST'].append(date_pred(dateUTC_pred, 8))
                info_dicts['vmax'].append(babj[5])
                info_dicts['grade'].append(get_type(babj[7]))
                info_dicts['lonTC'].append(babj[2])
                info_dicts['latTC'].append(babj[3])
                info_dicts['mslp'].append(babj[4])
                info_dicts['attr'].append('forecast')
            except (IndexError, ValueError, TypeError) as e:
                print(f"处理预报数据时发生错误: {e}")
                continue

    tc_info = pd.DataFrame(info_dicts)
    return tc_info

if __name__ == "__main__":

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' +
                      'AppleWebKit/537.36 (KHTML, like Gecko) ' +
                      'Chrome/91.0.4472.124 Safari/537.36'
    }

    # 获取目前存在的西北太平洋台风，包括未编号但是给出路径预报的
    t = int(round(time.time() * 1000))  # 13位时间戳
    url = f'http://typhoon.nmc.cn/weatherservice/typhoon/jsons/list_default?t={t}&callback=typhoon_jsons_list_default'
    item_list = get_current_tc_list(url, headers, min_num=2401, max_num=2421)

    all_tc_data = pd.DataFrame()

    if len(item_list) > 0:  # 存在台风
        print(f'目前西北太平洋存在{len(item_list)}个符合编号2401-2421的热带气旋!')
        for item in item_list:
            print(f"热带气旋编号: {item['tc_num']}, 名字: {item['name_cn']}")
            data = get_tc_info(item, headers)
            if not data.empty:
                all_tc_data = pd.concat([all_tc_data, data], ignore_index=True)
                print(data)
    else:
        print('目前西北太平洋没有符合编号2401-2421的热带气旋生成!')

    if not all_tc_data.empty:
        # 保存为CSV文件
        all_tc_data.to_csv('typhoon_2401_2421_2024_data1.csv', index=False, encoding='utf-8-sig')
        print("台风数据已保存至 typhoon_2401_2421_2024_data.csv")

        # 或者保存为Excel文件
        # all_tc_data.to_excel('typhoon_2401_2421_2024_data.xlsx', index=False)
        # print("台风数据已保存至 typhoon_2401_2421_2024_data.xlsx")
    else:
        print("没有符合条件的台风数据需要保存。")
