import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 定义根据风速将台风分类的函数
def classify_typhoon_by_wind_grade(vmax_knots):
    vmax_ms = vmax_knots * 0.51444  # 将风速从节转换为米/秒
    if vmax_ms < 10.8:
        return '非台风'
    elif 10.8 <= vmax_ms < 17.2:
        return '热带低压'
    elif 17.2 <= vmax_ms < 24.5:
        return '热带风暴'
    elif 24.5 <= vmax_ms < 32.7:
        return '强热带风暴'
    elif 32.7 <= vmax_ms < 41.5:
        return '台风'
    elif 41.5 <= vmax_ms < 56.1:
        return '强台风'
    else:
        return '超强台风'

# 加载台风数据
data_path_2024 = 'typhoon_2401_2421_2024_data1.csv'
typhoon_data_2024 = pd.read_csv(data_path_2024)
typhoon_data_2024['dateUTC'] = pd.to_datetime(typhoon_data_2024['dateUTC'], format='%Y%m%d%H%M')

# 筛选出7月和9月的数据
typhoon_data_jul_sep = typhoon_data_2024[(typhoon_data_2024['dateUTC'].dt.month == 7) | (typhoon_data_2024['dateUTC'].dt.month == 9)].copy()
typhoon_data_jul_sep['Typhoon_Category'] = typhoon_data_jul_sep['vmax'].apply(classify_typhoon_by_wind_grade)

# 加载中国省份地理信息
provinces_path = 'china_provinces.shp'
china_provinces = gpd.read_file(provinces_path)

# 创建点数据集
typhoon_data_jul_sep['geometry'] = typhoon_data_jul_sep.apply(lambda row: Point(row['lonTC'], row['latTC']), axis=1)
typhoon_points = gpd.GeoDataFrame(typhoon_data_jul_sep, geometry='geometry', crs=china_provinces.crs)

# 空间连接，确定每个台风点所在的省份
typhoon_with_province = gpd.sjoin(typhoon_points, china_provinces, how='left', predicate='intersects')

# 汇总输出，使用正确的省份列名 '省'
summary_data = typhoon_with_province[['name_cn', 'dateCST', 'Typhoon_Category', '省']].drop_duplicates()
summary_file_path = 'Typhoon_Summary_2024_Jul_Sep.csv'
summary_data.to_csv(summary_file_path, index=False)

print(f"数据已汇总并保存到 '{summary_file_path}'")
