# 导入必要的库
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 绘图
import seaborn as sns  # 数据可视化
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器
from sklearn.metrics import classification_report, confusion_matrix  # 模型评估
from sklearn.preprocessing import StandardScaler  # 特征标准化

# 1. 数据导入与清洗

# 读取台风数据
typhoon_data = pd.read_csv('1945-2023.csv')  # 请替换为实际文件路径

# 查看数据基本信息
print(typhoon_data.info())

# 处理缺失值
numeric_cols = ['风速', '气压', '移动速度']
for col in numeric_cols:
    typhoon_data[col] = typhoon_data[col].fillna(typhoon_data[col].mean())

# 删除关键列的缺失值
typhoon_data = typhoon_data.dropna(subset=['台风编号', '台风中文名称', '台风起始时间', '台风结束时间'])

# 处理异常值
for col in numeric_cols:
    lower = typhoon_data[col].quantile(0.01)
    upper = typhoon_data[col].quantile(0.99)
    typhoon_data[col] = typhoon_data[col].clip(lower, upper)

# 转换时间格式
typhoon_data['台风起始时间'] = pd.to_datetime(typhoon_data['台风起始时间'])
typhoon_data['台风结束时间'] = pd.to_datetime(typhoon_data['台风结束时间'])
typhoon_data['当前台风时间'] = pd.to_datetime(typhoon_data['当前台风时间'])

# 提取月份信息
typhoon_data['月份'] = typhoon_data['当前台风时间'].dt.month

# 2. 特征工程

# 区分夏台风（6-8月）和秋台风（9-11月）
def categorize_season(month):
    if 6 <= month <= 8:
        return '夏台风'
    elif 9 <= month <= 11:
        return '秋台风'
    else:
        return '其他'

typhoon_data['季节'] = typhoon_data['月份'].apply(categorize_season)

# 过滤出夏台风和秋台风的数据
filtered_typhoon = typhoon_data[typhoon_data['季节'].isin(['夏台风', '秋台风'])].copy()

# 3. 台风分类评价模型的建立

# 类别划分标准
def categorize_typhoon(wind_speed):
    if 17 <= wind_speed < 34:
        return 1  # 弱台风
    elif 34 <= wind_speed < 48:
        return 2  # 中台风
    elif 48 <= wind_speed < 64:
        return 3  # 强台风
    elif wind_speed >= 64:
        return 4  # 超强台风
    else:
        return 0  # 未分类

# 处理SettingWithCopyWarning：使用 .loc 进行赋值
filtered_typhoon.loc[:, '类别'] = filtered_typhoon['风速'].apply(categorize_typhoon)

# 特征选择
X = filtered_typhoon[['风速', '气压', '移动速度']]
y = filtered_typhoon['类别']

# 删除类别为0的记录（未分类）
X = X[y != 0]
y = y[y != 0]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 4. 模型训练与验证

# 进行预测
y_pred = rf_classifier.predict(X_test)

# 打印分类报告
print("分类报告：")
print(classification_report(y_test, y_pred))

# 打印混淆矩阵
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

# 5. 确定台风途经省份

def get_provinces(longitude, latitude):
    provinces = []
    # 广东省
    if 108 <= longitude <= 124 and 18 <= latitude <= 25:
        provinces.append('广东')
    # 福建省
    if 117 <= longitude <= 126 and 23 <= latitude <= 28:
        provinces.append('福建')
    # 浙江省
    if 118 <= longitude <= 123 and 27 <= latitude <= 31:
        provinces.append('浙江')
    # 江苏省
    if 118 <= longitude <= 130 and 30 <= latitude <= 35:
        provinces.append('江苏')
    # 上海市
    if 120 <= longitude <= 122 and 30 <= latitude <= 32:
        provinces.append('上海')
    # 海南省
    if 108 <= longitude <= 110 and 18 <= latitude <= 20:
        provinces.append('海南')
    # 广西壮族自治区
    if 105 <= longitude <= 110 and 20 <= latitude <= 25:
        provinces.append('广西')
    # 台湾省（简化）
    if 119 <= longitude <= 122 and 21 <= latitude <= 25:
        provinces.append('台湾')
    # 辽宁省
    if 120 <= longitude <= 125 and 38 <= latitude <= 43:
        provinces.append('辽宁')
    # 山东省
    if 117 <= longitude <= 122 and 34 <= latitude <= 38:
        provinces.append('山东')
    # 河北省
    if 114 <= longitude <= 120 and 38 <= latitude <= 42:
        provinces.append('河北')
    # 天津市
    if 117 <= longitude <= 120 and 38 <= latitude <= 40:
        provinces.append('天津')
    # 黑龙江省
    if 122 <= longitude <= 135 and 43 <= latitude <= 53:
        provinces.append('黑龙江')
    # 北京市
    if 115 <= longitude <= 117 and 39 <= latitude <= 41:
        provinces.append('北京')
    # 山西省
    if 110 <= longitude <= 121 and 34 <= latitude <= 42:
        provinces.append('山西')
    # 内蒙古自治区
    if 97 <= longitude <= 126 and 37 <= latitude <= 53:
        provinces.append('内蒙古')
    # 四川省
    if 97 <= longitude <= 110 and 25 <= latitude <= 35:
        provinces.append('四川')
    # 贵州省
    if 103 <= longitude <= 109 and 24 <= latitude <= 29:
        provinces.append('贵州')
    # 云南省
    if 97 <= longitude <= 106 and 21 <= latitude <= 30:
        provinces.append('云南')
    # 陕西省
    if 105 <= longitude <= 112 and 34 <= latitude <= 42:
        provinces.append('陕西')
    # 甘肃省
    if 95 <= longitude <= 108 and 32 <= latitude <= 42:
        provinces.append('甘肃')
    # 青海省
    if 80 <= longitude <= 100 and 32 <= latitude <= 39:
        provinces.append('青海')
    # 西藏自治区
    if 78 <= longitude <= 98 and 26 <= latitude <= 38:
        provinces.append('西藏')
    # 宁夏回族自治区
    if 104 <= longitude <= 110 and 35 <= latitude <= 40:
        provinces.append('宁夏')
    # 新疆维吾尔自治区
    if 73 <= longitude <= 135 and 34 <= latitude <= 49:
        provinces.append('新疆')
    # 重庆市
    if 105 <= longitude <= 111 and 28 <= latitude <= 33:
        provinces.append('重庆')
    # 贵州省
    if 103 <= longitude <= 109 and 24 <= latitude <= 29:
        provinces.append('贵州')
    # 河北省
    if 114 <= longitude <= 120 and 38 <= latitude <= 42:
        provinces.append('河北')
    # 内蒙古自治区
    if 97 <= longitude <= 126 and 37 <= latitude <= 53:
        provinces.append('内蒙古')
    # 去重
    provinces = list(set(provinces))
    return provinces


# 为每条记录添加途经省份
filtered_typhoon['途经省份'] = filtered_typhoon.apply(
    lambda row: get_provinces(row['经度'], row['纬度']), axis=1
)

# 聚合每个台风的途经省份，并使用最大类别
province_group = filtered_typhoon.groupby('台风编号')['途经省份'].apply(
    lambda lists: list(set([province for sublist in lists for province in sublist]))
).reset_index()

# 合并类别信息，使用最大类别
category_group = filtered_typhoon.groupby('台风编号')['类别'].max().reset_index()

# 合并结果
result_table = pd.merge(category_group, province_group, on='台风编号')

# 显示部分结果
print(result_table.head())

# 6. 夏台风与秋台风的区别分析

# 重新筛选类别为0的台风
# (已在之前步骤删除类别为0的记录)

# 聚合夏台风和秋台风的类别
summer_typhoons = filtered_typhoon[filtered_typhoon['季节'] == '夏台风']
autumn_typhoons = filtered_typhoon[filtered_typhoon['季节'] == '秋台风']

summer_categories = summer_typhoons.groupby('台风编号')['类别'].max().reset_index()
autumn_categories = autumn_typhoons.groupby('台风编号')['类别'].max().reset_index()

summer_categories['季节'] = '夏台风'
autumn_categories['季节'] = '秋台风'

combined_categories = pd.concat([summer_categories, autumn_categories])

# 绘制类别分布图
plt.figure(figsize=(10, 6))
sns.countplot(x='类别', hue='季节', data=combined_categories)
plt.title('夏台风与秋台风类别分布对比')
plt.xlabel('台风类别')
plt.ylabel('数量')
plt.legend(title='季节')
plt.show()

# 统计夏台风与秋台风的平均风速和气压
summer_stats = summer_typhoons[['风速', '气压']].mean()
autumn_stats = autumn_typhoons[['风速', '气压']].mean()

print("夏台风平均风速：", summer_stats['风速'])
print("夏台风平均气压：", summer_stats['气压'])
print("秋台风平均风速：", autumn_stats['风速'])
print("秋台风平均气压：", autumn_stats['气压'])

# 7. 保存结果

# 将分类结果与途经省份保存为CSV文件
result_table.to_csv('typhoon_classification_1945_2023_with_provinces.csv', index=False)

# 完成
