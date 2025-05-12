# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import math
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 定义Haversine公式计算距离
def haversine(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度之间的距离（公里）
    """
    R = 6371  # 地球半径，单位公里
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

# 读取数据
data = pd.read_csv('1945-2023.csv')

# 查看数据结构
print("数据预览：")
print(data.head())

# 数据预处理
# 检查关键列是否存在
required_columns = ['经度', '纬度', '风速']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"数据中缺少必要的列: {col}")

# 模拟降水量数据
reference_lon = 120.0  # 示例经度
reference_lat = 25.0    # 示例纬度

# 计算每条记录与参考点的距离
data['距离'] = data.apply(lambda row: haversine(reference_lon, reference_lat, row['经度'], row['纬度']), axis=1)

# 模拟降水量数据
# 假设降水量 P = P0 * exp(-gamma * d) + 噪声
P0_true = 200  # 真正的P0值，用于模拟
gamma_true = 0.03  # 真正的gamma值，用于模拟

# 生成降水量
np.random.seed(42)  # 设置随机种子以保证结果可重复
noise = np.random.normal(0, 10, size=len(data))  # 正态分布噪声
data['降水量'] = P0_true * np.exp(-gamma_true * data['距离']) + noise

# 确保降水量非负
data['降水量'] = data['降水量'].apply(lambda x: max(x, 0))

# 处理缺失值
data = data.dropna(subset=['经度', '纬度', '风速', '降水量', '距离'])

# 移除风速和降水量为零或负数的记录，以避免取对数时出错
data = data[(data['风速'] > 0) & (data['降水量'] > 0) & (data['距离'] > 0)]

# 检查是否有数据被移除
print(f"数据量（移除无效记录后）：{len(data)}")

# 建立风速与距离的关系模型
# 指数衰减模型
data['ln_V'] = np.log(data['风速'])
exp_model = LinearRegression()
exp_model.fit(data[['距离']], data['ln_V'])
V0_exp = math.exp(exp_model.intercept_)
alpha = -exp_model.coef_[0]
V_pred_exp = V0_exp * np.exp(-alpha * data['距离'])

# 幂律衰减模型
data['ln_d'] = np.log(data['距离'])
pow_model = LinearRegression()
pow_model.fit(data[['ln_d']], data['ln_V'])
ln_V0 = pow_model.intercept_
beta = -pow_model.coef_[0]
V0_pow = math.exp(ln_V0)
V_pred_pow = V0_pow * (data['距离'] ** (-beta))

# 评估风速模型
R2_V_exp = r2_score(data['ln_V'], exp_model.predict(data[['距离']]))
MSE_V_exp = mean_squared_error(data['ln_V'], exp_model.predict(data[['距离']]))
R2_V_pow = r2_score(data['ln_V'], pow_model.predict(data[['ln_d']]))
MSE_V_pow = mean_squared_error(data['ln_V'], pow_model.predict(data[['ln_d']]))

print("\n风速模型评估：")
print(f"风速指数衰减模型: R² = {R2_V_exp:.4f}, MSE = {MSE_V_exp:.4f}")
print(f"风速幂律衰减模型: R² = {R2_V_pow:.4f}, MSE = {MSE_V_pow:.4f}")

# 选择最佳风速模型
if R2_V_exp > R2_V_pow:
    print("选择风速指数衰减模型")
    V0 = V0_exp
    alpha_final = alpha
    V_model = '指数衰减模型'
    V_pred_final = V_pred_exp
else:
    print("选择风速幂律衰减模型")
    V0 = V0_pow
    beta_final = beta
    V_model = '幂律衰减模型'
    V_pred_final = V_pred_pow

# 建立降水量与距离的关系模型
# 指数衰减模型
data['ln_P'] = np.log(data['降水量'])
exp_model_p = LinearRegression()
exp_model_p.fit(data[['距离']], data['ln_P'])
P0_exp = math.exp(exp_model_p.intercept_)
gamma = -exp_model_p.coef_[0]
P_pred_exp = P0_exp * np.exp(-gamma * data['距离'])

# 幂律衰减模型
pow_model_p = LinearRegression()
pow_model_p.fit(data[['ln_d']], data['ln_P'])
ln_P0 = pow_model_p.intercept_
delta = -pow_model_p.coef_[0]
P0_pow = math.exp(ln_P0)
P_pred_pow = P0_pow * (data['距离'] ** (-delta))

# 评估降水量模型
R2_P_exp = r2_score(data['ln_P'], exp_model_p.predict(data[['距离']]))
MSE_P_exp = mean_squared_error(data['ln_P'], exp_model_p.predict(data[['距离']]))
R2_P_pow = r2_score(data['ln_P'], pow_model_p.predict(data[['ln_d']]))
MSE_P_pow = mean_squared_error(data['ln_P'], pow_model_p.predict(data[['ln_d']]))

print("\n降水量模型评估：")
print(f"降水量指数衰减模型: R² = {R2_P_exp:.4f}, MSE = {MSE_P_exp:.4f}")
print(f"降水量幂律衰减模型: R² = {R2_P_pow:.4f}, MSE = {MSE_P_pow:.4f}")

# 选择最佳降水量模型
if R2_P_exp > R2_P_pow:
    print("选择降水量指数衰减模型")
    P0 = P0_exp
    gamma_final = gamma
    P_model = '指数衰减模型'
    P_pred_final = P_pred_exp
else:
    print("选择降水量幂律衰减模型")
    P0 = P0_pow
    delta_final = delta
    P_model = '幂律衰减模型'
    P_pred_final = P_pred_pow

# 打印最终模型参数
print("\n最终风速模型: {} \n参数: {}".format(V_model,
    {"V0": V0, "alpha" if V_model=="指数衰减模型" else "beta": alpha_final if V_model=="指数衰减模型" else beta_final}))
print("最终降水量模型: {} \n参数: {}".format(P_model,
    {"P0": P0, "gamma" if P_model=="指数衰减模型" else "delta": gamma_final if P_model=="指数衰减模型" else delta_final}))

# 绘制风速模型拟合图
plt.figure(figsize=(12, 6))
plt.scatter(data['距离'], data['风速'], color='blue', alpha=0.5, label='实际风速')
if V_model == '指数衰减模型':
    plt.plot(data['距离'], V_pred_final, color='red', label='指数衰减拟合')
else:
    plt.plot(data['距离'], V_pred_final, color='green', label='幂律衰减拟合')
plt.xlabel('距离 (km)')
plt.ylabel('风速 (m/s)')
plt.title('风速与距离的关系模型')
plt.legend()
plt.grid(True)
plt.show()

# 绘制降水量模型拟合图
plt.figure(figsize=(12, 6))
plt.scatter(data['距离'], data['降水量'], color='blue', alpha=0.5, label='实际降水量')
if P_model == '指数衰减模型':
    plt.plot(data['距离'], P_pred_final, color='red', label='指数衰减拟合')
else:
    plt.plot(data['距离'], P_pred_final, color='green', label='幂律衰减拟合')
plt.xlabel('距离 (km)')
plt.ylabel('降水量 (mm)')
plt.title('降水量与距离的关系模型')
plt.legend()
plt.grid(True)
plt.show()

# 台风贝碧嘉的预测
# 假设贝碧嘉的中心位置在指定时间的距离如下（示例数据）
typhoon_forecast = pd.DataFrame({
    '时间': ['2024-09-16 14:00', '2024-09-17 14:00', '2024-09-18 14:00'],
    '距离': [30, 50, 70]  # 单位：公里
})

# 根据选择的模型进行预测
def predict_V(d, model_type, params):
    if model_type == '指数衰减模型':
        return params['V0'] * np.exp(-params['alpha'] * d)
    else:
        return params['V0'] * (d ** (-params['beta']))

def predict_P(d, model_type, params):
    if model_type == '指数衰减模型':
        return params['P0'] * np.exp(-params['gamma'] * d)
    else:
        return params['P0'] * (d ** (-params['delta']))

# 准备模型参数
if V_model == '指数衰减模型':
    V_params = {'V0': V0, 'alpha': alpha_final}
else:
    V_params = {'V0': V0, 'beta': beta_final}

if P_model == '指数衰减模型':
    P_params = {'P0': P0, 'gamma': gamma_final}
else:
    P_params = {'P0': P0, 'delta': delta_final}

# 进行预测
typhoon_forecast['预测风速 (m/s)'] = typhoon_forecast['距离'].apply(lambda d: predict_V(d, V_model, V_params))
typhoon_forecast['预测降水量 (mm)'] = typhoon_forecast['距离'].apply(lambda d: predict_P(d, P_model, P_params))

# 显示预测结果
print("\n台风贝碧嘉的风速与降水量预测：")
print(typhoon_forecast)

# 绘制预测结果
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('时间')
ax1.set_ylabel('预测风速 (m/s)', color=color)
ax1.plot(typhoon_forecast['时间'], typhoon_forecast['预测风速 (m/s)'], color=color, marker='o', label='预测风速')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  # 共享x轴
color = 'tab:red'
ax2.set_ylabel('预测降水量 (mm)', color=color)
ax2.plot(typhoon_forecast['时间'], typhoon_forecast['预测降水量 (mm)'], color=color, marker='x', label='预测降水量')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title('台风贝碧嘉行进途中的风速与降水量预测')
plt.grid(True)
plt.show()
