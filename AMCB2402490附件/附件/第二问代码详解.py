import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tslearn.metrics import dtw, dtw_path
from geopy.distance import geodesic
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import logging
from pathlib import Path
import sys

# 设置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置中文字体和负号显示
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS']  # 添加备用字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 全局变量
SEQ_LENGTH = 3  # 使用前三个数据点进行预测

# 1. 数据加载与预处理

def load_data(historical_path: Path, current_year_path: Path):
    try:
        historical_data = pd.read_csv(historical_path)
        current_year_data = pd.read_csv(current_year_path)
        logging.info("数据加载成功。")
        return historical_data, current_year_data
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"数据加载出错: {e}")
        sys.exit(1)

# 2. 数据清洗与对齐

def preprocess_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['台风编号', '当前台风时间', '经度', '纬度', '风速', '气压', '移动速度']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"历史数据缺少必要的列: {missing_columns}")

    # 解析日期时间
    df['当前台风时间'] = pd.to_datetime(df['当前台风时间'], errors='coerce')

    # 检查日期解析结果
    if df['当前台风时间'].isnull().any():
        logging.warning("部分日期时间解析失败，可能存在缺失或格式错误。")

    # 处理缺失值
    df[['风速', '气压', '移动速度']] = df[['风速', '气压', '移动速度']].apply(pd.to_numeric, errors='coerce')

    # 检查缺失值处理结果
    if df[['风速', '气压', '移动速度']].isnull().any().any():
        logging.warning("部分风速、气压或移动速度数据仍然存在缺失。")

    # 使用赋值操作代替 in-place=True，避免警告
    df[['风速', '气压', '移动速度']] = df[['风速', '气压', '移动速度']].fillna(method='ffill').fillna(method='bfill')

    # 计算移动方向（向量化）
    df = df.sort_values('当前台风时间').reset_index(drop=True).copy()
    delta_lon = df['经度'].diff()
    delta_lat = df['纬度'].diff()
    direction_rad = np.arctan2(delta_lon, delta_lat)  # 经度为x，纬度为y
    direction_deg = (np.degrees(direction_rad) + 360) % 360
    df['移动方向'] = direction_deg.fillna(method='ffill').fillna(0)

    # 选择需要的列，包括 '台风编号' 用于后续分组
    df = df[['台风编号', '当前台风时间', '经度', '纬度', '风速', '气压', '移动方向', '移动速度']].copy()

    return df

def preprocess_current_year_data(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['tc_num', 'name_cn', 'name_en', 'date', 'vmax', 'grade', 'latTC', 'lonTC', 'mslp', 'attr']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"当前年数据缺少必要的列: {missing_columns}")

    # 解析日期时间
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M', errors='coerce')

    # 检查日期解析结果
    if df['date'].isnull().any():
        logging.warning("部分日期时间解析失败，可能存在缺失或格式错误。")

    # 处理缺失值
    df[['vmax', 'mslp']] = df[['vmax', 'mslp']].apply(pd.to_numeric, errors='coerce')

    # 检查缺失值处理结果
    if df[['vmax', 'mslp']].isnull().any().any():
        logging.warning("部分 vmax 或 mslp 数据仍然存在缺失。")

    # 使用赋值操作代替 in-place=True，避免警告
    df[['vmax', 'mslp']] = df[['vmax', 'mslp']].fillna(method='ffill').fillna(method='bfill')

    # 重命名列以统一
    df = df.rename(columns={
        'lonTC': '经度',
        'latTC': '纬度',
        'vmax': '风速',
        'mslp': '气压'
    })

    # 选择需要的列
    df = df[['date', '经度', '纬度', '风速', '气压', 'tc_num', 'name_cn']].copy()

    # 计算移动方向和速度（向量化）
    df = df.sort_values('date').reset_index(drop=True).copy()

    # 计算距离和时间差
    coords_prev = df[['纬度', '经度']].shift(1)
    df['纬度_prev'] = coords_prev['纬度']
    df['经度_prev'] = coords_prev['经度']

    # 计算距离
    def calculate_distance(row):
        if pd.notnull(row['纬度_prev']) and pd.notnull(row['经度_prev']):
            return geodesic((row['纬度_prev'], row['经度_prev']), (row['纬度'], row['经度'])).kilometers
        else:
            return 0.0

    df['距离'] = df.apply(calculate_distance, axis=1)

    # 计算时间差（小时）
    df['时间差'] = df['date'].diff().dt.total_seconds() / 3600
    df['时间差'] = df['时间差'].replace(0, np.nan).fillna(method='ffill').fillna(3)  # 默认3小时

    # 计算移动速度
    df['移动速度'] = df['距离'] / df['时间差']

    # 计算移动方向
    delta_lon = df['经度'] - df['经度_prev']
    delta_lat = df['纬度'] - df['纬度_prev']
    avg_lat = (df['纬度'] + df['纬度_prev']) / 2
    direction_rad = np.arctan2(delta_lon * np.cos(np.deg2rad(avg_lat)), delta_lat)
    direction_deg = (np.degrees(direction_rad) + 360) % 360
    df['移动方向'] = direction_deg.fillna(0)

    # 填补第一个时间步的移动方向和速度
    df['移动方向'] = df['移动方向'].fillna(0)
    df['移动速度'] = df['移动速度'].fillna(0)

    # 选择需要的列
    df = df[['date', '经度', '纬度', '风速', '气压', '移动方向', '移动速度', 'tc_num', 'name_cn']].copy()

    return df

# 3. 特征工程

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # 计算路径惯性（前一时刻的方向）
    df['路径惯性'] = df['移动方向'].shift(1).fillna(0)

    # 计算气压梯度（气压变化率）
    df['气压梯度'] = df['气压'].diff().fillna(0)

    # 计算速度变化率（可选）
    df['速度变化率'] = df['移动速度'].diff().fillna(0)

    return df

# 4. 筛选台风数据

def select_typhoon_data(current_year_df: pd.DataFrame, tc_num: int, name_cn: str) -> pd.DataFrame:
    typhoon = current_year_df[
        (current_year_df['tc_num'] == tc_num) &
        (current_year_df['name_cn'] == name_cn)
        ].copy()

    logging.info(f"{name_cn}台风数据点数: {len(typhoon)}")

    if len(typhoon) < SEQ_LENGTH:
        raise ValueError(f"台风数据不足以创建序列。提供的数据点数: {len(typhoon)}")

    return typhoon

# 5. 构建时序数据集

def create_sequences(df: pd.DataFrame, features: list, target: list, seq_length: int = 3):
    X = []
    y = []
    # 按台风编号分组，确保序列不跨台风
    if '台风编号' in df.columns:
        group_col = '台风编号'
    elif 'tc_num' in df.columns:
        group_col = 'tc_num'
    else:
        raise ValueError("DataFrame中缺少台风编号列（'台风编号'或'tc_num'）。")

    for name, group in df.groupby(group_col):
        # 确定时间列名称
        time_col = '当前台风时间' if '当前台风时间' in group.columns else 'date'
        group = group.sort_values(time_col).reset_index(drop=True).copy()
        for i in range(len(group) - seq_length):
            X.append(group[features].iloc[i:i + seq_length].values)
            y.append(group[target].iloc[i + seq_length].values)

    return np.array(X), np.array(y)

# 6. 模型构建

def build_model(input_shape: tuple, output_dim: int) -> Sequential:
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))  # 修改激活函数为 'linear'

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    return model

# 7. 模型训练

def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray) -> Sequential:
    history = model.fit(
        X_train, y_train,
        epochs=200,  # 增加训练轮次
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ],
        verbose=1
    )
    return history

# 8. 模型评估

def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray,
                  target_scaler: MinMaxScaler):
    loss, mae = model.evaluate(X_test, y_test)
    logging.info(f"测试集均方误差 (MSE): {loss}")
    logging.info(f"测试集平均绝对误差 (MAE): {mae}")

    # 计算额外的评估指标
    y_test_inv = target_scaler.inverse_transform(y_test)
    y_pred = model.predict(X_test)
    y_pred_inv = target_scaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)

    logging.info(f"测试集均方误差 (MSE): {mse}")
    logging.info(f"测试集均方根误差 (RMSE): {rmse}")
    logging.info(f"测试集决定系数 (R²): {r2}")

    return y_test_inv, y_pred_inv

# 9. 台风路径预测

def predict_typhoon_path(model: Sequential, typhoon_df: pd.DataFrame, feature_scaler: MinMaxScaler,
                         target_scaler: MinMaxScaler, features: list, seq_length: int = 3):
    # 确保台风有足够的数据点
    if len(typhoon_df) < seq_length:
        raise ValueError("台风数据不足以创建序列。请提供更多数据。")

    # 排序数据
    typhoon_df = typhoon_df.sort_values('date').reset_index(drop=True).copy()

    # 选择特征
    typhoon_features = typhoon_df[features].copy()

    # 填补缺失值
    typhoon_features = typhoon_features.fillna(method='ffill').fillna(method='bfill')

    # 确保数据按3小时间隔采样
    typhoon_df = typhoon_df.set_index('date').resample('3H').ffill().reset_index()
    typhoon_features_scaled = feature_scaler.transform(typhoon_features)

    # 获取预测时间间隔
    time_diffs = typhoon_df['date'].diff().dropna()
    if not time_diffs.empty:
        freq = time_diffs.mode()[0]
    else:
        freq = pd.Timedelta(hours=3)  # 默认3小时

    # 获取预测时间起点
    last_actual_time = typhoon_df['date'].iloc[seq_length - 1]

    # 获取实际路径用于比较
    actual_positions = typhoon_df[['经度', '纬度']].iloc[seq_length:].values

    # 初始化输入序列为前三个真实数据点
    current_input = typhoon_features_scaled[:seq_length].reshape((1, seq_length, len(features)))

    # 初始化预测结果列表
    predictions_original = []
    prediction_times = []
    prediction_longitudes = []
    prediction_latitudes = []

    # 遍历每一个时间步
    for i in range(seq_length, len(typhoon_features_scaled)):
        # 预测下一个点
        pred_scaled = model.predict(current_input)
        pred_original = target_scaler.inverse_transform(pred_scaled)[0]

        # 保存预测结果
        predictions_original.append(pred_original)

        # 生成预测时间
        prediction_time = last_actual_time + freq
        prediction_times.append(prediction_time)

        # 保存预测经度和纬度
        prediction_longitudes.append(pred_original[0])  # 经度
        prediction_latitudes.append(pred_original[1])  # 纬度

        # 日志记录预测值与实际值对比
        actual_index = i - seq_length
        if actual_index < len(actual_positions):
            actual_lon = actual_positions[actual_index][0]
            actual_lat = actual_positions[actual_index][1]
            logging.info(
                f"预测时间: {prediction_time}, 预测经度: {pred_original[0]:.5f}, 预测纬度: {pred_original[1]:.5f}, "
                f"实际经度: {actual_lon}, 实际纬度: {actual_lat}")

        # 更新 last_actual_time
        last_actual_time = prediction_time

        # 准备下一步的输入序列
        if actual_index < len(actual_positions):
            # 使用实际的下一个点来更新输入序列
            actual_next = typhoon_df[features].iloc[i].values.reshape(1, -1)
            actual_next_scaled = feature_scaler.transform(actual_next)
            new_features_scaled = actual_next_scaled.copy()
        else:
            # 使用预测的点来更新输入序列
            new_features_scaled = typhoon_features_scaled[i].copy()
            new_features_scaled[:2] = pred_scaled  # 仅替换经度和纬度

        # 更新输入序列：移除最旧的数据点，添加当前预测或实际的数据点
        current_input = np.append(current_input[:, 1:, :], new_features_scaled.reshape(1, 1, -1), axis=1)

    predictions_original = np.array(predictions_original)

    # 确保预测结果与实际数据长度一致
    assert len(predictions_original) == len(actual_positions), "预测结果与实际数据长度不一致。"

    return predictions_original, prediction_times, prediction_longitudes, prediction_latitudes, actual_positions

# 10. 动态时间规整（DTW）比较

def compute_dtw(predictions: np.ndarray, actual: np.ndarray) -> float:
    distance = dtw(predictions, actual)
    logging.info(f"DTW距离: {distance}")
    return distance

def compute_dtw_path(predictions: np.ndarray, actual: np.ndarray):
    path, dtw_distance = dtw_path(predictions, actual)
    logging.info(f"DTW Path 长度: {len(path)}")

    # 检查 path 是否为列表且每个元素为长度为2的元组
    if not isinstance(path, list) or not all(isinstance(p, tuple) and len(p) == 2 for p in path):
        raise ValueError("DTW Path 的格式不正确。")

    aligned_pred = [predictions[p[0]] for p in path]
    aligned_actual = [actual[p[1]] for p in path]

    aligned_pred = np.array(aligned_pred)
    aligned_actual = np.array(aligned_actual)

    return aligned_pred, aligned_actual

# 11. 可视化

def plot_paths(predictions: np.ndarray, actual: np.ndarray, title: str = '台风路径预测与实际路径比较'):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions[:, 0], predictions[:, 1], marker='o', label='预测路径', color='red')  # 经度为x，纬度为y
    plt.plot(actual[:, 0], actual[:, 1], marker='x', label='实际路径', color='blue')  # 经度为x，纬度为y
    plt.title(title)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend()
    plt.grid()
    plt.show()

def plot_aligned_paths(aligned_pred: np.ndarray, aligned_actual: np.ndarray, title: str = 'DTW对齐后的预测路径与实际路径'):
    plt.figure(figsize=(10, 6))
    plt.plot(aligned_pred[:, 0], aligned_pred[:, 1], marker='o', label='预测路径对齐', color='green')
    plt.plot(aligned_actual[:, 0], aligned_actual[:, 1], marker='x', label='实际路径对齐', color='purple')
    plt.title(title)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend()
    plt.grid()
    plt.show()

def plot_paths_on_map(predicted: np.ndarray, actual: np.ndarray, title: str = '台风路径预测与实际路径地图比较'):
    # 计算经纬度范围
    all_lons = np.concatenate([predicted[:, 0], actual[:, 0]])
    all_lats = np.concatenate([predicted[:, 1], actual[:, 1]])

    min_lon, max_lon = all_lons.min() - 1, all_lons.max() + 1
    min_lat, max_lat = all_lats.min() - 1, all_lats.max() + 1

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 绘制预测路径
    plt.plot(predicted[:, 0], predicted[:, 1], marker='o', color='red', label='预测路径',
             transform=ccrs.PlateCarree())

    # 绘制实际路径
    plt.plot(actual[:, 0], actual[:, 1], marker='x', color='blue', label='实际路径', transform=ccrs.PlateCarree())

    # 标注起点和终点
    plt.scatter(predicted[0, 0], predicted[0, 1], color='green', s=100, label='预测起点',
                transform=ccrs.PlateCarree())
    plt.scatter(actual[0, 0], actual[0, 1], color='purple', s=100, label='实际起点', transform=ccrs.PlateCarree())
    plt.scatter(predicted[-1, 0], predicted[-1, 1], color='orange', s=100, label='预测终点',
                transform=ccrs.PlateCarree())
    plt.scatter(actual[-1, 0], actual[-1, 1], color='cyan', s=100, label='实际终点', transform=ccrs.PlateCarree())

    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()

# 12. 预测结果展示与CSV保存

def save_predictions(prediction_times: list, prediction_longitudes: list,
                    prediction_latitudes: list, filename: str = 'predicted_path.csv'):
    prediction_times = pd.to_datetime(prediction_times)

    prediction_df = pd.DataFrame({
        '时间': prediction_times.strftime('%Y-%m-%d %H:%M'),
        '预测经度': prediction_longitudes,
        '预测纬度': prediction_latitudes
    })

    logging.info("预测结果：")
    logging.info(prediction_df.head())  # 只打印前几行以确认格式

    # 将预测结果保存到CSV文件
    prediction_df.to_csv(filename, index=False, encoding='utf-8-sig')
    logging.info(f"预测结果已保存到 '{filename}'")

# 13. 总结

def summarize_performance(distance: float, threshold: float = 1000):
    if distance < threshold:
        logging.info("模型预测精度较高。")
    else:
        logging.info("模型预测精度较低，需要进一步优化。")

# 14. 绘制训练与验证曲线

def plot_training_history(history):
    # 绘制训练与验证损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失 (MSE)')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制训练与验证 MAE 曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['mae'], label='训练 MAE')
    plt.plot(history.history['val_mae'], label='验证 MAE')
    plt.title('训练与验证 MAE 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid()
    plt.show()

# 15. 主函数

def main():
    # 文件路径（请根据实际情况修改）
    historical_data_path =  '1945-2023.csv'
    current_year_data_path = 'typhoon_2401_2421_2024_data.csv'  # 请确认文件名

    # 加载数据
    historical_data, current_year_data = load_data(historical_data_path, current_year_data_path)

    # 预处理数据
    historical_data_clean = preprocess_historical_data(historical_data)
    current_year_data_clean = preprocess_current_year_data(current_year_data)

    # 特征工程
    historical_data_feat = feature_engineering(historical_data_clean)
    current_year_data_feat = feature_engineering(current_year_data_clean)

    # 选择特定台风数据
    tc_num = 2413
    name_cn = "贝碧嘉"
    typhoon_beijia = select_typhoon_data(current_year_data_feat, tc_num, name_cn)

    # 确保数据按3小时间隔采样
    typhoon_beijia = typhoon_beijia.set_index('date').resample('3H').ffill().reset_index()

    # 选择用于训练的数据（历史数据）
    # 确保历史数据中包含台风编号以便分组
    if '台风编号' not in historical_data_feat.columns:
        logging.error("历史数据中缺少'台风编号'列，无法按台风分组。")
        sys.exit(1)

    # 选择特征和目标
    features = ['经度', '纬度', '风速', '气压', '移动方向', '移动速度', '路径惯性', '气压梯度']
    target = ['经度', '纬度']  # 确保顺序为 [经度, 纬度]

    # 构建时序数据集
    X, y = create_sequences(historical_data_feat, features, target, seq_length=SEQ_LENGTH)

    # 检查X和y的形状
    logging.info(f"创建的序列数量: {X.shape[0]}")
    logging.info(f"每个序列的形状: {X.shape[1:]}")

    # 划分训练集、验证集和测试集（时间序列分割）
    split1 = int(len(X) * 0.7)
    split2 = int(len(X) * 0.85)
    X_train, X_val, X_test = X[:split1], X[split1:split2], X[split2:]
    y_train, y_val, y_test = y[:split1], y[split1:split2], y[split2:]

    logging.info(f"训练集大小: {X_train.shape}")
    logging.info(f"验证集大小: {X_val.shape}")
    logging.info(f"测试集大小: {X_test.shape}")

    # 初始化缩放器
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # 对训练集进行拟合和转换
    X_train_reshaped = X_train.reshape(-1, len(features))
    feature_scaler.fit(X_train_reshaped)
    X_train_scaled = feature_scaler.transform(X_train_reshaped).reshape(X_train.shape)

    # 对验证集和测试集进行转换
    X_val_scaled = feature_scaler.transform(X_val.reshape(-1, len(features))).reshape(X_val.shape)
    X_test_scaled = feature_scaler.transform(X_test.reshape(-1, len(features))).reshape(X_test.shape)

    # 对目标变量进行拟合和转换（仅在训练集上）
    target_scaler.fit(y_train)
    y_train_scaled = target_scaler.transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    # 构建模型
    model = build_model(input_shape=(SEQ_LENGTH, len(features)), output_dim=len(target))

    # 打印模型摘要
    model.summary()

    # 训练模型
    logging.info("开始训练模型...")
    history = train_model(model, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
    logging.info("模型训练完成。")

    # 评估模型
    y_test_inv, y_pred_inv = evaluate_model(model, X_test_scaled, y_test_scaled, target_scaler)

    # 预测台风路径
    predictions_original, prediction_times, prediction_longitudes, prediction_latitudes, actual_positions = predict_typhoon_path(
        model, typhoon_beijia, feature_scaler, target_scaler, features, seq_length=SEQ_LENGTH)

    # 动态时间规整（DTW）比较
    distance = compute_dtw(predictions_original, actual_positions)

    # 可视化预测路径与实际路径
    plot_paths(predictions_original, actual_positions, title='台风贝碧嘉预测路径与实际路径比较')

    # 动态时间规整（DTW）路径对齐可视化
    aligned_pred, aligned_actual = compute_dtw_path(predictions_original, actual_positions)
    plot_aligned_paths(aligned_pred, aligned_actual, title='DTW对齐后的预测路径与实际路径')

    # 在地图上进行路径比较
    plot_paths_on_map(predictions_original, actual_positions, title='台风贝碧嘉预测路径与实际路径地图比较')

    # 预测结果展示与CSV保存
    save_predictions(prediction_times, prediction_longitudes, prediction_latitudes, filename='predicted_path_beijia.csv')

    # 总结
    summarize_performance(distance, threshold=1000)

    # 绘制训练与验证损失曲线
    plot_training_history(history)

if __name__ == "__main__":
    main()
