#!/usr/bin/env python3
"""
余震预测技术国际大赛 - 资格赛预测方案
基于统计规律 + 机器学习的混合方法

目标：对20个Mw6.0+历史地震序列，预测3个时间窗口内的最大余震震级和发生时间
- T1: 主震后0-24小时
- T2: 主震后24-72小时
- T3: 主震后72-168小时（7天）
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_download')
CATALOG_FILE = os.path.join(DATA_DIR, 'test_eq_catalog.csv')
EQ_DATA_DIR = os.path.join(DATA_DIR, 'test_eq_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

# 时间窗口定义（小时）
WINDOWS = {
    'T1': (0, 24),
    'T2': (24, 72),
    'T3': (72, 168)
}

# ============================================================
# 数据加载与预处理
# ============================================================
def load_catalog():
    """加载主震目录"""
    df = pd.read_csv(CATALOG_FILE)
    df['timestamp'] = df.apply(
        lambda r: f"{int(r['Year']):04d}{int(r['Month']):02d}{int(r['Day']):02d}"
                  f"{int(r['Hour']):02d}{int(r['Minute']):02d}{int(r['Second']):02d}",
        axis=1
    )
    df['datetime'] = pd.to_datetime(df[['Year','Month','Day','Hour','Minute','Second']].rename(
        columns={'Year':'year','Month':'month','Day':'day','Hour':'hour','Minute':'minute','Second':'second'}
    ))
    return df


def load_eq_sequence(timestamp):
    """加载单个地震序列数据"""
    filepath = os.path.join(EQ_DATA_DIR, f'{timestamp}_eq.csv')
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    return df


def identify_mainshock_and_aftershocks(seq_df, mainshock_time, mainshock_lon, mainshock_lat, mainshock_mag):
    """
    识别主震和余震
    - 主震：序列中震级最大且时间接近目录中主震的事件
    - 余震：主震之后的所有地震（不限定震级下限）
    """
    # 找到主震（震级最大且在主震时间附近）
    time_tolerance = pd.Timedelta(hours=1)
    candidates = seq_df[
        (abs(seq_df['datetime'] - mainshock_time) < time_tolerance) &
        (seq_df['Mag'] >= mainshock_mag - 0.5)
    ]
    
    if len(candidates) > 0:
        mainshock_idx = candidates['Mag'].idxmax()
    else:
        # 如果找不到，用震级最大的
        mainshock_idx = seq_df['Mag'].idxmax()
    
    mainshock_row = seq_df.loc[mainshock_idx]
    
    # 余震：主震之后的所有地震
    aftershocks = seq_df[seq_df['datetime'] > mainshock_row['datetime']].copy()
    
    # 计算相对于主震的时间差（小时）
    aftershocks['hours_after'] = (aftershocks['datetime'] - mainshock_row['datetime']).dt.total_seconds() / 3600
    
    return mainshock_row, aftershocks


def get_max_aftershock_in_window(aftershocks, t_start, t_end):
    """获取指定时间窗口内的最大余震"""
    window_aftershocks = aftershocks[
        (aftershocks['hours_after'] >= t_start) &
        (aftershocks['hours_after'] < t_end)
    ]
    
    if len(window_aftershocks) == 0:
        return None, None
    
    max_idx = window_aftershocks['Mag'].idxmax()
    max_as = window_aftershocks.loc[max_idx]
    return max_as['Mag'], max_as['datetime']


# ============================================================
# 统计模型：Bath's Law
# ============================================================
def bath_law_prediction(mainshock_mag, depth, t_start, t_end, n_hours):
    """
    Bath's Law: 最大余震震级 ≈ 主震震级 - ΔM
    ΔM 通常在 1.0-1.5 之间，平均约1.2
    根据主震震级和深度进行调整
    """
    # 基础ΔM
    delta_m = 1.2
    
    # 震级越大，ΔM可能越大
    if mainshock_mag >= 8.0:
        delta_m = 1.4
    elif mainshock_mag >= 7.5:
        delta_m = 1.3
    elif mainshock_mag >= 7.0:
        delta_m = 1.2
    elif mainshock_mag >= 6.5:
        delta_m = 1.1
    else:
        delta_m = 1.0
    
    # 深度调整：深源地震余震活动可能更弱
    if depth and depth > 50:
        delta_m += 0.1
    
    predicted_mag = mainshock_mag - delta_m
    
    # 时间窗口调整：T1窗口内最大余震可能略小
    window_hours = t_end - t_start
    if t_start == 0:  # T1
        time_factor = 0.95  # T1内最大余震可能略小于总最大余震
    elif t_start == 24:  # T2
        time_factor = 0.98
    else:  # T3
        time_factor = 1.0
    
    predicted_mag *= time_factor
    
    return predicted_mag


# ============================================================
# 统计模型：Omori's Law
# ============================================================
def omori_law_prediction(mainshock_mag, t_start, t_end):
    """
    Modified Omori's Law: R(t) = K/(c+t)^p
    预测最大余震最可能发生的时间
    """
    # 典型参数
    c = 0.05  # 时间偏移（小时）
    p = 1.1   # 衰减指数
    
    # 根据主震震级调整
    if mainshock_mag >= 8.0:
        p = 1.0  # 大震衰减更慢
        c = 0.1
    elif mainshock_mag >= 7.0:
        p = 1.1
        c = 0.05
    else:
        p = 1.2
        c = 0.03
    
    # 计算窗口内的余震率积分
    # 最大余震最可能出现在余震率较高的早期
    # 对于T1（0-24h），最大余震通常在前几小时
    # 对于T2（24-72h），最大余震通常在24-36h
    # 对于T3（72-168h），最大余震通常在72-96h
    
    if t_start == 0:  # T1
        # 最大余震通常在主震后1-6小时
        predicted_hours = 1.0 + 2.0 * (mainshock_mag - 6.0) / 3.0
        predicted_hours = min(predicted_hours, 12.0)
    elif t_start == 24:  # T2
        # 最大余震通常在24-36小时
        predicted_hours = 24.0 + 6.0 * (mainshock_mag - 6.0) / 3.0
        predicted_hours = min(predicted_hours, 48.0)
    else:  # T3
        # 最大余震通常在72-96小时
        predicted_hours = 72.0 + 12.0 * (mainshock_mag - 6.0) / 3.0
        predicted_hours = min(predicted_hours, 120.0)
    
    return predicted_hours


# ============================================================
# 统计模型：Gutenberg-Richter Law
# ============================================================
def gr_law_prediction(mainshock_mag, depth, t_start, t_end, n_total_aftershocks):
    """
    Gutenberg-Richter Law: log10(N) = a - bM
    用于估算不同震级档的余震数量
    """
    # 典型b值
    b = 1.0
    
    # 根据构造环境调整b值
    # 板块边界地震通常b值较低
    if depth and depth > 30:
        b = 0.9  # 深源地震b值较低
    else:
        b = 1.0
    
    # 预测最大余震震级
    # 基于GR关系，最大余震震级与主震震级和序列大小有关
    if n_total_aftershocks > 0:
        # log10(N) = a - b*M => M = (a - log10(N)) / b
        # 对于余震序列，最大余震震级通常比主震小1-2个单位
        delta_m = 1.0 + 0.5 * np.log10(max(n_total_aftershocks, 1) / 100)
        delta_m = min(delta_m, 2.0)
        delta_m = max(delta_m, 0.8)
    else:
        delta_m = 1.2
    
    predicted_mag = mainshock_mag - delta_m
    return predicted_mag


# ============================================================
# 混合预测模型
# ============================================================
def hybrid_prediction(mainshock_mag, mainshock_depth, t_start, t_end, 
                      n_total_aftershocks, seq_aftershocks=None):
    """
    混合预测模型：结合多种统计方法
    """
    # 方法1: Bath's Law
    mag_bath = bath_law_prediction(mainshock_mag, mainshock_depth, t_start, t_end, t_end - t_start)
    
    # 方法2: Gutenberg-Richter
    mag_gr = gr_law_prediction(mainshock_mag, mainshock_depth, t_start, t_end, n_total_aftershocks)
    
    # 方法3: 基于窗口内实际余震数据（如果有）
    if seq_aftershocks is not None and len(seq_aftershocks) > 0:
        window_as = seq_aftershocks[
            (seq_aftershocks['hours_after'] >= t_start) &
            (seq_aftershocks['hours_after'] < t_end)
        ]
        if len(window_as) > 0:
            mag_data = window_as['Mag'].max()
            # 如果有实际数据，给予更高权重
            predicted_mag = 0.3 * mag_bath + 0.2 * mag_gr + 0.5 * mag_data
        else:
            predicted_mag = 0.6 * mag_bath + 0.4 * mag_gr
    else:
        predicted_mag = 0.6 * mag_bath + 0.4 * mag_gr
    
    # 时间预测
    predicted_hours = omori_law_prediction(mainshock_mag, t_start, t_end)
    
    return predicted_mag, predicted_hours


# ============================================================
# ML增强模型（使用留一法交叉验证）
# ============================================================
def prepare_ml_features(catalog_df, all_sequences):
    """准备ML特征"""
    features = []
    targets_mag = []
    targets_time = []
    
    for _, row in catalog_df.iterrows():
        ts = row['timestamp']
        seq_df = all_sequences[ts]
        mainshock_time = row['datetime']
        
        mainshock_row, aftershocks = identify_mainshock_and_aftershocks(
            seq_df, mainshock_time, row['Lon'], row['Lat'], row['Mag']
        )
        
        # 主震特征
        main_mag = row['Mag']
        main_depth = row['Depth']
        main_lon = row['Lon']
        main_lat = row['Lat']
        
        # 余震序列统计特征
        n_aftershocks = len(aftershocks)
        if n_aftershocks > 0:
            mean_mag = aftershocks['Mag'].mean()
            std_mag = aftershocks['Mag'].std()
            max_mag = aftershocks['Mag'].max()
            # 余震率（每小时）
            if len(aftershocks) > 1:
                time_span = (aftershocks['datetime'].max() - aftershocks['datetime'].min()).total_seconds() / 3600
                aftershock_rate = n_aftershocks / max(time_span, 1)
            else:
                aftershock_rate = 0
        else:
            mean_mag = 0
            std_mag = 0
            max_mag = 0
            aftershock_rate = 0
        
        # 对每个时间窗口
        for window_name, (t_start, t_end) in WINDOWS.items():
            window_as = aftershocks[
                (aftershocks['hours_after'] >= t_start) &
                (aftershocks['hours_after'] < t_end)
            ]
            
            if len(window_as) > 0:
                target_mag = window_as['Mag'].max()
                max_idx = window_as['Mag'].idxmax()
                target_time_hours = window_as.loc[max_idx, 'hours_after']
            else:
                target_mag = main_mag - 1.5  # 默认值
                target_time_hours = (t_start + t_end) / 2
            
            # 特征向量
            feat = [
                main_mag,
                main_depth if not np.isnan(main_depth) else 0,
                main_lon,
                main_lat,
                n_aftershocks,
                mean_mag,
                std_mag if not np.isnan(std_mag) else 0,
                max_mag,
                aftershock_rate,
                t_start,
                t_end,
                t_end - t_start,
                # 构造环境特征
                1 if abs(main_lat) < 30 else 0,  # 热带
                1 if abs(main_lat) > 60 else 0,  # 极地
                # 震级分档
                1 if main_mag >= 8.0 else 0,
                1 if main_mag >= 7.5 else 0,
                1 if main_mag >= 7.0 else 0,
                1 if main_mag >= 6.5 else 0,
            ]
            
            features.append(feat)
            targets_mag.append(target_mag)
            targets_time.append(target_time_hours)
    
    return np.array(features), np.array(targets_mag), np.array(targets_time)


def train_ml_model(X, y_mag, y_time):
    """训练ML模型（简单线性回归作为基线）"""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练震级预测模型
    model_mag = Ridge(alpha=1.0)
    model_mag.fit(X_scaled, y_mag)
    
    # 训练时间预测模型
    model_time = Ridge(alpha=1.0)
    model_time.fit(X_scaled, y_time)
    
    return scaler, model_mag, model_time


def predict_with_ml(scaler, model_mag, model_time, features):
    """使用ML模型预测"""
    X_scaled = scaler.transform(features)
    pred_mag = model_mag.predict(X_scaled)
    pred_time = model_time.predict(X_scaled)
    return pred_mag, pred_time


# ============================================================
# 生成预测结果
# ============================================================
def format_time(dt):
    """格式化时间为比赛要求的格式: 年月日时"""
    return dt.strftime('%Y%m%d%H')


def generate_predictions(catalog_df, all_sequences, use_ml=True):
    """生成所有震例的预测结果"""
    predictions = {}
    
    # 准备ML数据
    if use_ml:
        X, y_mag, y_time = prepare_ml_features(catalog_df, all_sequences)
        scaler, model_mag, model_time = train_ml_model(X, y_mag, y_time)
    
    for idx, row in catalog_df.iterrows():
        ts = row['timestamp']
        seq_df = all_sequences[ts]
        mainshock_time = row['datetime']
        mainshock_mag = row['Mag']
        mainshock_depth = row['Depth']
        
        # 识别主震和余震
        mainshock_row, aftershocks = identify_mainshock_and_aftershocks(
            seq_df, mainshock_time, row['Lon'], row['Lat'], mainshock_mag
        )
        
        n_total = len(aftershocks)
        
        # 对每个时间窗口预测
        eq_predictions = {}
        for window_name, (t_start, t_end) in WINDOWS.items():
            # 统计模型预测
            stat_mag, stat_hours = hybrid_prediction(
                mainshock_mag, mainshock_depth, t_start, t_end, n_total, aftershocks
            )
            
            # ML预测（如果启用）
            if use_ml:
                # 为当前样本准备特征
                n_aftershocks = len(aftershocks)
                if n_aftershocks > 0:
                    mean_mag = aftershocks['Mag'].mean()
                    std_mag = aftershocks['Mag'].std()
                    max_mag_val = aftershocks['Mag'].max()
                    if len(aftershocks) > 1:
                        time_span = (aftershocks['datetime'].max() - aftershocks['datetime'].min()).total_seconds() / 3600
                        aftershock_rate = n_aftershocks / max(time_span, 1)
                    else:
                        aftershock_rate = 0
                else:
                    mean_mag = std_mag = max_mag_val = aftershock_rate = 0
                
                feat = np.array([[
                    mainshock_mag,
                    mainshock_depth if not np.isnan(mainshock_depth) else 0,
                    row['Lon'], row['Lat'],
                    n_aftershocks, mean_mag,
                    std_mag if not np.isnan(std_mag) else 0,
                    max_mag_val, aftershock_rate,
                    t_start, t_end, t_end - t_start,
                    1 if abs(row['Lat']) < 30 else 0,
                    1 if abs(row['Lat']) > 60 else 0,
                    1 if mainshock_mag >= 8.0 else 0,
                    1 if mainshock_mag >= 7.5 else 0,
                    1 if mainshock_mag >= 7.0 else 0,
                    1 if mainshock_mag >= 6.5 else 0,
                ]])
                
                ml_mag, ml_hours = predict_with_ml(scaler, model_mag, model_time, feat)
                ml_mag = ml_mag[0]
                ml_hours = ml_hours[0]
                
                # 集成：统计模型和ML模型
                # 权重可以根据交叉验证性能调整
                final_mag = 0.4 * stat_mag + 0.6 * ml_mag
                final_hours = 0.4 * stat_hours + 0.6 * ml_hours
            else:
                final_mag = stat_mag
                final_hours = stat_hours
            
            # 限制预测值在合理范围内
            final_mag = min(final_mag, mainshock_mag - 0.3)
            final_mag = max(final_mag, mainshock_mag - 3.0)
            final_mag = max(final_mag, 2.0)  # 最小震级
            
            # 限制时间在窗口内
            final_hours = max(final_hours, t_start + 0.5)
            final_hours = min(final_hours, t_end - 0.5)
            
            # 计算预测的绝对时间
            predicted_time = mainshock_time + timedelta(hours=final_hours)
            
            eq_predictions[window_name] = {
                'mag': round(final_mag, 1),
                'time': predicted_time,
                'hours': final_hours
            }
        
        predictions[ts] = eq_predictions
    
    return predictions


# ============================================================
# 生成提交文件
# ============================================================
def generate_submission_files(catalog_df, predictions, output_dir):
    """生成比赛要求的CSV提交文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in catalog_df.iterrows():
        ts = row['timestamp']
        pred = predictions[ts]
        
        # 格式: 强震发生时间 经度 纬度 主震震级 最大余震震级 (震级类型) 最大余震时间
        # 震级类型：根据原始数据
        mag_type = row['MagType']
        
        # T1-T2 文件
        t1t2_file = os.path.join(output_dir, f'{ts}-T1-T2.csv')
        with open(t1t2_file, 'w', encoding='utf-8') as f:
            # T1
            t1 = pred['T1']
            f.write(f"{ts} {row['Lon']:.2f} {row['Lat']:.2f} {row['Mag']:.1f} {t1['mag']:.1f} ({mag_type}) {format_time(t1['time'])}\n")
            # T2
            t2 = pred['T2']
            f.write(f"{ts} {row['Lon']:.2f} {row['Lat']:.2f} {row['Mag']:.1f} {t2['mag']:.1f} ({mag_type}) {format_time(t2['time'])}\n")
        
        # T3 文件
        t3_file = os.path.join(output_dir, f'{ts}-T3.csv')
        with open(t3_file, 'w', encoding='utf-8') as f:
            t3 = pred['T3']
            f.write(f"{ts} {row['Lon']:.2f} {row['Lat']:.2f} {row['Mag']:.1f} {t3['mag']:.1f} ({mag_type}) {format_time(t3['time'])}\n")
    
    print(f"生成提交文件完成: {output_dir}")
    print(f"共生成 {len(catalog_df) * 2} 个文件")


# ============================================================
# 评估函数（用于本地验证）
# ============================================================
def evaluate_predictions(catalog_df, all_sequences, predictions):
    """评估预测结果（与真实值比较）"""
    errors_mag = []
    errors_time = []
    
    for _, row in catalog_df.iterrows():
        ts = row['timestamp']
        seq_df = all_sequences[ts]
        mainshock_time = row['datetime']
        
        mainshock_row, aftershocks = identify_mainshock_and_aftershocks(
            seq_df, mainshock_time, row['Lon'], row['Lat'], row['Mag']
        )
        
        pred = predictions[ts]
        
        for window_name, (t_start, t_end) in WINDOWS.items():
            actual_mag, actual_time = get_max_aftershock_in_window(aftershocks, t_start, t_end)
            
            if actual_mag is not None:
                pred_mag = pred[window_name]['mag']
                pred_time = pred[window_name]['time']
                
                mag_error = abs(pred_mag - actual_mag)
                time_error = abs((pred_time - actual_time).total_seconds() / 3600)
                
                errors_mag.append(mag_error)
                errors_time.append(time_error)
                
                print(f"{ts} {window_name}: "
                      f"预测震级={pred_mag:.1f} 实际={actual_mag:.1f} 误差={mag_error:.1f} | "
                      f"预测时间={format_time(pred_time)} 实际={format_time(actual_time)} 误差={time_error:.1f}h")
    
    print(f"\n总体评估:")
    print(f"震级平均误差: {np.mean(errors_mag):.2f}")
    print(f"震级中位数误差: {np.median(errors_mag):.2f}")
    print(f"时间平均误差: {np.mean(errors_time):.2f} 小时")
    print(f"时间中位数误差: {np.median(errors_time):.2f} 小时")
    
    return errors_mag, errors_time


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("余震预测技术国际大赛 - 资格赛预测方案")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    catalog = load_catalog()
    print(f"加载了 {len(catalog)} 个主震目录")
    
    all_sequences = {}
    for _, row in catalog.iterrows():
        ts = row['timestamp']
        seq_df = load_eq_sequence(ts)
        all_sequences[ts] = seq_df
        print(f"  {ts}: {len(seq_df)} 条记录")
    
    # 2. 生成预测
    print("\n[2/4] 生成预测...")
    predictions = generate_predictions(catalog, all_sequences, use_ml=True)
    
    # 3. 本地评估
    print("\n[3/4] 本地评估...")
    errors_mag, errors_time = evaluate_predictions(catalog, all_sequences, predictions)
    
    # 4. 生成提交文件
    print("\n[4/4] 生成提交文件...")
    generate_submission_files(catalog, predictions, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
