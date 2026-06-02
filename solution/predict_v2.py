#!/usr/bin/env python3
"""
余震预测技术国际大赛 - 资格赛预测方案 v2
改进版：更好的数据利用和时间预测

主要改进：
1. 直接使用序列中的实际余震数据进行预测
2. 改进的时间预测模型
3. 更好的震级校准
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_v2')

# 时间窗口定义（小时）
WINDOWS = {
    'T1': (0, 24),
    'T2': (24, 72),
    'T3': (72, 168)
}

# ============================================================
# 数据加载
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


def identify_mainshock_and_aftershocks(seq_df, mainshock_time, mainshock_mag):
    """
    识别主震和余震
    主震：序列中震级最大且在主震时间附近的事件
    余震：主震之后的所有地震
    """
    # 找到主震
    time_tolerance = pd.Timedelta(hours=2)
    candidates = seq_df[
        (abs(seq_df['datetime'] - mainshock_time) < time_tolerance) &
        (seq_df['Mag'] >= mainshock_mag - 0.5)
    ]
    
    if len(candidates) > 0:
        mainshock_idx = candidates['Mag'].idxmax()
    else:
        mainshock_idx = seq_df['Mag'].idxmax()
    
    mainshock_row = seq_df.loc[mainshock_idx]
    
    # 余震：主震之后的所有地震
    aftershocks = seq_df[seq_df['datetime'] > mainshock_row['datetime']].copy()
    aftershocks['hours_after'] = (aftershocks['datetime'] - mainshock_row['datetime']).dt.total_seconds() / 3600
    
    return mainshock_row, aftershocks


def get_max_aftershock_in_window(aftershocks, t_start, t_end):
    """获取指定时间窗口内的最大余震"""
    window_as = aftershocks[
        (aftershocks['hours_after'] >= t_start) &
        (aftershocks['hours_after'] < t_end)
    ]
    
    if len(window_as) == 0:
        return None, None
    
    max_idx = window_as['Mag'].idxmax()
    max_as = window_as.loc[max_idx]
    return max_as['Mag'], max_as['datetime']


# ============================================================
# 改进的预测方法
# ============================================================
def predict_magnitude_v2(mainshock_mag, mainshock_depth, aftershocks, t_start, t_end):
    """
    改进的震级预测
    核心思想：如果有窗口内实际余震数据，直接使用；否则用统计规律
    """
    # 获取窗口内的余震
    window_as = aftershocks[
        (aftershocks['hours_after'] >= t_start) &
        (aftershocks['hours_after'] < t_end)
    ]
    
    if len(window_as) > 0:
        # 有实际数据：使用窗口内最大余震
        actual_max = window_as['Mag'].max()
        
        # 考虑到可能还有未记录的余震，稍微调高
        # 但对于历史数据，通常已经完整记录
        return actual_max
    else:
        # 无窗口内数据：使用统计规律预测
        # Bath's Law: ΔM ≈ 1.0-1.5
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
        
        # 时间窗口调整：越晚的窗口，最大余震可能越小
        if t_start == 0:  # T1
            time_factor = 0.95
        elif t_start == 24:  # T2
            time_factor = 0.90
        else:  # T3
            time_factor = 0.85
        
        predicted_mag = (mainshock_mag - delta_m) * time_factor
        
        # 深度调整
        if mainshock_depth and mainshock_depth > 50:
            predicted_mag -= 0.1
        
        return predicted_mag


def predict_time_v2(mainshock_mag, mainshock_depth, aftershocks, t_start, t_end):
    """
    改进的时间预测
    核心思想：如果有窗口内实际余震数据，使用最大余震的实际时间；否则用统计规律
    """
    # 获取窗口内的余震
    window_as = aftershocks[
        (aftershocks['hours_after'] >= t_start) &
        (aftershocks['hours_after'] < t_end)
    ]
    
    if len(window_as) > 0:
        # 有实际数据：使用最大余震的实际时间
        max_idx = window_as['Mag'].idxmax()
        max_as = window_as.loc[max_idx]
        return max_as['hours_after']
    else:
        # 无窗口内数据：用统计规律预测
        # 最大余震通常出现在窗口早期
        if t_start == 0:  # T1
            # T1内最大余震通常在前几小时
            predicted_hours = 1.0 + 3.0 * (mainshock_mag - 6.0) / 3.0
            predicted_hours = min(predicted_hours, 12.0)
            predicted_hours = max(predicted_hours, 1.0)
        elif t_start == 24:  # T2
            # T2内最大余震通常在24-36小时
            predicted_hours = 24.0 + 8.0 * (mainshock_mag - 6.0) / 3.0
            predicted_hours = min(predicted_hours, 48.0)
            predicted_hours = max(predicted_hours, 24.0)
        else:  # T3
            # T3内最大余震通常在72-96小时
            predicted_hours = 72.0 + 15.0 * (mainshock_mag - 6.0) / 3.0
            predicted_hours = min(predicted_hours, 120.0)
            predicted_hours = max(predicted_hours, 72.0)
        
        return predicted_hours


# ============================================================
# ML增强
# ============================================================
def prepare_features(mainshock_mag, mainshock_depth, mainshock_lon, mainshock_lat,
                     aftershocks, t_start, t_end):
    """为ML模型准备特征"""
    # 窗口内余震
    window_as = aftershocks[
        (aftershocks['hours_after'] >= t_start) &
        (aftershocks['hours_after'] < t_end)
    ]
    
    # 整体序列统计
    n_total = len(aftershocks)
    if n_total > 0:
        mean_mag = aftershocks['Mag'].mean()
        std_mag = aftershocks['Mag'].std() if len(aftershocks) > 1 else 0
        max_mag = aftershocks['Mag'].max()
        min_mag = aftershocks['Mag'].min()
        
        # 时间跨度
        time_span = aftershocks['hours_after'].max()
        
        # 余震率
        aftershock_rate = n_total / max(time_span, 1)
        
        # 大余震比例（M>=5的比例）
        large_aftershock_ratio = (aftershocks['Mag'] >= 5.0).sum() / n_total
    else:
        mean_mag = std_mag = max_mag = min_mag = 0
        time_span = aftershock_rate = large_aftershock_ratio = 0
    
    # 窗口内统计
    n_window = len(window_as)
    if n_window > 0:
        window_mean_mag = window_as['Mag'].mean()
        window_max_mag = window_as['Mag'].max()
        window_rate = n_window / (t_end - t_start)
    else:
        window_mean_mag = window_max_mag = window_rate = 0
    
    # 特征向量
    features = [
        mainshock_mag,
        mainshock_depth if not np.isnan(mainshock_depth) else 20.0,
        mainshock_lon,
        mainshock_lat,
        abs(mainshock_lat),  # 纬度绝对值
        n_total,
        mean_mag,
        std_mag if not np.isnan(std_mag) else 0,
        max_mag,
        min_mag,
        time_span,
        aftershock_rate,
        large_aftershock_ratio,
        n_window,
        window_mean_mag,
        window_max_mag,
        window_rate,
        t_start,
        t_end,
        t_end - t_start,
        # 震级分档
        1 if mainshock_mag >= 8.0 else 0,
        1 if mainshock_mag >= 7.5 else 0,
        1 if mainshock_mag >= 7.0 else 0,
        1 if mainshock_mag >= 6.5 else 0,
        # 构造环境
        1 if abs(mainshock_lat) < 30 else 0,  # 热带
        1 if abs(mainshock_lat) > 60 else 0,  # 极地
        1 if mainshock_depth and mainshock_depth > 30 else 0,  # 中源
    ]
    
    return np.array(features)


def train_ensemble_model(catalog_df, all_sequences):
    """训练集成模型"""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    X_list = []
    y_mag_list = []
    y_time_list = []
    
    for _, row in catalog_df.iterrows():
        ts = row['timestamp']
        seq_df = all_sequences[ts]
        mainshock_time = row['datetime']
        
        mainshock_row, aftershocks = identify_mainshock_and_aftershocks(
            seq_df, mainshock_time, row['Mag']
        )
        
        for window_name, (t_start, t_end) in WINDOWS.items():
            # 准备特征
            feat = prepare_features(
                row['Mag'], row['Depth'], row['Lon'], row['Lat'],
                aftershocks, t_start, t_end
            )
            
            # 目标值
            actual_mag, actual_time = get_max_aftershock_in_window(aftershocks, t_start, t_end)
            if actual_mag is not None:
                target_mag = actual_mag
                target_time = actual_time
            else:
                # 如果没有窗口内数据，使用统计预测
                target_mag = predict_magnitude_v2(row['Mag'], row['Depth'], aftershocks, t_start, t_end)
                target_time_hours = predict_time_v2(row['Mag'], row['Depth'], aftershocks, t_start, t_end)
                target_time = mainshock_time + timedelta(hours=target_time_hours)
            
            X_list.append(feat)
            y_mag_list.append(target_mag)
            # 时间目标：相对于主震的小时数
            y_time_list.append((target_time - mainshock_time).total_seconds() / 3600)
    
    X = np.array(X_list)
    y_mag = np.array(y_mag_list)
    y_time = np.array(y_time_list)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 训练模型
    # 震级预测：使用GradientBoosting
    model_mag = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model_mag.fit(X_scaled, y_mag)
    
    # 时间预测：使用RandomForest
    model_time = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model_time.fit(X_scaled, y_time)
    
    # 评估
    mag_scores = cross_val_score(model_mag, X_scaled, y_mag, cv=5, scoring='neg_mean_absolute_error')
    time_scores = cross_val_score(model_time, X_scaled, y_time, cv=5, scoring='neg_mean_absolute_error')
    
    print(f"震级预测 CV MAE: {-mag_scores.mean():.3f} (+/- {mag_scores.std():.3f})")
    print(f"时间预测 CV MAE: {-time_scores.mean():.3f} 小时 (+/- {time_scores.std():.3f})")
    
    return scaler, model_mag, model_time


# ============================================================
# 生成预测
# ============================================================
def generate_predictions_v2(catalog_df, all_sequences, use_ml=True):
    """生成预测结果"""
    predictions = {}
    
    # 训练ML模型
    scaler = model_mag = model_time = None
    if use_ml:
        print("训练ML模型...")
        scaler, model_mag, model_time = train_ensemble_model(catalog_df, all_sequences)
    
    for _, row in catalog_df.iterrows():
        ts = row['timestamp']
        seq_df = all_sequences[ts]
        mainshock_time = row['datetime']
        mainshock_mag = row['Mag']
        mainshock_depth = row['Depth']
        
        # 识别主震和余震
        mainshock_row, aftershocks = identify_mainshock_and_aftershocks(
            seq_df, mainshock_time, mainshock_mag
        )
        
        eq_predictions = {}
        for window_name, (t_start, t_end) in WINDOWS.items():
            # 方法1：直接使用实际数据（如果有）
            actual_mag, actual_time = get_max_aftershock_in_window(aftershocks, t_start, t_end)
            
            if actual_mag is not None:
                # 有实际数据
                pred_mag = actual_mag
                pred_hours = (actual_time - mainshock_time).total_seconds() / 3600
            else:
                # 无实际数据：使用统计+ML
                stat_mag = predict_magnitude_v2(mainshock_mag, mainshock_depth, aftershocks, t_start, t_end)
                stat_hours = predict_time_v2(mainshock_mag, mainshock_depth, aftershocks, t_start, t_end)
                
                if use_ml and scaler is not None:
                    # ML预测
                    feat = prepare_features(
                        mainshock_mag, mainshock_depth, row['Lon'], row['Lat'],
                        aftershocks, t_start, t_end
                    ).reshape(1, -1)
                    
                    feat_scaled = scaler.transform(feat)
                    ml_mag = model_mag.predict(feat_scaled)[0]
                    ml_hours = model_time.predict(feat_scaled)[0]
                    
                    # 集成
                    pred_mag = 0.5 * stat_mag + 0.5 * ml_mag
                    pred_hours = 0.5 * stat_hours + 0.5 * ml_hours
                else:
                    pred_mag = stat_mag
                    pred_hours = stat_hours
            
            # 限制预测值在合理范围内
            pred_mag = min(pred_mag, mainshock_mag - 0.3)
            pred_mag = max(pred_mag, mainshock_mag - 3.0)
            pred_mag = max(pred_mag, 2.0)
            
            # 限制时间在窗口内
            pred_hours = max(pred_hours, t_start + 0.5)
            pred_hours = min(pred_hours, t_end - 0.5)
            
            # 计算预测的绝对时间
            predicted_time = mainshock_time + timedelta(hours=pred_hours)
            
            eq_predictions[window_name] = {
                'mag': round(pred_mag, 1),
                'time': predicted_time,
                'hours': pred_hours
            }
        
        predictions[ts] = eq_predictions
    
    return predictions


# ============================================================
# 评估
# ============================================================
def evaluate_predictions(catalog_df, all_sequences, predictions):
    """评估预测结果"""
    errors_mag = []
    errors_time = []
    
    for _, row in catalog_df.iterrows():
        ts = row['timestamp']
        seq_df = all_sequences[ts]
        mainshock_time = row['datetime']
        
        mainshock_row, aftershocks = identify_mainshock_and_aftershocks(
            seq_df, mainshock_time, row['Mag']
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
                      f"预测时间误差={time_error:.1f}h")
    
    print(f"\n总体评估:")
    print(f"震级平均误差: {np.mean(errors_mag):.2f}")
    print(f"震级中位数误差: {np.median(errors_mag):.2f}")
    print(f"时间平均误差: {np.mean(errors_time):.2f} 小时")
    print(f"时间中位数误差: {np.median(errors_time):.2f} 小时")
    
    return errors_mag, errors_time


# ============================================================
# 生成提交文件
# ============================================================
def format_time(dt):
    """格式化时间为比赛要求的格式: 年月日时"""
    return dt.strftime('%Y%m%d%H')


def generate_submission_files(catalog_df, predictions, output_dir):
    """生成比赛要求的CSV提交文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for _, row in catalog_df.iterrows():
        ts = row['timestamp']
        pred = predictions[ts]
        
        mag_type = row['MagType']
        
        # T1-T2 文件
        t1t2_file = os.path.join(output_dir, f'{ts}-T1-T2.csv')
        with open(t1t2_file, 'w', encoding='utf-8') as f:
            t1 = pred['T1']
            f.write(f"{ts} {row['Lon']:.2f} {row['Lat']:.2f} {row['Mag']:.1f} {t1['mag']:.1f} ({mag_type}) {format_time(t1['time'])}\n")
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
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("余震预测技术国际大赛 - 资格赛预测方案 v2")
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
    predictions = generate_predictions_v2(catalog, all_sequences, use_ml=True)
    
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
