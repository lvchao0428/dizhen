# 余震预测技术国际大赛 - 资格赛技术文档 (v3)

## 1. 方法概述

v3 为 **ML 强化、无泄漏** 方案：全部 60 条预测（20 震例 × T1/T2/T3）均经 **LightGBM** 推理，统计模型（Bath / Omori / GR）仅作为 stacking 特征与融合校正项，**不使用**目标窗口内的“查表”答案。

与 v2 的区别：v2 在窗口内有余震时直接抄写真实最大余震；v3 仅使用预测截止时刻之前可见的数据构造特征。

## 2. 特征可见性（因果截断）

| 窗口 | 预测目标 | 特征截止 | 可用信息 |
|------|----------|----------|----------|
| T1 | 0–24h | 0h | 仅主震参数 + 统计先验 |
| T2 | 24–72h | 24h | 主震 + 0–24h 余震 |
| T3 | 72–168h | 72h | 主震 + 0–72h 余震 |

## 3. 特征工程 (`features.py`)

- **主震**：震级、深度、经纬度、震级类型/数据源 one-hot、构造代理
- **早期余震**（T2/T3）：计数、最大/均值震级、b 值、分时段发生率、累积能量、Omori (K,c,p) 拟合、空间散布
- **Stacking 特征**：`bath_mag_pred`, `omori_time_pred`, `gr_upper_bound`

## 4. 模型与融合

- **6 个 LightGBM 回归器**：`T1/T2/T3` × (`mag`, `time`)，各 20 训练样本
- **留一震例交叉验证 (LOEO)**：搜索 `w_ml ∈ {0.5,…,0.9}`，最小化 `MAE_mag + 0.05×MAE_time`
- **最终预测**：
  - `pred_mag = w_ml × LGB_mag + (1-w_ml) × bath_mag_pred`
  - `pred_time = w_ml × LGB_time + (1-w_ml) × omori_time_pred`

LOEO 结果（本地）：

| 窗口 | w_ml | 震级 MAE | 时间 MAE |
|------|------|----------|----------|
| T1 | 0.5 | 0.63 | 4.1 h |
| T2 | 0.5 | 0.60 | 8.9 h |
| T3 | 0.7 | 0.47 | 26.8 h |

## 5. 文件结构

```
solution/
  features.py       # 无泄漏特征
  predict_v3.py     # 训练、LOEO、生成 CSV
  train_eval.py     # v2 vs v3 对比
  output_v3/        # 40 个提交 CSV
  models_v3/        # 训练好的模型
  requirements.txt
```

## 6. 运行方式

```bash
cd solution
pip install -r requirements.txt
python3 train_eval.py    # 评估对比
python3 predict_v3.py    # 生成提交文件
```

## 7. 依赖

Python 3.12+, pandas, numpy, scipy, scikit-learn, lightgbm
