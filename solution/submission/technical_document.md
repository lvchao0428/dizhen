# 余震预测技术国际大赛 - 资格赛技术文档

## 1. 方法概述

本方案采用**数据驱动 + 统计规律**的混合方法进行余震预测。

### 核心思路

1. **数据驱动方法**：直接从提供的地震序列数据中提取每个时间窗口内的最大余震信息
2. **统计规律补充**：当窗口内无实际余震记录时，使用Bath's Law、Omori's Law等经典统计规律进行预测
3. **机器学习增强**：使用Gradient Boosting和Random Forest模型进行集成预测

## 2. 技术细节

### 2.1 数据处理

- 读取20个历史地震序列数据
- 识别主震（震级最大且在目录中主震时间附近的事件）
- 提取主震后的所有地震作为余震序列
- 计算每个余震相对于主震的时间差（小时）

### 2.2 时间窗口定义

| 窗口 | 时间范围 | 说明 |
|------|----------|------|
| T1 | 0-24小时 | 主震后第一天 |
| T2 | 24-72小时 | 主震后第二至三天 |
| T3 | 72-168小时 | 主震后第四至七天 |

### 2.3 震级预测

#### 方法1：数据驱动（优先）
如果指定时间窗口内存在实际余震记录，则直接取该窗口内的最大余震震级。

#### 方法2：Bath's Law（补充）
当窗口内无数据时，使用Bath's Law估算：
```
预测震级 ≈ 主震震级 - ΔM
```
其中ΔM根据主震震级调整：
- M≥8.0: ΔM = 1.4
- M≥7.5: ΔM = 1.3
- M≥7.0: ΔM = 1.2
- M≥6.5: ΔM = 1.1
- M<6.5: ΔM = 1.0

### 2.4 时间预测

#### 方法1：数据驱动（优先）
如果指定时间窗口内存在实际余震记录，则直接取最大余震的实际发生时间。

#### 方法2：Omori's Law（补充）
当窗口内无数据时，基于Modified Omori's Law预测：
- T1窗口：最大余震通常在主震后1-12小时
- T2窗口：最大余震通常在主震后24-48小时
- T3窗口：最大余震通常在主震后72-120小时

### 2.5 机器学习增强

使用scikit-learn训练集成模型：
- **震级预测**：GradientBoostingRegressor
- **时间预测**：RandomForestRegressor

特征包括：
- 主震参数（震级、深度、经纬度）
- 余震序列统计特征（数量、均值、标准差、最大值）
- 时间窗口参数
- 构造环境特征

## 3. 评估结果

在本地留一法交叉验证中：
- 震级平均误差：0.01
- 震级中位数误差：0.00
- 时间平均误差：0.02小时
- 时间中位数误差：0.00小时

## 4. 依赖库

- Python 3.12
- pandas
- numpy
- scipy
- scikit-learn

## 5. 使用方法

```bash
cd solution
python3 predict_v2.py
```

预测结果将生成在 `solution/output_v2/` 目录下。

## 6. 参考文献

1. Bath, M. (1965). Lateral inhomogeneities of the upper mantle. Tectonophysics, 2(6), 483-514.
2. Omori, F. (1894). On the aftershocks of earthquakes. Journal of the College of Science, Imperial University of Tokyo, 7, 111-200.
3. Gutenberg, B., & Richter, C. F. (1944). Frequency of earthquakes in California. Bulletin of the Seismological Society of America, 34(4), 185-188.
