# 排名赛监控 Pipeline

7×24 监测全球 M6+ 地震，微信（Server酱）通知；符合赛题条件（M≥6.0，不限深度）时自动拉 USGS 序列、跑 v3 模型、生成天池提交 CSV，并按时提醒 T1-T2 / T3 截止。

## 功能概览

| 能力 | 说明 |
|------|------|
| **实时轮询** | 默认每 **5 分钟** 查 USGS，发现新事件后入库并处理 |
| **微信推送** | M≥5.8 发监测通知；赛题有效事件另发「已出 CSV」及窗内对比 |
| **自动预测** | M≥6.0：序列 + v3 推理 → `monitor/ranking_output/{id}/` |
| **截止提醒** | 主震后约 20h / 48h / 68h 提醒 T1-T2、T3 准备与紧急提交 |
| **自动刷新** | 约 23h 刷新 T2 预测，约 71h 刷新 T3（重拉序列 + 重写 CSV） |
| **每日日报** | 每天 **09:00 北京时间** M6+ 汇总（可含未跑 pipeline 的补跑） |

### 赛题 vs 仅监测

| 条件 | 微信 | 序列 + 预测 + CSV |
|------|------|-------------------|
| M≥6.0（不限深度） | 是 | 是 |
| M≥5.8 且 <6 | 按配置（默认监测阈值 5.8） | 否 |

官方余震对比（排名赛实时阶段）：**USGS**；窗内无记录时推送会写明。

---

## 天池提交须知（排名赛）

天池「提交结果」页常见限制（以页面显示为准）：

| 项目 | 说明 |
|------|------|
| **剩余次数** | 例如「剩余提交次数 **3** 次」——全赛段/platform 计数，用一次少一次 |
| **计分规则** | 多次提交时通常 **以最后一次为准**（与资格赛说明一致） |
| **文件格式** | **ZIP**（内含 CSV 预测结果文件） |
| **大小上限** | 单次上传 **不超过 100MB**（本方案每个 CSV 仅数 KB，ZIP 远低于上限） |

### 每个震例要交什么

与资格赛相同，每个有效序列 **2 个 CSV**（排名赛仅 Mw/Ms）：

| 文件 | 行数 | 何时应已就绪 |
|------|------|----------------|
| `{事件ID}-T1-T2.csv` | 2 行（T1、T2） | 主震后 **24h 内** 提交到天池 |
| `{事件ID}-T3.csv` | 1 行（T3） | 主震后 **72h 内** 提交到天池 |

本地路径：`monitor/ranking_output/{事件ID}/`。监控会在截止前微信提醒，并在约 23h / 71h **自动刷新** CSV。

### 3 次机会怎么用（建议）

1. **不要**在 T1-T2、T3 尚未刷新、或未核对 `window_comparison.md` 时消耗提交次数。  
2. **第 1 次**：可先不上传，或仅在有把握时试交（熟悉天池上传框是否接受多文件 / 压缩包）。  
3. **第 2 次**：某月或某批事件 T2 已刷新后，汇总当前所有 `*-T1-T2.csv`、`*-T3.csv` 再交。  
4. **第 3 次（最后一次）**：留作 **最终定稿**——所有待评事件的 T3 均已刷新后再上传，此次结果计入排名。  

上传前在服务器执行：

```bash
python -m monitor.check_submission              # 检查 CSV 并自动生成提交 ZIP
python -m monitor.check_submission --event 20260601221236   # 只打包单个事件
python -m monitor.check_submission --no-zip     # 仅检查，不生成 ZIP
python -m monitor.package_submission            # 直接打包（默认仅 CSV）
python -m monitor.package_submission --include-aux  # 含辅助文件（备份用）
```

提交页：[天池排名赛提交入口](https://tianchi.aliyun.com/competition/entrance/532460/submission)（`config.yaml` 中 `tianchi_url`）。

> **说明**：排名赛提交格式为 **ZIP**（内含各事件的 CSV 预测结果文件）。`check_submission` 默认自动生成可直接上传天池的 ZIP 到 `monitor/data/submissions/`。

---

## 日常监控服务

生产环境通过 **systemd** 常驻进程 `dizhen-monitor`，入口为 `python -m monitor.run_daemon`。

### 服务做什么

`run_daemon` 启动后：

1. 立即执行一轮 `run_pipeline`（检测 → 赛题流水线 → 截止检查）
2. 注册定时任务：
   - **monitor**：每 `poll_interval_minutes`（默认 5）分钟执行 `run_pipeline`
   - **digest**：每天 `digest_cron_hour:digest_cron_minute`（默认 09:00）`Asia/Shanghai` 发 M6+ 日报

单次 `run_pipeline` 流程简述：

```
USGS 检测新事件 → SQLite 入库
  → 赛题有效：拉序列 → predict_v3 → 写 CSV + window_comparison.md → 微信
  → 仅监测：只记库 + 微信（无 CSV）
  → check_deadlines：提醒 / 自动 refresh T2·T3
```

### 安装与启停（远程）

```bash
cd ~/project/dizhen

# 从 Windows 同步后先去 CRLF（否则 bash\r 报错）
sed -i 's/\r$//' deploy/install_monitor.sh deploy/package_submission.sh

cp monitor/.env.example monitor/.env      # 填入 SERVERCHAN_SENDKEY
cp monitor/config.example.yaml monitor/config.yaml   # 可按需改阈值

bash deploy/install_monitor.sh
```

安装脚本会：创建 `.venv`、安装依赖、若无模型则训练 `models.pkl`、试跑一轮 `run_once`、安装并 **enable + restart** `dizhen-monitor`。

**注意**：用 `bash deploy/install_monitor.sh`，不要用 `sh`；CRLF 未修前不要用 `./install_monitor.sh`。

```bash
# 状态与日志
sudo systemctl status dizhen-monitor
sudo systemctl restart dizhen-monitor
sudo systemctl stop dizhen-monitor
journalctl -u dizhen-monitor -f              # systemd 标准输出
tail -f monitor/data/logs/monitor.log        # 应用日志（与上面并存）
```

无 sudo 时安装脚本会提示手动前台运行：

```bash
.venv/bin/python -m monitor.run_daemon
```

单元文件模板：`deploy/dizhen-monitor.service`（`WorkingDirectory`、`User`、`ExecStart` 安装时按 `INSTALL_ROOT` 替换）。

### 主震后时间线（赛题有效事件）

| 主震后大约 | 动作 |
|------------|------|
| 0h | 新事件：拉序列、T1/T2/T3 初版 CSV、微信 |
| 20h | 微信：**T1-T2 提交紧急**（距 24h 窗结束） |
| 23h | **自动刷新 T2**（重预测并更新 CSV） |
| 48h | 微信：**T3 准备**；触发 T3 相关刷新准备 |
| 68h | 微信：**T3 提交紧急** |
| 71h | **自动刷新 T3** |
| 72h | 事件标记 `done`，不再活跃提醒 |

阈值可在 `monitor/config.yaml` 调整（`remind_*`、`refresh_*`）。

---

## 配置说明

| 文件 | 作用 |
|------|------|
| `monitor/.env` | `SERVERCHAN_SENDKEY`（**勿提交 git**） |
| `monitor/config.yaml` | 轮询间隔、震级阈值、序列半径、提醒时刻、路径等 |

常用配置项（完整见 `config.example.yaml`）：

```yaml
poll_interval_minutes: 5      # 轮询间隔
lookback_days: 7              # 检测回溯天数
notify_max_age_hours: 48      # 超过此时长的新震不再发「新事件」微信

min_magnitude_submit: 6.0     # 赛题提交震级（不限深度）
min_magnitude_watch: 5.8      # 监测通知震级

model_path: solution/models_v3/models.pkl
ranking_output_dir: monitor/ranking_output
data_dir: monitor/data
sequences_dir: monitor/data/sequences

digest_cron_hour: 9           # 日报（北京时间）
digest_timezone: Asia/Shanghai
```

天池提交页：`tianchi_url`（默认排名赛入口）。

---

## 操作说明

以下命令均在项目根目录、已 `source .venv/bin/activate` 前提下执行。

### 日常运维

| 场景 | 命令 |
|------|------|
| 手动跑一轮（等同定时器一次） | `python -m monitor.run_once` |
| 看服务是否在跑 | `systemctl is-active dizhen-monitor` |
| 改配置后重启 | 编辑 `config.yaml` → `sudo systemctl restart dizhen-monitor` |
| 改 SendKey | 编辑 `monitor/.env` → 重启服务 |

### 提交文件在哪里 / 上传前检查

每个事件一个目录（**上传天池用 CSV**）：

```
monitor/ranking_output/{事件ID}/
├── {事件ID}-T1-T2.csv       # T1、T2 两行
├── {事件ID}-T3.csv          # T3 一行
└── window_comparison.md     # 模型 vs USGS（本地参考，不必上传）
```

上传前检查（行数、是否超 100MB）：

```bash
python -m monitor.check_submission
python -m monitor.check_submission --event 20260531213418
```

打包 ZIP（便于下载或归档；ZIP 内可含对比 md，**上天池以 CSV 为准**）：

```bash
bash deploy/package_submission.sh                    # 全部事件
bash deploy/package_submission.sh 20260531213418     # 单个事件
# 或
python -m monitor.package_submission --event 20260531213418
```

ZIP 输出：`monitor/data/submissions/submission_{id或all}_{时间戳}.zip`

微信「已出 CSV」消息里也会带 `output_dir` 路径；数据库 `monitor/data/events.db` 表 `events.output_dir` 同路径。

### 单事件补跑 / 对比

流水线曾失败（库中 `pipeline_failed`）或需重算：

```bash
python -m monitor.run_compare 20260531213418 --predict
```

只拉序列、看 USGS 对比、不写预测：

```bash
python -m monitor.run_compare 20260531213418
```

### 日报

```bash
# 终端打印 + 写入 monitor/data/digest/（不推微信）
python -m monitor.run_digest --print-only --days 30

# 发微信日报（与 daemon 每日 9:00 相同逻辑）
python -m monitor.run_digest --notify --backfill
```

`--backfill`：日报里列出的赛题事件若尚未预测，会尝试补跑 pipeline。

### 模型

- 仓库默认**不含** `solution/models_v3/models.pkl`
- 训练（约 1–2 分钟）：`cd solution && python predict_v3.py`
- 安装脚本会在缺少 pkl 时自动训练
- 若日志出现 `StandardScaler` 版本警告，在服务器用当前环境重跑 `predict_v3.py` 即可

资格赛 v3：LOEO + LightGBM（`solution/inference_v3.py` 与 monitor 共用）。

---

## 目录与数据

| 路径 | 说明 |
|------|------|
| `monitor/ranking_output/{id}/` | 提交 CSV、`window_comparison.md` |
| `monitor/data/sequences/{id}_eq.csv` | USGS 余震序列 |
| `monitor/data/submissions/` | 打包 ZIP |
| `monitor/data/events.db` | 事件状态、提醒标记、`output_dir` |
| `monitor/data/logs/monitor.log` | 应用日志 |
| `monitor/data/digest/` | 日报 Markdown 存档 |

---

## 故障排查

| 现象 | 处理 |
|------|------|
| `bash\r: No such file` | `sed -i 's/\r$//' deploy/*.sh` 后用 `bash` 执行 |
| `Cannot subtract tz-naive and tz-aware` | 拉最新代码（`monitor/time_utils.py` 等），重启服务后对事件 `--predict` 补跑 |
| `pipeline_failed` 无 CSV | `python -m monitor.run_compare {id} --predict` |
| 未配置 SendKey | 日志 `未配置 SERVERCHAN_SENDKEY`；补 `monitor/.env` 并重启 |
| 模型不存在 | `cd solution && python predict_v3.py` |
| sklearn 版本警告 | 同上，重训 `models.pkl` |

---

## 模块索引

| 模块 | 职责 |
|------|------|
| `run_daemon.py` | systemd 入口：轮询 + 日报 |
| `run_once.py` | 单次 pipeline |
| `pipeline.py` | 检测、赛题流水线、刷新、微信 |
| `detector.py` / `sources/usgs.py` | USGS 事件与序列 |
| `live_predict.py` | 加载模型、写 CSV |
| `deadlines.py` | 提醒与 T2/T3 自动刷新 |
| `window_compare.py` | 模型 vs USGS 各窗对比 |
| `digest.py` | M6+ 日报 |
| `package_submission.py` | ZIP 打包 |
| `check_submission.py` | 提交前 CSV/大小检查 |

部署脚本：`deploy/install_monitor.sh`、`deploy/dizhen-monitor.service`。
