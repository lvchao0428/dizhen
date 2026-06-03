# 排名赛监控 Pipeline

全球 M6+ 地震监控、Server酱微信通知、v3 模型自动生成提交 CSV。

## 快速开始（远程服务器）

```bash
cd /path/to/dizhen

# 1. 配置密钥
cp monitor/.env.example monitor/.env
# 编辑 SERVERCHAN_SENDKEY=...

cp monitor/config.example.yaml monitor/config.yaml

# 2. 安装并启动
chmod +x deploy/install_monitor.sh
./deploy/install_monitor.sh

# 3. 启动服务（若已安装 systemd）
sudo systemctl start dizhen-monitor
```

## 手动单次运行

```bash
source .venv/bin/activate
python -m monitor.run_once
```

## 输出

- 预测 CSV：`monitor/ranking_output/{event_id}/`
- 余震序列：`monitor/data/sequences/{event_id}_eq.csv`
- 状态库：`monitor/data/events.sqlite`
- 日志：`monitor/data/logs/monitor.log`

## 训练模型

若 `solution/models_v3/models.pkl` 不存在：

```bash
cd solution && python predict_v3.py
```

## 配置说明

见 `monitor/config.example.yaml`：

- `poll_interval_minutes`: 轮询间隔（默认 5 分钟）
- `notify_max_age_hours`: 仅对主震后 48h 内新事件推送（避免首次启动刷屏）
- `SERVERCHAN_SENDKEY`: 放在 `monitor/.env`
