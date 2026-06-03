# 排名赛监控 Pipeline

- **实时**：M≥5.8 微信通知（不限深度）；浅源 M≥6 自动拉序列、v3 预测、生成 CSV
- **微信内容**：模型预估 vs 官方(USGS) 各窗对比；无余震时提示「数据为空」
- **每日 09:00 北京时间**：M6+ 汇总日报
- **手动**：`run_digest` / `run_compare` / `package_submission`

## 部署（远程）

```bash
cd ~/project/dizhen

# Windows 同步后先去 CRLF
sed -i 's/\r$//' deploy/install_monitor.sh deploy/package_submission.sh

cp monitor/.env.example monitor/.env   # SERVERCHAN_SENDKEY
cp monitor/config.example.yaml monitor/config.yaml

bash deploy/install_monitor.sh

sudo systemctl status dizhen-monitor
```

勿用 `sh install_monitor.sh`；勿在 CRLF 未修时用 `./install_monitor.sh`。

## 模型

- 仓库默认**无** `solution/models_v3/models.pkl`
- 训练：`cd solution && python predict_v3.py`（约 **1–2 分钟**，20 震例 × 3 窗）
- 资格赛 v3 方案，LOEO + LightGBM

## 常用命令

```bash
source .venv/bin/activate

python -m monitor.run_once              # 单次轮询
python -m monitor.run_digest --print-only --days 30
python -m monitor.run_digest --notify --backfill
python -m monitor.run_compare 20260531213418 --predict
bash deploy/package_submission.sh
```

## 目录

| 路径 | 说明 |
|------|------|
| `monitor/ranking_output/{id}/` | `*-T1-T2.csv`, `*-T3.csv`, `window_comparison.md` |
| `monitor/data/sequences/` | USGS 余震序列 |
| `monitor/data/submissions/` | 打包 ZIP |

## 逻辑

| 条件 | 微信 | 序列+预测+CSV |
|------|------|----------------|
| M≥6 深度≤70km | 是 | 是 |
| M≥6 深度>70km | 是（仅监测） | 否 |

官方余震来源：排名赛实时阶段用 **USGS**；观测为空时推送中会写明。
