# Research automation

代码位于 `shared_tools/automation/`，运行状态位于同级实验根目录的
`automation_state/`。不要把脚本复制进状态目录。

首次接管已有实验时，先 dry-run：

```bash
BASE=/home/luban/rank-ssl/chenpinyuan/tf_rank_BR
PY=/nfs/volume-100003-1/raochongzhi/envs/tf_a6000/bin/python

$PY $BASE/shared_tools/automation/init_existing_runs.py \
  --base-dir $BASE \
  --state-dir $BASE/automation_state \
  --config $BASE/shared_tools/automation/experiments.json \
  --dry-run
```

确认结果后，将 `--dry-run` 改为 `--apply`。初始化只接管已有目录、进程和产物，
不会执行 `submit_luban.sh`，也不会重跑实验。

正式 tick：

```bash
bash $BASE/shared_tools/automation/automation_tick.sh
```

`automation_tick.sh` 在入口显式执行
`source /home/luban/rank-ssl/chenpinyuan/miniconda_base/bin/activate SFT_A6000`，确保
crontab 与手动执行使用同一个依赖环境。可通过 `CONDA_ACTIVATE`、`AUTOMATION_ENV` 和
`PYTHON_BIN` 环境变量覆盖默认值。

确认手动 tick 正常后，再使用 `flock` 配置 crontab。不要同时启用旧的自动提交
crontab 和新的控制器。

## 新实验注册

新实验分支根目录必须包含 `experiment.json`。控制器扫描远程分支，跳过
`main/shared_tools`；只有配置合法且 `enabled=true` 的新分支才会被 clone，并由外层
启动该分支原有的 `submit_luban.sh`。已有 run JSON 的分支不会因后续 commit 自动重跑。

```json
{
  "enabled": true,
  "baseline": "tf_train_base_new_try_roll_test_predict_time",
  "train_start_day": "20260303",
  "train_end_day": "20260720",
  "test_start_day": "20260721",
  "test_end_day": "20260724",
  "auto_test_start_ckpt_day": "20260724",
  "auto_test_end_day": "20260801",
  "allowed_missing_train_days": [],
  "allowed_missing_test_days": [],
  "require_inference_benchmark": true
}
```

rolling 数据存在已确认的上游缺口时，可按实验显式声明允许缺失日期。例如：

```json
{
  "allowed_missing_train_days": ["20260728"],
  "allowed_missing_test_days": ["20260728"]
}
```

验证器不会要求缺失训练日的 checkpoint，也不会要求缺失测试日的日志或 TSV 行；
后续有效测试日必须使用该测试日之前日期最大的真实 checkpoint。例如 20260728 同时
缺少训练和测试数据时，20260729 必须由 checkpoint 20260727 评估。缺失日不得通过
复制旧 checkpoint 或伪造指标行补齐。允许缺失日期必须位于 rolling 区间内，且不会
放宽 fixed-window 验证。

新实验在已有 active batch 运行期间只创建 run 状态，不会混入当前批次。当前批次完成后，
未入批的运行会由 `manage_batch.py` 冻结为下一批。

只有 fixed-window、没有 rolling 的实验使用：

```json
{
  "enabled": true,
  "rolling_enabled": false,
  "batch_group": "manual_feature_factorial_20260827",
  "baseline": "fixed_window_control_branch",
  "train_start_day": "20260303",
  "train_end_day": "20260820",
  "test_start_day": "20260821",
  "test_end_day": "20260823",
  "require_inference_benchmark": true
}
```

fixed-only run 只校验目标 checkpoint、fixed test 四任务日志、测试截止日期和
inference benchmark，不要求 `auto_test_*`、rolling 日志或 `rolling_metrics.tsv`；
batch 聚合也只生成 fixed 与 inference 结果。

可选 `batch_group` 用于隔离并发发现的实验组合。同一组会冻结为一个 batch；不同组
按最早发现顺序分别聚合，不会把 User-mandated Runs 与后续 Agent-generated Runs
混在同一 batch。未配置时使用 `default`。
