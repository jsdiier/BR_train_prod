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
  "require_inference_benchmark": true
}
```

新实验在已有 active batch 运行期间只创建 run 状态，不会混入当前批次。当前批次完成后，
未入批的运行会由 `manage_batch.py` 冻结为下一批。
