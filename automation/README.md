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

确认手动 tick 正常后，再使用 `flock` 配置 crontab。不要同时启用旧的自动提交
crontab 和新的控制器。
