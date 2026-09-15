# W&B monitoring contract

- Project: `br-rank-prod`
- Group: `BR_train_prod_baseline_w_wandb`
- Run name: `<group>__train_<start_day>_<end_day>`
- Online logging interval: every 100 process-local training steps
- Parameter norm interval: every 1000 process-local training steps
- Multi-task gradient diagnostics: disabled by default (`WANDB_TASK_GRAD_INTERVAL=0`)

The integration is observational only. Gradient norms are measured before
`apply_gradients`; no clipping or gradient replacement is performed.

For this private experiment, `WANDB_API_KEY` is exported directly by `train.sh`,
matching the established TensorFlow rank integration. If W&B cannot be imported,
initialized, or reached, training continues and the same scalar records are
appended to `log/wandb_metrics.jsonl`.

Before launching on Luban, verify the selected Python environment:

```bash
source ./common.conf
"$python" -c 'import wandb; print(wandb.__version__)'
grep -q '^export WANDB_API_KEY=' train.sh && echo WANDB_API_KEY_CONFIGURED
```

The default run records losses, label rates, prediction distributions,
throughput, learning rate, global and module gradient norms, periodic parameter
norms, numerical-health counters, and daily AUC/GAUC/MAE. W&B system monitoring
adds GPU, CPU, and memory telemetry when supported by the installed client.
