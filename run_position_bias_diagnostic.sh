#!/usr/bin/env bash
set -euo pipefail

cd /home/luban/rank-ssl/chenpinyuan/tf_rank_BR_prod/BR_train_prod_ema_pal_position_bias_diagnostic

/nfs/volume-100003-1/raochongzhi/envs/tf_a6000/bin/python -u \
    diagnose_position_bias.py \
    --checkpoint /home/luban/rank-ssl/chenpinyuan/tf_rank_BR_prod/BR_train_prod_bs_lr_ema_weights/model/checkpoints/20260826 \
    --data-root hdfs://DClusterUS1/user/prod_soda_trade_strategy/rank/jiazhuo/hash_fea_new/train \
    --days 20260827,20260828,20260829 \
    --parts-per-day 2,2,1 \
    --output-dir log/pal_position_bias_diagnostic_ckpt_20260826
