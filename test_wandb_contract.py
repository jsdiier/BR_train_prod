#!/usr/bin/env python3
"""Static regression contract; intentionally does not import TensorFlow."""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    train_text = (ROOT / 'train.py').read_text()
    monitor_text = (ROOT / 'wandb_monitor.py').read_text()
    common_text = (ROOT / 'common.conf').read_text()
    train_shell_text = (ROOT / 'train.sh').read_text()
    experiment = json.loads((ROOT / 'experiment.json').read_text())

    ast.parse(train_text)
    ast.parse(monitor_text)

    assert 'tf.linalg.global_norm' in train_text
    assert 'model.optimizer.apply_gradients' in train_text
    assert 'tf.clip_by_global_norm' not in train_text
    assert 'wandb.watch' not in train_text
    assert 'WANDB_API_KEY=' not in common_text
    assert 'export WANDB_API_KEY=' in train_shell_text
    assert 'WANDB_PROJECT="chen1109487007-a123/br-rank-prod"' in common_text
    assert train_text.index("tf.config.experimental.set_memory_growth(gpu, True)") < train_text.index("solver = Learner()")
    assert 'WANDB_TASK_GRAD_INTERVAL=0' in common_text
    assert experiment['branch'] == 'BR_train_prod_baseline_w_wandb'
    assert experiment['change'] == 'wandb_observability_only'
    assert experiment['initialization'] == 'fresh_end_to_end'
    assert experiment['enabled'] is True
    assert experiment['manual_only'] is False
    print('WANDB_CONTRACT: PASS')


if __name__ == '__main__':
    main()
