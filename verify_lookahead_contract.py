#!/usr/bin/env python3
"""Static contract checks that do not require TensorFlow on the local host."""
from pathlib import Path


def require(condition, message):
    if not condition:
        raise AssertionError(message)


train_source = Path('train.py').read_text(encoding='utf-8')
model_source = Path('model.py').read_text(encoding='utf-8')
test_source = Path('test.py').read_text(encoding='utf-8')

require('k=5, alpha=0.5, ema_decay=0.999' in train_source,
        'Lookahead hyperparameters changed')
require('model.optimizer.apply_gradients' in train_source,
        'Adam fast update is missing')
require('self.optimizer_trajectory.after_fast_update' in train_source,
        'Lookahead/EMA post-update hook is missing')
require('optimizer_trajectory=self.optimizer_trajectory' in train_source,
        'checkpoint does not preserve optimizer trajectory state')
require('optimizer_trajectory' not in model_source,
        'training-only Lookahead state leaked into serving Model')
require('serve_model([dummy_sids, dummy_fids])' in train_source,
        'serving signature is no longer the original two tensors')
require('assert_existing_objects_matched' in test_source,
        'inference restore does not safely ignore training-only state')

print('LOOKAHEAD_CONTRACT_OK')
