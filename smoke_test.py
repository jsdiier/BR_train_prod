#!/usr/bin/env python
import argparse
import os

import tensorflow as tf

import utils as ut
from model import Model


def gradient_is_finite(gradient):
    values = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
    return bool(tf.reduce_all(tf.math.is_finite(values)).numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=(
        'hdfs://DClusterUS1/user/prod_soda_trade_strategy/rank/'
        'jiazhuo/hash_fea_new/train/'))
    parser.add_argument('--day', default='20260720')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    pattern = args.data_root.rstrip('/') + '/' + args.day + '/part*'
    files = sorted(tf.io.gfile.glob(pattern))
    if not files:
        raise RuntimeError('no TFRecord parts found: %s' % pattern)
    print('SMOKE_DATA part=%s' % files[0])

    dataset = ut.ReadTFRecordV2(
        [files[0]], shuffle_size=1, batch_size=args.batch_size,
        fetch_size=1, num_parallel=1)
    feat = next(iter(dataset))
    model = Model(training=True)

    with tf.GradientTape() as tape:
        predictions = model([feat['fea_ids'], feat['fea_vals']])
        labels = [
            feat['cvr_label'], feat['cat_label'],
            feat['clk_label'], feat['ext_label']]
        task_losses = [
            tf.reduce_mean(model.loss_bc(tf.expand_dims(label, 1), prediction))
            for label, prediction in zip(labels, predictions)]
        total_loss = tf.add_n(task_losses)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    missing = [
        variable.name for variable, gradient
        in zip(model.trainable_variables, gradients)
        if gradient is None]
    nonfinite = [
        variable.name for variable, gradient
        in zip(model.trainable_variables, gradients)
        if gradient is not None and not gradient_is_finite(gradient)]
    if missing:
        print('SMOKE_WARNING missing_baseline_gradients=%d' % len(missing))
    if nonfinite:
        raise RuntimeError('non-finite gradients: %s' % nonfinite)
    for prediction in predictions:
        tf.debugging.assert_all_finite(prediction, 'non-finite prediction')
    tf.debugging.assert_all_finite(total_loss, 'non-finite loss')

    experiment_variables = [
        variable.name for variable in model.trainable_variables
        if 'lowrank_' in variable.name.lower()]
    experiment_missing = [
        variable.name for variable, gradient
        in zip(model.trainable_variables, gradients)
        if 'lowrank_' in variable.name.lower() and gradient is None]
    experiment_gradients = [
        variable.name for variable, gradient
        in zip(model.trainable_variables, gradients)
        if 'lowrank_' in variable.name.lower() and gradient is not None]
    if not experiment_variables:
        raise RuntimeError('experiment variables received no gradients: lowrank_')
    if experiment_missing:
        raise RuntimeError(
            'experiment variables with missing gradients: %s' %
            experiment_missing)

    print('DCNV2_AUDIT layers=%d low_rank=%d projection_dim=%d' % (
        model.lowrank_crossnet.num_layers,
        model.lowrank_crossnet.low_rank,
        model.lowrank_cross_projection.units))
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    parameter_count = sum(
        int(tf.size(variable).numpy()) for variable in model.trainable_variables)
    print('SMOKE_OK loss=%.8f parameters=%d experiment_gradients=%d' % (
        float(total_loss.numpy()), parameter_count, len(experiment_gradients)))


if __name__ == '__main__':
    main()
