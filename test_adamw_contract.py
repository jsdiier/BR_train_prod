#!/usr/bin/env python
import numpy as np
import tensorflow as tf

from module.selective_adamw import SelectiveAdamW


def main():
    dense = tf.keras.layers.Dense(2, use_bias=True, name='dense_probe')
    embedding = tf.keras.layers.Embedding(8, 2, name='embedding_probe')
    batch_norm = tf.keras.layers.BatchNormalization(name='batch_norm_probe')
    layer_norm = tf.keras.layers.LayerNormalization(name='layer_norm_probe')

    _ = dense(tf.ones([2, 3]))
    _ = embedding(tf.constant([[1, 2]]))
    _ = batch_norm(tf.ones([2, 2]), training=True)
    _ = layer_norm(tf.ones([2, 2]))
    variables = (
        dense.trainable_variables + embedding.trainable_variables +
        batch_norm.trainable_variables + layer_norm.trainable_variables)

    optimizer = SelectiveAdamW(
        learning_rate=0.1, decoupled_weight_decay=0.01,
        beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    decayed, excluded = optimizer.decay_audit(variables)
    assert decayed == [dense.kernel.name], decayed
    assert dense.bias.name in excluded
    assert embedding.embeddings.name in excluded

    before = [v.numpy().copy() for v in variables]
    optimizer.apply_gradients([(tf.zeros_like(v), v) for v in variables])
    after = [v.numpy().copy() for v in variables]

    kernel_index = next(
        index for index, variable in enumerate(variables)
        if variable is dense.kernel)
    expected_kernel = before[kernel_index] * (1.0 - 0.1 * 0.01)
    np.testing.assert_allclose(
        after[kernel_index], expected_kernel, rtol=1e-6, atol=1e-7)
    for index, variable in enumerate(variables):
        if index == kernel_index:
            continue
        np.testing.assert_allclose(
            after[index], before[index], rtol=0.0, atol=0.0,
            err_msg='excluded variable changed: %s' % variable.name)
    print('ADAMW_CONTRACT_OK decayed=%s excluded=%d' % (
        ','.join(decayed), len(excluded)))


if __name__ == '__main__':
    main()
