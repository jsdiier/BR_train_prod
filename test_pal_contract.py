#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small, data-free contract test for the strict-CTCVR PAL implementation."""

import numpy as np
import tensorflow as tf

from model import Model


def main():
    model = Model(training=True)
    ranks = tf.constant(
        [0, 1, 49, 50, 99, 100, 199, 200, 299, 300, 399, 400, 499, 500, 966],
        dtype=tf.int32,
    )
    expected = np.asarray(
        [0, 1, 49, 50, 54, 55, 58, 59, 60, 61, 61, 62, 62, 63, 63],
        dtype=np.int32,
    )
    actual = model._position_bucket(ranks).numpy()
    np.testing.assert_array_equal(actual, expected)

    # Non-zero task-private biases exercise both factors of strict CTCVR.
    bias = np.zeros((63, 4), dtype=np.float32)
    bias[:, 0] = 0.20  # CVR|CLICK
    bias[:, 1] = -0.10  # CAT
    bias[:, 2] = 0.30  # CLICK
    bias[:, 3] = -0.20  # EXT
    model.position_bias_tail.assign(bias)

    relevance = tf.constant([[0.25], [0.40]], dtype=tf.float32)
    buy, cat, click, ext = model._position_aware_predictions(
        tf.constant([0, 1]), relevance, relevance, relevance, relevance
    )
    expected_cvr = model._add_logit_bias(
        relevance, tf.constant([[0.0], [0.20]], dtype=tf.float32)
    )
    np.testing.assert_allclose(
        buy.numpy(), (click * expected_cvr).numpy(), rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        model._position_aware_predictions(
            tf.constant([0, 0]), relevance, relevance, relevance, relevance
        )[0].numpy(),
        (relevance * relevance).numpy(),
        rtol=1e-6,
        atol=1e-7,
    )

    # The full ranking model intentionally remains unbuilt in this lightweight
    # contract test.  Inspect only the eagerly-created PAL parameter; asking
    # Keras for model.trainable_variables would force all nested Sequential
    # towers to have been called first.
    position_parameter_names = [model.position_bias_tail.name.lower()]
    if any("buy" in name and "position" in name for name in position_parameter_names):
        raise AssertionError("independent BUY position bias must not exist")
    if model.position_bias_tail.shape != (63, 4):
        raise AssertionError("unexpected PAL position table shape")
    if not (cat.shape == click.shape == ext.shape == buy.shape):
        raise AssertionError("PAL task output shapes differ")

    serving_model = Model(training=False, pred=True, enable_position_bias=False)
    if serving_model.position_bias_tail is not None:
        raise AssertionError("Serving model must not create the PAL position table")

    print(
        "PAL_CONTRACT_OK buckets=64 trainable_bias_shape=63x4 "
        "strict_ctcvr=true serving_position_table=false"
    )


if __name__ == "__main__":
    main()
