#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fail-fast audit that a SavedModel has no PAL position dependency."""

import argparse

import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("serving_model", help="path to exported SavedModel")
    args = parser.parse_args()

    loaded = tf.saved_model.load(args.serving_model)
    if "serving_default" not in loaded.signatures:
        raise RuntimeError("serving_default signature is missing")
    signature = loaded.signatures["serving_default"]
    positional, keyword = signature.structured_input_signature
    input_specs = tf.nest.flatten(positional) + tf.nest.flatten(keyword)
    if len(input_specs) != 2:
        raise RuntimeError("Serving must expose exactly two tensors, found %d" % len(input_specs))

    variable_names = [variable.name for variable in loaded.variables]
    forbidden_variables = [
        name for name in variable_names if "pal_position_bias" in name.lower()
    ]
    if forbidden_variables:
        raise RuntimeError("Serving contains PAL position variables: %s" % forbidden_variables)

    node_names = [node.name for node in signature.graph.as_graph_def().node]
    forbidden_nodes = [
        name for name in node_names if "pal_position_bias" in name.lower()
    ]
    if forbidden_nodes:
        raise RuntimeError("Serving graph reads PAL position nodes: %s" % forbidden_nodes)

    print("SERVING_NO_POSITION_OK")
    print("input_tensor_count: 2")
    print("input_signature: %s" % signature.structured_input_signature)
    print("pal_position_variable_count: 0")
    print("pal_position_graph_node_count: 0")


if __name__ == "__main__":
    main()
