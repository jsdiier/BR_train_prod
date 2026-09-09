#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run one experiment in its own module environment and dump aligned predictions."""

from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np


ADD_INFO_WIDTH = 24
TRACE_ID_INDEX = 0
SHOP_ID_INDEX = 1
UID_INDEX = 5
REC_ID_INDEX = 6
AD_SLOT_ID = 1044
TASKS = (
    ("buy", "cvr_label", 0),
    ("cat", "cat_label", 1),
    ("click", "clk_label", 2),
    ("ext", "ext_label", 3),
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--start-day", default="20260826")
    parser.add_argument("--end-day", default="20260828")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--progress-batches", type=int, default=100)
    return parser.parse_args()


def bootstrap(experiment_dir):
    experiment_dir = os.path.abspath(experiment_dir)
    required = ("model.py", "model_conf.py", "utils.py")
    for name in required:
        if not os.path.isfile(os.path.join(experiment_dir, name)):
            raise RuntimeError("missing %s in %s" % (name, experiment_dir))
    sys.path.insert(0, experiment_dir)
    os.chdir(experiment_dir)
    import tensorflow as tf
    import model_conf
    import utils as ut
    from model import Model
    return tf, model_conf, ut, Model


def date_range(start_day, end_day):
    day = datetime.datetime.strptime(start_day, "%Y%m%d")
    end = datetime.datetime.strptime(end_day, "%Y%m%d")
    while day <= end:
        yield day.strftime("%Y%m%d")
        day += datetime.timedelta(days=1)


def data_files(tf, root, start_day, end_day):
    files = []
    by_day = {}
    for day in date_range(start_day, end_day):
        paths = sorted(tf.io.gfile.glob("%s/%s/part*" % (root.rstrip("/"), day)))
        if not paths:
            raise RuntimeError("no TFRecord parts found for day=%s" % day)
        by_day[day] = len(paths)
        files.extend(paths)
    print("DATA files=%d by_day=%s" % (len(files), by_day))
    return files, by_day


def make_dataset(ut, files, batch_size):
    return ut.ReadTFRecordV2(
        files,
        shuffle_size=1,
        batch_size=batch_size,
        fetch_size=2,
        num_parallel=4,
    )


def restore(tf, Model, checkpoint_dir, first_batch):
    model = Model(training=True)
    model([first_batch["fea_ids"], first_batch["fea_vals"]])
    zeros = [tf.zeros_like(variable) for variable in model.trainable_variables]
    model.optimizer.apply_gradients(zip(zeros, model.trainable_variables))
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if not latest:
        raise RuntimeError("checkpoint missing: %s" % checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    checkpoint.restore(latest).assert_consumed()
    model.training = False
    print("CHECKPOINT_RESTORED %s" % latest)
    return model, latest


def decode(value):
    return value.decode("utf-8", errors="replace")


def hash64(text):
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def metadata_and_ad_fid(tf, feat, batch_size):
    values = feat["add_info_list"].values.numpy()
    expected = batch_size * ADD_INFO_WIDTH
    if values.size != expected:
        raise RuntimeError(
            "add_info width mismatch actual=%d expected=%d" % (values.size, expected)
        )
    matrix = values.reshape(batch_size, ADD_INFO_WIDTH)

    sids = tf.sparse.to_dense(feat["fea_ids"]).numpy()
    fids = tf.sparse.to_dense(feat["fea_vals"]).numpy()
    ad_fids = np.empty(batch_size, dtype=np.int64)
    sample_hashes = np.empty(batch_size, dtype=np.uint64)
    request_hashes = np.empty(batch_size, dtype=np.uint64)
    uid_hashes = np.empty(batch_size, dtype=np.uint64)

    for row_index in range(batch_size):
        positions = np.flatnonzero(sids[row_index] == AD_SLOT_ID)
        if positions.size != 1:
            raise RuntimeError(
                "sample must contain slot %d exactly once; row=%d count=%d"
                % (AD_SLOT_ID, row_index, positions.size)
            )
        ad_fids[row_index] = int(fids[row_index, positions[0]])
        traceid = decode(matrix[row_index, TRACE_ID_INDEX])
        shopid = decode(matrix[row_index, SHOP_ID_INDEX])
        uid = decode(matrix[row_index, UID_INDEX])
        recid = decode(matrix[row_index, REC_ID_INDEX])
        sample_hashes[row_index] = hash64("\t".join((traceid, recid, shopid)))
        request_hashes[row_index] = hash64(recid)
        uid_hashes[row_index] = hash64(uid)
    return ad_fids, sample_hashes, request_hashes, uid_hashes


def main():
    args = arguments()
    tf, model_conf, ut, Model = bootstrap(args.experiment_dir)
    files, files_by_day = data_files(tf, args.data_root, args.start_day, args.end_day)
    batch_size = int(model_conf.batch_size)
    first_batch = next(iter(make_dataset(ut, files[:1], batch_size)))
    model, checkpoint_path = restore(tf, Model, args.checkpoint, first_batch)

    arrays = {"ad_fid": [], "sample_hash": [], "request_hash": [], "uid_hash": []}
    for task, _, _ in TASKS:
        arrays[task + "_label"] = []
        arrays[task + "_score"] = []

    total = 0
    started = time.time()
    for batch_index, feat in enumerate(make_dataset(ut, files, batch_size), 1):
        actual_batch = int(feat["cvr_label"].shape[0])
        outputs = model([feat["fea_ids"], feat["fea_vals"]])
        if len(outputs) != 4:
            raise RuntimeError("expected 4 model outputs, got=%d" % len(outputs))
        ad_fid, sample_hash, request_hash, uid_hash = metadata_and_ad_fid(
            tf, feat, actual_batch
        )
        arrays["ad_fid"].append(ad_fid)
        arrays["sample_hash"].append(sample_hash)
        arrays["request_hash"].append(request_hash)
        arrays["uid_hash"].append(uid_hash)
        for task, label_key, output_index in TASKS:
            arrays[task + "_label"].append(
                feat[label_key].numpy().reshape(-1).astype(np.float32)
            )
            arrays[task + "_score"].append(
                outputs[output_index].numpy().reshape(-1).astype(np.float32)
            )
        total += actual_batch
        if batch_index % args.progress_batches == 0:
            elapsed = max(time.time() - started, 1e-9)
            print(
                "INFERENCE_PROGRESS label=%s batches=%d samples=%d samples_per_s=%.2f"
                % (args.label, batch_index, total, total / elapsed)
            )

    result = {name: np.concatenate(values) for name, values in arrays.items()}
    unique_fids, fid_counts = np.unique(result["ad_fid"], return_counts=True)
    if unique_fids.size != 2:
        raise RuntimeError(
            "slot 1044 must contain exactly 2 FIDs; got=%s"
            % dict(zip(unique_fids.tolist(), fid_counts.tolist()))
        )
    result["metadata_json"] = np.asarray(
        json.dumps(
            {
                "label": args.label,
                "experiment_dir": os.path.abspath(args.experiment_dir),
                "checkpoint": checkpoint_path,
                "data_root": args.data_root,
                "start_day": args.start_day,
                "end_day": args.end_day,
                "files_by_day": files_by_day,
                "samples": total,
                "is_ad_fid_counts": dict(
                    (str(fid), int(count)) for fid, count in zip(unique_fids, fid_counts)
                ),
            },
            sort_keys=True,
        )
    )
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    np.savez(args.output, **result)
    print("INFERENCE_COMPLETED label=%s samples=%d output=%s" % (args.label, total, args.output))


if __name__ == "__main__":
    main()
