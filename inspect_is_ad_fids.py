#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit the two FIDs emitted by the binary is_ad slot before segmentation."""

from __future__ import print_function

import argparse
import csv
import datetime
import os
import sys
from collections import Counter, defaultdict


ADD_INFO_WIDTH = 24
TRACE_ID_INDEX = 0
SHOP_ID_INDEX = 1
REC_ID_INDEX = 6
REQUEST_TIME_INDEX = 21


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--start-day", default="20260826")
    parser.add_argument("--end-day", default="20260828")
    parser.add_argument("--slot-id", type=int, default=1044)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--examples-per-fid", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def bootstrap(experiment_dir):
    experiment_dir = os.path.abspath(experiment_dir)
    if not os.path.isfile(os.path.join(experiment_dir, "utils.py")):
        raise RuntimeError("experiment utils.py missing: %s" % experiment_dir)
    sys.path.insert(0, experiment_dir)
    os.chdir(experiment_dir)
    import tensorflow as tf
    import utils as ut
    return tf, ut


def date_range(start_day, end_day):
    day = datetime.datetime.strptime(start_day, "%Y%m%d")
    end = datetime.datetime.strptime(end_day, "%Y%m%d")
    while day <= end:
        yield day.strftime("%Y%m%d")
        day += datetime.timedelta(days=1)


def data_files(tf, root, start_day, end_day):
    files = []
    for day in date_range(start_day, end_day):
        paths = sorted(tf.io.gfile.glob("%s/%s/part*" % (root.rstrip("/"), day)))
        if not paths:
            raise RuntimeError("no TFRecord parts found for day=%s" % day)
        print("DATA_DAY day=%s parts=%d" % (day, len(paths)))
        files.extend(paths)
    return files


def decode(value):
    return value.decode("utf-8", errors="replace")


def main():
    args = arguments()
    output_path = os.path.abspath(args.output)
    tf, ut = bootstrap(args.experiment_dir)
    files = data_files(tf, args.data_root, args.start_day, args.end_day)
    dataset = ut.ReadTFRecordV2(
        files,
        shuffle_size=1,
        batch_size=1024,
        fetch_size=2,
        num_parallel=4,
    )

    counts = Counter()
    examples = defaultdict(list)
    missing = 0
    multiple = 0
    total = 0

    for batch_index, feat in enumerate(dataset, 1):
        sids = tf.sparse.to_dense(feat["fea_ids"]).numpy()
        fids = tf.sparse.to_dense(feat["fea_vals"]).numpy()
        batch_size = int(sids.shape[0])
        add_info = feat["add_info_list"].values.numpy()
        expected = batch_size * ADD_INFO_WIDTH
        if add_info.size != expected:
            raise RuntimeError(
                "add_info width mismatch actual=%d expected=%d"
                % (add_info.size, expected)
            )
        add_info = add_info.reshape(batch_size, ADD_INFO_WIDTH)

        for row_index in range(batch_size):
            if total >= args.max_samples:
                break
            positions = (sids[row_index] == args.slot_id).nonzero()[0]
            if positions.size == 0:
                missing += 1
                total += 1
                continue
            if positions.size != 1:
                multiple += 1
                total += 1
                continue
            fid = int(fids[row_index, positions[0]])
            counts[fid] += 1
            if len(examples[fid]) < args.examples_per_fid:
                examples[fid].append(
                    (
                        decode(add_info[row_index, TRACE_ID_INDEX]),
                        decode(add_info[row_index, REC_ID_INDEX]),
                        decode(add_info[row_index, SHOP_ID_INDEX]),
                        decode(add_info[row_index, REQUEST_TIME_INDEX]),
                    )
                )
            total += 1
        if batch_index % 20 == 0 or total >= args.max_samples:
            print(
                "PROGRESS batches=%d samples=%d fid_counts=%s missing=%d multiple=%d"
                % (batch_index, total, dict(counts), missing, multiple)
            )
        if total >= args.max_samples:
            break

    if missing or multiple:
        raise RuntimeError(
            "slot cardinality invalid samples=%d missing=%d multiple=%d"
            % (total, missing, multiple)
        )
    if len(counts) != 2:
        raise RuntimeError(
            "binary is_ad slot must have exactly 2 FIDs; got=%s" % dict(counts)
        )

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["fid", "count", "ratio", "traceid", "recid", "shopid", "request_time"]
        )
        for fid, count in counts.most_common():
            for traceid, recid, shopid, request_time in examples[fid]:
                writer.writerow(
                    [fid, count, "%.10f" % (float(count) / total), traceid, recid, shopid, request_time]
                )

    print("IS_AD_FID_AUDIT samples=%d slot=%d" % (total, args.slot_id))
    for fid, count in counts.most_common():
        print("IS_AD_FID fid=%d count=%d ratio=%.6f" % (fid, count, float(count) / total))
    print("IMPORTANT map one FID to raw is_ad=1 using the printed sample IDs; do not guess")
    print("OUTPUT %s" % output_path)


if __name__ == "__main__":
    main()
