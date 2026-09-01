#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare baseline and PAL relevance predictions on the same logged samples."""

from __future__ import print_function

import argparse
import csv
import datetime
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import model_conf
import utils as ut
from model import Model


TASKS = (
    ("buy", "cvr_label", 0),
    ("cat", "cat_label", 3),
    ("click", "clk_label", 2),
    ("ext", "ext_label", 4),
)
POSITION_NAMES = ("1-3", "4-6", "7-10", "11-20", "21+")
ADD_INFO_WIDTH = 24
RANK_INDEX = 3
REC_ID_INDEX = 6


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-checkpoint",
        default=(
            "/home/luban/rank-ssl/chenpinyuan/tf_rank_BR_prod/"
            "BR_train_prod_bs_lr_ema_weights/model/checkpoints/20260720"
        ),
    )
    parser.add_argument(
        "--pal-checkpoint",
        default=(
            "/home/luban/rank-ssl/chenpinyuan/tf_rank_BR_prod/"
            "BR_train_prod_ema_pal_position_debias/model/checkpoints/20260720"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=(
            "hdfs://DClusterUS1/user/prod_soda_trade_strategy/rank/"
            "jiazhuo/hash_fea_new/train"
        ),
    )
    parser.add_argument("--start-day", default="20260721")
    parser.add_argument("--end-day", default="20260724")
    parser.add_argument("--batch-size", type=int, default=model_conf.batch_size)
    parser.add_argument("--output-dir", default="log/posthoc_pal_ckpt_20260720_20260721_20260724")
    return parser.parse_args()


def date_range(start_day, end_day):
    day = datetime.datetime.strptime(start_day, "%Y%m%d")
    end = datetime.datetime.strptime(end_day, "%Y%m%d")
    while day <= end:
        yield day.strftime("%Y%m%d")
        day += datetime.timedelta(days=1)


def data_files(root, start_day, end_day):
    files = []
    by_day = {}
    for day in date_range(start_day, end_day):
        paths = sorted(tf.io.gfile.glob("%s/%s/part*" % (root.rstrip("/"), day)))
        if not paths:
            raise RuntimeError("no TFRecord parts found for day=%s" % day)
        by_day[day] = len(paths)
        files.extend(paths)
    return files, by_day


def dataset(files, batch_size):
    return ut.ReadTFRecordV2(
        files,
        shuffle_size=1,
        batch_size=batch_size,
        fetch_size=2,
        num_parallel=4,
    )


def restore(checkpoint_dir, first_batch, enable_position_bias, label):
    # pred=True exposes [BUY, CVR|CLICK, CLICK, CAT, EXT] while the call still
    # receives only [fea_ids, fea_vals], hence both models are relevance-only.
    model = Model(training=True, pred=True, enable_position_bias=enable_position_bias)
    model([first_batch["fea_ids"], first_batch["fea_vals"]])
    zeros = [tf.zeros_like(variable) for variable in model.trainable_variables]
    model.optimizer.apply_gradients(zip(zeros, model.trainable_variables))
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if not latest:
        raise RuntimeError("checkpoint missing: %s" % checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    checkpoint.restore(latest).assert_consumed()
    model.training = False
    print("CHECKPOINT_RESTORED label=%s path=%s" % (label, latest))
    return model, latest


def metadata(feat, batch_size):
    values = feat["add_info_list"].values.numpy()
    expected = batch_size * ADD_INFO_WIDTH
    if values.size != expected:
        raise RuntimeError(
            "add_info schema mismatch: actual=%d expected=%d" % (values.size, expected)
        )
    matrix = values.reshape(batch_size, ADD_INFO_WIDTH)
    rank = np.asarray(
        [int(value.decode("utf-8")) for value in matrix[:, RANK_INDEX]],
        dtype=np.int32,
    )
    if np.any(rank < 0):
        raise RuntimeError("negative final display rank detected")
    rec_id = [value.decode("utf-8", errors="replace") for value in matrix[:, REC_ID_INDEX]]
    return rank, rec_id


def coarse_position(rank):
    group = np.full(rank.shape, 4, dtype=np.int16)
    group[rank <= 2] = 0
    group[(rank >= 3) & (rank <= 5)] = 1
    group[(rank >= 6) & (rank <= 9)] = 2
    group[(rank >= 10) & (rank <= 19)] = 3
    return group


def auc(label, score):
    if label.size == 0 or np.unique(label).size < 2:
        return float("nan")
    return float(roc_auc_score(label, score))


def stratified_auc(label, score, group):
    rows = []
    weighted_sum = 0.0
    total_weight = 0.0
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    sorted_label = label[order]
    sorted_score = score[order]
    values, starts, counts = np.unique(
        sorted_group, return_index=True, return_counts=True
    )
    for value, start, count in zip(values, starts, counts):
        end = start + count
        group_label = sorted_label[start:end]
        positives = int(np.sum(group_label))
        negatives = int(group_label.size - positives)
        pair_weight = float(positives * negatives)
        group_auc = auc(group_label, sorted_score[start:end])
        rows.append((value, int(group_label.size), positives, group_auc, pair_weight))
        if np.isfinite(group_auc) and pair_weight > 0:
            weighted_sum += group_auc * pair_weight
            total_weight += pair_weight
    result = weighted_sum / total_weight if total_weight else float("nan")
    return result, rows


def grouped_auc(label, score, group_id):
    labels_by_group = defaultdict(list)
    scores_by_group = defaultdict(list)
    for y, prediction, key in zip(label, score, group_id):
        labels_by_group[key].append(float(y))
        scores_by_group[key].append(float(prediction))
    weighted_sum = 0.0
    sample_sum = 0
    unweighted_sum = 0.0
    valid_groups = 0
    for key, labels in labels_by_group.items():
        values = np.asarray(labels, dtype=np.float32)
        if np.unique(values).size < 2:
            continue
        group_auc = float(roc_auc_score(values, scores_by_group[key]))
        weighted_sum += group_auc * values.size
        sample_sum += values.size
        unweighted_sum += group_auc
        valid_groups += 1
    return (
        weighted_sum / sample_sum if sample_sum else float("nan"),
        unweighted_sum / valid_groups if valid_groups else float("nan"),
        valid_groups,
        sample_sum,
    )


def main():
    args = arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    files, files_by_day = data_files(args.data_root, args.start_day, args.end_day)
    print("DATA files=%d by_day=%s" % (len(files), files_by_day))

    first_batch = next(iter(dataset(files[:1], args.batch_size)))
    baseline_model, baseline_path = restore(
        args.baseline_checkpoint, first_batch, False, "baseline"
    )
    pal_model, pal_path = restore(args.pal_checkpoint, first_batch, True, "pal")

    arrays = {"rank": []}
    for task, label_key, output_index in TASKS:
        del label_key, output_index
        arrays[task + "_label"] = []
        arrays[task + "_baseline"] = []
        arrays[task + "_pal"] = []
    arrays["cvr_baseline"] = []
    arrays["cvr_pal"] = []
    clicked_rec_ids = []

    total = 0
    started = time.time()
    for batch_index, feat in enumerate(dataset(files, args.batch_size), 1):
        batch_size = int(feat["cvr_label"].shape[0])
        baseline_outputs = baseline_model([feat["fea_ids"], feat["fea_vals"]])
        pal_outputs = pal_model([feat["fea_ids"], feat["fea_vals"]])
        rank, rec_id = metadata(feat, batch_size)
        arrays["rank"].append(rank)

        for task, label_key, output_index in TASKS:
            arrays[task + "_label"].append(
                feat[label_key].numpy().reshape(-1).astype(np.float32)
            )
            arrays[task + "_baseline"].append(
                baseline_outputs[output_index].numpy().reshape(-1).astype(np.float32)
            )
            arrays[task + "_pal"].append(
                pal_outputs[output_index].numpy().reshape(-1).astype(np.float32)
            )

        click_mask = feat["clk_label"].numpy().reshape(-1) > 0
        arrays["cvr_baseline"].append(
            baseline_outputs[1].numpy().reshape(-1)[click_mask].astype(np.float32)
        )
        arrays["cvr_pal"].append(
            pal_outputs[1].numpy().reshape(-1)[click_mask].astype(np.float32)
        )
        clicked_rec_ids.extend(value for value, keep in zip(rec_id, click_mask) if keep)

        total += batch_size
        if batch_index % 100 == 0:
            speed = total / max(time.time() - started, 0.001)
            print(
                "\rbatches=%d samples=%d speed=%.1f samples/s"
                % (batch_index, total, speed),
                end="",
            )
            sys.stdout.flush()
    print("\nINFERENCE_COMPLETED samples=%d" % total)

    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    position = coarse_position(arrays["rank"])
    report = {
        "baseline_checkpoint": baseline_path,
        "pal_checkpoint": pal_path,
        "data_root": args.data_root,
        "start_day": args.start_day,
        "end_day": args.end_day,
        "samples": int(total),
        "tasks": {},
    }
    position_rows = []

    for task, _, _ in TASKS:
        label = arrays[task + "_label"]
        task_report = {}
        print("\n" + "=" * 90)
        print("TASK=%s" % task.upper())
        for model_name in ("baseline", "pal"):
            score = arrays[task + "_" + model_name]
            exact_auc, exact_rows = stratified_auc(label, score, arrays["rank"])
            bucket_auc, bucket_rows = stratified_auc(label, score, position)
            task_report[model_name] = {
                "global_auc": auc(label, score),
                "exact_rank_auc": exact_auc,
                "five_bucket_auc": bucket_auc,
            }
            for value, samples, positives, value_auc, pair_weight in exact_rows:
                position_rows.append(
                    (task, model_name, "exact", str(value), samples, positives, value_auc, pair_weight)
                )
            for value, samples, positives, value_auc, pair_weight in bucket_rows:
                position_rows.append(
                    (
                        task,
                        model_name,
                        "five_bucket",
                        POSITION_NAMES[int(value)],
                        samples,
                        positives,
                        value_auc,
                        pair_weight,
                    )
                )
            print(
                "%s global_auc=%.10f exact_rank_auc=%.10f five_bucket_auc=%.10f"
                % (model_name, task_report[model_name]["global_auc"], exact_auc, bucket_auc)
            )
        task_report["delta_pal_minus_baseline"] = {
            key: task_report["pal"][key] - task_report["baseline"][key]
            for key in ("global_auc", "exact_rank_auc", "five_bucket_auc")
        }
        print(
            "PAL_MINUS_BASELINE global=%+.3f‰ exact_rank=%+.3f‰ five_bucket=%+.3f‰"
            % tuple(
                task_report["delta_pal_minus_baseline"][key] * 1000.0
                for key in ("global_auc", "exact_rank_auc", "five_bucket_auc")
            )
        )
        report["tasks"][task] = task_report

    clicked = arrays["click_label"] > 0
    clicked_buy_label = arrays["buy_label"][clicked]
    if clicked_buy_label.size != arrays["cvr_baseline"].size:
        raise RuntimeError("clicked CVR arrays are not aligned")
    cvr_report = {
        "clicked_samples": int(clicked_buy_label.size),
        "buy_positives": int(np.sum(clicked_buy_label)),
    }
    print("\n" + "=" * 90)
    print("CVR_GIVEN_CLICK")
    for model_name in ("baseline", "pal"):
        score = arrays["cvr_" + model_name]
        gauc, uauc, groups, grouped_samples = grouped_auc(
            clicked_buy_label, score, clicked_rec_ids
        )
        cvr_report[model_name] = {
            "auc": auc(clicked_buy_label, score),
            "gauc": gauc,
            "uauc": uauc,
            "valid_groups": groups,
            "grouped_samples": grouped_samples,
        }
        print(
            "%s auc=%.10f gauc=%.10f uauc=%.10f valid_groups=%d grouped_samples=%d"
            % (
                model_name,
                cvr_report[model_name]["auc"],
                gauc,
                uauc,
                groups,
                grouped_samples,
            )
        )
    cvr_report["delta_pal_minus_baseline"] = {
        key: cvr_report["pal"][key] - cvr_report["baseline"][key]
        for key in ("auc", "gauc", "uauc")
    }
    print(
        "PAL_MINUS_BASELINE auc=%+.3f‰ gauc=%+.3f‰ uauc=%+.3f‰"
        % tuple(
            cvr_report["delta_pal_minus_baseline"][key] * 1000.0
            for key in ("auc", "gauc", "uauc")
        )
    )
    report["cvr_given_click"] = cvr_report

    with open(os.path.join(args.output_dir, "posthoc_report.json"), "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "within_position_auc.tsv"), "w") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("task", "model", "stratum_type", "stratum", "samples", "positives", "auc", "pair_weight")
        )
        writer.writerows(position_rows)
    print("OUTPUT_DIR %s" % os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
