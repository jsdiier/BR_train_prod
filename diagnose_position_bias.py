#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen-checkpoint diagnostic for residual position dependence."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

import utils as ut
from model import Model


TASKS = (
    ("buy", "cvr_label"),
    ("cat", "cat_label"),
    ("click", "clk_label"),
    ("ext", "ext_label"),
)
POSITION_NAMES = ("1-3", "4-6", "7-10", "11-20", "21+")
FIELD_COUNT = 24
HADOOP = "/usr/local/hadoop-current/bin/hadoop"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "/home/luban/rank-ssl/chenpinyuan/tf_rank_BR_prod/"
            "BR_train_prod_bs_lr_ema_weights/model/checkpoints/20260826"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=(
            "hdfs://DClusterUS1/user/prod_soda_trade_strategy/rank/"
            "jiazhuo/hash_fea_new/train"
        ),
    )
    parser.add_argument("--days", default="20260827,20260828,20260829")
    parser.add_argument("--parts-per-day", default="2,2,1")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--score-bins", type=int, default=20)
    parser.add_argument("--min-cell-samples", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default="log/pal_position_bias_diagnostic_ckpt_20260826",
    )
    return parser.parse_args()


def choose_parts(root, days, counts, seed):
    if len(days) != len(counts):
        raise RuntimeError("days/counts length mismatch")
    rng = random.Random(seed)
    selected = []
    for day, count in zip(days, counts):
        pattern = "%s/%s/part*" % (root.rstrip("/"), day)
        process = subprocess.run(
            [HADOOP, "fs", "-ls", pattern],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "hadoop fs -ls failed for %s:\n%s" % (pattern, process.stderr)
            )
        paths = []
        for line in process.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 8 and "/part-" in fields[-1]:
                paths.append((fields[-1], int(fields[4])))
        paths.sort(key=lambda item: item[0])
        if len(paths) < count:
            raise RuntimeError("day=%s found=%d requested=%d" % (day, len(paths), count))
        selected.extend((day, path, size) for path, size in sorted(rng.sample(paths, count)))
    return selected


def stage_hdfs_part(path, size, cache_dir, part_index, part_count):
    os.makedirs(cache_dir, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes < size + 1024 * 1024 * 1024:
        raise RuntimeError(
            "insufficient local space: free=%d required_at_least=%d"
            % (free_bytes, size + 1024 * 1024 * 1024)
        )
    local_path = os.path.join(cache_dir, "current_part.tfrecord.gz")
    if os.path.exists(local_path):
        os.remove(local_path)

    process = subprocess.Popen([HADOOP, "fs", "-cat", path], stdout=subprocess.PIPE)
    copied = 0
    started = time.time()
    with open(local_path, "wb") as output:
        while True:
            chunk = process.stdout.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            copied += len(chunk)
            ratio = min(copied / float(max(size, 1)), 1.0)
            speed = copied / max(time.time() - started, 0.001) / 1024.0 / 1024.0
            print(
                "\rstage part=%d/%d [%s%s] %6.2f%% %.1f/%.1f MB %.1f MB/s"
                % (
                    part_index,
                    part_count,
                    "#" * int(ratio * 30),
                    "-" * (30 - int(ratio * 30)),
                    ratio * 100.0,
                    copied / 1024.0 / 1024.0,
                    size / 1024.0 / 1024.0,
                    speed,
                ),
                end="",
            )
            sys.stdout.flush()
    process.stdout.close()
    return_code = process.wait()
    print()
    if return_code != 0:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise RuntimeError("hadoop fs -cat failed with return code %d: %s" % (return_code, path))
    if copied != size:
        raise RuntimeError("staged size mismatch: copied=%d hdfs_size=%d" % (copied, size))
    return local_path


def make_dataset(path, batch_size):
    return ut.ReadTFRecordV2(
        [path],
        shuffle_size=1,
        batch_size=batch_size,
        fetch_size=2,
        num_parallel=4,
    )


def restore_model(checkpoint_dir, first_batch):
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


def validation_trace(traceid):
    if not isinstance(traceid, bytes):
        traceid = str(traceid).encode("utf-8", errors="replace")
    return hashlib.sha1(traceid).digest()[0] < 51


def metadata(feat, batch_size):
    values = feat["add_info_list"].values.numpy()
    expected = batch_size * FIELD_COUNT
    if values.size != expected:
        raise RuntimeError(
            "add_info schema mismatch: actual=%d expected=%d" % (values.size, expected)
        )
    matrix = values.reshape(batch_size, FIELD_COUNT)
    if any(not value for value in matrix[:, 0]):
        raise RuntimeError("empty traceid detected; trace-disjoint split is unsafe")
    ranks = np.asarray([int(value.decode("utf-8")) for value in matrix[:, 3]], np.int32)
    if np.any(ranks < 0):
        raise RuntimeError("negative rank detected")
    is_validation = np.fromiter(
        (validation_trace(value) for value in matrix[:, 0]),
        dtype=np.bool_,
        count=batch_size,
    )
    return ranks, is_validation


def five_position_groups(rank):
    group = np.full(rank.shape, 4, np.int16)
    group[rank <= 2] = 0
    group[(rank >= 3) & (rank <= 5)] = 1
    group[(rank >= 6) & (rank <= 9)] = 2
    group[(rank >= 10) & (rank <= 19)] = 3
    return group


def pal_position_buckets(rank):
    """64 buckets: exact head, increasingly coarse tail."""
    bucket = np.empty(rank.shape, np.int16)
    bucket[rank <= 49] = rank[rank <= 49]
    mask = (rank >= 50) & (rank <= 99)
    bucket[mask] = 50 + (rank[mask] - 50) // 10
    mask = (rank >= 100) & (rank <= 199)
    bucket[mask] = 55 + (rank[mask] - 100) // 25
    mask = (rank >= 200) & (rank <= 299)
    bucket[mask] = 59 + (rank[mask] - 200) // 50
    bucket[(rank >= 300) & (rank <= 399)] = 61
    bucket[(rank >= 400) & (rank <= 499)] = 62
    bucket[rank >= 500] = 63
    return bucket


def logits(score):
    score = np.clip(score.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(score) - np.log1p(-score)


def auc(label, prediction):
    if label.size == 0 or np.unique(label).size < 2:
        return float("nan")
    return float(roc_auc_score(label, prediction))


def logloss(label, prediction):
    prediction = np.clip(prediction, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(label * np.log(prediction) + (1.0 - label) * np.log(1.0 - prediction)))


def fit_probe(x, y, position=None, initial=None):
    """Fit score calibration, optionally plus 63 anchored position biases."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    n = float(y.size)
    parameter_count = 2 if position is None else 65
    if initial is None:
        initial = np.zeros(parameter_count, np.float64)
        initial[0] = 1.0
    if position is not None:
        position = position.astype(np.int32)

    def objective(parameter):
        z = parameter[0] * x + parameter[1]
        if position is not None:
            bias = np.concatenate(([0.0], parameter[2:]))
            z = z + bias[position]
        error = (expit(z) - y) / n
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        gradient = np.zeros_like(parameter)
        gradient[0] = np.sum(error * x)
        gradient[1] = np.sum(error)
        if position is not None:
            gradient[2:] = np.bincount(position, weights=error, minlength=64)[1:]
            loss += 0.5e-5 * np.sum(parameter[2:] ** 2)
            gradient[2:] += 1e-5 * parameter[2:]
        return float(loss), gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError("probe fit failed: %s" % result.message)
    return result.x


def probe_prediction(parameter, x, position=None):
    z = parameter[0] * x + parameter[1]
    if position is not None:
        bias = np.concatenate(([0.0], parameter[2:]))
        z = z + bias[position]
    return expit(z)


def within_position_auc(label, score, group):
    rows = []
    weighted_sum = 0.0
    total_weight = 0.0
    for group_id, name in enumerate(POSITION_NAMES):
        mask = group == group_id
        group_label = label[mask]
        group_auc = auc(group_label, score[mask])
        positive = int(np.sum(group_label))
        negative = int(group_label.size - positive)
        weight = float(positive * negative)
        rows.append((name, int(group_label.size), positive, group_auc, weight))
        if np.isfinite(group_auc) and weight > 0:
            weighted_sum += group_auc * weight
            total_weight += weight
    return weighted_sum / total_weight, rows


def conditional_table(
    path,
    task,
    train_score,
    valid_score,
    valid_label,
    valid_calibrated,
    valid_position,
    score_bin_count,
    min_samples,
):
    edges = np.quantile(train_score, np.linspace(0.0, 1.0, score_bin_count + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    score_bin = np.searchsorted(edges[1:-1], valid_score, side="right")
    spreads = []
    with open(path, "a") as output:
        if output.tell() == 0:
            output.write(
                "task\tscore_bin\tposition\tsamples\tlabel_rate\tmean_score\t"
                "score_only_calibrated\tresidual\n"
            )
        for bin_id in range(score_bin_count):
            rates, weights = [], []
            for position_id, position_name in enumerate(POSITION_NAMES):
                mask = (score_bin == bin_id) & (valid_position == position_id)
                count = int(np.sum(mask))
                if count == 0:
                    continue
                rate = float(np.mean(valid_label[mask]))
                calibrated = float(np.mean(valid_calibrated[mask]))
                output.write(
                    "%s\t%d\t%s\t%d\t%.10f\t%.10f\t%.10f\t%.10f\n"
                    % (
                        task,
                        bin_id,
                        position_name,
                        count,
                        rate,
                        np.mean(valid_score[mask]),
                        calibrated,
                        rate - calibrated,
                    )
                )
                if count >= min_samples:
                    rates.append(rate)
                    weights.append(count)
            if len(rates) >= 2:
                spreads.append((max(rates) - min(rates), sum(weights)))
    if not spreads:
        return float("nan")
    return sum(value * weight for value, weight in spreads) / sum(weight for _, weight in spreads)


def main():
    args = arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    days = [value.strip() for value in args.days.split(",")]
    counts = [int(value) for value in args.parts_per_day.split(",")]
    parts = choose_parts(args.data_root, days, counts, args.seed)

    print("=" * 100)
    print("FROZEN BASELINE POSITION DIAGNOSTIC")
    print("checkpoint=%s" % args.checkpoint)
    for index, (day, path, size) in enumerate(parts, 1):
        print(
            "part=%d/%d day=%s size=%.1fMB path=%s"
            % (index, len(parts), day, size / 1024.0 / 1024.0, path)
        )
    print("=" * 100)

    with open(os.path.join(args.output_dir, "sampled_parts.tsv"), "w") as output:
        output.write("day\tpath\tsize_bytes\n")
        for day, path, size in parts:
            output.write("%s\t%s\t%d\n" % (day, path, size))

    cache_dir = os.path.join(args.output_dir, "cache")
    first_local_path = stage_hdfs_part(
        parts[0][1], parts[0][2], cache_dir, 1, len(parts)
    )
    first_batch = next(iter(make_dataset(first_local_path, args.batch_size)))
    model, checkpoint_path = restore_model(args.checkpoint, first_batch)

    arrays = {"rank": [], "validation": [], "day": []}
    for task, _ in TASKS:
        arrays[task + "_label"] = []
        arrays[task + "_score"] = []

    total = 0
    started = time.time()
    for part_index, (day, path, size) in enumerate(parts, 1):
        if part_index == 1:
            local_path = first_local_path
        else:
            local_path = stage_hdfs_part(path, size, cache_dir, part_index, len(parts))
        part_total = 0
        try:
            for batch_index, feat in enumerate(make_dataset(local_path, args.batch_size), 1):
                batch_size = int(feat["cvr_label"].shape[0])
                prediction = model([feat["fea_ids"], feat["fea_vals"]])
                rank, validation = metadata(feat, batch_size)
                arrays["rank"].append(rank)
                arrays["validation"].append(validation)
                arrays["day"].append(np.full(batch_size, int(day), np.int32))
                for task_index, (task, label_key) in enumerate(TASKS):
                    arrays[task + "_label"].append(
                        feat[label_key].numpy().reshape(-1).astype(np.float32)
                    )
                    arrays[task + "_score"].append(
                        prediction[task_index].numpy().reshape(-1).astype(np.float32)
                    )
                total += batch_size
                part_total += batch_size
                if batch_index % 20 == 0:
                    speed = total / max(time.time() - started, 0.001)
                    print(
                        "\rinference part=%d/%d day=%s batches=%d part_samples=%d "
                        "total=%d speed=%.1f samples/s"
                        % (part_index, len(parts), day, batch_index, part_total, total, speed),
                        end="",
                    )
                    sys.stdout.flush()
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)
        print("\npart completed %d/%d samples=%d" % (part_index, len(parts), part_total))

    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    np.savez_compressed(os.path.join(args.output_dir, "baseline_predictions.npz"), **arrays)
    train = ~arrays["validation"]
    valid = arrays["validation"]
    print("train_samples=%d validation_samples=%d" % (np.sum(train), np.sum(valid)))

    pal_bucket = pal_position_buckets(arrays["rank"])
    five_group = five_position_groups(arrays["rank"])
    conditional_path = os.path.join(args.output_dir, "conditional_position_rates.tsv")
    if os.path.exists(conditional_path):
        os.remove(conditional_path)
    rng = np.random.RandomState(args.seed)
    report = {"checkpoint": checkpoint_path, "samples": int(total), "tasks": {}}
    metrics_rows, daily_rows, within_rows = [], [], []

    for task, _ in TASKS:
        label = arrays[task + "_label"].astype(np.float64)
        score = arrays[task + "_score"].astype(np.float64)
        x = logits(score)
        score_parameter = fit_probe(x[train], label[train])
        position_parameter = fit_probe(
            x[train],
            label[train],
            pal_bucket[train],
            np.concatenate((score_parameter, np.zeros(63))),
        )
        valid_label, valid_score, valid_x = label[valid], score[valid], x[valid]
        valid_bucket = pal_bucket[valid]
        score_prediction = probe_prediction(score_parameter, valid_x)
        position_prediction = probe_prediction(position_parameter, valid_x, valid_bucket)
        shuffled_bucket = valid_bucket.copy()
        rng.shuffle(shuffled_bucket)
        shuffled_prediction = probe_prediction(position_parameter, valid_x, shuffled_bucket)

        score_auc = auc(valid_label, score_prediction)
        position_auc = auc(valid_label, position_prediction)
        score_loss = logloss(valid_label, score_prediction)
        position_loss = logloss(valid_label, position_prediction)
        weighted_auc, bucket_rows = within_position_auc(
            valid_label, valid_score, five_group[valid]
        )
        for row in bucket_rows:
            within_rows.append((task,) + row)
        rate_spread = conditional_table(
            conditional_path,
            task,
            score[train],
            valid_score,
            valid_label,
            score_prediction,
            five_group[valid],
            args.score_bins,
            args.min_cell_samples,
        )
        row = {
            "task": task,
            "global_auc": auc(valid_label, valid_score),
            "within_position_auc": weighted_auc,
            "score_only_auc": score_auc,
            "score_position_auc": position_auc,
            "delta_auc": position_auc - score_auc,
            "score_only_logloss": score_loss,
            "score_position_logloss": position_loss,
            "relative_logloss_reduction": (score_loss - position_loss) / score_loss,
            "shuffled_position_auc": auc(valid_label, shuffled_prediction),
            "shuffled_position_logloss": logloss(valid_label, shuffled_prediction),
            "same_score_rate_spread": rate_spread,
        }
        metrics_rows.append(row)
        print("\nTASK=%s" % task.upper())
        print("global_auc                       %.10f" % row["global_auc"])
        print("within_position_auc              %.10f" % row["within_position_auc"])
        print("score_plus_position_delta_auc    %+.3f‰" % (row["delta_auc"] * 1000.0))
        print("relative_logloss_reduction       %+.6f%%" % (row["relative_logloss_reduction"] * 100.0))
        print("same_score_rate_spread           %.10f" % row["same_score_rate_spread"])
        print("shuffled_position_auc            %.10f" % row["shuffled_position_auc"])

        daily_positive = True
        for day in sorted(np.unique(arrays["day"])):
            mask = valid & (arrays["day"] == day)
            day_score = probe_prediction(score_parameter, x[mask])
            day_position = probe_prediction(position_parameter, x[mask], pal_bucket[mask])
            daily = (
                task,
                int(day),
                int(np.sum(mask)),
                auc(label[mask], day_score),
                auc(label[mask], day_position),
                logloss(label[mask], day_score),
                logloss(label[mask], day_position),
            )
            daily_rows.append(daily)
            daily_positive = daily_positive and daily[4] > daily[3]
            print("day=%d delta_auc=%+.3f‰" % (day, (daily[4] - daily[3]) * 1000.0))
        report["tasks"][task] = dict(row)
        report["tasks"][task].pop("task")
        report["tasks"][task]["daily_delta_auc_all_positive"] = bool(daily_positive)

    buy = report["tasks"]["buy"]
    if (
        buy["delta_auc"] >= 0.001
        and buy["relative_logloss_reduction"] >= 0.005
        and buy["daily_delta_auc_all_positive"]
    ):
        decision = "STRONG_POSITION_RESIDUAL_PAL_WORTH_RUNNING"
    elif buy["delta_auc"] >= 0.0003 or buy["relative_logloss_reduction"] >= 0.002:
        decision = "MODERATE_POSITION_RESIDUAL_PAL_OPTIONAL"
    else:
        decision = "WEAK_POSITION_RESIDUAL_PAL_LOW_PRIORITY"
    report["decision"] = decision

    metric_columns = list(metrics_rows[0].keys())
    with open(os.path.join(args.output_dir, "probe_metrics.tsv"), "w") as output:
        output.write("\t".join(metric_columns) + "\n")
        for row in metrics_rows:
            output.write("\t".join(str(row[key]) for key in metric_columns) + "\n")

    with open(os.path.join(args.output_dir, "probe_metrics_by_day.tsv"), "w") as output:
        output.write(
            "task\tday\tsamples\tscore_only_auc\tscore_position_auc\t"
            "score_only_logloss\tscore_position_logloss\n"
        )
        for row in daily_rows:
            output.write("\t".join(str(value) for value in row) + "\n")

    with open(os.path.join(args.output_dir, "within_position_auc.tsv"), "w") as output:
        output.write("task\tposition\tsamples\tpositives\tauc\tpair_weight\n")
        for row in within_rows:
            output.write("\t".join(str(value) for value in row) + "\n")

    with open(os.path.join(args.output_dir, "diagnostic_report.json"), "w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "SUMMARY.txt"), "w") as output:
        output.write("decision: %s\n" % decision)
        output.write("samples: %d\n" % total)
        output.write("buy_delta_auc: %+.8f\n" % buy["delta_auc"])
        output.write(
            "buy_relative_logloss_reduction: %+.8f\n"
            % buy["relative_logloss_reduction"]
        )
        output.write(
            "buy_daily_delta_auc_all_positive: %s\n"
            % buy["daily_delta_auc_all_positive"]
        )

    print("\n" + "=" * 100)
    print("FINAL_DECISION %s" % decision)
    print("OUTPUT_DIR %s" % os.path.abspath(args.output_dir))
    print("=" * 100)


if __name__ == "__main__":
    main()
