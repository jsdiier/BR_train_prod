#!/usr/bin/env python3
"""Probe the stability of BUY = CTR**alpha * CVR across multiple days."""

import argparse
import csv
import gzip
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint-day", required=True)
    parser.add_argument("--days", required=True, help="Comma-separated YYYYMMDD days")
    parser.add_argument("--alpha-start", type=float, default=0.8)
    parser.add_argument("--alpha-end", type=float, default=3.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).resolve().parent.parent),
        help="Directory containing experiment folders",
    )
    parser.add_argument(
        "--hdfs-root",
        default="/user/prod_soda_trade_strategy/rank/jiazhuo/hash_fea_new/train",
    )
    parser.add_argument("--hadoop-bin", default="/usr/local/hadoop-current/bin/hadoop")
    parser.add_argument("--gpu-id", default="0")
    return parser.parse_args()


def first_hdfs_part(hadoop_bin, hdfs_root, day):
    pattern = "%s/%s/part-*.tfrecord.gz" % (hdfs_root.rstrip("/"), day)
    result = subprocess.run(
        [hadoop_bin, "fs", "-ls", pattern],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("HDFS ls failed for %s: %s" % (day, result.stderr.strip()))
    paths = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[-1].endswith(".tfrecord.gz"):
            paths.append(fields[-1])
    if not paths:
        raise RuntimeError("No TFRecord part found for day %s" % day)
    return sorted(paths)[0]


def download_part(hadoop_bin, remote_path, local_path):
    if local_path.exists() and local_path.stat().st_size > 0:
        try:
            with gzip.open(str(local_path), "rb") as stream:
                stream.read(1)
            print("[CACHE] %s" % local_path, flush=True)
            return
        except OSError:
            local_path.unlink()

    temp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    print("[DOWNLOAD] %s -> %s" % (remote_path, local_path), flush=True)
    with temp_path.open("wb") as output:
        result = subprocess.run(
            [hadoop_bin, "fs", "-cat", remote_path],
            stdout=output,
        )
    if result.returncode != 0:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError("HDFS cat failed: %s" % remote_path)
    with gzip.open(str(temp_path), "rb") as stream:
        stream.read(1)
    temp_path.replace(local_path)


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.alpha_step <= 0 or args.alpha_end < args.alpha_start:
        raise ValueError("Invalid alpha range")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_id)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import log_loss, roc_auc_score

    tf.get_logger().setLevel("ERROR")

    base_dir = Path(args.base_dir).expanduser().resolve()
    experiment_dir = base_dir / args.experiment
    if not experiment_dir.is_dir():
        raise RuntimeError("Experiment directory not found: %s" % experiment_dir)

    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = experiment_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    days = [value.strip() for value in args.days.split(",") if value.strip()]
    if not days:
        raise ValueError("--days is empty")

    local_files = {}
    for day in days:
        remote = first_hdfs_part(args.hadoop_bin, args.hdfs_root, day)
        local_path = cache_dir / (day + ".tfrecord.gz")
        download_part(args.hadoop_bin, remote, local_path)
        local_files[day] = local_path

    os.chdir(str(experiment_dir))
    sys.path.insert(0, str(experiment_dir))
    from model import Model

    checkpoint_dir = experiment_dir / "model" / "checkpoints" / args.checkpoint_day
    latest_checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
    if latest_checkpoint is None:
        raise RuntimeError("Checkpoint not found: %s" % checkpoint_dir)

    feature_spec = {
        "cvr_label": tf.io.FixedLenFeature([], tf.float32),
        "cat_label": tf.io.FixedLenFeature([], tf.float32),
        "clk_label": tf.io.FixedLenFeature([], tf.float32),
        "ext_label": tf.io.FixedLenFeature([], tf.float32),
        "fea_ids": tf.io.VarLenFeature(tf.int64),
        "fea_vals": tf.io.VarLenFeature(tf.int64),
    }

    def make_dataset(path):
        def parse_record(serialized):
            return tf.io.parse_single_example(serialized, feature_spec)

        dataset = tf.data.TFRecordDataset(
            [str(path)], compression_type="GZIP", buffer_size=16 * 1024 * 1024
        )
        dataset = dataset.map(parse_record, num_parallel_calls=2)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.batch(args.batch_size, drop_remainder=False)
        return dataset.prefetch(1)

    first_batch = next(iter(make_dataset(local_files[days[0]])))
    model = Model(training=False, pred=True)
    model([first_batch["fea_ids"], first_batch["fea_vals"]])

    # Model creates an Adam optimizer in __init__, and Keras registers it as a
    # trackable child.  A model-only Checkpoint would otherwise still discover
    # that child and recreate every Adam slot while restoring, which is both
    # unnecessary for inference and can consume several times the model memory.
    if hasattr(model, "_delete_tracking"):
        model._delete_tracking("optimizer")
    object.__setattr__(model, "optimizer", None)
    tracked_children = model._trackable_children()
    if "optimizer" in tracked_children:
        raise RuntimeError("Failed to detach optimizer from inference model")

    tf.train.Checkpoint(model=model).restore(latest_checkpoint).expect_partial()
    model.training = False
    model.pred = True
    print("[MODEL] restored %s" % latest_checkpoint, flush=True)

    cache = {}
    aggregate = {key: [] for key in ("buy", "cat", "click", "ctr", "cvr")}
    for day in days:
        parts = {key: [] for key in aggregate}
        sample_count = 0
        for batch_index, feat in enumerate(make_dataset(local_files[day]), 1):
            _, cvr, ctr, _, _ = model([feat["fea_ids"], feat["fea_vals"]])
            values = {
                "buy": feat["cvr_label"].numpy().reshape(-1),
                "cat": feat["cat_label"].numpy().reshape(-1),
                "click": feat["clk_label"].numpy().reshape(-1),
                "ctr": ctr.numpy().reshape(-1),
                "cvr": cvr.numpy().reshape(-1),
            }
            for key, value in values.items():
                parts[key].append(value)
            sample_count += len(values["buy"])
            if batch_index % 500 == 0:
                print("[PROGRESS] day=%s samples=%d" % (day, sample_count), flush=True)

        if not parts["buy"]:
            raise RuntimeError("No records parsed for day %s" % day)
        cache[day] = {}
        for key, values in parts.items():
            merged = np.concatenate(values).astype(np.float64)
            cache[day][key] = merged
            aggregate[key].append(merged)
        print(
            "[DAY] %s samples=%d buy_pos=%d buy_rate=%.6f"
            % (day, len(cache[day]["buy"]), cache[day]["buy"].sum(), cache[day]["buy"].mean()),
            flush=True,
        )

    cache["ALL"] = {key: np.concatenate(values) for key, values in aggregate.items()}
    alphas = np.round(
        np.arange(args.alpha_start, args.alpha_end + args.alpha_step * 0.5, args.alpha_step), 10
    )
    epsilon = 1e-7

    def auc(labels, scores):
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))

    rows = []
    for scope, data in cache.items():
        buy = data["buy"]
        log_ctr = np.log(np.clip(data["ctr"], epsilon, 1.0))
        log_cvr = np.log(np.clip(data["cvr"], epsilon, 1.0))
        baseline = np.clip(np.exp(log_ctr + log_cvr), epsilon, 1.0 - epsilon)
        baseline_auc = auc(buy, baseline)
        click_mask = data["click"] == 1
        cat_mask = data["cat"] == 1
        for alpha in alphas:
            score = np.clip(np.exp(alpha * log_ctr + log_cvr), epsilon, 1.0 - epsilon)
            all_auc = auc(buy, score)
            rows.append(
                {
                    "scope": scope,
                    "alpha": float(alpha),
                    "samples": len(buy),
                    "buy_pos": int(buy.sum()),
                    "buy_rate": float(buy.mean()),
                    "all_auc": all_auc,
                    "all_delta_permille": (all_auc - baseline_auc) * 1000.0,
                    "click1_auc": auc(buy[click_mask], score[click_mask]),
                    "cat1_auc": auc(buy[cat_mask], score[cat_mask]),
                    "logloss": float(log_loss(buy, score, labels=[0, 1])),
                    "mean_pred": float(score.mean()),
                    "calibration_gap": float(score.mean() - buy.mean()),
                }
            )

    result_path = output_dir / "probe_ctr_alpha.tsv"
    with result_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.tsv"
    summary_rows = []
    print("\nscope\tbest_alpha\tall_auc\tdelta_permille\tclick1_auc\tcat1_auc\tlogloss\tmean_pred\tcalibration_gap")
    for scope in days + ["ALL"]:
        candidates = [row for row in rows if row["scope"] == scope]
        best = max(candidates, key=lambda row: row["all_auc"])
        summary_rows.append(best)
        print(
            "%s\t%.2f\t%.6f\t%+.3f\t%.6f\t%.6f\t%.6f\t%.6f\t%+.6f"
            % (
                scope,
                best["alpha"],
                best["all_auc"],
                best["all_delta_permille"],
                best["click1_auc"],
                best["cat1_auc"],
                best["logloss"],
                best["mean_pred"],
                best["calibration_gap"],
            ),
            flush=True,
        )
    with summary_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n[RESULT] %s" % result_path, flush=True)
    print("[SUMMARY] %s" % summary_path, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("[ERROR] %s: %s" % (type(error).__name__, error), file=sys.stderr, flush=True)
        raise
