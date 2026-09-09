#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare baseline and Top1700 on organic/ad segments and mixed requests."""

from __future__ import print_function

import argparse
import csv
import json
import math
import os

import numpy as np
from sklearn.metrics import roc_auc_score


TASKS = ("buy", "cat", "click", "ext")
MODELS = ("baseline", "top1700")
SEGMENTS = ("organic", "ad")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--top1700", required=True)
    parser.add_argument("--ad-fid", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def auc(labels, scores):
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def logloss(labels, scores):
    if labels.size == 0:
        return float("nan")
    clipped = np.clip(scores.astype(np.float64), 1e-7, 1.0 - 1e-7)
    labels = labels.astype(np.float64)
    return float(np.mean(-(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))))


def finite(value):
    return None if not math.isfinite(value) else value


def segment_metrics(labels, scores, mask):
    selected_labels = labels[mask]
    selected_scores = scores[mask]
    positives = int(np.sum(selected_labels))
    samples = int(selected_labels.size)
    return {
        "samples": samples,
        "positives": positives,
        "positive_rate": float(positives) / samples if samples else None,
        "auc": finite(auc(selected_labels, selected_scores)),
        "logloss": finite(logloss(selected_labels, selected_scores)),
        "score_mean": float(np.mean(selected_scores)) if samples else None,
        "score_p10": float(np.percentile(selected_scores, 10)) if samples else None,
        "score_p50": float(np.percentile(selected_scores, 50)) if samples else None,
        "score_p90": float(np.percentile(selected_scores, 90)) if samples else None,
    }


def request_gap(request_hash, is_ad, scores):
    order = np.argsort(request_hash, kind="stable")
    request_hash = request_hash[order]
    is_ad = is_ad[order].astype(np.float64)
    scores = scores[order].astype(np.float64)
    _, starts = np.unique(request_hash, return_index=True)
    ad_count = np.add.reduceat(is_ad, starts)
    organic_count = np.add.reduceat(1.0 - is_ad, starts)
    ad_sum = np.add.reduceat(scores * is_ad, starts)
    organic_sum = np.add.reduceat(scores * (1.0 - is_ad), starts)
    valid = (ad_count > 0) & (organic_count > 0)
    gaps = ad_sum[valid] / ad_count[valid] - organic_sum[valid] / organic_count[valid]
    if gaps.size == 0:
        return {"mixed_requests": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "mixed_requests": int(gaps.size),
        "mean": float(np.mean(gaps)),
        "p10": float(np.percentile(gaps, 10)),
        "p50": float(np.percentile(gaps, 50)),
        "p90": float(np.percentile(gaps, 90)),
    }


def assert_aligned(baseline, top1700):
    keys = ("sample_hash", "request_hash", "uid_hash", "ad_fid")
    keys += tuple(task + "_label" for task in TASKS)
    for key in keys:
        if baseline[key].shape != top1700[key].shape or not np.array_equal(
            baseline[key], top1700[key]
        ):
            raise RuntimeError("prediction files are not aligned at key=%s" % key)


def delta(candidate, baseline):
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def fmt(value, scale=1.0, digits=6):
    if value is None:
        return "NA"
    return ("%%+.%df" % digits) % (value * scale)


def main():
    args = arguments()
    baseline = np.load(args.baseline, allow_pickle=False)
    top1700 = np.load(args.top1700, allow_pickle=False)
    assert_aligned(baseline, top1700)

    unique_fids = np.unique(baseline["ad_fid"])
    if unique_fids.size != 2 or args.ad_fid not in unique_fids:
        raise RuntimeError(
            "--ad-fid must be one of the two observed FIDs: %s" % unique_fids.tolist()
        )
    is_ad = baseline["ad_fid"] == args.ad_fid
    masks = {"organic": ~is_ad, "ad": is_ad}

    summary = {
        "contract": {
            "baseline": json.loads(baseline["metadata_json"].item()),
            "top1700": json.loads(top1700["metadata_json"].item()),
            "ad_slot_id": 1044,
            "ad_fid": args.ad_fid,
            "observed_fids": [int(value) for value in unique_fids],
            "aligned_samples": int(is_ad.size),
        },
        "segments": {},
        "request_ad_minus_organic_score": {},
    }

    for task in TASKS:
        summary["segments"][task] = {}
        labels = baseline[task + "_label"]
        for segment in SEGMENTS:
            summary["segments"][task][segment] = {}
            mask = masks[segment]
            for model_name, source in (("baseline", baseline), ("top1700", top1700)):
                summary["segments"][task][segment][model_name] = segment_metrics(
                    labels, source[task + "_score"], mask
                )
            base_metrics = summary["segments"][task][segment]["baseline"]
            candidate_metrics = summary["segments"][task][segment]["top1700"]
            summary["segments"][task][segment]["delta"] = {
                key: delta(candidate_metrics[key], base_metrics[key])
                for key in ("auc", "logloss", "score_mean", "score_p10", "score_p50", "score_p90")
            }

        summary["request_ad_minus_organic_score"][task] = {}
        for model_name, source in (("baseline", baseline), ("top1700", top1700)):
            summary["request_ad_minus_organic_score"][task][model_name] = request_gap(
                baseline["request_hash"], is_ad, source[task + "_score"]
            )
        base_gap = summary["request_ad_minus_organic_score"][task]["baseline"]
        candidate_gap = summary["request_ad_minus_organic_score"][task]["top1700"]
        summary["request_ad_minus_organic_score"][task]["delta"] = {
            key: delta(candidate_gap[key], base_gap[key]) for key in ("mean", "p10", "p50", "p90")
        }

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    json_path = os.path.join(args.output_dir, "ad_segment_summary.json")
    with open(json_path, "w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    tsv_path = os.path.join(args.output_dir, "ad_segment_metrics.tsv")
    with open(tsv_path, "w") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "task", "segment", "samples", "positives", "positive_rate",
                "baseline_auc", "top1700_auc", "delta_auc",
                "baseline_logloss", "top1700_logloss", "delta_logloss",
                "baseline_score_mean", "top1700_score_mean", "delta_score_mean",
            ]
        )
        for task in TASKS:
            for segment in SEGMENTS:
                item = summary["segments"][task][segment]
                base = item["baseline"]
                candidate = item["top1700"]
                writer.writerow(
                    [
                        task, segment, base["samples"], base["positives"], base["positive_rate"],
                        base["auc"], candidate["auc"], item["delta"]["auc"],
                        base["logloss"], candidate["logloss"], item["delta"]["logloss"],
                        base["score_mean"], candidate["score_mean"], item["delta"]["score_mean"],
                    ]
                )

    print("=" * 100)
    print("AD / ORGANIC SEGMENT COMPARISON")
    print("aligned_samples=%d ad_fid=%d observed_fids=%s" % (is_ad.size, args.ad_fid, unique_fids.tolist()))
    for task in TASKS:
        print("\nTASK=%s" % task.upper())
        print("segment   samples    positives  baseline_auc top1700_auc delta_auc")
        for segment in SEGMENTS:
            item = summary["segments"][task][segment]
            base = item["baseline"]
            candidate = item["top1700"]
            print(
                "%-9s %-10d %-10d %-12s %-11s %s"
                % (
                    segment,
                    base["samples"],
                    base["positives"],
                    "NA" if base["auc"] is None else "%.6f" % base["auc"],
                    "NA" if candidate["auc"] is None else "%.6f" % candidate["auc"],
                    fmt(item["delta"]["auc"], scale=1000.0, digits=3) + " permille",
                )
            )
        gap = summary["request_ad_minus_organic_score"][task]
        print(
            "request ad-organic mean score gap: baseline=%s top1700=%s delta=%s mixed_requests=%d"
            % (
                fmt(gap["baseline"]["mean"], digits=7),
                fmt(gap["top1700"]["mean"], digits=7),
                fmt(gap["delta"]["mean"], digits=7),
                gap["baseline"]["mixed_requests"],
            )
        )
    print("\nOUTPUT_JSON %s" % json_path)
    print("OUTPUT_TSV %s" % tsv_path)


if __name__ == "__main__":
    main()
