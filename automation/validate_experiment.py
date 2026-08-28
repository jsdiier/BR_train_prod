#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
import sys

from common import dates, next_day


TASKS = {"buy", "cat", "click", "ext"}
LOG_RE = re.compile(
    r"(?:rolling|fixed)_test_ckpt_(\d{8})_from_(\d{8})_to_(\d{8})_(\d{12,14})$"
)
METRIC_RE = re.compile(
    r"test_(buy|cat|click|ext)\s+auc:([\d.]+)\s+gauc:([\d.]+)\s+"
    r"uauc:([\d.]+).*?size:(\d+).*?pos:\s*(\d+)"
)
PERF_CONFIG_RE = re.compile(
    r"\[INFERENCE_BENCHMARK\]\s+device:(.*?)\s+batch_size:(\d+)\s+"
    r"warmup_batches:(\d+)\s+measure_batches:(\d+)\s+samples:(\d+)"
)


def checkpoint_ok(exp_dir, day):
    root = os.path.join(exp_dir, "model", "checkpoints", day)
    return (os.path.isfile(os.path.join(root, "checkpoint"))
            and bool(glob.glob(os.path.join(root, "*.index")))
            and bool(glob.glob(os.path.join(root, "*.data-*"))))


def latest_logs(exp_dir):
    selected = {}
    candidates = glob.glob(os.path.join(exp_dir, "log", "rolling_test_ckpt_*"))
    candidates.extend(glob.glob(os.path.join(exp_dir, "log", "fixed_test_ckpt_*")))
    for path in candidates:
        match = LOG_RE.match(os.path.basename(path))
        if not match or not os.path.isfile(path):
            continue
        ckpt, start, end, stamp = match.groups()
        key = (ckpt, start, end)
        if key not in selected or stamp > selected[key][0]:
            selected[key] = (stamp, path)
    return {key: value[1] for key, value in selected.items()}


def parse_log(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read()
    rows = {}
    for task, auc, gauc, uauc, size, pos in METRIC_RE.findall(content):
        rows[task] = {
            "auc": float(auc), "gauc": float(gauc), "uauc": float(uauc),
            "size": int(size), "pos": int(pos)
        }
    perf_complete = all(token in content for token in (
        "[INFERENCE_BENCHMARK] device:",
        "[INFERENCE_BENCHMARK] model throughput_samples_s:",
        "[INFERENCE_BENCHMARK] end_to_end throughput_samples_s:",
    ))
    configs = list(PERF_CONFIG_RE.finditer(content))
    perf_config = None
    if configs:
        match = configs[-1]
        perf_config = {"device": match.group(1).strip(), "batch_size": int(match.group(2)),
                       "warmup_batches": int(match.group(3)),
                       "measure_batches": int(match.group(4)), "samples": int(match.group(5))}
    return rows, perf_complete, perf_config


def configured_days(config, key):
    value = config.get(key, [])
    return set(value if isinstance(value, list) else [])


def rolling_expectations(seed, rolling_end, missing_train_days,
                         missing_test_days):
    """Return real checkpoints and test->latest-prior-checkpoint relationships."""
    required_ckpts = [seed]
    expected_logs = []
    latest_ckpt = seed
    for test_day in dates(next_day(seed), rolling_end):
        if test_day not in missing_test_days:
            expected_logs.append((latest_ckpt, test_day, test_day))
        if test_day < rolling_end and test_day not in missing_train_days:
            required_ckpts.append(test_day)
            latest_ckpt = test_day
    return required_ckpts, expected_logs


def validate(exp_dir, config):
    result = {"ok": False, "checks": {}, "errors": [], "artifacts": {}}
    train_end = config["train_end_day"]
    rolling_enabled = config.get("rolling_enabled", True) is not False
    seed = config.get("auto_test_start_ckpt_day")
    rolling_end = config.get("auto_test_end_day")
    missing_train_days = configured_days(config, "allowed_missing_train_days")
    missing_test_days = configured_days(config, "allowed_missing_test_days")

    required_ckpts = [train_end]
    rolling_expected = []
    if rolling_enabled:
        rolling_ckpts, rolling_expected = rolling_expectations(
            seed, rolling_end, missing_train_days, missing_test_days)
        required_ckpts.extend(rolling_ckpts)
    required_ckpts = sorted(set(required_ckpts))
    missing_ckpts = [day for day in required_ckpts if not checkpoint_ok(exp_dir, day)]
    result["checks"]["checkpoints"] = not missing_ckpts
    if missing_ckpts:
        result["errors"].append("missing checkpoints: " + ",".join(missing_ckpts))

    logs = latest_logs(exp_dir)
    fixed_key = (train_end, config["test_start_day"], config["test_end_day"])
    expected = [fixed_key]
    if rolling_enabled:
        expected.extend(rolling_expected)
    missing_logs, invalid_logs = [], []
    fixed_perf, fixed_perf_config = False, None
    for key in expected:
        path = logs.get(key)
        if not path:
            missing_logs.append("/".join(key))
            continue
        metrics, perf, perf_config = parse_log(path)
        if set(metrics) != TASKS:
            invalid_logs.append("%s tasks=%s" % ("/".join(key), sorted(metrics)))
        if key == fixed_key:
            fixed_perf = perf
            fixed_perf_config = perf_config
            result["artifacts"]["fixed_log"] = path
    result["checks"]["fixed_log_parseable"] = fixed_key in logs and not any(
        item.startswith("/".join(fixed_key)) for item in invalid_logs)
    if rolling_enabled:
        result["checks"]["rolling_logs_complete"] = not missing_logs and not invalid_logs
    if missing_logs:
        result["errors"].append("missing test logs: " + ",".join(missing_logs))
    if invalid_logs:
        result["errors"].append("invalid test logs: " + ";".join(invalid_logs))

    if rolling_enabled:
        summary = os.path.join(exp_dir, "model", "rolling_metrics.tsv")
        summary_keys = set()
        try:
            with open(summary, "r", encoding="utf-8", errors="replace") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    summary_keys.add((row["checkpoint_day"], row["test_start_day"],
                                      row["test_end_day"], row["task"]))
            wanted = {(a, b, c, task) for a, b, c in expected for task in TASKS}
            summary_ok = wanted.issubset(summary_keys)
        except (OSError, KeyError, TypeError):
            summary_ok = False
        result["checks"]["rolling_summary_complete"] = summary_ok
        last_expected_test_day = rolling_expected[-1][2] if rolling_expected else None
        result["checks"]["last_test_day_correct"] = bool(
            last_expected_test_day and
            any(key[2] == last_expected_test_day for key in summary_keys))
        result["artifacts"]["rolling_metrics"] = summary
        if not summary_ok:
            result["errors"].append("rolling_metrics.tsv is missing required rows")
    else:
        result["checks"]["last_test_day_correct"] = (
            fixed_key in logs and fixed_key[2] == config["test_end_day"])
        result["artifacts"]["rolling_metrics"] = None

    require_perf = bool(config.get("require_inference_benchmark", True))
    result["checks"]["inference_benchmark_complete"] = fixed_perf or not require_perf
    result["artifacts"]["inference_benchmark_config"] = fixed_perf_config
    result["artifacts"]["allowed_missing_train_days"] = sorted(missing_train_days)
    result["artifacts"]["allowed_missing_test_days"] = sorted(missing_test_days)
    if require_perf and not fixed_perf:
        result["errors"].append("fixed-window inference benchmark is incomplete")

    result["ok"] = all(result["checks"].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--config-json", required=True,
                        help="JSON object or path containing resolved experiment config")
    args = parser.parse_args()
    if os.path.isfile(args.config_json):
        with open(args.config_json, encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        config = json.loads(args.config_json)
    result = validate(os.path.abspath(args.experiment_dir), config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
