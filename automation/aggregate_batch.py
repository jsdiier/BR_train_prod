#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

from common import atomic_json, load_json, now


def write_structured_metrics(base_dir, baseline, states, out, test_end_day):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import test_multi
    names = [baseline] + [state["branch"] for state in states if state["branch"] != baseline]
    records, perf_records = [], []
    baseline_perf = None
    for name in names:
        exp_dir = os.path.join(base_dir, name)
        log = test_multi.find_latest_test_log(exp_dir, dt=test_end_day)
        metrics = test_multi.parse_test_log(log)
        perf = test_multi.parse_inference_benchmark(log)
        if name == baseline:
            baseline_perf = perf
        for task, values in metrics.items():
            records.append({"experiment": name, "task": task, **values, "log_path": log})
        perf_records.append({"experiment": name, **perf, "log_path": log})
    with open(os.path.join(out, "fixed_window.tsv"), "w", newline="", encoding="utf-8") as handle:
        fields = ["experiment", "task", "auc", "gauc", "uauc", "pos", "log_path"]
        writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    perf_fields = ["experiment", "device", "batch_size", "warmup_batches", "measure_batches",
                   "samples", "model_throughput", "model_latency_sample", "model_p50",
                   "model_p95", "e2e_throughput", "e2e_p50", "e2e_p95", "log_path"]
    with open(os.path.join(out, "inference_benchmark.tsv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, perf_fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(perf_records)


def write_combined_rolling(base_dir, baseline, states, out):
    rows = []
    sources = [(baseline, os.path.join(base_dir, baseline))]
    sources.extend((state["branch"], state["experiment_dir"])
                   for state in states if state["branch"] != baseline)
    for branch, exp_dir in sources:
        source = os.path.join(exp_dir, "model", "rolling_metrics.tsv")
        if not os.path.isfile(source):
            continue
        with open(source, "r", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows.append({"experiment": branch, **row})
        shutil.copy2(source, os.path.join(out, branch + "_rolling_metrics.tsv"))
    if rows:
        fields = ["experiment", "checkpoint_day", "test_start_day", "test_end_day", "task",
                  "auc", "gauc", "uauc", "size", "pos", "log_path"]
        with open(os.path.join(out, "rolling_auc.tsv"), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)


def run_capture(command, output):
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(proc.stdout)
    if proc.returncode:
        raise RuntimeError("command failed (%d): %s; see %s" % (
            proc.returncode, " ".join(command), output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--tools-dir", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    if not args.no_push:
        subprocess.run(["git", "pull", "--rebase"], cwd=args.tools_dir, check=True)
    batch = load_json(args.batch)
    states = [load_json(os.path.join(args.state_dir, "runs", run_id + ".json"))
              for run_id in batch["members"]]
    done = [state for state in states if state["status"] == "done"]
    failed = [state for state in states if state["status"] == "failed"]
    baseline_dir = os.path.join(args.base_dir, batch["baseline"])
    baseline_config = done[0]["config"] if done else states[0]["config"]
    from validate_experiment import validate
    baseline_validation = validate(baseline_dir, baseline_config)
    baseline_perf = baseline_validation["artifacts"].get("inference_benchmark_config")
    performance_mismatch = []
    for state in done:
        if state.get("require_inference_benchmark"):
            current = state["validation"]["artifacts"].get("inference_benchmark_config")
            if current != baseline_perf:
                performance_mismatch.append(state["branch"])
    if not baseline_validation["ok"]:
        raise RuntimeError("baseline validation failed: %s" % "; ".join(baseline_validation["errors"]))
    if performance_mismatch:
        print("[BATCH] warning: inference benchmark config differs from baseline; "
              "fixed/rolling metrics will still be aggregated, but inference performance "
              "is not directly comparable: %s" % ",".join(performance_mismatch))
    out = os.path.join(args.tools_dir, "result", batch["batch_id"])
    os.makedirs(os.path.join(out, "failures"), exist_ok=True)

    manifest = {"batch_id": batch["batch_id"], "baseline": batch["baseline"],
                "generated_at": now(), "baseline_validation": baseline_validation,
                "inference_benchmark_config_mismatch": performance_mismatch,
                "members": states}
    atomic_json(os.path.join(out, "manifest.json"), manifest)
    with open(os.path.join(out, "status.tsv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["run_id", "branch", "commit", "status", "failure_reason"])
        for state in states:
            writer.writerow([state["run_id"], state["branch"], state["commit"],
                             state["status"], state.get("failure_reason") or ""])
    with open(os.path.join(out, "validation.tsv"), "w", newline="", encoding="utf-8") as handle:
        checks = sorted({key for state in states for key in state["validation"]["checks"]})
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["branch"] + checks)
        for state in states:
            writer.writerow([state["branch"]] + [
                "yes" if state["validation"]["checks"].get(key) else "no" for key in checks])
    for state in failed:
        with open(os.path.join(out, "failures", state["branch"] + ".txt"),
                  "w", encoding="utf-8") as handle:
            handle.write((state.get("failure_reason") or "unknown failure") + "\n")

    if done:
        branches = [state["branch"] for state in done if state["branch"] != batch["baseline"]]
        if branches:
            write_structured_metrics(args.base_dir, batch["baseline"], done, out,
                                     states[0]["config"]["test_end_day"])
            fixed = [sys.executable, os.path.join(args.tools_dir, "test_multi.py"),
                     batch["baseline"]] + branches + [states[0]["config"]["test_end_day"]]
            run_capture(fixed, os.path.join(out, "fixed_window.txt"))
            rolling = [sys.executable, os.path.join(args.tools_dir, "rolling_auc_compare.py"),
                       batch["baseline"]] + branches + ["--base-dir", args.base_dir,
                       "-o", os.path.join(out, "rolling_auc.html")]
            run_capture(rolling, os.path.join(out, "rolling_auc_generation.txt"))
            write_combined_rolling(args.base_dir, batch["baseline"], done, out)

    batch["status"] = "completed_with_failures" if failed else "completed"
    batch["result_dir"], batch["completed_at"] = out, now()
    atomic_json(args.batch, batch)
    active = os.path.join(args.state_dir, "active_batch.json")
    if os.path.isfile(active):
        os.unlink(active)
    if not args.no_push:
        subprocess.run(["git", "add", os.path.relpath(out, args.tools_dir)], cwd=args.tools_dir, check=True)
        subprocess.run(["git", "commit", "-m", "result: %s" % batch["batch_id"]],
                       cwd=args.tools_dir, check=True)
        subprocess.run(["git", "push"], cwd=args.tools_dir, check=True)
    print("[BATCH] %s -> %s" % (batch["batch_id"], batch["status"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
