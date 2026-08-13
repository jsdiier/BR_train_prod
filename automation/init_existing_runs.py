#!/usr/bin/env python3
import argparse
import json
import os
import sys

from common import atomic_json, git_head, load_json, now, related_processes
from validate_experiment import validate


def resolved(config, entry):
    value = dict(config["defaults"])
    value["baseline"] = config.get("default_baseline")
    value.update(entry)
    return value


def main():
    parser = argparse.ArgumentParser(description="Adopt existing experiment directories without submitting jobs")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    candidates = []
    for entry in config["experiments"]:
        branch = entry["branch"]
        exp_dir = os.path.join(os.path.abspath(args.base_dir), branch)
        if not os.path.isdir(exp_dir):
            print("[INVALID] %s: directory missing" % branch)
            continue
        try:
            commit = git_head(exp_dir)
        except Exception as exc:
            print("[INVALID] %s: cannot read git HEAD: %s" % (branch, exc))
            continue
        exp_config = resolved(config, entry)
        report = validate(exp_dir, exp_config)
        processes = related_processes(exp_dir)
        if report["ok"]:
            status, label = "done", "DONE_CANDIDATE"
        elif processes:
            status, label = "running", "RUNNING_CANDIDATE"
        else:
            status, label = "failed", "FAILED_CANDIDATE"
        run_id = "%s@%s@attempt_1" % (branch, commit[:12])
        state = {
            "schema_version": 1, "run_id": run_id, "branch": branch,
            "commit": commit, "attempt": 1, "baseline": exp_config["baseline"],
            "status": status, "adopted": True, "job_uuid": None,
            "require_inference_benchmark": exp_config.get("require_inference_benchmark", True),
            "config": exp_config, "experiment_dir": exp_dir, "created_at": now(),
            "updated_at": now(), "processes": processes, "validation": report,
            "failure_reason": None if status != "failed" else "; ".join(report["errors"])
        }
        print("[%s] %s commit=%s processes=%d errors=%s" % (
            label, branch, commit[:12], len(processes),
            "none" if report["ok"] else " | ".join(report["errors"])))
        candidates.append(state)

    if args.dry_run:
        print("[DRY-RUN] no state files were written")
        return 0

    runs_dir = os.path.join(args.state_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    members = []
    for state in candidates:
        path = os.path.join(runs_dir, state["run_id"] + ".json")
        if os.path.exists(path):
            print("[SKIP] state already exists: %s" % path)
        else:
            atomic_json(path, state)
        members.append(state["run_id"])
    batch_id = "batch_%s_initial" % __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    batch = {"schema_version": 1, "batch_id": batch_id, "created_at": now(),
             "baseline": config["default_baseline"], "members": members,
             "status": "waiting", "result_dir": None}
    batch_path = os.path.join(args.state_dir, "batches", batch_id + ".json")
    atomic_json(batch_path, batch)
    atomic_json(os.path.join(args.state_dir, "active_batch.json"), {"batch_id": batch_id})
    print("[APPLY] initialized %d runs and %s" % (len(members), batch_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
