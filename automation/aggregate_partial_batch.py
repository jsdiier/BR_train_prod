#!/usr/bin/env python3
"""Create an immutable closing snapshot for selected completed batch members.

Unlike aggregate_batch.py, this command never mutates run state, batch state, or
active_batch.json.  It is intended for closing a completed logical portfolio
while unrelated/user-mandated members of the source batch are still running.
"""

import argparse
import csv
import os
import re
import subprocess
import sys

from aggregate_batch import (
    run_capture,
    write_combined_rolling,
    write_structured_metrics,
)
from common import atomic_json, load_json, now
from validate_experiment import validate


def validation_complete(state):
    checks = state.get("validation", {}).get("checks", {})
    return state.get("status") == "done" and checks and all(checks.values())


def compact_state(state):
    return {
        "run_id": state.get("run_id"),
        "branch": state.get("branch"),
        "commit": state.get("commit"),
        "status": state.get("status"),
        "attempt": state.get("attempt"),
        "validation": state.get("validation"),
        "failure_reason": state.get("failure_reason"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--tools-dir", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--branch", action="append", dest="branches", required=True)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    if not re.match(r"^batch_[A-Za-z0-9_.-]+$", args.snapshot_id):
        raise RuntimeError("snapshot id must start with batch_ and contain only safe characters")

    if not args.no_push:
        subprocess.run(["git", "pull", "--rebase"], cwd=args.tools_dir, check=True)

    source_batch = load_json(args.batch)
    states = [
        load_json(os.path.join(args.state_dir, "runs", run_id + ".json"))
        for run_id in source_batch["members"]
    ]
    by_branch = {state["branch"]: state for state in states}

    duplicate_branches = sorted({name for name in args.branches
                                 if args.branches.count(name) > 1})
    if duplicate_branches:
        raise RuntimeError("duplicate --branch values: %s" % ",".join(duplicate_branches))

    missing = [name for name in args.branches if name not in by_branch]
    if missing:
        raise RuntimeError("branches not found in source batch: %s" % ",".join(missing))

    selected = [by_branch[name] for name in args.branches]
    not_ready = [state["branch"] for state in selected if not validation_complete(state)]
    if not_ready:
        raise RuntimeError("selected branches are not done with complete validation: %s" %
                           ",".join(not_ready))

    deferred = [state for state in states if state["branch"] not in set(args.branches)]
    baseline = source_batch["baseline"]
    baseline_dir = os.path.join(args.base_dir, baseline)
    baseline_config = selected[0]["config"]
    baseline_validation = validate(baseline_dir, baseline_config)
    if not baseline_validation["ok"]:
        raise RuntimeError("baseline validation failed: %s" %
                           "; ".join(baseline_validation["errors"]))

    baseline_perf = baseline_validation["artifacts"].get("inference_benchmark_config")
    performance_mismatch = []
    for state in selected:
        current = state["validation"]["artifacts"].get("inference_benchmark_config")
        if current != baseline_perf:
            performance_mismatch.append(state["branch"])

    final_out = os.path.join(args.tools_dir, "result", args.snapshot_id)
    if os.path.exists(final_out):
        raise RuntimeError("snapshot output already exists: %s" % final_out)
    out = final_out + ".tmp.%d" % os.getpid()
    os.makedirs(out)

    manifest = {
        "schema_version": 1,
        "batch_id": args.snapshot_id,
        "source_batch_id": source_batch["batch_id"],
        "snapshot_type": "partial_closing",
        "closing_scope": "agent_generated_completed_portfolio",
        "generated_at": now(),
        "evidence_cutoff": now(),
        "baseline": baseline,
        "source_batch_status": source_batch.get("status"),
        "baseline_validation": baseline_validation,
        "inference_benchmark_config_mismatch": performance_mismatch,
        "members": selected,
        "deferred_members": [compact_state(state) for state in deferred],
        "source_batch_untouched": True,
    }
    atomic_json(os.path.join(out, "manifest.json"), manifest)

    with open(os.path.join(out, "status.tsv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["run_id", "branch", "commit", "status", "failure_reason"])
        for state in selected:
            writer.writerow([state["run_id"], state["branch"], state["commit"],
                             state["status"], state.get("failure_reason") or ""])

    with open(os.path.join(out, "validation.tsv"), "w", newline="", encoding="utf-8") as handle:
        checks = sorted({key for state in selected
                         for key in state["validation"]["checks"]})
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["branch"] + checks)
        for state in selected:
            writer.writerow([state["branch"]] + [
                "yes" if state["validation"]["checks"].get(key) else "no"
                for key in checks
            ])

    with open(os.path.join(out, "deferred_members.tsv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["run_id", "branch", "commit", "status", "attempt"])
        for state in deferred:
            writer.writerow([state.get("run_id"), state.get("branch"), state.get("commit"),
                             state.get("status"), state.get("attempt")])

    test_end_day = selected[0]["config"]["test_end_day"]
    write_structured_metrics(args.base_dir, baseline, selected, out, test_end_day)
    branches = [state["branch"] for state in selected]
    fixed = [sys.executable, os.path.join(args.tools_dir, "test_multi.py"), baseline]
    fixed.extend(branches)
    fixed.append(test_end_day)
    run_capture(fixed, os.path.join(out, "fixed_window.txt"))

    rolling = [sys.executable, os.path.join(args.tools_dir, "rolling_auc_compare.py"),
               baseline]
    rolling.extend(branches)
    rolling.extend(["--base-dir", args.base_dir,
                    "-o", os.path.join(out, "rolling_auc.html")])
    run_capture(rolling, os.path.join(out, "rolling_auc_generation.txt"))
    write_combined_rolling(args.base_dir, baseline, selected, out)
    os.rename(out, final_out)

    if not args.no_push:
        relative_out = os.path.relpath(final_out, args.tools_dir)
        subprocess.run(["git", "add", relative_out], cwd=args.tools_dir, check=True)
        subprocess.run(["git", "commit", "-m", "result: %s" % args.snapshot_id],
                       cwd=args.tools_dir, check=True)
        subprocess.run(["git", "push"], cwd=args.tools_dir, check=True)

    print("[PARTIAL BATCH] %s (%d selected, %d deferred) -> %s" %
          (args.snapshot_id, len(selected), len(deferred), final_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
