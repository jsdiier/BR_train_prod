#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

from common import TERMINAL, atomic_json, load_json, now, run_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--tools-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    active_path = os.path.join(args.state_dir, "active_batch.json")
    if not os.path.isfile(active_path):
        batched = set()
        batches_dir = os.path.join(args.state_dir, "batches")
        if os.path.isdir(batches_dir):
            for name in os.listdir(batches_dir):
                if name.endswith(".json"):
                    batched.update(load_json(os.path.join(batches_dir, name)).get("members", []))
        candidates = [load_json(path) for path in run_files(args.state_dir)]
        candidates = [state for state in candidates if state["run_id"] not in batched]
        if not candidates:
            print("[BATCH] no unbatched runs")
            return 0
        # Freeze one explicit group at a time. This keeps user-mandated
        # fixed-only factorials separate from later agent-generated runs.
        candidates.sort(key=lambda state: state.get("created_at", ""))
        batch_group = candidates[0].get("batch_group", "default")
        baseline = candidates[0]["baseline"]
        candidates = [state for state in candidates
                      if state.get("batch_group", "default") == batch_group
                      and state["baseline"] == baseline]
        batch_id = "batch_%s" % __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        baselines = {state["baseline"] for state in candidates}
        batch = {"schema_version": 1, "batch_id": batch_id, "created_at": now(),
                 "baseline": baselines.pop(), "members": [state["run_id"] for state in candidates],
                 "batch_group": batch_group,
                 "status": "waiting", "result_dir": None}
        batch_path = os.path.join(batches_dir, batch_id + ".json")
        atomic_json(batch_path, batch)
        atomic_json(active_path, {"batch_id": batch_id})
        print("[BATCH] froze %d runs from group %s with baseline %s into %s" % (
            len(candidates), batch_group, baseline, batch_id))
        active = {"batch_id": batch_id}
    else:
        active = load_json(active_path)
    batch_path = os.path.join(args.state_dir, "batches", active["batch_id"] + ".json")
    batch = load_json(batch_path)
    states = []
    for run_id in batch["members"]:
        path = os.path.join(args.state_dir, "runs", run_id + ".json")
        if not os.path.isfile(path):
            print("[BATCH] missing run state: %s" % run_id)
            return 1
        states.append(load_json(path))
    pending = [state["run_id"] for state in states if state["status"] not in TERMINAL]
    if pending:
        print("[BATCH] waiting for %d runs: %s" % (len(pending), ", ".join(pending)))
        return 0
    if batch["status"] in {"completed", "completed_with_failures"}:
        print("[BATCH] already completed: %s" % batch["batch_id"])
        return 0
    batch["status"], batch["updated_at"] = "aggregating", now()
    atomic_json(batch_path, batch)
    command = [args.python, os.path.join(args.tools_dir, "automation", "aggregate_batch.py"),
               "--base-dir", args.base_dir, "--state-dir", args.state_dir,
               "--tools-dir", args.tools_dir, "--batch", batch_path]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
