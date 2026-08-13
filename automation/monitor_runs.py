#!/usr/bin/env python3
import argparse
import os
import sys

from common import TERMINAL, atomic_json, load_json, now, related_processes, run_files
from validate_experiment import validate


def update(path):
    state = load_json(path)
    if state.get("status") in TERMINAL:
        return state
    report = validate(state["experiment_dir"], state["config"])
    processes = related_processes(state["experiment_dir"])
    state["validation"], state["processes"], state["updated_at"] = report, processes, now()
    if report["ok"]:
        state["status"], state["failure_reason"] = "done", None
        state.setdefault("finished_at", now())
    elif processes:
        state["status"] = "running"
    else:
        state["status"] = "failed"
        state["failure_reason"] = "; ".join(report["errors"])
        state["finished_at"] = now()
    atomic_json(path, state)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", help="kept for CLI compatibility")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    states = [update(path) for path in run_files(args.state_dir)]
    print("%-52s %-10s %-5s %s" % ("run_id", "status", "pids", "validation"))
    for state in states:
        checks = state["validation"]["checks"]
        passed = sum(bool(value) for value in checks.values())
        print("%-52s %-10s %-5d %d/%d" % (
            state["run_id"], state["status"], len(state.get("processes", [])),
            passed, len(checks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
