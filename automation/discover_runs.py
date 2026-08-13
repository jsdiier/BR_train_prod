#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

from common import atomic_json, git_head, load_json, now, run_files


def main():
    parser = argparse.ArgumentParser(description="Register and launch newly configured experiments")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-url", default="https://github.com/jsdiier/tf_rank_BR.git")
    args = parser.parse_args()
    config = load_json(args.config)
    known = {load_json(path)["branch"] for path in run_files(args.state_dir)}
    for entry in config["experiments"]:
        branch = entry["branch"]
        if branch in known:
            continue
        exp_dir = os.path.join(args.base_dir, branch)
        if not os.path.isdir(exp_dir):
            clone = subprocess.run(["git", "clone", "-b", branch, args.repo_url, exp_dir])
            if clone.returncode:
                print("[DISCOVER] clone failed: %s" % branch)
                continue
        commit = git_head(exp_dir)
        resolved = dict(config["defaults"])
        resolved["baseline"] = config["default_baseline"]
        resolved.update(entry)
        run_id = "%s@%s@attempt_1" % (branch, commit[:12])
        log_path = os.path.join(exp_dir, "nohup_submit_automation.log")
        log = open(log_path, "ab", buffering=0)
        process = subprocess.Popen(["bash", "submit_luban.sh"], cwd=exp_dir,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        log.close()
        state = {"schema_version": 1, "run_id": run_id, "branch": branch,
                 "commit": commit, "attempt": 1, "baseline": resolved["baseline"],
                 "status": "running", "adopted": False, "job_uuid": None,
                 "require_inference_benchmark": resolved.get("require_inference_benchmark", True),
                 "config": resolved, "experiment_dir": exp_dir, "created_at": now(),
                 "updated_at": now(), "launcher_pid": process.pid, "processes": [],
                 "validation": {"ok": False, "checks": {}, "errors": ["not validated yet"]},
                 "failure_reason": None}
        atomic_json(os.path.join(args.state_dir, "runs", run_id + ".json"), state)
        print("[DISCOVER] launched %s pid=%d" % (run_id, process.pid))
    return 0


if __name__ == "__main__":
    sys.exit(main())
