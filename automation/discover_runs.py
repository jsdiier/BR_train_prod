#!/usr/bin/env python3
"""Discover opt-in experiment branches and launch their existing submit_luban.sh."""
import argparse
import json
import os
import re
import subprocess
import sys

from common import atomic_json, load_json, now, run_files


BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
FIXED_REQUIRED = (
    "baseline", "train_start_day", "train_end_day", "test_start_day", "test_end_day",
)
ROLLING_REQUIRED = ("auto_test_start_ckpt_day", "auto_test_end_day")


def sync_remote_heads(repo_url, tools_dir):
    subprocess.run(
        ["git", "fetch", "--quiet", "--prune", "--no-tags", repo_url,
         "+refs/heads/*:refs/automation-discovery/*"],
        cwd=tools_dir, check=True, timeout=180,
    )
    output = subprocess.check_output(
        ["git", "for-each-ref", "--format=%(objectname) %(refname)",
         "refs/automation-discovery/"], cwd=tools_dir, text=True
    )
    heads = []
    for line in output.splitlines():
        commit, ref = line.split(None, 1)
        prefix = "refs/automation-discovery/"
        if ref.startswith(prefix):
            heads.append((ref[len(prefix):], commit))
    return sorted(heads)


def read_remote_config(tools_dir, branch):
    """Read experiment.json from the already synchronized discovery ref."""
    ref = "refs/automation-discovery/%s" % branch
    show = subprocess.run(
        ["git", "show", "%s:experiment.json" % ref],
        cwd=tools_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if show.returncode:
        return None
    try:
        return json.loads(show.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid experiment.json: %s" % exc)


def validate_config(branch, value):
    if not isinstance(value, dict):
        raise ValueError("experiment.json must contain a JSON object")
    if value.get("enabled") is not True:
        return None
    rolling_enabled = value.get("rolling_enabled", True) is not False
    required = FIXED_REQUIRED + (ROLLING_REQUIRED if rolling_enabled else ())
    missing = [key for key in required if not value.get(key)]
    if missing:
        raise ValueError("missing required fields: %s" % ",".join(missing))
    for key in required[1:]:
        if not re.fullmatch(r"\d{8}", str(value[key])):
            raise ValueError("%s must be YYYYMMDD" % key)
    resolved = dict(value)
    resolved["branch"] = branch
    resolved["rolling_enabled"] = rolling_enabled
    resolved["require_inference_benchmark"] = bool(
        value.get("require_inference_benchmark", True))
    return resolved


def clone_exact(repo_url, branch, commit, exp_dir):
    if os.path.exists(exp_dir):
        if not os.path.isdir(os.path.join(exp_dir, ".git")):
            raise RuntimeError("target exists but is not a git clone: %s" % exp_dir)
        local = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=exp_dir, text=True).strip()
        if local != commit:
            raise RuntimeError(
                "existing directory HEAD %s differs from discovered commit %s" %
                (local[:12], commit[:12]))
        return
    subprocess.run(["git", "clone", "--branch", branch, "--single-branch",
                    repo_url, exp_dir], check=True)
    local = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=exp_dir, text=True).strip()
    if local != commit:
        raise RuntimeError("cloned HEAD changed during discovery; retry on next tick")


def main():
    parser = argparse.ArgumentParser(description="Discover opt-in branches and launch new experiments")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--tools-dir", required=True)
    parser.add_argument("--repo-url", default="https://github.com/jsdiier/tf_rank_BR.git")
    parser.add_argument("--exclude", default="main,shared_tools")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    excluded = {item.strip() for item in args.exclude.split(",") if item.strip()}
    known = {load_json(path)["branch"] for path in run_files(args.state_dir)}
    try:
        heads = sync_remote_heads(args.repo_url, args.tools_dir)
    except Exception as exc:
        print("[DISCOVER_ERROR] cannot synchronize remote heads: %s" % exc, file=sys.stderr)
        return 1
    for branch, commit in heads:
        if branch in excluded or branch in known:
            continue
        if not BRANCH_RE.fullmatch(branch) or branch.startswith("/") or ".." in branch.split("/"):
            print("[DISCOVER] unsafe branch name, skipped: %s" % branch)
            continue
        try:
            raw = read_remote_config(args.tools_dir, branch)
            if raw is None:
                continue
            config = validate_config(branch, raw)
            if config is None:
                print("[DISCOVER] disabled: %s" % branch)
                continue
            run_id = "%s@%s@attempt_1" % (branch.replace("/", "__"), commit[:12])
            if args.dry_run:
                print("[DISCOVER_DRY_RUN] eligible %s commit=%s" % (branch, commit[:12]))
                continue
            exp_dir = os.path.join(os.path.abspath(args.base_dir), branch)
            clone_exact(args.repo_url, branch, commit, exp_dir)
            if not os.path.isfile(os.path.join(exp_dir, "submit_luban.sh")):
                raise RuntimeError("submit_luban.sh is missing")
            state_path = os.path.join(args.state_dir, "runs", run_id + ".json")
            state = {"schema_version": 1, "run_id": run_id, "branch": branch,
                     "commit": commit, "attempt": 1, "baseline": config["baseline"],
                     "batch_group": config.get("batch_group", "default"),
                     "status": "launching", "adopted": False, "job_uuid": None,
                     "require_inference_benchmark": config["require_inference_benchmark"],
                     "config": config, "experiment_dir": exp_dir, "created_at": now(),
                     "updated_at": now(), "launcher_pid": None, "processes": [],
                     "validation": {"ok": False, "checks": {},
                                    "errors": ["submit_luban.sh has not finished"]},
                     "failure_reason": None}
            atomic_json(state_path, state)
            log_path = os.path.join(exp_dir, "nohup_submit_automation.log")
            with open(log_path, "ab", buffering=0) as log:
                process = subprocess.Popen(
                    ["bash", "submit_luban.sh"], cwd=exp_dir, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            state["status"], state["launcher_pid"], state["updated_at"] = (
                "running", process.pid, now())
            atomic_json(state_path, state)
            known.add(branch)
            print("[DISCOVER] launched %s pid=%d" % (run_id, process.pid))
        except Exception as exc:
            print("[DISCOVER_ERROR] %s: %s" % (branch, exc), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
