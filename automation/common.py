#!/usr/bin/env python3
import datetime as dt
import json
import os
import subprocess
import tempfile


TERMINAL = {"done", "failed"}


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state-", dir=os.path.dirname(path), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def git_head(exp_dir):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=exp_dir, text=True
    ).strip()


def dates(start, end):
    cur = dt.datetime.strptime(start, "%Y%m%d").date()
    last = dt.datetime.strptime(end, "%Y%m%d").date()
    while cur <= last:
        yield cur.strftime("%Y%m%d")
        cur += dt.timedelta(days=1)


def previous_day(day):
    value = dt.datetime.strptime(day, "%Y%m%d").date() - dt.timedelta(days=1)
    return value.strftime("%Y%m%d")


def next_day(day):
    value = dt.datetime.strptime(day, "%Y%m%d").date() + dt.timedelta(days=1)
    return value.strftime("%Y%m%d")


def run_files(state_dir):
    root = os.path.join(state_dir, "runs")
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, name) for name in os.listdir(root) if name.endswith(".json"))


def related_processes(exp_dir):
    target = os.path.realpath(exp_dir)
    found = []
    proc = "/proc"
    if not os.path.isdir(proc):
        return found
    for entry in os.listdir(proc):
        if not entry.isdigit():
            continue
        try:
            cwd = os.path.realpath(os.readlink(os.path.join(proc, entry, "cwd")))
            with open(os.path.join(proc, entry, "cmdline"), "rb") as handle:
                cmd = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except (OSError, PermissionError):
            continue
        if cwd == target and any(name in cmd for name in (
                "submit_luban.sh", "rolling_test.sh", "train.py", "test.py")):
            found.append({"pid": int(entry), "cmd": cmd.strip()})
    return found
