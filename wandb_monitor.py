#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Small, failure-tolerant Weights & Biases adapter for training telemetry."""

from __future__ import print_function

import os
import socket
import json
import datetime


def _env_flag(name, default):
    value = os.environ.get(name, default)
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


class WandbMonitor(object):
    """Keep W&B failures from changing the outcome of a training job."""

    def __init__(self):
        self.enabled = _env_flag("WANDB_ENABLED", "1")
        self.run = None
        self.wandb = None
        self.disabled_reason = None
        self.local_log_path = os.environ.get(
            "WANDB_LOCAL_LOG", "./log/wandb_metrics.jsonl"
        )

    def _write_local(self, event, values):
        record = dict(values)
        record['_event'] = event
        record['_time_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'
        try:
            parent = os.path.dirname(self.local_log_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent)
            with open(self.local_log_path, 'a') as output:
                output.write(json.dumps(record, sort_keys=True) + '\n')
        except Exception as exc:
            print("WANDB_MONITOR local audit warning: %s" % exc)

    def start(self, config, start_day, end_day):
        self._write_local('start', {
            'train_start_day': start_day,
            'train_end_day': end_day,
            'config': config,
        })
        if not self.enabled:
            self.disabled_reason = "WANDB_ENABLED=0"
            print("WANDB_MONITOR disabled: %s" % self.disabled_reason)
            return

        try:
            import wandb
        except Exception as exc:
            self.enabled = False
            self.disabled_reason = "import failed: %s" % exc
            print("WANDB_MONITOR unavailable; training continues: %s" % self.disabled_reason)
            return

        project = os.environ.get("WANDB_PROJECT", "br-rank-prod")
        group = os.environ.get("WANDB_GROUP", "BR_train_prod_baseline_w_wandb")
        job_type = os.environ.get("WANDB_JOB_TYPE", "train")
        default_name = "%s__train_%s_%s" % (group, start_day, end_day)
        run_name = os.environ.get("WANDB_RUN_NAME", default_name)

        run_config = dict(config)
        run_config.update({
            "train_start_day": start_day,
            "train_end_day": end_day,
            "git_branch": os.environ.get("GIT_BRANCH", group),
            "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
            "host": socket.gethostname(),
        })

        try:
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                group=group,
                job_type=job_type,
                name=run_name,
                config=run_config,
                reinit=True,
            )
            # W&B's internal history step remains append-only. All charts use the
            # restored optimizer iteration as their explicit x-axis instead.
            wandb.define_metric("optimizer_step")
            wandb.define_metric("*", step_metric="optimizer_step")
            print(
                "WANDB_MONITOR started project=%s group=%s name=%s url=%s"
                % (project, group, run_name, getattr(self.run, "url", ""))
            )
        except Exception as exc:
            self.enabled = False
            self.run = None
            self.disabled_reason = "init failed: %s" % exc
            print("WANDB_MONITOR init failed; training continues: %s" % self.disabled_reason)

    def log(self, values):
        self._write_local('metrics', values)
        if not self.enabled or self.run is None:
            return
        try:
            self.wandb.log(values)
        except Exception as exc:
            self.enabled = False
            self.disabled_reason = "log failed: %s" % exc
            print("WANDB_MONITOR log failed; further W&B logging disabled: %s" % exc)

    def finish(self):
        self._write_local('finish', {})
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:
            print("WANDB_MONITOR finish warning: %s" % exc)
        finally:
            self.run = None
