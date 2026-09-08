#!/usr/bin/env python3
"""Fail fast when fixed-window or rolling metrics are absent from the TSV."""

import argparse
import csv
import datetime as dt


TASKS = ("buy", "cat", "click", "ext")


def next_day(day):
    value = dt.datetime.strptime(day, "%Y%m%d").date() + dt.timedelta(days=1)
    return value.strftime("%Y%m%d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--fixed-checkpoint", required=True)
    parser.add_argument("--fixed-start", required=True)
    parser.add_argument("--fixed-end", required=True)
    parser.add_argument("--rolling-seed", required=True)
    parser.add_argument("--rolling-end", required=True)
    args = parser.parse_args()

    expected = {
        (args.fixed_checkpoint, args.fixed_start, args.fixed_end, task)
        for task in TASKS
    }

    checkpoint_day = args.rolling_seed
    test_day = next_day(checkpoint_day)
    while test_day <= args.rolling_end:
        expected.update(
            (checkpoint_day, test_day, test_day, task) for task in TASKS
        )
        checkpoint_day = test_day
        test_day = next_day(test_day)

    actual = set()
    with open(args.input, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            actual.add((
                row["checkpoint_day"],
                row["test_start_day"],
                row["test_end_day"],
                row["task"],
            ))

    missing = sorted(expected - actual)
    if missing:
        print("METRICS_CONTRACT_FAILED missing_rows=%d" % len(missing))
        for checkpoint, start, end, task in missing:
            print(
                "MISSING checkpoint=%s test=%s..%s task=%s"
                % (checkpoint, start, end, task)
            )
        raise SystemExit(2)

    print(
        "METRICS_CONTRACT_OK required_rows=%d fixed_rows=4 rolling_rows=%d"
        % (len(expected), len(expected) - 4)
    )


if __name__ == "__main__":
    main()
