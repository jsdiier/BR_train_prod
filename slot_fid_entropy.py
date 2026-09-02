#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import gzip
import math
import os
import struct
import subprocess
import sys

import tensorflow as tf


HADOOP = "/usr/local/hadoop-current/bin/hadoop"

SOURCES = {
    "cpy": (
        "hdfs://DClusterUS1/user/prod_soda_trade_strategy/rank/"
        "chenpinyuan/hash_fea_new_fixed/train/20260830"
    ),
    "jz": (
        "hdfs://DClusterUS1/user/prod_soda_trade_strategy/rank/"
        "jiazhuo/hash_fea_new/train/20260830"
    ),
}


def choose_first_part(directory):
    output = subprocess.check_output(
        [HADOOP, "fs", "-ls", directory],
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    paths = []
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        path = fields[-1]
        name = os.path.basename(path)
        if name.startswith("part-") and name.endswith(".tfrecord.gz"):
            paths.append(path)

    if not paths:
        raise RuntimeError("没有找到 TFRecord part: %s" % directory)

    return sorted(paths)[0]


def read_exact(stream, size):
    chunks = []
    remaining = size

    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if not chunks:
                return None
            raise EOFError("TFRecord truncated: expected=%d" % size)
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def iter_tfrecord(stream):
    while True:
        length_bytes = read_exact(stream, 8)
        if length_bytes is None:
            return

        length = struct.unpack("<Q", length_bytes)[0]

        # 跳过 length CRC
        read_exact(stream, 4)

        record = read_exact(stream, length)

        # 跳过 data CRC
        read_exact(stream, 4)

        yield record


def collect(source_name, hdfs_path, limit, progress_every):
    # slot -> Counter(fid -> occurrence count)
    slot_counts = collections.defaultdict(collections.Counter)

    process = subprocess.Popen(
        [HADOOP, "fs", "-cat", hdfs_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    records = 0
    raw_pairs = 0
    dedup_pairs = 0

    try:
        with gzip.GzipFile(fileobj=process.stdout, mode="rb") as stream:
            for serialized in iter_tfrecord(stream):
                example = tf.train.Example()
                example.ParseFromString(serialized)

                features = example.features.feature
                slots = list(features["fea_ids"].int64_list.value)
                fids = list(features["fea_vals"].int64_list.value)

                if len(fids) != len(slots):
                    raise RuntimeError(
                        "%s record=%d fea_ids=%d fea_vals=%d"
                        % (source_name, records + 1, len(fids), len(slots))
                    )

                raw_pairs += len(fids)

                # 同一样本内，相同 slot + fid 只计一次
                sample_pairs = set()
                for slot_value, fid_value in zip(slots, fids):
                    slot_id = int(round(slot_value))
                    fid = int(fid_value)

                    # 过滤 padding 或非法值
                    if slot_id <= 0 or fid == 0:
                        continue

                    sample_pairs.add((slot_id, fid))

                dedup_pairs += len(sample_pairs)

                for slot_id, fid in sample_pairs:
                    slot_counts[slot_id][fid] += 1

                records += 1

                if records % progress_every == 0:
                    print(
                        "[%s] records=%d raw_pairs=%d dedup_pairs=%d slots=%d"
                        % (
                            source_name,
                            records,
                            raw_pairs,
                            dedup_pairs,
                            len(slot_counts),
                        ),
                        file=sys.stderr,
                        flush=True,
                    )

                if limit > 0 and records >= limit:
                    break

    finally:
        if process.stdout:
            process.stdout.close()

        # limit 模式下管道会被提前关闭，hadoop 可能返回非零，属于预期
        process.wait()

    print(
        "[%s] COMPLETED records=%d raw_pairs=%d dedup_pairs=%d slots=%d"
        % (source_name, records, raw_pairs, dedup_pairs, len(slot_counts)),
        file=sys.stderr,
        flush=True,
    )

    return slot_counts, records


def entropy_bits(counter):
    total = sum(counter.values())
    if total <= 0:
        return None

    entropy = 0.0
    for count in counter.values():
        probability = float(count) / float(total)
        entropy -= probability * math.log(probability, 2)

    return entropy


def fmt_float(value):
    if value is None:
        return ""
    return "%.10f" % value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="每个数据源读取多少条；0 表示完整 part",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()

    selected_parts = {}
    all_counts = {}
    record_counts = {}

    for source_name in ("cpy", "jz"):
        selected_parts[source_name] = choose_first_part(SOURCES[source_name])
        print(
            "[%s] selected_part=%s" %
            (source_name, selected_parts[source_name]),
            file=sys.stderr,
            flush=True,
        )

        all_counts[source_name], record_counts[source_name] = collect(
            source_name,
            selected_parts[source_name],
            args.limit,
            args.progress_every,
        )

    all_slots = sorted(
        set(all_counts["cpy"].keys()) | set(all_counts["jz"].keys())
    )

    with open(args.output, "w") as output:
        output.write(
            "slot_id\t"
            "e_cpy\t"
            "e_jz\t"
            "gap\t"
            "cpy_occurrences\t"
            "jz_occurrences\t"
            "cpy_unique_fids\t"
            "jz_unique_fids\n"
        )

        for slot_id in all_slots:
            cpy_counter = all_counts["cpy"].get(slot_id)
            jz_counter = all_counts["jz"].get(slot_id)

            e_cpy = entropy_bits(cpy_counter) if cpy_counter else None
            e_jz = entropy_bits(jz_counter) if jz_counter else None

            if e_cpy is not None and e_jz is not None:
                gap = e_cpy - e_jz
            else:
                gap = None

            output.write(
                "%d\t%s\t%s\t%s\t%d\t%d\t%d\t%d\n"
                % (
                    slot_id,
                    fmt_float(e_cpy),
                    fmt_float(e_jz),
                    fmt_float(gap),
                    sum(cpy_counter.values()) if cpy_counter else 0,
                    sum(jz_counter.values()) if jz_counter else 0,
                    len(cpy_counter) if cpy_counter else 0,
                    len(jz_counter) if jz_counter else 0,
                )
            )

    print("OUTPUT=%s" % os.path.abspath(args.output))
    print("CPY_PART=%s" % selected_parts["cpy"])
    print("JZ_PART=%s" % selected_parts["jz"])
    print("CPY_RECORDS=%d" % record_counts["cpy"])
    print("JZ_RECORDS=%d" % record_counts["jz"])


if __name__ == "__main__":
    main()
