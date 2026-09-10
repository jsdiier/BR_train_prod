#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""只读验证 OC Step4 的 20260831 全部 24 个小时分区。"""

from __future__ import print_function

import json

from pyspark import StorageLevel
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


COUNTRY = "BR"
BUSINESS_DATE = "20260831"

NEW_BASE = (
    "/user/prod_soda_trade_strategy/rank/chenpinyuan/"
    "sailing_nearby_br_new_tmp_w_oc/BR/20260831"
)
OLD_BASE = (
    "/user/prod_soda_trade_strategy/rank/dudunying/"
    "sailing_nearby_br_new_tmp/BR/20260831"
)
STEP3_BASE = (
    "/user/prod_soda_trade_strategy/rank/dudunying/"
    "sailing_nearby_br_feature_parse/BR"
)
ACTION_TABLE = (
    "soda_international_trade_stg."
    "sailing_nearby_br_u_act_t_w_oc"
)

# 老 Step4 的 0831 历史输出尚未包含该字段；sample_domain 则是本次新增。
OLD_OUTPUT_ABSENT_COLUMNS = {"user_ride_feat", "sample_domain"}

KEY_COLUMNS = [
    "rec_id",
    "shop_id",
    "rank",
    "rank_type",
    "order_id",
    "sample_domain",
]
REQUIRED_COLUMNS = [
    "recID",
    "s_id",
    "rank",
    "rank_type",
    "orderid",
    "traceid",
    "label",
    "is_imp",
    "is_clk",
    "is_cat",
    "is_cov",
    "sample_domain",
]


def route(hour):
    hour_text = "%02d" % hour
    logical_hour = BUSINESS_DATE + hour_text

    if hour <= 10:
        biz_hour = "20260902" + hour_text
        step3_hour = "20260901" + hour_text
    elif hour <= 22:
        biz_hour = "20260901" + hour_text
        step3_hour = BUSINESS_DATE + hour_text
    else:
        biz_hour = BUSINESS_DATE + "23"
        step3_hour = BUSINESS_DATE + "23"

    return {
        "logical_hour": logical_hour,
        "biz_hour": biz_hour,
        "new_path": NEW_BASE + "/" + logical_hour,
        "old_path": OLD_BASE + "/" + logical_hour,
        "step3_path": STEP3_BASE + "/" + step3_hour,
        "schema_log_path": (
            NEW_BASE + "/" + logical_hour + "/_SCHEMA_LOG.json"
        ),
    }


ROUTES = [route(hour) for hour in range(24)]

spark = (
    SparkSession.builder
    .appName("validate_step4_oc_full_day_20260831")
    .enableHiveSupport()
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

failures = []


def title(text):
    print("")
    print("=" * 120)
    print(text)
    print("=" * 120)


def check(name, passed, observed, expected):
    status = "PASS" if passed else "FAIL"
    print(
        "%s name=%s observed=%s expected=%s"
        % (status, name, observed, expected)
    )
    if not passed:
        failures.append(
            "%s observed=%s expected=%s" % (name, observed, expected)
        )


def get_fs(path_text):
    path = spark._jvm.org.apache.hadoop.fs.Path(path_text)
    fs = path.getFileSystem(spark._jsc.hadoopConfiguration())
    return fs, path


def exists(path_text):
    fs, path = get_fs(path_text)
    return bool(fs.exists(path))


def read_hdfs_text(path_text):
    fs, path = get_fs(path_text)
    stream = fs.open(path)
    reader = spark._jvm.java.io.BufferedReader(
        spark._jvm.java.io.InputStreamReader(stream, "UTF-8")
    )
    lines = []
    try:
        while True:
            line = reader.readLine()
            if line is None:
                break
            lines.append(line)
    finally:
        reader.close()
    return "\n".join(lines)


def normalize(column_name):
    value = F.trim(F.col(column_name))
    return F.when(
        value.isNull() | (value == "") | (value == "\\N"),
        F.lit(None).cast("string"),
    ).otherwise(value)


def different(left, right):
    return ~left.eqNullSafe(right)


def derive_hour():
    return F.regexp_extract(
        F.input_file_name(), r"/(20260831[0-2][0-9])/", 1
    )


def read_hourly_text(paths):
    return (
        spark.read.text(paths)
        .withColumn("logical_hour", derive_hour())
        .withColumn("parts", F.split(F.col("value"), "\\t", -1))
        .withColumn("field_count", F.size("parts"))
    )


# ------------------------------------------------------------------
# 1. 预检 24 小时路径和 JSON Schema 日志
# ------------------------------------------------------------------

title("PREFLIGHT AND SCHEMA LOG VALIDATION")

logs = {}
reference_columns = None

for item in ROUTES:
    hour = item["logical_hour"]

    for path_name in ["new_path", "old_path", "step3_path", "schema_log_path"]:
        present = exists(item[path_name])
        check(
            "%s_%s_exists" % (hour, path_name),
            present,
            present,
            True,
        )

    if not exists(item["schema_log_path"]):
        continue

    try:
        log = json.loads(read_hdfs_text(item["schema_log_path"]))
    except Exception as error:
        check("%s_schema_log_json" % hour, False, error, "valid JSON")
        continue

    logs[hour] = log
    final_columns = log.get("final_output_columns", [])
    effective_columns = log.get("effective_output_columns", [])
    filled_columns = log.get("compatibility_filled_columns", [])
    physical_columns = log.get("step3_physical_columns", [])

    check(
        "%s_biz_hour" % hour,
        str(log.get("biz_hour")) == item["biz_hour"],
        log.get("biz_hour"),
        item["biz_hour"],
    )
    check(
        "%s_business_date" % hour,
        str(log.get("business_date")) == BUSINESS_DATE,
        log.get("business_date"),
        BUSINESS_DATE,
    )
    check(
        "%s_logical_hour" % hour,
        str(log.get("logical_partition_hour")) == hour,
        log.get("logical_partition_hour"),
        hour,
    )
    check(
        "%s_output_path" % hour,
        str(log.get("output_path")).rstrip("/") == item["new_path"],
        log.get("output_path"),
        item["new_path"],
    )
    check(
        "%s_step3_path" % hour,
        [str(value).rstrip("/") for value in log.get("step3_input_paths", [])]
        == [item["step3_path"]],
        log.get("step3_input_paths"),
        [item["step3_path"]],
    )
    check(
        "%s_final_count" % hour,
        log.get("final_output_column_count") == len(final_columns),
        log.get("final_output_column_count"),
        len(final_columns),
    )
    check(
        "%s_effective_and_filled" % hour,
        set(effective_columns).isdisjoint(set(filled_columns))
        and set(effective_columns) | set(filled_columns) == set(final_columns),
        "%d+%d" % (len(effective_columns), len(filled_columns)),
        len(final_columns),
    )
    check(
        "%s_filled_absent_from_step3" % hour,
        set(filled_columns).isdisjoint(set(physical_columns)),
        filled_columns,
        "absent from physical Step3",
    )
    check(
        "%s_sample_domain_last" % hour,
        bool(final_columns) and final_columns[-1] == "sample_domain",
        final_columns[-1] if final_columns else None,
        "sample_domain",
    )

    if reference_columns is None:
        reference_columns = final_columns
    else:
        check(
            "%s_final_schema_same" % hour,
            final_columns == reference_columns,
            len(final_columns),
            len(reference_columns),
        )

check("schema_log_count", len(logs) == 24, len(logs), 24)

if reference_columns is None:
    print("OVERALL_RESULT=FAIL")
    spark.stop()
    raise RuntimeError("No valid Step4 schema log")

missing_required = sorted(set(REQUIRED_COLUMNS) - set(reference_columns))
check("required_columns", not missing_required, missing_required, [])

if failures:
    title("PREFLIGHT FAILED")
    for failure in failures:
        print("FAILURE=%s" % failure)
    print("OVERALL_RESULT=FAIL")
    spark.stop()
    raise RuntimeError("Preflight failed; data scan was not started")

index = {name: position for position, name in enumerate(reference_columns)}


# ------------------------------------------------------------------
# 2. 新输出：字段宽度及兼容补列值
# ------------------------------------------------------------------

new_raw = read_hourly_text([item["new_path"] for item in ROUTES])

compatibility_invalid = F.lit(False)
for item in ROUTES:
    hour = item["logical_hour"]
    for column_name in logs[hour].get("compatibility_filled_columns", []):
        compatibility_invalid = compatibility_invalid | (
            (F.col("logical_hour") == hour)
            & (F.col("parts").getItem(index[column_name]) != "\\N")
        )

raw_metrics = (
    new_raw.groupBy("logical_hour", "field_count")
    .agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(compatibility_invalid, 1).otherwise(0)).alias(
            "invalid_compatibility_rows"
        ),
    )
    .orderBy("logical_hour", "field_count")
    .collect()
)

title("NEW OUTPUT FIELD COUNT")
for row in raw_metrics:
    print(row)

metrics_by_hour = {}
for row in raw_metrics:
    metrics_by_hour.setdefault(row["logical_hour"], []).append(row)

for item in ROUTES:
    hour = item["logical_hour"]
    rows = metrics_by_hour.get(hour, [])
    expected_count = logs[hour]["final_output_column_count"]
    valid = (
        len(rows) == 1
        and rows[0]["field_count"] == expected_count
        and rows[0]["rows"] > 0
    )
    check(
        "%s_fixed_width" % hour,
        valid,
        [(row["field_count"], row["rows"]) for row in rows],
        [(expected_count, ">0")],
    )
    invalid_filled = sum(
        (row["invalid_compatibility_rows"] or 0) for row in rows
    )
    check(
        "%s_compatibility_values" % hour,
        invalid_filled == 0,
        invalid_filled,
        0,
    )


# ------------------------------------------------------------------
# 3. 新输出：基础质量、Label、去重键
# ------------------------------------------------------------------

new_data = (
    new_raw.select(
        "logical_hour",
        *[
            F.col("parts").getItem(index[name]).alias(name)
            for name in REQUIRED_COLUMNS
        ]
    )
    .select(
        "logical_hour",
        normalize("recID").alias("rec_id"),
        normalize("s_id").alias("shop_id"),
        normalize("rank").alias("rank"),
        normalize("rank_type").alias("rank_type"),
        normalize("orderid").alias("order_id"),
        normalize("traceid").alias("trace_id"),
        normalize("sample_domain").alias("sample_domain"),
        F.col("label").cast("int").alias("label"),
        F.col("is_imp").cast("int").alias("is_imp"),
        F.col("is_clk").cast("int").alias("is_clk"),
        F.col("is_cat").cast("int").alias("is_cat"),
        F.col("is_cov").cast("int").alias("is_cov"),
    )
    .persist(StorageLevel.DISK_ONLY)
)

expected_label = (
    F.when(F.col("is_cov") > 0, 3)
    .when(F.col("is_cat") > 0, 2)
    .when(F.col("is_clk") > 0, 1)
    .otherwise(0)
)
invalid_behavior = (
    F.col("is_imp").isNull()
    | (~F.col("is_imp").isin(0, 1))
    | F.col("is_clk").isNull()
    | (~F.col("is_clk").isin(0, 1))
    | F.col("is_cat").isNull()
    | (~F.col("is_cat").isin(0, 1))
    | F.col("is_cov").isNull()
    | (~F.col("is_cov").isin(0, 1))
)

quality_rows = (
    new_data.groupBy("logical_hour", "sample_domain")
    .agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct(F.struct("rec_id", "shop_id")).alias("pairs"),
        F.sum(F.when(F.col("is_clk") > 0, 1).otherwise(0)).alias("clk"),
        F.sum(F.when(F.col("is_cat") > 0, 1).otherwise(0)).alias("cat"),
        F.sum(F.when(F.col("is_cov") > 0, 1).otherwise(0)).alias("cov"),
        F.sum(
            F.when(
                F.col("rec_id").isNull()
                | F.col("shop_id").isNull()
                | F.col("trace_id").isNull(),
                1,
            ).otherwise(0)
        ).alias("invalid_keys"),
        F.sum(F.when(invalid_behavior, 1).otherwise(0)).alias(
            "invalid_behavior"
        ),
        F.sum(F.when(different(F.col("label"), expected_label), 1).otherwise(0)).alias(
            "label_mismatch"
        ),
        F.sum(
            F.when(
                (F.col("is_cov") > F.col("is_cat"))
                | (F.col("is_cat") > F.col("is_clk"))
                | (F.col("is_clk") > F.col("is_imp")),
                1,
            ).otherwise(0)
        ).alias("hierarchy_errors"),
    )
    .orderBy("logical_hour", "sample_domain")
    .collect()
)

title("DOMAIN AND LABEL QUALITY")
quality_map = {}
for row in quality_rows:
    print(row)
    quality_map[(row["logical_hour"], row["sample_domain"])] = row

for item in ROUTES:
    for domain in ["nearby", "oldcustomer"]:
        key = (item["logical_hour"], domain)
        row = quality_map.get(key)
        check("%s_%s_present" % key, row is not None and row["rows"] > 0,
              row["rows"] if row else 0, ">0")
        if row:
            check("%s_%s_one_pair_one_row" % key, row["rows"] == row["pairs"],
                  row["rows"], row["pairs"])
            for metric in [
                "invalid_keys", "invalid_behavior", "label_mismatch", "hierarchy_errors"
            ]:
                check("%s_%s_%s" % (key[0], key[1], metric),
                      (row[metric] or 0) == 0, row[metric] or 0, 0)

invalid_domains = new_data.where(
    F.col("sample_domain").isNull()
    | (~F.col("sample_domain").isin("nearby", "oldcustomer"))
).count()
check("invalid_domain_rows", invalid_domains == 0, invalid_domains, 0)

duplicate = (
    new_data.groupBy("logical_hour", *KEY_COLUMNS)
    .count()
    .where(F.col("count") > 1)
    .agg(
        F.count(F.lit(1)).alias("groups"),
        F.sum(F.col("count") - 1).alias("extra_rows"),
        F.max("count").alias("max_rows"),
    )
    .first()
)
title("DEDUP KEY VALIDATION")
print(duplicate)
check("duplicated_key_groups", duplicate["groups"] == 0, duplicate["groups"], 0)
check("extra_duplicate_rows", (duplicate["extra_rows"] or 0) == 0,
      duplicate["extra_rows"] or 0, 0)


# ------------------------------------------------------------------
# 4. 输出样本追溯到 Step1，并核对 Label/行为字段
# ------------------------------------------------------------------

action = (
    spark.table(ACTION_TABLE)
    .where((F.col("dt") == BUSINESS_DATE) & (F.col("country_code") == COUNTRY))
    .select(
        "sample_domain",
        F.col("rec_id").cast("string").alias("rec_id"),
        F.col("shopid").cast("string").alias("shop_id"),
        F.col("label").cast("int").alias("action_label"),
        F.col("is_imp").cast("int").alias("action_imp"),
        F.col("is_clk").cast("int").alias("action_clk"),
        F.col("is_cat").cast("int").alias("action_cat"),
        F.col("is_cov").cast("int").alias("action_cov"),
    )
    .dropDuplicates(["sample_domain", "rec_id", "shop_id"])
    .withColumn("in_action", F.lit(1))
)

comparison = new_data.alias("o").join(
    action.alias("a"), ["sample_domain", "rec_id", "shop_id"], "left"
)
trace_rows = (
    comparison.groupBy("logical_hour", "sample_domain")
    .agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col("a.in_action").isNull(), 1).otherwise(0)).alias("orphans"),
        F.sum(F.when(F.col("a.in_action").isNotNull() & different(
            F.col("o.label"), F.col("a.action_label")), 1).otherwise(0)).alias("label_diff"),
        F.sum(F.when(F.col("a.in_action").isNotNull() & different(
            F.col("o.is_imp"), F.col("a.action_imp")), 1).otherwise(0)).alias("imp_diff"),
        F.sum(F.when(F.col("a.in_action").isNotNull() & different(
            F.col("o.is_clk"), F.col("a.action_clk")), 1).otherwise(0)).alias("clk_diff"),
        F.sum(F.when(F.col("a.in_action").isNotNull() & different(
            F.col("o.is_cat"), F.col("a.action_cat")), 1).otherwise(0)).alias("cat_diff"),
        F.sum(F.when(F.col("a.in_action").isNotNull() & different(
            F.col("o.is_cov"), F.col("a.action_cov")), 1).otherwise(0)).alias("cov_diff"),
    )
    .orderBy("logical_hour", "sample_domain")
    .collect()
)

title("STEP1 TRACEABILITY")
for row in trace_rows:
    print(row)
    for metric in ["orphans", "label_diff", "imp_diff", "clk_diff", "cat_diff", "cov_diff"]:
        check("%s_%s_%s" % (row["logical_hour"], row["sample_domain"], metric),
              (row[metric] or 0) == 0, row[metric] or 0, 0)


# ------------------------------------------------------------------
# 5. 新 nearby 与老 Step4 基线逐小时对比
# ------------------------------------------------------------------

old_columns = [
    name for name in reference_columns if name not in OLD_OUTPUT_ABSENT_COLUMNS
]
old_index = {name: position for position, name in enumerate(old_columns)}
baseline_source = [
    "recID", "s_id", "rank", "rank_type", "orderid",
    "label", "is_imp", "is_clk", "is_cat", "is_cov",
]
missing_old_mapping = sorted(set(baseline_source) - set(old_index))
check("old_baseline_mapping", not missing_old_mapping, missing_old_mapping, [])

old_raw = read_hourly_text([item["old_path"] for item in ROUTES])
old_width = (
    old_raw.groupBy("logical_hour", "field_count").count()
    .orderBy("logical_hour", "field_count").collect()
)
title("OLD BASELINE FIELD COUNT")
for row in old_width:
    print(row)

old_width_map = {}
for row in old_width:
    old_width_map.setdefault(row["logical_hour"], []).append(row)
for item in ROUTES:
    rows = old_width_map.get(item["logical_hour"], [])
    check(
        "%s_old_fixed_width" % item["logical_hour"],
        len(rows) == 1 and rows[0]["field_count"] == len(old_columns) and rows[0]["count"] > 0,
        [(row["field_count"], row["count"]) for row in rows],
        [(len(old_columns), ">0")],
    )

old_data = (
    old_raw.select(
        "logical_hour",
        *[F.col("parts").getItem(old_index[name]).alias(name) for name in baseline_source]
    )
    .select(
        "logical_hour",
        normalize("recID").alias("rec_id"),
        normalize("s_id").alias("shop_id"),
        normalize("rank").alias("rank"),
        normalize("rank_type").alias("rank_type"),
        normalize("orderid").alias("order_id"),
        F.col("label").cast("int").alias("label"),
        F.col("is_imp").cast("int").alias("is_imp"),
        F.col("is_clk").cast("int").alias("is_clk"),
        F.col("is_cat").cast("int").alias("is_cat"),
        F.col("is_cov").cast("int").alias("is_cov"),
    )
    .persist(StorageLevel.DISK_ONLY)
)

baseline_key = ["logical_hour", "rec_id", "shop_id", "rank", "rank_type", "order_id"]
normalized_key = [
    F.coalesce(F.col(name), F.lit("__NULL__")).alias(name) for name in baseline_key
]
old_base = (
    old_data.select(
        *normalized_key,
        F.col("label").alias("old_label"),
        F.col("is_imp").alias("old_imp"),
        F.col("is_clk").alias("old_clk"),
        F.col("is_cat").alias("old_cat"),
        F.col("is_cov").alias("old_cov"),
    )
    .dropDuplicates(baseline_key)
    .withColumn("in_old", F.lit(1))
)
new_base = (
    new_data.where(F.col("sample_domain") == "nearby")
    .select(
        *normalized_key,
        F.col("label").alias("new_label"),
        F.col("is_imp").alias("new_imp"),
        F.col("is_clk").alias("new_clk"),
        F.col("is_cat").alias("new_cat"),
        F.col("is_cov").alias("new_cov"),
    )
    .dropDuplicates(baseline_key)
    .withColumn("in_new", F.lit(1))
)

baseline_rows = (
    old_base.join(new_base, baseline_key, "full_outer")
    .groupBy("logical_hour")
    .agg(
        F.sum(F.when(F.col("in_old").isNotNull() & F.col("in_new").isNotNull(), 1)
              .otherwise(0)).alias("shared"),
        F.sum(F.when(F.col("in_old").isNotNull() & F.col("in_new").isNull(), 1)
              .otherwise(0)).alias("old_only"),
        F.sum(F.when(F.col("in_old").isNull() & F.col("in_new").isNotNull(), 1)
              .otherwise(0)).alias("new_only"),
        F.sum(F.when(
            F.col("in_old").isNotNull() & F.col("in_new").isNotNull()
            & (different(F.col("old_label"), F.col("new_label"))
               | different(F.col("old_imp"), F.col("new_imp"))
               | different(F.col("old_clk"), F.col("new_clk"))
               | different(F.col("old_cat"), F.col("new_cat"))
               | different(F.col("old_cov"), F.col("new_cov"))), 1
        ).otherwise(0)).alias("behavior_diff"),
    )
    .orderBy("logical_hour")
    .collect()
)

title("NEW NEARBY VS OLD BASELINE")
for row in baseline_rows:
    print(row)
    for metric in ["old_only", "new_only", "behavior_diff"]:
        check("%s_%s" % (row["logical_hour"], metric),
              (row[metric] or 0) == 0, row[metric] or 0, 0)


# ------------------------------------------------------------------
# 6. 最终结果
# ------------------------------------------------------------------

new_data.unpersist()
old_data.unpersist()

title("FINAL RESULT")
print("checked_hours=24")
print("failed_checks=%d" % len(failures))
for failure in failures:
    print("FAILURE=%s" % failure)
print("OVERALL_RESULT=%s" % ("PASS" if not failures else "FAIL"))

spark.stop()

if failures:
    raise RuntimeError(
        "Step4 full-day validation failed: %d checks failed" % len(failures)
    )
