"""
compare_test_results.py
比较两个实验（最多两组）的 test 结果，输出 buy / cat / click / ext 四个指标表格
额外验证两个实验各 tower 的 pos 数量是否一致
用法:
    python compare_test_results.py <baseline_exp> <new_exp> [dt]
示例:

    # 两个都用默认 BASE_DIR，各自取自己 log 目录下启动时间最新的 test_log
    python compare_test_results.py br_nearby_rank_base br_nearby_rank_dev

    # baseline 用绝对路径，new 用默认 BASE_DIR
    python compare_test_results.py /home/luban/rank-ssl/chenpinyuan/tf_rank_BR/br_nearby_rank_base br_nearby_rank_dev

    # 两个都用绝对路径
    python compare_test_results.py /path/to/exp_a /path/to/exp_b

    # 指定 dt（test 的 end day）：只在该 dt（test_log_<dt>_<启动时间> 的第一段）下的日志里，取启动时间最新的那份
    python compare_test_results.py br_nearby_rank_base br_nearby_rank_dev 20260724
"""

import os
import re
import sys
from tabulate import tabulate

BASE_DIR = "/home/luban/rank-ssl/chenpinyuan/tf_rank_BR"


# ──────────────────────────────────────────────
# 0. 将实验名/路径解析为完整实验目录
# ──────────────────────────────────────────────
def resolve_exp_dir(exp: str) -> str:
    """
    若 exp 是绝对路径则直接使用，否则拼接 BASE_DIR。
    返回实验根目录（不含 /log）。
    """
    if os.path.isabs(exp):
        return exp
    return os.path.join(BASE_DIR, exp)


# ──────────────────────────────────────────────
# 1. 找最新的 test_log 文件
# ──────────────────────────────────────────────
def find_latest_test_log(exp_dir: str, dt: str = None) -> str:
    """
    在 <exp_dir>/log/ 下找 test_log_* 文件。
    文件名格式: test_log_<dt数据截止日>_<启动时间戳>

    - dt 为 None: 在全部 test_log_* 里，按启动时间戳取最大的一份（原有行为）。
    - dt 指定时: 只在第一段等于 dt（即 test 的 end day）的 test_log_* 里，按启动时间戳取最大的一份。
    """
    log_dir = os.path.join(exp_dir, "log")
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"log 目录不存在: {log_dir}")

    pattern = re.compile(r"^test_log_(\d+)_(\d+)$")
    candidates = []
    for fname in os.listdir(log_dir):
        m = pattern.match(fname)
        if not m:
            continue
        file_dt, start_ts = m.group(1), m.group(2)
        if dt is not None and file_dt != dt:
            continue
        candidates.append((start_ts, fname))

    if not candidates:
        if dt is not None:
            raise FileNotFoundError(f"在 {log_dir} 下未找到 dt={dt} 的 test_log_* 文件")
        raise FileNotFoundError(f"在 {log_dir} 下未找到 test_log_* 文件")

    candidates.sort(key=lambda x: x[0])
    latest_fname = candidates[-1][1]
    return os.path.join(log_dir, latest_fname)


# ──────────────────────────────────────────────
# 2. 解析日志内容
# ──────────────────────────────────────────────
def parse_test_log(log_path: str) -> dict:
    """
    读取 log 文件末尾，解析各任务的 auc / gauc / uauc / pos。
    返回结构:
    {
        "buy":   {"auc": float, "gauc": float, "uauc": float, "pos": int or None},
        "cat":   {...},
        "click": {...},
        "ext":   {...},
    }
    """
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    tail_lines = lines[-50:]

    results = {}

    # 日志格式:
    #   test_buy auc:0.812845 gauc:0.650200 uauc:0.656800 size:6737113 loss:0.173427, pos: 363942
    # pos 字段可能不存在（online_xxx 行就没有）
    metric_pattern = re.compile(
        r"test_(\w+)\s+"
        r"auc:([\d.]+)\s+"
        r"gauc:([\d.]+)\s+"
        r"uauc:([\d.]+)"
        r"(?:.*?pos:\s*(\d+))?"          # pos 可选
    )

    for line in tail_lines:
        m = metric_pattern.search(line)
        if m:
            task = m.group(1)
            results[task] = {
                "auc":  float(m.group(2)),
                "gauc": float(m.group(3)),
                "uauc": float(m.group(4)),
                "pos":  int(m.group(5)) if m.group(5) is not None else None,
            }
    return results


# ──────────────────────────────────────────────
# 3. 打印对比表格（tabulate）
# ──────────────────────────────────────────────
def format_delta(delta: float) -> str:
    """将千分数绝对提升格式化为带千分号的字符串，如 +2.092‰ 或 -1.500‰"""
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}‰"


def print_comparison_table(
    task: str,
    baseline_name: str,
    new_name: str,
    baseline_metrics: dict,
    new_metrics: dict,
):
    bm = baseline_metrics.get(task)
    nm = new_metrics.get(task)

    col_base  = f"baseline\n{baseline_name}"
    col_new   = f"new\n{new_name}"
    col_delta = "absolutely \ndelta"

    if bm is None or nm is None:
        missing = baseline_name if bm is None else new_name
        print(f"  ⚠  缺少数据: {missing}")
        return

    rows = []
    for metric in ("auc", "gauc", "uauc"):
        b_val = bm[metric]
        n_val = nm[metric]
        delta = (n_val - b_val) * 1000
        rows.append([
            f"test_{metric}",
            f"{b_val:.6f}",
            f"{n_val:.6f}",
            format_delta(delta),
        ])

    headers = [task.upper(), col_base, col_new, col_delta]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", stralign="center", numalign="center"))


# ──────────────────────────────────────────────
# 4. 验证各 tower 的 pos 是否一致
# ──────────────────────────────────────────────
def print_pos_validation(
    baseline_name: str,
    new_name: str,
    baseline_metrics: dict,
    new_metrics: dict,
):
    """
    打印两个实验各 tower 的 pos 对比表，并标注是否一致。
    """
    print("\n" + "=" * 60)
    print("  POS 一致性验证")
    print("=" * 60)

    tasks = ("buy", "cat", "click", "ext")
    rows = []
    all_match = True

    for task in tasks:
        bm = baseline_metrics.get(task)
        nm = new_metrics.get(task)

        b_pos = bm["pos"] if bm and bm.get("pos") is not None else "N/A"
        n_pos = nm["pos"] if nm and nm.get("pos") is not None else "N/A"

        if b_pos == "N/A" or n_pos == "N/A":
            status = "miss"
            all_match = False
        elif b_pos == n_pos:
            status = "yes"
        else:
            status = "no"
            all_match = False

        rows.append([task.upper(), b_pos, n_pos, status])

    headers = [
        "TOWER",
        f"baseline pos\n{baseline_name}",
        f"new pos\n{new_name}",
        "pos_num",
    ]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", stralign="center", numalign="center"))

    if all_match:
        print("\n  ✅ 所有 tower 的 pos 完全一致，数据对齐正常。\n")
    else:
        print("\n  ❌ 存在 pos 不一致的 tower，请检查数据分片是否对齐！\n")


# ──────────────────────────────────────────────
# 5. 主流程
# ──────────────────────────────────────────────
def compare_experiments(baseline_exp: str, new_exp: str, dt: str = None):
    baseline_dir = resolve_exp_dir(baseline_exp)
    new_dir      = resolve_exp_dir(new_exp)

    # 显示名取路径最后一段，方便表格对齐
    baseline_name = os.path.basename(baseline_dir)
    new_name      = os.path.basename(new_dir)

    if dt is not None:
        print(f"[INFO] 指定 dt (test end day): {dt}（仅在该 dt 下取启动时间最新的 test_log）")

    print(f"\n[INFO] baseline 实验: {baseline_name}")
    print(f"[INFO] 实验目录:       {baseline_dir}")
    baseline_log = find_latest_test_log(baseline_dir, dt=dt)
    print(f"[INFO] 使用 log 文件:  {baseline_log}")

    print(f"\n[INFO] new 实验:      {new_name}")
    print(f"[INFO] 实验目录:       {new_dir}")
    new_log = find_latest_test_log(new_dir, dt=dt)
    print(f"[INFO] 使用 log 文件:  {new_log}")

    baseline_metrics = parse_test_log(baseline_log)
    new_metrics      = parse_test_log(new_log)

    # ── 指标对比表格 ──
    for task in ("buy", "cat", "click", "ext"):
        print_comparison_table(
            task,
            baseline_name,
            new_name,
            baseline_metrics,
            new_metrics,
        )

    # ── pos 一致性验证 ──
    print_pos_validation(baseline_name, new_name, baseline_metrics, new_metrics)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("用法: python compare_test_results.py <baseline_exp> <new_exp> [dt]")
        print("示例: python compare_test_results.py br_nearby_rank_base br_nearby_rank_dev")
        print("      python compare_test_results.py /abs/path/to/exp_a br_nearby_rank_dev")
        print("      python compare_test_results.py br_nearby_rank_base br_nearby_rank_dev 20260724")
        sys.exit(1)

    dt_arg = sys.argv[3] if len(sys.argv) == 4 else None
    compare_experiments(sys.argv[1], sys.argv[2], dt=dt_arg)
