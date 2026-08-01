"""
test_multi.py
比较一个基线实验和多个实验（1对多）的 test 结果，输出 buy / cat / click / ext 四个指标表格
每个实验行用两行展示：数值 + 相对基线的千分位提升（论文风格）
额外验证各实验各 tower 的 pos 数量是否与基线一致

用法:
    python test_multi.py <baseline_exp> <exp1> [exp2] [exp3] ... [dt]

示例:
    # 基线 + 4 个实验，各自取自己 log 目录下启动时间最新的 test_log
    python test_multi.py tf_train_base_new_try br_nearby_rank_deep_tower br_nearby_rank_mmoe br_nearby_rank_din_wide br_nearby_rank_cross_net

    # 指定 dt（test 的 end day）：只在该 dt 下取启动时间最新的 test_log，dt 放最后一个参数即可
    python test_multi.py tf_train_base_new_try br_nearby_rank_deep_tower br_nearby_rank_mmoe 20260724
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
# 3. 千分位提升格式化
# ──────────────────────────────────────────────
def format_delta(delta: float) -> str:
    """将千分数绝对提升格式化为带千分号的字符串，如 (+2.092‰) 或 (-1.500‰)"""
    sign = "+" if delta >= 0 else ""
    return f"({sign}{delta:.3f}‰)"


# ──────────────────────────────────────────────
# 4. 打印单任务对比表格（基线 + N 个实验）
# ──────────────────────────────────────────────
def print_comparison_table(
    task: str,
    baseline_name: str,
    exp_names: list,
    baseline_metrics: dict,
    all_exp_metrics: dict,
):
    bm = baseline_metrics.get(task)

    print("\n" + "=" * 70)
    print(f"  任务: {task.upper()}")
    print("=" * 70)

    if bm is None:
        print(f"  ⚠  基线缺少 {task} 任务数据: {baseline_name}")
        return

    rows = [[
        f"{baseline_name}(baseline)",
        f"{bm['auc']:.6f}",
        f"{bm['gauc']:.4f}",
        f"{bm['uauc']:.4f}",
    ]]

    for exp_name in exp_names:
        nm = all_exp_metrics[exp_name].get(task)
        if nm is None:
            rows.append([exp_name, "缺失数据", "", ""])
            continue
        auc_cell  = f"{nm['auc']:.6f}\n{format_delta((nm['auc'] - bm['auc']) * 1000)}"
        gauc_cell = f"{nm['gauc']:.4f}\n{format_delta((nm['gauc'] - bm['gauc']) * 1000)}"
        uauc_cell = f"{nm['uauc']:.4f}\n{format_delta((nm['uauc'] - bm['uauc']) * 1000)}"
        rows.append([exp_name, auc_cell, gauc_cell, uauc_cell])

    headers = ["experiment", "auc", "gauc", "uauc"]
    print(tabulate(
        rows, headers=headers, tablefmt="fancy_grid",
        stralign="center", colalign=("left", "center", "center", "center"),
    ))


# ──────────────────────────────────────────────
# 5. 验证各 tower 的 pos 是否与基线一致
# ──────────────────────────────────────────────
def print_pos_validation(
    baseline_name: str,
    exp_names: list,
    baseline_metrics: dict,
    all_exp_metrics: dict,
):
    print("\n" + "=" * 70)
    print("  POS 一致性验证")
    print("=" * 70)

    tasks = ("buy", "cat", "click", "ext")
    rows = []
    all_match = True

    def pos_row(name, metrics):
        vals = []
        for task in tasks:
            m = metrics.get(task)
            vals.append(m["pos"] if m and m.get("pos") is not None else "N/A")
        return vals

    base_vals = pos_row(baseline_name, baseline_metrics)
    rows.append([f"{baseline_name}(baseline)"] + base_vals + ["-"])

    for exp_name in exp_names:
        exp_vals = pos_row(exp_name, all_exp_metrics[exp_name])
        if any(v == "N/A" for v in base_vals + exp_vals):
            status = "miss"
            all_match = False
        elif exp_vals == base_vals:
            status = "yes"
        else:
            status = "no"
            all_match = False
        rows.append([exp_name] + exp_vals + [status])

    headers = ["experiment", "buy_pos", "cat_pos", "click_pos", "ext_pos", "status"]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", stralign="center", numalign="center"))

    if all_match:
        print("\n  ✅ 所有实验各 tower 的 pos 与基线完全一致，数据对齐正常。\n")
    else:
        print("\n  ❌ 存在 pos 与基线不一致（或缺失）的实验，请检查数据分片是否对齐！\n")


# ──────────────────────────────────────────────
# 6. 主流程
# ──────────────────────────────────────────────
def compare_experiments(baseline_exp: str, exps: list, dt: str = None):
    baseline_dir = resolve_exp_dir(baseline_exp)
    baseline_name = os.path.basename(baseline_dir)

    exp_dirs = [resolve_exp_dir(e) for e in exps]
    exp_names = [os.path.basename(d) for d in exp_dirs]

    if dt is not None:
        print(f"[INFO] 指定 dt (test end day): {dt}（仅在该 dt 下取启动时间最新的 test_log）")

    print(f"\n[INFO] 基线实验: {baseline_name}")
    print(f"[INFO] 实验目录:  {baseline_dir}")
    baseline_log = find_latest_test_log(baseline_dir, dt=dt)
    print(f"[INFO] 使用 log 文件: {baseline_log}")
    baseline_metrics = parse_test_log(baseline_log)

    all_exp_metrics = {}
    for name, dir_ in zip(exp_names, exp_dirs):
        print(f"\n[INFO] 对比实验: {name}")
        print(f"[INFO] 实验目录:  {dir_}")
        log = find_latest_test_log(dir_, dt=dt)
        print(f"[INFO] 使用 log 文件: {log}")
        all_exp_metrics[name] = parse_test_log(log)

    # ── 指标对比表格 ──
    for task in ("buy", "cat", "click", "ext"):
        print_comparison_table(task, baseline_name, exp_names, baseline_metrics, all_exp_metrics)

    # ── pos 一致性验证 ──
    print_pos_validation(baseline_name, exp_names, baseline_metrics, all_exp_metrics)


if __name__ == "__main__":
    args = sys.argv[1:]

    # 最后一个参数若是 8 位纯数字，视为 dt（test end day），否则视为实验名
    dt_arg = None
    if args and re.fullmatch(r"\d{8}", args[-1]):
        dt_arg = args[-1]
        args = args[:-1]

    if len(args) < 2:
        print("用法: python test_multi.py <baseline_exp> <exp1> [exp2] ... [dt]")
        print("示例: python test_multi.py tf_train_base_new_try br_nearby_rank_deep_tower br_nearby_rank_mmoe")
        print("      python test_multi.py tf_train_base_new_try br_nearby_rank_deep_tower br_nearby_rank_mmoe 20260724")
        sys.exit(1)

    compare_experiments(args[0], args[1:], dt=dt_arg)
