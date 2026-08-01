# shared_tools

跟模型代码无关的公用脚本存放分支，不属于任何具体实验分支。

## 用法

在 k8s 上把这个分支单独 clone 成一个跟各实验分支同级的目录，例如：

```bash
cd ~/rank-ssl/chenpinyuan/tf_rank_BR
git clone -b shared_tools https://github.com/jsdiier/tf_rank_BR.git shared_tools
```

目录结构会是：

```
tf_rank_BR/
├── br_nearby_rank_base/
├── br_nearby_rank_dev/
├── tf_train_base_new_try/
└── shared_tools/          <- 这个分支
    └── compare_test_results.py
```

`compare_test_results.py` 默认用 `BASE_DIR = "/home/luban/rank-ssl/chenpinyuan/tf_rank_BR"` 拼实验目录，
实验名传绝对路径也可以。依赖 `tabulate`，没装的话先 `pip3 install tabulate`。

```bash
cd shared_tools
python3 compare_test_results.py br_nearby_rank_base br_nearby_rank_dev

# 指定 dt（test 的 end day），只在该 dt 下取启动时间最新的 test_log
python3 compare_test_results.py br_nearby_rank_base br_nearby_rank_dev 20260724
```

`run_exp.sh` 依次拉起多个实验目录下的 `submit_luban.sh`（全部放后台并行跑）：

```bash
bash run_exp.sh br_nearby_rank_lhuc_gate br_nearby_rank_mmoe br_nearby_rank_din_wide br_nearby_rank_ext_focal_loss
```

`test_multi.py` 支持**一个基线 + 多个实验**同时对比（`compare_test_results.py` 只能两两对比）：

```bash
cd shared_tools
python3 test_multi.py tf_train_base_new_try br_nearby_rank_deep_tower br_nearby_rank_mmoe br_nearby_rank_din_wide br_nearby_rank_cross_net

# 指定 dt（test 的 end day），放在参数最后即可，只要是 8 位纯数字就会被识别为 dt 而不是实验名
python3 test_multi.py tf_train_base_new_try br_nearby_rank_deep_tower br_nearby_rank_mmoe 20260724
```

## 目前包含的工具

- `compare_test_results.py`：对比**两个**实验分支的 `test.py` 输出日志，按 buy/cat/click/ext 四个任务输出 auc/gauc/uauc 对比表（含千分制绝对提升），并校验两边各 tower 的 pos 数量是否一致；支持指定 dt（test 的 end day）只在该 dt 的日志里取最新一份
- `test_multi.py`：对比**一个基线 + 任意多个实验**的 `test.py` 输出日志，按 buy/cat/click/ext 四个任务输出对比表；每个实验行用两行展示（数值 + 相对基线的千分位提升，论文风格），并校验各实验各 tower 的 pos 是否与基线一致；同样支持指定 dt
- `run_exp.sh`：只支持显式传实验文件夹名列表，逐个 `cd` 进对应目录 `nohup bash submit_luban.sh &`，日志落在各自实验文件夹内部（`nohup_submit.log`），最后打印 PID 汇总表（风格参照 `tf_rank/run_test_zs.sh`）
