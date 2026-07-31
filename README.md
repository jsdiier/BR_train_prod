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

## 目前包含的工具

- `compare_test_results.py`：对比**两个**实验分支的 `test.py` 输出日志，按 buy/cat/click/ext 四个任务输出 auc/gauc/uauc 对比表（含千分制绝对提升），并校验两边各 tower 的 pos 数量是否一致；支持指定 dt（test 的 end day）只在该 dt 的日志里取最新一份
- `run_exp.sh`：只支持显式传实验文件夹名列表，逐个 `cd` 进对应目录 `nohup bash submit_luban.sh &`，日志落在各自实验文件夹内部（`nohup_submit.log`），最后打印 PID 汇总表（风格参照 `tf_rank/run_test_zs.sh`）
