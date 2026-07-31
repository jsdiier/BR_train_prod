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

## 目前包含的工具

- `compare_test_results.py`：对比**两个**实验分支的 `test.py` 输出日志，按 buy/cat/click/ext 四个任务输出 auc/gauc/uauc 对比表（含千分制绝对提升），并校验两边各 tower 的 pos 数量是否一致；支持指定 dt（test 的 end day）只在该 dt 的日志里取最新一份
