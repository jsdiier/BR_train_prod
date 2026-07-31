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

`compare_test_results.py` 默认去脚本所在目录的**上一级**（也就是 `tf_rank_BR/`）里找 `<实验名>/log/test_log_*`，
所以只要 `shared_tools` 和各实验目录是兄弟目录，直接跑就行：

```bash
cd shared_tools
python3 compare_test_results.py br_nearby_rank_base br_nearby_rank_dev tf_train_base_new_try
```

## 目前包含的工具

- `compare_test_results.py`：对比多个实验分支的 `test.py` 输出日志，按 buy/cat/click/ext 四个任务输出 auc/gauc/uauc 等指标对比表
