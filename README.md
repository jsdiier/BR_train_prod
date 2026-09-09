# BR_train_prod_remain1700_ad_segment_diagnostic

隔离的 test-only 诊断，用于解释 Top1700 整体离线收益与线上广告指标负向之间的差异。

固定口径：

- baseline：`BR_train_prod_bs_lr_ema_weights` 的 `20260825` checkpoint；
- candidate：`BR_train_prod_ema_interest_feature_tfrecord_fixed_remain1700_e2e` 的 `20260825` checkpoint；
- 数据：`chenpinyuan/hash_fea_new_fixed/train/20260826～20260828`；
- 广告字段：`is_ad`，由 slot 1044 的二值 FID 表示；
- 输出：广告/自然四任务 AUC、LogLoss、分数分布及同请求广告相对自然店铺分差。

slot 1044 在每条样本中都存在，并由两个 FID 分别表示 `is_ad=0/1`。当前配置的
`dummy,1` 语义是：原始值 1 命中枚举列表的第 0 桶，原始值 0 进入第 1 个兜底桶。
20260826～20260828 的 10 万样本审计确认：

- `293859875685924864`：`is_ad=1`，广告样本；
- `293859875685924865`：`is_ad=0`，自然样本。

禁止使用“slot 是否存在”区分广告。若生产分桶配置发生变化，必须重新审计映射。

先执行 FID 审计：

```bash
bash run_ad_segment_diagnostic.sh inspect
```

根据输出样本到原始特征日志确认哪个 FID 对应 `is_ad=1` 后，再运行完整诊断：

```bash
nohup bash run_ad_segment_diagnostic.sh run 293859875685924864 \
  > nohup_ad_segment_diagnostic.log 2>&1 &
```

最终结果位于：

```text
log/ad_segment_ckpt_20260825_20260826_20260828/ad_segment_metrics.tsv
log/ad_segment_ckpt_20260825_20260826_20260828/ad_segment_summary.json
```
