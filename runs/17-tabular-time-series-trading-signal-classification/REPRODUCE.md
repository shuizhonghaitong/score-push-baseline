# 17-tabular-time-series-trading-signal-classification

此处对应原 `score-push/runs/17-tabular-time-series-trading-signal-classification/`，保存原历史代码及轻量记录；不包含后来新增的建模 skill 或效果工具。

历史参考分数：0.36833。
当前核验：源码保留，完整训练未重跑。

原始入口和结果请从本目录代码、`RESULT.json`（如有）与 `experiments.csv`（如有）核对。源码可能包含多次尝试，不能把最后修改的文件自动当成历史最佳入口。

完整训练复现仍需核对实际数据、划分、环境与最佳候选入口；本快照没有把原先未完成的工作改写成“已复现”。

已记录的缺口：
- Resolve v2b clipping repair and source-to-submission identity
- Persist actual time blocks; tuning blocks are not independent test

原机保留的大产物（不会随源码克隆）：
- `data/archive_btc_train.csv`，SHA256 `6cc0bbae56fea945a5641e4a02d5adfc24516906f25f4bfd6d47490ab9b36829`。
- `outputs/submission_causal_v2b.csv`，SHA256 `b0ba0ba85cb0b10cf46fa79cb805f8cbbfb3451d09e25e2eb5a08ee0e23924ff`。
