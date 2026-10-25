# 历史 baseline 索引

这些是原 agent 自主完成一场比赛过程中的代码与尝试；失败和未完成任务也保留。历史公榜分数不是本次重新取得的官方分数。

| 原 runs/ 目录 | 历史参考分数 | 已验证范围 |
|---|---:|---|
| [00-titanic-smoke](../runs/00-titanic-smoke/REPRODUCE.md) | 未确认 | 无已确认的此处历史公榜成绩；保留原代码／失败或未完成状态 |
| [00-titanic-smoke-v2](../runs/00-titanic-smoke-v2/REPRODUCE.md) | 0.77751 | 源码保留，完整训练未重跑 |
| [01-predicting-housing-affordability-in-the-u-s](../runs/01-predicting-housing-affordability-in-the-u-s/REPRODUCE.md) | 6.98 | 90 次 CPU 模型拟合已重跑，最终 8,534 行提交与历史最佳逐字节一致 |
| [02-soil-grain-size-from-photos](../runs/02-soil-grain-size-from-photos/REPRODUCE.md) | 未确认 | 无已确认的此处历史公榜成绩；保留原代码／失败或未完成状态 |
| [03-low-cost-motor-imagery-decoding-for-rehab](../runs/03-low-cost-motor-imagery-decoding-for-rehab/REPRODUCE.md) | 0.79 | 源码保留，完整训练未重跑 |
| [04-low-cost-motor-imagery-decoding-for-rehab-cross-subject](../runs/04-low-cost-motor-imagery-decoding-for-rehab-cross-subject/REPRODUCE.md) | 0.76 | 最佳提交文件 360 行逐字节重建；上游 ML/DL 未重训 |
| [05-misconception-detection-in-k-12-student-math-workings](../runs/05-misconception-detection-in-k-12-student-math-workings/REPRODUCE.md) | 0.50849 | 源码保留，完整训练未重跑 |
| [06-name-that-llm](../runs/06-name-that-llm/REPRODUCE.md) | 0.444 | 源码保留，完整训练未重跑 |
| [07-2026-ieee-big-data-traffic-flow-bench](../runs/07-2026-ieee-big-data-traffic-flow-bench/REPRODUCE.md) | 0.86515 | 历史成绩来自公开方案复现；本次未重训 |
| [08-umud-challenge-muscle-architecture-in-ultrasound-data](../runs/08-umud-challenge-muscle-architecture-in-ultrasound-data/REPRODUCE.md) | 0.44638 | 历史记录含回退行为，不能当作目标模型已成功运行 |
| [09-filament-segmentation-2026](../runs/09-filament-segmentation-2026/REPRODUCE.md) | 未确认 | 无已确认的此处历史公榜成绩；保留原代码／失败或未完成状态 |
| [10-the-3lc-low-light-detection-challenge](../runs/10-the-3lc-low-light-detection-challenge/REPRODUCE.md) | 0.59039 | 源码保留，完整训练未重跑 |
| [11-memecoins-pump-or-dump](../runs/11-memecoins-pump-or-dump/REPRODUCE.md) | 未确认 | 无已确认的此处历史公榜成绩；保留原代码／失败或未完成状态 |
| [12-neurips-2026-smart-buildings](../runs/12-neurips-2026-smart-buildings/REPRODUCE.md) | 未确认 | 无已确认的此处历史公榜成绩；保留原代码／失败或未完成状态 |
| [13-enveda-CASMI26-molecule-id-mass-spectra](../runs/13-enveda-CASMI26-molecule-id-mass-spectra/REPRODUCE.md) | 未确认 | 无已确认的此处历史公榜成绩；保留原代码／失败或未完成状态 |
| [14-gemma-4-developer-agent](../runs/14-gemma-4-developer-agent/REPRODUCE.md) | 未确认 | 无已确认的此处历史公榜成绩；保留原代码／失败或未完成状态 |
| [15-arc-prize-2026-arc-agi-2](../runs/15-arc-prize-2026-arc-agi-2/REPRODUCE.md) | 未确认 | 无已确认的此处历史公榜成绩；保留原代码／失败或未完成状态 |
| [16-arc-prize-2026-arc-agi-3](../runs/16-arc-prize-2026-arc-agi-3/REPRODUCE.md) | 未确认 | 无已确认的此处历史公榜成绩；保留原代码／失败或未完成状态 |
| [17-tabular-time-series-trading-signal-classification](../runs/17-tabular-time-series-trading-signal-classification/REPRODUCE.md) | 0.36833 | 源码保留，完整训练未重跑 |
| [18-climx-main-track](../runs/18-climx-main-track/REPRODUCE.md) | -0.07306 | 源码保留，完整训练未重跑 |
| [19-soil-grain-size-from-photos](../runs/19-soil-grain-size-from-photos/REPRODUCE.md) | 42.30133 | 源码保留，完整训练未重跑 |
| [20-squares-in-squares](../runs/20-squares-in-squares/REPRODUCE.md) | 未确认 | 无已确认的此处历史公榜成绩；保留原代码／失败或未完成状态 |

完整原始文件身份见 `source-manifest.json`，逐场声明与外部产物身份见 `baselines.json`。
完整训练的实际核验结果将保存在 `verification/housing-training.json`；提交文件重建报告在同目录。
