# 01-predicting-housing-affordability-in-the-u-s

此处对应原 `score-push/runs/01-predicting-housing-affordability-in-the-u-s/`，保存原历史代码及轻量记录；不包含后来新增的建模 skill 或效果工具。

历史参考分数：6.98。
当前核验：90 次 CPU 模型拟合已重跑，最终 8,534 行提交与历史最佳逐字节一致。

原始入口和结果请从本目录代码、`RESULT.json`（如有）与 `experiments.csv`（如有）核对。源码可能包含多次尝试，不能把最后修改的文件自动当成历史最佳入口。

在仓库根目录使用 `python reproduce.py rebuild housing` 重建最佳提交。需要先恢复官方 train/test CSV；实际指令见根目录 README。

完整训练入口：`python reproduce.py train-housing --output-root artifacts/housing-training`。它在新目录执行原 `scripts/oof_stack.py`、`scripts/stack_full.py`，随后应用 v4b 的三 seed 平均修正。原脚本保留错误的 v4 生成过程，未被悄悄改写。

已记录的边界：
- Official hidden-score replay not performed; preserved historical public score is 6.98
- Ridge was selected on development OOF labels; this is not an independent final test

原机保留的大产物（不会随源码克隆）：
- `subs/v4b_stack.csv`，SHA256 `d12d391ff310fc500158803542a0f5642cb2fb047f1dd44a8f0e08355b73277f`。
- `models/oof_stack/oof.json`，SHA256 `341f0985eca5a678e231ee661e1727f051b28ea5a23cd0e978f1ed84ef80c7e3`。
- `models/oof_stack/summary.json`，SHA256 `333e39b56ba67109944849ba88e273c27b582b7a233bbf1bfb2db9e0a894e5b7`。
- `data/train.csv`，SHA256 `3a74b5fc7edb44e2e060ae0a300af538161494cd337330c65a7f95d680fc5cb2`。
- `data/test.csv`，SHA256 `1f56c675fb742db79f8bad1ca859c8b4db118b3976717c72b1c41a34834740ea`。
- `models/oof_stack/test_3seed.npz.npy`，SHA256 `e22afa2ae2cd09e3283173d5408c5e1e3643d152da4a09b15ca3c5589639ba55`。
- `models/oof_stack/blend_3seed_fixed.npy`，SHA256 `2c3a0d0ecbde1ff1dade2d927948611d93758060ec33ef26e13316c57293a9c4`。
