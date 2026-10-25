# 04-low-cost-motor-imagery-decoding-for-rehab-cross-subject

此处对应原 `score-push/runs/04-low-cost-motor-imagery-decoding-for-rehab-cross-subject/`，保存原历史代码及轻量记录；不包含后来新增的建模 skill 或效果工具。

历史参考分数：0.76。
当前核验：最佳提交文件 360 行逐字节重建；上游 ML/DL 未重训。

原始入口和结果请从本目录代码、`RESULT.json`（如有）与 `experiments.csv`（如有）核对。源码可能包含多次尝试，不能把最后修改的文件自动当成历史最佳入口。

在仓库根目录执行 `python reproduce.py rebuild eeg`，只使用本仓库保留的两份概率文件即可重建历史最佳提交；不需原始 EEG、GPU、登录或访问旧目录。

已记录的缺口：
- Verify DL OOF/test provenance for public-best v5 blend
- Metric replay does not prove training exclusion

原机保留的大产物（不会随源码克隆）：
- `submissions/blend_v5_cpu.csv`，SHA256 `9cdcbee0421bc33c667917cf12c8925baf74cac57d0a551c4cf79ed49e1ab4ae`。
- `submissions/nb_v3f_out/oof_ml.csv`，SHA256 `961ee3b73988755ae5a3a4158e12d982132973405305416de9c813bf82aa8da0`。
- `submissions/nb_dlcv_out/oof_dl.csv`，SHA256 `3aceb0c7f7c03b10563d8dc6e8a2ca05b0442a2bb2f35bac0551e5ed939cfb00`。
- `submissions/nb_dlcpu_out/test_probs.csv`，SHA256 `0e5bfe4926da0708903d808fb85076c39b192fc0f6dfc83f1cac8a8986dbffa4`。
- `submissions/nb_v3f_out/test_probs.csv`，SHA256 `71bb724f45f7bf5a47671c5ab78ec94a853fef757bde77ed966253bf2489a594`。
