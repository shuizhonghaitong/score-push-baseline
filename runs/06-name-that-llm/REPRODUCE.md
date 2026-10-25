# 06-name-that-llm

此处对应原 `score-push/runs/06-name-that-llm/`，保存原历史代码及轻量记录；不包含后来新增的建模 skill 或效果工具。

历史参考分数：0.444。
当前核验：源码保留，完整训练未重跑。

原始入口和结果请从本目录代码、`RESULT.json`（如有）与 `experiments.csv`（如有）核对。源码可能包含多次尝试，不能把最后修改的文件自动当成历史最佳入口。

完整训练复现仍需核对实际数据、划分、环境与最佳候选入口；本快照没有把原先未完成的工作改写成“已复现”。

已记录的缺口：
- Recover row IDs and label-column mapping
- svc4_val.npz overlaps prefit training: excluded as trustworthy holdout
- Map public-best 0.444 to exact source and predictions

原机保留的大产物（不会随源码克隆）：
- `work/svc_proxy.npz`，SHA256 `977285642f066f926faa4805bfbc8654f4bffa8ad2cedaa81e4661789cf58941`。
