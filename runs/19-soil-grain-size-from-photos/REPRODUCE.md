# 19-soil-grain-size-from-photos

此处对应原 `score-push/runs/19-soil-grain-size-from-photos/`，保存原历史代码及轻量记录；不包含后来新增的建模 skill 或效果工具。

历史参考分数：42.30133。
当前核验：源码保留，完整训练未重跑。

原始入口和结果请从本目录代码、`RESULT.json`（如有）与 `experiments.csv`（如有）核对。源码可能包含多次尝试，不能把最后修改的文件自动当成历史最佳入口。

完整训练复现仍需核对实际数据、划分、环境与最佳候选入口；本快照没有把原先未完成的工作改写成“已复现”。

已记录的缺口：
- Recover sample IDs/labels for OOF; top10 differs from hand3 candidate
- Link recovery to failed run02 without discarding failure evidence

原机保留的大产物（不会随源码克隆）：
- `state/checkpoint.json`，SHA256 `80032659556447237da359ef3d103509e7c7c4d9472fe4a407a580499b2c7522`。
- `submissions/ens67_hand_grai.csv`，SHA256 `2323bca407fd9dd3358547fc4ba808d040d67dd8e508ae6e4e21213a0ba6bdb0`。
- `state/oof_top10.npz`，SHA256 `66e3d61597fcda1818a96f676dcd9f8c2568cf9eb10e7aaf2e986b14d2273d15`。
