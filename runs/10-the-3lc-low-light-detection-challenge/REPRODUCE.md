# 10-the-3lc-low-light-detection-challenge

此处对应原 `score-push/runs/10-the-3lc-low-light-detection-challenge/`，保存原历史代码及轻量记录；不包含后来新增的建模 skill 或效果工具。

历史参考分数：0.59039。
当前核验：源码保留，完整训练未重跑。

原始入口和结果请从本目录代码、`RESULT.json`（如有）与 `experiments.csv`（如有）核对。源码可能包含多次尝试，不能把最后修改的文件自动当成历史最佳入口。

完整训练复现仍需核对实际数据、划分、环境与最佳候选入口；本快照没有把原先未完成的工作改写成“已复现”。

已记录的缺口：
- Locate validation detections and actual 733-image split
- Model retained locally; not loaded during audit

原机保留的大产物（不会随源码克隆）：
- `outputs/tuned200/submission.csv`，SHA256 `dee219f5674880c74e6672d2ecd5c80283ca99c0b091ac200f914660f1ff1ed4`。
- `models/runs/tuned200/weights/best.pt`，SHA256 `cd06341eee2a1a617ff2d02ffaac0ead17c69f7a7ab50100a9ae4d89137519f7`。
