# score-push baseline

这是原工程 **`score-push/` 暂停时工作树的独立源码快照**，供同事阅读和保存已有 baseline。GitHub 仓库根目录直接对应原 `score-push/` 的结构；本机维护副本是相邻的 `score-push-baseline/`。

这里保留原调度代码、原 Kaggle 平台 skill、22 次历史任务的代码和轻量结果。原 baseline 内的多轮建模尝试、失败和恢复记录也保留；它们不全是最终胜出的模型。
后来新增的 `ml-experiment-loop` skill、`competition_loop` 工具原型、效果实验设计、项目聊天梳理均不在本仓库，也没有带入包含这些草稿的旧 Git 历史。

## 同事从哪里看

| 位置 | 内容 |
|---|---|
| **[比赛索引](reproduce/BASELINES.md)** | 历史分数、原目录入口和实际复现状态 |
| `tools/` | 原调度、训练队列、状态与记录工具，逐字节保存，未做 lab 版重构 |
| `skills/kaggle/` | 原 baseline 使用的平台 skill |
| `runs/<原任务目录>/` | 原建模代码、Notebook、结果与实验表；每场有 `REPRODUCE.md` |
| `plan.json`、`followup-plan.json`、`prompt.template.txt` | 暂停时的原计划和任务指导，保持历史内容 |
| `reproduce.py`、`reproduce/` | 本次新增的独立离线核验、输入恢复与训练复现入口 |

原 `AGENTS.md` 改名为 `AGENTS.original.txt`，避免旧运行指令在同事阅读目录时自动生效。Notebook 代码单元保留，输出、执行计数和 Notebook 级运行元数据已去除。所有变换及原始哈希均记录于 [source-manifest.json](reproduce/source-manifest.json)。

## 克隆后先做的两项检查

Python 3.10 以上即可，不需要 GPU、Kaggle 登录或访问原机目录：

```bash
python -B reproduce.py verify
python -B reproduce.py rebuild eeg
```

第一条核对保留文件的 SHA256。第二条使用随仓库保存的 ML/DL 概率，重建 EEG Cross 历史最佳提交：360 行、SHA256 `9cdcbee0421bc33c667917cf12c8925baf74cac57d0a551c4cf79ed49e1ab4ae`。
输出写入忽略目录 `artifacts/`；不会正式提交，也不会改动 `runs/` 中的参考文件。这证明提交文件生成过程可复现，不代表重新训练了 EEG 模型或重新取得公榜 0.76。

## Housing：重建历史最佳提交

使用 Python 3.11，依赖版本在 [replay-requirements.txt](reproduce/environments/replay-requirements.txt)。本机直接复用已有 `kaggle` 环境，无需安装或下载。

原机可从相邻原目录恢复已核验的官方 CSV：

```bash
python -B reproduce.py restore --source-root ../score-push --task housing
python -B reproduce.py rebuild housing
```

换到其他机器时，将有权使用的官方文件放到：

```text
runs/01-predicting-housing-affordability-in-the-u-s/data/train.csv
runs/01-predicting-housing-affordability-in-the-u-s/data/test.csv
```

文件身份见 [required-inputs.json](reproduce/required-inputs.json)。官方数据来源是 [Housing 比赛](https://www.kaggle.com/competitions/predicting-housing-affordability-in-the-u-s/data)；此脚本只验证和使用本地输入，不自动下载。

重建使用随仓库保存的小型预测与 OOF，恢复 Ridge 融合及已见 ZIP 漂移校正，得到与历史 `v4b_stack.csv` 逐字节相同的 8,534 行提交。它对应历史公榜 6.98；本次不重新正式提交。

## Housing：从原始输入重训基础模型

依赖在 [housing-training-requirements.txt](reproduce/environments/housing-training-requirements.txt)。除官方 train/test CSV，还需当时使用的 `external/us-zip-codes-demographics/zips.csv`；精确哈希在输入清单中。外部数据源按原 `RESULT.json` 记录为 [cybermaxtools/us-zip-codes-demographics](https://www.kaggle.com/datasets/cybermaxtools/us-zip-codes-demographics)，不同修订须先核对哈希。

```bash
python -B reproduce.py restore --source-root ../score-push --task housing --purpose training
python -B reproduce.py train-housing --output-root artifacts/housing-training
```

这是一次明确的 CPU 重训：原 `scripts/oof_stack.py` 的 30 次拟合、原 `scripts/stack_full.py` 的 60 次拟合，使用原 16 线程设置；输出目录必须不存在。脚本和数据会复制到该新目录，原代码及参考产物不被覆盖。

原 `stack_full.py` 保存的是三个 seed 预测之和，曾导致错误的 v4 提交。本次入口保留原脚本，再应用历史 v4b 的除以 3 修正和 Ridge 截距，最后生成正确候选；该修正不是新建模策略。
**2026-10-25 已实际完成全部 90 次拟合**，OOF RMSE 为 7.926081285926268，最后生成的文件与历史最佳提交逐字节一致，SHA256 为 `d12d391ff310fc500158803542a0f5642cb2fb047f1dd44a8f0e08355b73277f`。训练和预测只使用本地数据，没有正式提交。
训练报告分别记录“是否执行完”和“与历史最佳文件是否逐字节一致”；完整证据见 [housing-training.json](reproduce/verification/housing-training.json) 及同目录的两个训练摘要日志。

## 数据、环境和可复现范围

- 原始数据、大模型、训练权重、完整会话、凭据均不上传。两场用于核验的小型参考预测和历史最终提交通过明确清单保留；这些不是训练权重。
- 本机恢复的原始数据仍在忽略目录中；克隆到另一台机器不会自动带上这些数据。大产物原位置和已有哈希见 [baselines.json](reproduce/baselines.json)。
- 原机另存 `.local/housing-inputs-20261025.tar.gz`，包含上述三份精确 Housing 输入和清单，供本地留档；它不上传 Git。需要恢复时先解压到可信输入目录，再以该目录作为 `restore --source-root`。
- [环境记录](reproduce/environments/) 保存当前 CPU/GPU 环境的 Python 和安装包版本，不冒充每次历史运行的完整环境锁。
- 其他比赛的原代码和历史结果已保存，但逐场完整训练、实际划分和最终入口仍有缺口，见比赛索引。历史公榜分数不作为本次新取得的分数。
- 原调度器包含原机路径、Qwen 服务和账号依赖，保留作源码参考。直接重跑整个自主参赛流程还需要配置这些服务及权限；它不是确定性重放同一条 agent 轨迹。日常离线核验使用上面的 `reproduce.py` 入口。

## 校验

```bash
python -B -m unittest discover -s tools -p 'test_*.py'
python -B -m unittest discover -s reproduce/tests -p 'test_*.py'
```

原始 `score-push/` 和研发仓库 `kaggle-experiment-lab` 不在此次发布中被修改或替换。

发布快照标签为 `baseline-20261025`。独立克隆、输入恢复、两项提交重建及测试结果见 [clean-clone.json](reproduce/verification/clean-clone.json)。该检查使用 Python 审计钩子拒绝原工程路径读取和网络连接，用于核对依赖可移植性，不作为对抗性安全沙箱证明。
