# Local GPU training

Prefer local GPU training and feature extraction. GPUs 2 and 3 form a shared
training pool, separate from the Qwen inference services on GPUs 4–7.
Each training GPU runs at most one job from this project at a time. Before
starting a job the worker checks utilization and leaves at least 8 GiB free
beyond the requested memory. It waits when resources are occupied.

Codex tool sandboxes hide NVIDIA devices. Use the queue below instead of assuming
that setting `CUDA_VISIBLE_DEVICES` grants access. The host worker exposes the
selected NVIDIA device and control nodes inside a separate sandbox. It permits
writes only in the requesting run and a private temporary directory. Training
runs without network access; prepare data and weights with ordinary tools first.

The cached CUDA environment is
`/mnt/disk3/liuyutong_home/miniconda3/envs/qwen/bin/python` (PyTorch 2.9.0 + CUDA 12.8).
Reuse it without modifying it. For extra packages, create a venv with this Python
and `--system-site-packages` in the run; retain its CUDA build of PyTorch.

From a run directory, submit a script:

```bash
/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python /mnt/disk3/liuyutong_home/workspace/skill/score-push/tools/local_training.py submit --memory-gib 16 train.py -- --epochs 10
```

The command returns a job directory. Query it periodically while doing other work:

```bash
/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python /mnt/disk3/liuyutong_home/workspace/skill/score-push/tools/local_training.py status /absolute/path/to/job
```

Use `--python /absolute/path/to/venv/bin/python` before the script argument if
using a run-local CUDA venv. Arguments after the script are passed to it unchanged.
The status records the actual GPU, exit code and timestamps; `stdout.log` and
`stderr.log` remain in the job directory. Save model outputs beneath the run.
Check `torch.cuda.is_available()` and fail explicitly if CUDA is unavailable.
Size the requested memory to the workload; a request larger than available
memory waits. Keep practical memory headroom inside the training script as well.

Kaggle remote execution remains available when required by competition rules or
when a workload cannot fit the local pool. Existing running Codex conversations
retain their original environment and instructions; this policy applies to new runs.
