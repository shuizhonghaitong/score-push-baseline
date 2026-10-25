import os, sys
print("exe:", sys.executable, flush=True)
print("prefix:", sys.prefix, flush=True)
print("path:", sys.path, flush=True)
p = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/10-the-3lc-low-light-detection-challenge/venv/lib/python3.11/site-packages"
print("site exists:", os.path.isdir(p), flush=True)
if os.path.isdir(p):
    print("site contents:", sorted(os.listdir(p))[:20], flush=True)
try:
    import ultralytics
    print("ultralytics OK:", ultralytics.__file__, flush=True)
except Exception as e:
    print("ultralytics FAIL:", repr(e), flush=True)
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
