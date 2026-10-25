import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
exec(open("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification/code/cv2.py").read().split('print("features"')[0])

def run(name, start_frac, K):
    folds = []
    start0 = int(N * start_frac)
    seglen = (N - start0) // K
    for i in range(K):
        a = start0 + i * seglen
        b = N if i == K - 1 else start0 + (i + 1) * seglen
        folds.append((a, b))
    aps = []
    for (a, b) in folds:
        g = (LR[a:b] >= PROM).astype(float)
        score = g * (1.0 + LR[a:b])
        aps.append(ap(Y[a:b], score))
    print(f"{name:28s} mean={np.mean(aps):.4f} all={[round(x,3) for x in aps]}")

run("gate+lr, last60% x10", 0.40, 10)
run("gate+lr, last50% x10", 0.50, 10)
run("gate+lr, last70% x10", 0.30, 10)
run("gate+lr, last40% x8", 0.60, 8)
run("gate+lr, last30% x6", 0.70, 6)
# sensitivity: PROM variants
for p2 in (0.0099, 0.00995, 0.01):
    folds = []
    start0 = int(N * 0.40); seglen = (N - start0) // 10
    for i in range(10):
        a = start0 + i * seglen; b = N if i == 9 else start0 + (i + 1) * seglen
        folds.append((a, b))
    aps = [ap(Y[a:b], ((LR[a:b] >= p2) * (1 + LR[a:b]))) for a, b in folds]
    print(f"gate+lr prom={p2}              mean={np.mean(aps):.4f}")
