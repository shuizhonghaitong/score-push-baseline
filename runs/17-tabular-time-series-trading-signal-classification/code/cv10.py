import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
exec(open("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification/code/cv2.py").read().split('print("features"')[0])

folds = []
start0 = int(N * 0.40); K = 10; seglen = (N - start0) // K
for i in range(K):
    a = start0 + i * seglen; b = N if i == K - 1 else start0 + (i + 1) * seglen
    folds.append((a, b))

def rep(name, fn):
    aps = []
    for (a, b) in folds:
        g = (LR[a:b] >= PROM).astype(float)
        aps.append(ap(Y[a:b], g * fn(LR[a:b])))
    print(f"{name:24s} mean={np.mean(aps):.4f} last3={[round(x,3) for x in aps[-3:]]}")

rep("linear", lambda x: 1 + x)
rep("sqrt", lambda x: 1 + np.sqrt(x))
rep("log1p10", lambda x: 1 + np.log1p(10 * x))
rep("log1p20", lambda x: 1 + np.log1p(20 * x))
rep("log1p50", lambda x: 1 + np.log1p(50 * x))
rep("log1p100", lambda x: 1 + np.log1p(100 * x))
rep("x^0.5 capped", lambda x: 1 + np.minimum(np.sqrt(x), 1.0) + 0.1 * (x > 0.01))
