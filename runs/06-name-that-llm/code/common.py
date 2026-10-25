import numpy as np, pandas as pd, re, os
LABELS = ['gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra',
          'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
          'claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
          'qwen3.8-max','glm-5.3','glm-5.3-flash','kimi-k3']
LAB2I = {l:i for i,l in enumerate(LABELS)}
ROOT = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm"
def load_data():
    train = pd.read_csv(f"{ROOT}/data/train.csv")
    pool = pd.read_parquet(f"{ROOT}/pool/v4/ntl_pool_v4.parquet")
    prv = pd.read_csv(f"{ROOT}/pool/passages.csv")
    return train, pool, prv
def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True)); return e/e.sum(1, keepdims=True)
def sinkhorn(P, K=18, iters=100):
    P = P.copy(); target = np.full(K, len(P)/K)
    for _ in range(iters):
        P *= target / P.sum(0); P /= P.sum(1, keepdims=True)
    return P
def ba(y, P):
    y = np.asarray(y); K = 18
    pred = P.argmax(1)
    rec = np.zeros(K)
    for k in range(K):
        m = y == k
        rec[k] = (pred[m] == k).mean() if m.any() else 0.0
    return float(rec.mean())
def ba_prob(y, P):  # probability-based sanity variant not used by metric
    return ba(y, P)
