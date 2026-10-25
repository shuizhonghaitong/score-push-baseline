import json
import numpy as np
import pandas as pd

K = 18
CANON = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
         'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
         'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
         'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']

def to_canon(arr, order_file, tag=""):
    """Reindex columns of arr (n x K log-probs/probs) into CANON order.
    order_file holds, per CANON label, the column index in arr's own order."""
    if order_file is not None:
        try:
            idx = np.load(order_file)
            assert idx.shape == (K,) and len(set(idx.tolist())) == K, f"{tag} order malformed"
            return np.asarray(arr)[:, idx]
        except FileNotFoundError:
            print(f"WARN: {tag} order file {order_file} missing; assuming canonical")
    return np.asarray(arr)

def sinkhorn(P, iters=100):
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)
    return P

def make_sub(ids, P):
    return pd.DataFrame({"id": ids, "model": np.array(CANON)[P.argmax(1)]})
