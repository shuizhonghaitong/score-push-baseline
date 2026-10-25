import numpy as np

def map3_from_topk(topk, labels):
    tot = 0.0
    for pred, y in zip(topk, labels):
        seen = set(); score = 0.0
        for k, p in enumerate(pred[:3], 1):
            if p in seen:
                continue
            seen.add(p)
            if p == y:
                score = 1.0 / k
                break
        tot += score
    return tot / len(labels)

def top3_from_scores(scores, codes, k=3):
    """scores: (n, c) array; returns list of top-k code lists (ties broken by index)."""
    codes = np.asarray(codes)
    idx = np.argsort(-scores, axis=1)[:, :k]
    return [list(codes[i]) for i in idx]
