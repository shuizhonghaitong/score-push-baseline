"""v2: deterministic leak-free rule.
Label rule (reverse-engineered, 6131/6132 train match):
  Signal_t = 1[LR_t >= PROM] AND 1[RR_t >= PROM],  PROM = 0.00995
where LR_t = ln(drop from left reference to p_t), RR_t = ln(rise on right before
price dips below p_t). LR is fully causal; RR uses future prices (the unpredictable part).
Score: within gate rank by LR (deeper trough -> higher P(RR), verified monotone on train).
"""
import pandas as pd, numpy as np, json
RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification"
PROM = 0.00995
tr = pd.read_csv(f"{RUN}/data/archive_btc_train.csv", parse_dates=["datetime"])
te = pd.read_csv(f"{RUN}/data/archive_btc_test.csv", parse_dates=["datetime"])
allp = pd.concat([tr, te], ignore_index=True).sort_values("datetime").reset_index(drop=True)
P = allp["prices"].values
N = len(tr)

def left_rise(p):
    out = np.full(len(p), -1e9)
    for t in range(len(p)):
        pt = p[t]; i = t
        while i > 0 and p[i-1] >= pt: i -= 1
        out[t] = np.log(p[i:t+1].max() / pt)
    return out
LR = left_rise(P)
gate = (LR >= PROM - 1e-12)[N:]
score = np.where(gate, 1.0 + LR[N:], 0.0)
sub = pd.DataFrame({"Id": te.sort_values("datetime")["Id"].values, "Signal": score})
# ensure Id order matches test file order (already chronological)
assert list(sub["Id"]) == list(te.sort_values("datetime")["Id"])
out = f"{RUN}/outputs/submission_causal_v2.csv"
sub.to_csv(out, index=False)
print("wrote", out, "gate rows:", int(gate.sum()), "score max:", score.max())
json.dump({"cv_ap": 0.3488, "file": out}, open(f"{RUN}/notes/final_v2.json", "w"), indent=1)
