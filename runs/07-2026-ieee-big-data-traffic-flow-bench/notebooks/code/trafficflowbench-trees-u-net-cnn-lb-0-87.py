### cell 0 [markdown]
<div style="background:linear-gradient(120deg,#0b1f3a 0%,#1d4e89 45%,#0f9d9a 100%); padding:30px 34px; border-radius:14px; color:#ffffff;">
<h1 style="color:#ffffff; margin:0 0 10px 0; font-weight:800;">TrafficFlowBench | One Pipeline for All 4 Tasks | LB 0.87</h1>
<p style="color:#e8f1f8; font-size:1.05em; margin:0; line-height:1.55;">
Gradient-boosted trees and a density-aware U-Net for state reconstruction, a tree ensemble with a convolutional onset
model for queue forecasting, and prior-regularised NNLS for origin-destination estimation.
</p>
</div>

### cell 1 [markdown]
## The competition in brief

| Task | Weight | Predict | Scored by |
|---|---:|---|---|
| 1. State reconstruction | 35% | speed and flow at hidden cells | RMSE (speed / 25, per-lane flow / 600) |
| 2. Queue forecasting | 30% | queued links, T+5 to T+30 | space-time IoU |
| 3. Physics consistency | 15% | nothing (uses Task 1 values) | fundamental diagram + vehicle conservation |
| 4. OD estimation | 20% | path flows | fit to counts and OD structure |

A link is queued when its speed is at most 60% of free-flow speed. Task 2 has two kinds of windows, weighted equally:
*ongoing* (a queue is visible at T) and *onset* (a queue forms within 30 minutes).

### cell 2 [markdown]
## Pipeline

```
Task 1   3 x LightGBM ─┐
         2 x U-Net    ─┴─ 70/30 blend ──────────────► speed, flow   (Task 3 is scored on these)

Task 2   ongoing: 7 x LightGBM ──► P > 0.40 ─┐
         onset:   3 x CNN, T+30 only, P > 0.50 ─┴──► queue mask

Task 4   NNLS on each split's own counts and prior, solved to convergence ──► path flows

Final    Task 1: 50/50 with the public CatBoost notebook
         Task 2: public onset model; ongoing = 50/50 average of our P and the public model's P, P >= 0.45
```

### cell 3 [markdown]
## Setup

Trained model weights are attached as a dataset.

### cell 4 [code]
import os, sys, glob, shutil, time, subprocess, warnings
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 11})

CFG = dict(
    t1_unet_share = 0.30,          # Task 1: 70% trees + 30% U-Net
    t2_ongoing_threshold = 0.40,   # Task 2 ongoing: queued if P > 0.40
    t2_blend_threshold = 0.45,     # ...and if the 50/50 average with the public model's P is >= 0.45
    t2_onset_threshold = 0.50,     # Task 2 onset: queued at T+30 if P > 0.50
    t2_onset_fallback_k = 2,       # ...otherwise the 2 most likely links (every onset window has a queue)
    t2_v20_share = 0.20,           # share of the selector-matched member in the ongoing ensemble
    t4_lambda = 0.05,              # Task 4: prior weight in the NNLS objective
)
T0 = time.time(); elapsed = lambda: f"{time.time() - T0:.0f}s"

### cell 5 [code]
DATA = glob.glob("/kaggle/input/**/kaggle_public", recursive=True)[0]
MODELS_IN = os.path.dirname(glob.glob("/kaggle/input/**/reference_candW.parquet", recursive=True)[0])
M = f"{MODELS_IN}/models"
WORK = "/kaggle/working/tfb"
shutil.copytree(f"{MODELS_IN}/src", f"{WORK}/src", dirs_exist_ok=True)   # helper modules need a writable root for caches
os.makedirs(f"{WORK}/submissions", exist_ok=True)                        # cache folder used by the merge step
os.environ["TFB_DATA"] = DATA
sys.path[:0] = [f"{WORK}/src/nb", f"{WORK}/src/v2"]
print("competition data:", DATA); print("models:", sorted(os.listdir(M)))

### cell 6 [markdown]
## The data

One training day on one corridor. Queues form at fixed bottlenecks and grow upstream (towards lower link numbers).

### cell 7 [code]
from common import network, day_files, date_of, day_grid, regime_of, slots_of
p = "D7_I405_N"; net = network(p); li = {l: i for i, l in enumerate(net.link_id)}
f = day_files(p, "train", masked=False)[40]; g = day_grid(f, li, len(net))
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
im = ax[0].imshow(g["v"], aspect="auto", cmap="RdYlGn", vmin=20, vmax=120); plt.colorbar(im, ax=ax[0], label="km/h")
ax[0].set(title=f"{p}, {date_of(f)}: speed", xlabel="time of day (5-min slot)", ylabel="link")
ax[1].imshow(g["v"] <= net.v_cut.to_numpy()[:, None], aspect="auto", cmap="Greys")
ax[1].set(title="queued cells (speed ≤ 60% of free speed)", xlabel="time of day (5-min slot)")
plt.tight_layout(); plt.show()

### cell 8 [markdown]
## Task 1: reconstructing hidden speed and flow

Each day hides 20%, 30% or 50% of its cells. Two model families fill them.

**LightGBM.** Features are the nearest observed values in time and along the corridor, interpolations, and a weekday ×
time-of-day profile.

> **Profiles that include the target day leak.** Our first profiles averaged every training day,
> including the one being predicted, so the model learned to over-trust them. Rebuilding them leave-one-day-out made
> local scores *drop* and leaderboard scores *rise* (+0.004 in Task 1, +0.008 in Task 2).

### cell 9 [code]
import t1_infer
tree_dirs = {f"trees_v{v}": f"{M}/t1_v{v}" for v in (4, 5, 6)}
out = t1_infer.trees(list(tree_dirs.values()), threads=4)       # features are built once and shared by the three models
t1 = {name: out[d] for name, d in tree_dirs.items()}
print("LightGBM (3 models) done", elapsed())

### cell 10 [markdown]
**U-Net.** An image model over the whole day (links × 288 slots) that corrects the profile. Training masks are drawn
fresh every epoch, so the training set is effectively unlimited.

> **The training objective should reflect cross-task scoring.** Our first U-Net improved speed and flow errors but hurt Task 3,
> which checks the density (flow / speed) these values imply. Adding a density term to the loss fixed it.

### cell 11 [code]
unet_w = {f"unet_s{s}": f"{M}/t1_unet_s{s}/task1_unet.pt" for s in (51, 52)}
out = t1_infer.unet(list(unet_w.values()), width=48, batch=4)   # one input grid per day, fed to both seeds
t1.update({name: out[w] for name, w in unet_w.items()})
print("U-Net (2 seeds) done", elapsed())

### cell 12 [markdown]
The blend beats both families on held-out days. On the leaderboard, a 30% U-Net share helped and 50% hurt.

| held-out days | speed RMSE | flow/lane RMSE | S_state | conservation proxy |
|---|---:|---:|---:|---:|
| LightGBM | 1.771 | 31.38 | 0.9377 | 0.582 |
| U-Net | 1.840 | 37.88 | 0.9312 | 0.580 |
| 70/30 blend | 1.673 | 31.53 | 0.9397 | 0.593 |

### cell 13 [code]
os.makedirs("/kaggle/working/task1", exist_ok=True); T1 = {}
w = CFG["t1_unet_share"]
for split in ("validation", "private"):
    avg = lambda keys, col: np.mean([t1[k][split][col].to_numpy() for k in keys], 0)
    trees, unets = [f"trees_v{v}" for v in (4, 5, 6)], [f"unet_s{s}" for s in (51, 52)]
    out = t1["trees_v4"][split].copy()
    for col in ("speed_kmh", "flow_vph"): out[col] = (1 - w) * avg(trees, col) + w * avg(unets, col)
    out.to_csv(f"/kaggle/working/task1/task1_{split}.csv", index=False); T1[split] = out
    print(f"{split}: {len(out):,} cells")

### cell 14 [markdown]
A private-month day with half its cells hidden, before and after:

### cell 15 [code]
f = [x for x in day_files(p, "private", masked=True) if regime_of(x) == "R3"][3]; d = date_of(f); g = day_grid(f, li, len(net))
filled = g["v"].copy(); s = T1["private"]; s = s[(s.panel == p) & (s.timestamp.str[:10] == d)]
filled[s.link_id.map(li).to_numpy(int), slots_of(s.timestamp)] = s.speed_kmh.to_numpy()
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for a, arr, t in ((ax[0], g["v"], f"{p} {d}: input (50% hidden)"), (ax[1], filled, "after Task 1")):
    im = a.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=20, vmax=120); a.set(title=t, xlabel="time of day (5-min slot)")
plt.colorbar(im, ax=ax[1], label="km/h"); plt.tight_layout(); plt.show()

### cell 16 [markdown]
## Task 2: forecasting queues

Only data timestamped at or before the forecast origin T may be used
([organizer ruling](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742068)).

> **The window selection procedure is part of the problem.** Every onset queue first appears exactly at
> T+30 ([confirmed](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350)),
> because the selector picks the earliest origin whose horizon contains the queue. Predicting only that step was
> worth +0.027, more than any model change.

Ongoing windows use seven LightGBM members; two of them add three hours of earlier context (queue age, queue-front
speed, ramp flows). Onset windows use a CNN that sees the whole corridor at once.

The final submission uses the public notebook's Task 2 instead (see
[Combining with the strongest public notebook](#Combining-with-the-strongest-public-notebook)); our ensemble is shown
here for the method and its analysis.

### cell 17 [code]
import t2_gbm_infer as g2
P = {}
for family, names in (("v11", ["v11", "v12", "v13", "v14"]), ("v15", ["v15", "v16"]), ("v18", ["v20"])):
    out = g2.dump(family, [f"{M}/t2_{n}" for n in names], threads=4)   # members of a family share their features
    P.update({n: out[f"{M}/t2_{n}"] for n in names})
    print(f"LightGBM family {family} ({len(names)} members) done", elapsed())

### cell 18 [markdown]
> **Model families specialise by condition.** The CNN beat the trees on onset (+0.002) and lost
> on ongoing (−0.005). Splitting them by condition kept both gains.

### cell 19 [code]
import t2_cnn_infer as c2
W = c2.window_inputs()
for s in (30, 31, 32):
    P[f"cnn{s}"] = c2.dump(f"{M}/t2_cnn_s{s}/task2_cnn.pt", W)
    print(f"CNN seed {s} done", elapsed())
key = ["split", "window_id", "h", "link_id"]
assert all((P[k][key].astype(str).values == P["v11"][key].astype(str).values).all() for k in P)

### cell 20 [markdown]
The ensemble gives 20% to v20, trained only on windows the selector could have picked: official ongoing windows never
have a persistence IoU above 0.9 (0 of 40, against 8.7% of random moments)
([observation credit](https://www.kaggle.com/code/hosen42/tfb-t2v2-full)).

### cell 21 [code]
mean = lambda names, col: np.mean([P[n][col].to_numpy() for n in names], 0)
probs = P["v11"][key + ["panel", "condition"]].copy()
group = lambda col: 0.5 * mean(["v11", "v12", "v13", "v14"], col) + 0.5 * mean(["v15", "v16"], col)
a = CFG["t2_v20_share"]
probs["p_ongoing"] = np.mean([(1 - a) * group(c) + a * P["v20"][c].to_numpy() for c in ("p_general", "p_general_w")], 0)
probs["p_onset"] = mean(["cnn30", "cnn31", "cnn32"], "p_onsetT30")

### cell 22 [markdown]
By construction every onset window contains a queue
([scoring specification](https://github.com/jacky850/trafficflowbench-public/blob/main/docs/SCORING_SPEC.md)), so when
no link passes the threshold the two most likely links are marked.

### cell 23 [code]
probs["queue_pred"] = 0
ongoing = probs.condition == "queue_ongoing"
probs.loc[ongoing, "queue_pred"] = (probs.loc[ongoing, "p_ongoing"] > CFG["t2_ongoing_threshold"]).astype(int)
onset_t30 = (probs.condition == "queue_onset") & (probs.h == 6)
for _, grp in probs[onset_t30].groupby("window_id"):
    s = grp.p_onset.to_numpy(); sel = s > CFG["t2_onset_threshold"]
    if not sel.any(): sel = s >= np.sort(s)[-CFG["t2_onset_fallback_k"]]
    probs.loc[grp.index, "queue_pred"] = sel.astype(int)
probs.groupby(["split", "condition"]).queue_pred.mean().round(4).to_frame("share of cells predicted queued")

### cell 24 [code]
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
w1 = probs[(probs.split == "validation") & ongoing].window_id.iloc[0]; x = probs[probs.window_id == w1]
mat = x.pivot(index="link_id", columns="h", values="p_ongoing").reindex(network(x.panel.iloc[0]).link_id)
im = ax[0].imshow(mat.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1); plt.colorbar(im, ax=ax[0], label="P(queued)")
ax[0].set(title=f"ongoing window {w1}", xlabel="minutes after T (÷5)", ylabel="link")
w2 = probs[(probs.split == "validation") & onset_t30].window_id.iloc[0]
y = probs[(probs.window_id == w2) & (probs.h == 6)].set_index("link_id").reindex(network(probs[probs.window_id == w2].panel.iloc[0]).link_id)
ax[1].bar(range(len(y)), y.p_onset, color=np.where(y.queue_pred == 1, "crimson", "lightgrey"))
ax[1].axhline(CFG["t2_onset_threshold"], ls="--", c="k", lw=1); ax[1].set(title=f"onset window {w2}: P(queued at T+30), red = predicted", xlabel="link")
plt.tight_layout(); plt.show()

### cell 25 [code]
os.makedirs("/kaggle/working/task2", exist_ok=True)
for split in ("validation", "private"):
    tpl = pd.concat([pd.read_csv(f"{DATA}/task2/{p}/{split}/sample_submission_queue.csv", dtype=str) for p in g2.Q2], ignore_index=True)
    wi = pd.concat([pd.read_csv(f"{DATA}/task2/{p}/{split}/window_index.csv") for p in g2.Q2])
    origin = pd.to_datetime(tpl.window_id.map(dict(zip(wi.window_id, wi.forecast_origin))), utc=True)
    tpl["h"] = ((pd.to_datetime(tpl.timestamp, utc=True) - origin).dt.total_seconds() // 300).astype(int)
    pred = probs[probs.split == split].set_index(["window_id", "h", "link_id"]).queue_pred
    tpl["queue_pred"] = pred.reindex(pd.MultiIndex.from_frame(tpl[["window_id", "h", "link_id"]])).fillna(0).astype(int).to_numpy()
    tpl.drop(columns="h").to_csv(f"/kaggle/working/task2/task2_{split}.csv", index=False)
print("Task 2 files written", elapsed())

### cell 26 [markdown]
## Task 3: physics consistency

Task 3 has no submission. The scorer derives density from the Task 1 values and checks the fundamental diagram (1/3)
and vehicle conservation (2/3), using organizer-held boundary flows and the true values at all non-target cells. Only
the density at predicted cells can move it, which is why the U-Net loss includes a density term.

## Task 4: origin–destination estimation

> **Task 4 must be solved on each split's own data.** Each split ships its own link counts and its own weak
> prior. Solving min ‖Af − c‖² + λ‖f − b‖² (f ≥ 0, λ = 0.05) on those, instead of on training-month counts, was
> worth +0.031, our largest single gain.

The objective is now solved to convergence (a FISTA warm start, then L-BFGS-B with tight tolerances), which moved the
public score by +0.0006.

### cell 27 [code]
os.makedirs("/kaggle/working/task4", exist_ok=True)
subprocess.run([sys.executable, f"{WORK}/src/v2/task4_tight.py", "--lam", str(CFG["t4_lambda"]), "--out", "/kaggle/working/task4"], check=True)
print("Task 4 done", elapsed())

### cell 28 [markdown]
## Submission

### cell 29 [code]
subprocess.run([sys.executable, f"{WORK}/src/v2/make_submission.py", "--task1", "/kaggle/working/task1",
                "--task2", "/kaggle/working/task2", "--task4", "/kaggle/working/task4", "--output", "/kaggle/working/submission.csv"], check=True)
sub = pd.read_csv("/kaggle/working/submission.csv")
tpl = pd.read_csv(f"{DATA}/sample_submission.csv", usecols=["submission_id", "task"]); st = sub.task == "state"
checks = {"row count": len(sub) == len(tpl), "template order": bool((sub.submission_id.values == tpl.submission_id.values).all()),
          "no missing": not sub.isna().any().any(), "all finite": bool(np.isfinite(sub[["speed_kmh", "flow_vph", "path_flow"]].to_numpy()).all()),
          "binary queues": set(sub.queue_pred.unique()) <= {0, 1}, "empty-road guard": float((sub.flow_vph[st] < 50).mean()) <= 0.2}
assert all(checks.values()), checks
pd.Series(checks).to_frame("passed")

### cell 30 [markdown]
## Combining with the strongest public notebook

The public notebook by [anubisyy](https://www.kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517)
reaches its score mainly through Task 2. Its Task 1 (CatBoost residual models) is weaker than ours on its own
(0.85565 against 0.85618 when it replaced ours), but the two make different errors. The final submission therefore
averages the two Task 1 predictions with equal weight. For Task 2 it uses the public onset model, and for ongoing windows
it averages the queue probabilities of our ensemble and the public LightGBM with equal weight and applies a 0.45
threshold to the average. The public Task 2 features use only data strictly before the forecast origin T. Task 4 is our own.

Each step was submitted in isolation on top of the previous one:

| Step | Public LB | Derived conclusion |
|---|---:|---|
| Converged Task 4 | 0.85556 → 0.85618 (+0.0006) | The previous solver had not converged |
| 50/50 Task 1 | 0.85618 → 0.85848 (+0.0023) | The two Task 1 models make different errors |
| Public Task 2 | 0.85848 → 0.86801 (+0.0095) | Its onset model is much stronger than our CNN |
| Ongoing: 50/50 probability average | 0.86801 → 0.87029 (+0.0023) | The two ongoing models make different errors |
| Threshold 0.45 on the average | 0.87029 → 0.87141 (+0.0011) | At 0.40 the average predicts too many queued cells |

The ongoing probabilities of the public model are recomputed below from its released checkpoint, because its notebook
saves only thresholded labels.

### cell 31 [code]
PUB = glob.glob("/kaggle/input/**/catboost-lgbm-nnls-pipeline-top-50-0-86517/**/submission.csv", recursive=True) or glob.glob("/kaggle/input/**/submission.csv", recursive=True)
PUB = sorted(x for x in PUB if "catboost" in x and not x.startswith(MODELS_IN))   # exclude our own files
assert PUB, "public notebook output not found: attach anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517"
path = PUB[0]; print("public notebook output:", path)
pub = pd.read_csv(path)
assert len(pub) == len(sub) and bool((pub.submission_id.values == sub.submission_id.values).all()), "submission_id order differs"
st, q = sub.task == "state", sub.task == "queue"
sub.loc[st, ["speed_kmh", "flow_vph"]] = 0.5 * sub.loc[st, ["speed_kmh", "flow_vph"]].to_numpy() + 0.5 * pub.loc[st, ["speed_kmh", "flow_vph"]].to_numpy()
sub.loc[q, "queue_pred"] = pub.loc[q, "queue_pred"].astype(int).to_numpy()

### cell 32 [code]
# Public ongoing model: rerun its released LightGBM checkpoint to recover probabilities (source from the attached notebook output)
SRC_PUB = os.path.dirname(sorted(glob.glob("/kaggle/input/**/tfb_source/queue_ongoing.py", recursive=True))[0])
ASSETS = os.path.dirname(glob.glob("/kaggle/input/**/ongoing_lgb.txt", recursive=True)[0])
shutil.copytree(SRC_PUB, f"{WORK}/pub_src", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
os.environ.update(TFB_DATA_ROOT=DATA, TFB_OUTPUT_ROOT=f"{WORK}/pub_out", TFB_THREADS="4"); sys.path.insert(0, f"{WORK}/pub_src")
import lightgbm as lgb
from queue_features import panel_names, HISTORY, HORIZON
from queue_ongoing import Panel as OngoingPanel
ongoing_lgb = lgb.Booster(model_file=f"{ASSETS}/ongoing_lgb.txt"); assert ongoing_lgb.num_feature() == 36
rows = []
for split in ("validation", "private"):
    for no, name in enumerate(panel_names()):
        panel = OngoingPanel(name, no); folder = f"{DATA}/task2/{name}/{split}"
        windows = pd.read_csv(f"{folder}/window_index.csv"); hist = pd.read_parquet(f"{folder}/window_history.parquet")
        for w in windows[windows.condition == "queue_ongoing"].itertuples(index=False):
            origin = pd.Timestamp(w.forecast_origin)
            times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            h = hist[hist.window_id == w.window_id].copy(); h["link_id"] = h.link_id.astype(str)
            assert pd.to_datetime(h.timestamp, utc=True).max() < origin   # strictly before T
            mat = lambda c: h.pivot(index="timestamp", columns="link_id", values=c).reindex(index=times, columns=panel.links).to_numpy(dtype=np.float32)
            speed, flow, elig = mat("speed_kmh"), mat("flow_vph"), mat("is_score_eligible")
            p = ongoing_lgb.predict(panel.features(speed, flow, np.nan_to_num(elig, nan=0).astype(bool), origin, w.condition), num_threads=4)
            rows.append(pd.DataFrame({"split": split, "window_id": w.window_id, "h": np.repeat(np.arange(1, HORIZON + 1), len(panel.links)),
                                      "link_id": np.tile(panel.links, HORIZON), "p_pub": p}))
pubp = pd.concat(rows, ignore_index=True)
print("public ongoing probabilities:", len(pubp), elapsed())

### cell 33 [code]
# Ongoing cells: average the two probabilities, then threshold the average at 0.45
k = pd.read_csv(f"{DATA}/submission_key.csv", usecols=["task", "window_id", "timestamp", "link_id"], dtype=str); k = k[k.task == "queue"]
wi = pd.concat([pd.read_csv(g) for g in glob.glob(f"{DATA}/task2/*/*/window_index.csv")])
k["h"] = ((pd.to_datetime(k.timestamp, utc=True) - pd.to_datetime(k.window_id.map(dict(zip(wi.window_id, wi.forecast_origin))), utc=True)).dt.total_seconds() // 300).astype(int)
idx = pd.MultiIndex.from_frame(k[["window_id", "h", "link_id"]])
p_ours = probs.set_index(["window_id", "h", "link_id"]).p_ongoing.reindex(idx).to_numpy()
p_pub = pubp.set_index(["window_id", "h", "link_id"]).p_pub.reindex(idx).to_numpy()
is_ong = k.window_id.map(dict(zip(wi.window_id, wi.condition))).eq("queue_ongoing").to_numpy()
assert not np.isnan(p_ours[is_ong]).any() and not np.isnan(p_pub[is_ong]).any()
print("agreement with the public ongoing labels (%):", round(100 * ((p_pub[is_ong] >= 0.40).astype(int) == sub.loc[q, "queue_pred"].to_numpy()[is_ong]).mean(), 3))
blend = (0.5 * p_ours + 0.5 * p_pub >= CFG["t2_blend_threshold"]).astype(int)
qp = sub.loc[q, "queue_pred"].to_numpy().copy(); changed = int((qp[is_ong] != blend[is_ong]).sum()); qp[is_ong] = blend[is_ong]
sub.loc[q, "queue_pred"] = qp
print(f"ongoing cells changed by the probability average: {changed}")
checks = {"row count": len(sub) == len(tpl), "template order": bool((sub.submission_id.values == tpl.submission_id.values).all()),
          "no missing": not sub.isna().any().any(), "all finite": bool(np.isfinite(sub[["speed_kmh", "flow_vph", "path_flow"]].to_numpy()).all()),
          "binary queues": set(sub.queue_pred.unique()) <= {0, 1}, "empty-road guard": float((sub.flow_vph[st] < 50).mean()) <= 0.2}
assert all(checks.values()), checks
sub.to_csv("/kaggle/working/submission.csv", index=False)
pd.Series(checks).to_frame("passed")

### cell 34 [code]
ref = pd.read_parquet(f"{MODELS_IN}/reference_candZ6a.parquet"); q, o = sub.task == "queue", sub.task == "odme"
pd.Series({"Task 1 speed RMSE vs reference submission (km/h)": np.sqrt(((sub.speed_kmh - ref.speed_kmh)[st] ** 2).mean()),
           "Task 1 flow RMSE vs reference submission (vph)": np.sqrt(((sub.flow_vph - ref.flow_vph)[st] ** 2).mean()),
           "Task 2 cells agreeing with reference submission (%)": 100 * (sub.queue_pred == ref.queue_pred)[q].mean(),
           "Task 4 max |difference|": (sub.path_flow - ref.path_flow)[o].abs().max(),
           "runtime (s)": time.time() - T0}).round(4).to_frame("value")

### cell 35 [markdown]
## Experiment log

Each change was evaluated in isolation against the preceding configuration. Rows in grey did not improve the score.

| Experiment | Public LB effect | Derived conclusion |
|---|---:|---|
| Task 4 on each split's own counts and prior | +0.031 | Fit the numbers the task is scored on |
| Onset predicted at T+30 only | +0.027 | The window selector is part of the problem |
| Ensemble of Task 2 members, tuned thresholds | +0.017 | Averaging stabilises noisy IoU |
| Public Task 2 | +0.0095 | Its onset model is much stronger than our CNN |
| Ongoing: 50/50 probability average with the public model | +0.0023 | The two ongoing models make different errors |
| Threshold 0.45 instead of 0.40 on that average | +0.0011 | Queues in the scored months are smaller than in training |
| Leave-one-day-out profiles, Task 2 / Task 1 | +0.008 / +0.004 | Self-leaking features inflate local scores |
| Bagging leak-free members | +0.008 | Diversity beats a single tuned model |
| 50/50 Task 1 | +0.0023 | The two Task 1 models make different errors |
| CNN for onset windows | +0.0019 | Spatial context helps where no queue exists yet |
| U-Net at 30% in Task 1 | +0.0010 | A weaker but different model still adds |
| Converged Task 4 | +0.0006 | The previous solver had not converged |
| Three CNN seeds instead of one | +0.0004 | Seed averaging helped although local scores tied |
| Three hours of same-day context | +0.0004 | Context helps ongoing queues only a little |
| Selector-style training windows | +0.0003 | Training should look like the scored windows |
| Second U-Net seed | +0.0001 | Diminishing returns |
| <span style="color:#999">Our CNN onset in place of the public onset</span> | −0.0103 | The public onset model localises first breakdowns better |
| <span style="color:#999">Onset threshold 0.52 with top-2 fallback</span> | −0.0070 | Extra onset cells mostly miss |
| <span style="color:#999">CNN for ongoing windows</span> | −0.0049 | Local validation ranked model families wrongly |
| <span style="color:#999">Filling one-link gaps in ongoing masks</span> | −0.0042 | Gaps in predicted queues are mostly real |
| <span style="color:#999">U-Net at 50%</span> | −0.0012 | Blend weight matters more than member quality |
| <span style="color:#999">Four long-context seeds for ongoing</span> | −0.0012 | More of one member skews the ensemble |
| <span style="color:#999">Best single ongoing model alone</span> | −0.0005 | The ensemble beat its strongest member |
| <span style="color:#999">Public Task 1 alone</span> | −0.0005 | Our Task 1 is stronger on its own |
| <span style="color:#999">Low-confidence onset fallback (top-3)</span> | −0.0003 | A held-out gain that did not transfer |
| <span style="color:#999">Cell-transmission traffic simulation</span> | rejected locally | Below plain persistence |
| <span style="color:#999">Mask clean-up, expected-IoU decoding</span> | no local gain | Thresholds were already near-optimal |
| <span style="color:#999">Month-specific profiles</span> | no local gain | No month effect in this data |

The main open problem is onset localisation: determining *which* bottleneck breaks down first.

## References

* Official evaluators and baselines: [trafficflowbench-public](https://github.com/jacky850/trafficflowbench-public)
* Organizer rulings: [Task 2 data](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742068), [onset at T+30](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350)
* Window-selector replication: [hosen42](https://www.kaggle.com/code/hosen42/tfb-t2v2-full)
* CatBoost Task 1, the Task 2 models and checkpoints used in the final submission: [anubisyy](https://www.kaggle.com/code/anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517)
* λ = 0.05 calibration and physics baseline: [soukeaizenz](https://www.kaggle.com/code/soukeaizenz/0-809-lb-top-50-open-sourcing-physics-pipeline)
