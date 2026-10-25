"""Memecoins Pump-or-Dump pipeline (single source of truth: local + Kaggle notebook).

Modes: smoke | wf | submit | both
Writes to OUT: features.csv, wf_results.csv, calibration.csv, submission.csv (submit mode)
"""
import gc
import glob
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

SEED = 42
COST = 0.3
TP_LEVELS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0, 13.0, 16.0, 20.0, 25.0, 31.0, 40.0, 50.0, 64.0, 80.0, 100.0]
TP_GRID = np.unique(np.concatenate([
    np.arange(3.0, 25.01, 0.5), np.arange(25.5, 40.01, 1.0),
    np.arange(42.5, 100.01, 2.5), [100.0]]))
Q_LEVELS = [0.10, 0.30, 0.50, 0.70, 0.90]
N_MC = int(os.environ.get("N_MC", "48"))
EVTHR = float(os.environ.get("EVTHR", "0"))
PBLEND = float(os.environ.get("PBLEND", "0"))
HGBW = float(os.environ.get("HGBW", "0"))
CASCADE_P = float(os.environ.get("CASCADE_P", "0"))
KILL = 1.5
WF_DAYS_DEFAULT = 8
MAX_PARTICIPANTS = 25
NJOBS = int(os.environ.get("NJOBS", "2"))


def feat_v2():
    return os.environ.get("FEAT_V2", "off") == "on"


def feat_v3():
    return os.environ.get("FEAT_V3", "off") == "on"


def feat_v4():
    return os.environ.get("FEAT_V4", "off") == "on"


REG_COLS_V3 = ["reg_n_1d", "reg_p5_1d", "reg_p2_1d", "reg_big_1d", "reg_fxm_1d",
               "reg_n_3d", "reg_p5_3d", "reg_big_3d"]

CURVE_COLS = ["mint", "ts_rel", "ts", "direction", "sol", "wallet", "failed", "new_price"]
CURVE_COLS_V2 = ["slot", "tip_index", "tok"]
AMM_COLS = ["mint", "ts_rel", "ts", "direction", "sol", "signing_wallet", "new_price",
            "pool_quote_token_reserves_after", "virtual_quote_reserves"]
AMM_COLS_V2 = ["fee_payer", "tok", "pool_base_token_reserves_after", "base_token"]


def _pq_names(path):
    try:
        import pyarrow.parquet as pq
        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        return None


def _read_cols(path, base, extra):
    cols = base + list(extra)
    names = _pq_names(path)
    if names is not None:
        cols = [c for c in cols if c in names]
    try:
        return pd.read_parquet(path, columns=cols)
    except Exception:
        return pd.read_parquet(path, columns=[c for c in base if c in (names or set(base))])


def read_amm(path, v2=False):
    df = _read_cols(path, AMM_COLS, AMM_COLS_V2 if v2 else [])
    return df.rename(columns={"signing_wallet": "wallet"})


def read_curve(path, v2=False):
    return _read_cols(path, CURVE_COLS, CURVE_COLS_V2 if v2 else [])

LGB_CLF = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
               subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
               n_jobs=NJOBS, verbose=-1)
LGB_REG = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=30,
               subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
               n_jobs=NJOBS, verbose=-1)


def log(*a):
    print(*a, flush=True)


def find_data_dir():
    try:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        here = os.getcwd()
    cands = ["/kaggle/input/competitions/memecoins-pump-or-dump",
             "/kaggle/input/memecoins-pump-or-dump",
             os.path.join(here, "data"), "data"]
    if os.environ.get("DATA_DIR", ""):
        cands.insert(0, os.environ["DATA_DIR"])
    for p in cands:
        if os.path.isdir(p) and (glob.glob(os.path.join(p, "day_*_pumpfun*.parquet")) or
                                 os.path.isdir(os.path.join(p, "trades"))):
            return p
    if os.environ.get("RESUME_FEATS", ""):
        for p in cands:
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "train.csv")):
                return p
    raise FileNotFoundError("no data dir found")


def day_of_path(path):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path)
    return m.group(1) if m else None


def list_tape_days(data_dir):
    days = {}
    trades = os.path.join(data_dir, "trades")
    if os.path.isdir(trades):
        for f in glob.glob(os.path.join(trades, "**", "*.parquet"), recursive=True):
            d = day_of_path(f)
            if not d:
                continue
            kind = "curve" if "pumpfun" in os.path.basename(f) else "amm"
            days.setdefault(d, {})[kind] = f
    else:
        for f in glob.glob(os.path.join(data_dir, "day_*_pumpfun_all_swaps.parquet")):
            days.setdefault(day_of_path(f), {})["curve"] = f
        for f in glob.glob(os.path.join(data_dir, "day_*_pumpswap_all_swaps.parquet")):
            days.setdefault(day_of_path(f), {})["amm"] = f
    return dict(sorted(days.items()))


def curve_features(c, v2=False):
    c = c.copy()
    dbuy = c["direction"].astype(str).str.lower().str.startswith("b")
    okm = c["failed"] == 0
    c["lp"] = np.where(c["new_price"] > 0, np.log(c["new_price"]), np.nan)
    mints = pd.Index(c["mint"].unique(), name="mint")
    F = pd.DataFrame(index=mints)
    ok = c[okm]
    g = ok.groupby("mint")
    F["c_n"] = g.size()
    F["c_sol"] = g["sol"].sum()
    F["c_uw"] = g["wallet"].nunique()
    F["c_dur"] = -g["ts_rel"].min()
    F["c_pchg"] = g["lp"].last() - g["lp"].first()
    F["c_max_order"] = g["sol"].max()
    F["c_mean_order"] = g["sol"].mean()
    F["c_buy_n"] = ok[dbuy[okm]].groupby("mint").size()
    okv = ok.assign(_b=np.where(dbuy[okm], ok["sol"], 0.0), _s=np.where(~dbuy[okm], ok["sol"], 0.0))
    F["c_buy_sol"] = okv.groupby("mint")["_b"].sum()
    F["c_sell_sol"] = okv.groupby("mint")["_s"].sum()
    for thr, tag in [(-60, "60"), (-300, "300")]:
        w = ok[ok["ts_rel"] > thr]
        F[f"c_n{tag}"] = w.groupby("mint").size()
        F[f"c_sol{tag}"] = w.groupby("mint")["sol"].sum()
    F["c_uw300"] = ok[ok["ts_rel"] > -300].groupby("mint")["wallet"].nunique()
    F["c_volfrac600"] = ok[ok["ts_rel"] > -600].groupby("mint")["sol"].sum() / F["c_sol"]
    fl = c[~okm]
    F["c_f_n"] = fl.groupby("mint").size()
    F["c_f_ratio"] = F["c_f_n"] / (F["c_f_n"].fillna(0) + F["c_n"].fillna(0))
    F["c_f60"] = fl[fl["ts_rel"] > -60].groupby("mint").size()
    ws = ok.groupby(["mint", "wallet"])["sol"].sum()
    tot = ws.groupby(level=0).sum()
    F["c_top1"] = ws.groupby(level=0).max() / tot
    F["c_hhi"] = (ws / ws.groupby(level=0).transform("sum")).pow(2).groupby(level=0).sum()
    rt = ok[dbuy[okm]][["mint", "wallet"]].merge(ok[~dbuy[okm]][["mint", "wallet"]], on=["mint", "wallet"], how="inner")
    F["c_rt_uw"] = rt.groupby("mint")["wallet"].nunique()
    if v2:
        if "slot" in c.columns:
            cnt = ok.groupby(["mint", "slot"])["sol"].transform("size")
            clust_sol = ok[cnt >= 2].groupby("mint")["sol"].sum()
            F["c_slot_clust"] = clust_sol / F["c_sol"].replace(0, np.nan)
        s300 = ok[ok["ts_rel"] > -300].groupby("mint")["sol"].sum()
        s600 = ok[ok["ts_rel"] > -600].groupby("mint")["sol"].sum()
        F["c_burst5of10"] = s300 / s600.replace(0, np.nan)
        F["c_f_buyfrac"] = fl.groupby("mint")["direction"].apply(
            lambda x: x.astype(str).str.lower().str.startswith("b").mean())
        if "tip_index" in c.columns:
            F["c_tip_mean"] = ok.groupby("mint")["tip_index"].mean()
        F["c_n600"] = ok[ok["ts_rel"] > -600].groupby("mint").size()
        F["c_plast"] = g["lp"].last()
    return F.fillna(0)


def amm_features(a, v2=False):
    a = a.copy()
    dbuy = a["direction"].astype(str).str.upper().str.startswith("B")
    a["p"] = a["new_price"].where(a["new_price"] > 0)
    mints = pd.Index(a["mint"].unique(), name="mint")
    F = pd.DataFrame(index=mints)
    g = a.groupby("mint")
    F["a_n"] = g.size()
    F["a_sol"] = g["sol"].sum()
    F["a_uw"] = g["wallet"].nunique()
    F["a_buyshare"] = a.assign(_v=np.where(dbuy, a["sol"], 0.0)).groupby("mint")["_v"].sum() / F["a_sol"]
    for lo, hi, tag in [(0, 15, "015"), (15, 30, "1530"), (30, 60, "3060")]:
        w = a[(a["ts_rel"] >= lo) & (a["ts_rel"] < hi)]
        F[f"a_n_{tag}"] = w.groupby("mint").size()
        F[f"a_sol_{tag}"] = w.groupby("mint")["sol"].sum()
        F[f"a_buyshare_{tag}"] = w.assign(_v=np.where(dbuy[w.index], w["sol"], 0.0)).groupby("mint")["_v"].sum()
    F["a_late_early"] = F["a_sol_3060"] / F["a_sol_015"].replace(0, np.nan)
    F["a_mid_early"] = F["a_sol_1530"] / F["a_sol_015"].replace(0, np.nan)
    ws = a.groupby(["mint", "wallet"])["sol"].sum()
    tot = ws.groupby(level=0).sum()
    F["a_top1"] = ws.groupby(level=0).max() / tot
    F["a_hhi"] = (ws / ws.groupby(level=0).transform("sum")).pow(2).groupby(level=0).sum()
    a["is_buy"] = dbuy
    buys = a[dbuy]
    F["a_max_order"] = buys.groupby("mint")["sol"].max()
    F["a_whale_n"] = buys[buys["sol"] >= 5.0].groupby("mint").size()
    p = a["p"].replace(0, np.nan)
    pf, pl, pm = p.groupby(a["mint"]).first(), p.groupby(a["mint"]).last(), p.groupby(a["mint"]).max()
    F["a_ret"] = pl / pf
    F["a_runup"] = pm / pf
    F["a_at_high"] = pl / pm
    F["a_vol"] = np.log(p).groupby(a["mint"]).std()
    eb = a[(a["ts_rel"] <= 15) & dbuy][["mint", "wallet"]].drop_duplicates()
    sl = a[~dbuy][["mint", "wallet"]].drop_duplicates()
    fl = eb.merge(sl, on=["mint", "wallet"], how="inner")
    F["a_flip_n"] = fl.groupby("mint")["wallet"].nunique()
    F["a_flip"] = F["a_flip_n"] / eb.groupby("mint")["wallet"].nunique()
    d0 = a.groupby("mint")["pool_quote_token_reserves_after"].first() + a.groupby("mint")["virtual_quote_reserves"].first()
    d1 = a.groupby("mint")["pool_quote_token_reserves_after"].last() + a.groupby("mint")["virtual_quote_reserves"].last()
    F["a_depth0"] = d0
    F["a_depth1"] = d1
    F["a_depth_chg"] = d1 / d0.replace(0, np.nan)
    if v2:
        if "fee_payer" in a.columns and "wallet" in a.columns:
            diff = a["fee_payer"] != a["wallet"]
            F["a_fpay_diff"] = a[diff].groupby("mint")["sol"].sum() / F["a_sol"].replace(0, np.nan)
            F["a_fpay_n"] = a[diff].groupby("mint")["wallet"].nunique()
        if "slot" in a.columns:
            cnt = a.groupby(["mint", "slot"])["sol"].transform("size")
            clust_sol = a[cnt >= 2].groupby("mint")["sol"].sum()
            F["a_slot_clust"] = clust_sol / F["a_sol"].replace(0, np.nan)
        if "pool_base_token_reserves_after" in a.columns:
            b0 = a.groupby("mint")["pool_base_token_reserves_after"].first()
            b1 = a.groupby("mint")["pool_base_token_reserves_after"].last()
            F["a_coinres_chg"] = b1 / b0.replace(0, np.nan)
        pmin = p.groupby(a["mint"]).min()
        F["a_draw"] = (pm - pmin) / pm.replace(0, np.nan)
        F["a_last10s"] = a[a["ts_rel"] >= 50].groupby("mint")["sol"].sum() / F["a_sol"].replace(0, np.nan)
        F["a_first10s"] = a[a["ts_rel"] < 10].groupby("mint")["sol"].sum() / F["a_sol"].replace(0, np.nan)
        if "tok" in a.columns:
            F["a_tok_max"] = np.log1p(buys.groupby("mint")["tok"].max())
        F["a_tpw"] = F["a_n"] / F["a_uw"].replace(0, np.nan)
        F["a_plast"] = pl
        ea = a[a["ts_rel"] < 15][["mint", "wallet"]].drop_duplicates()
        la = a[a["ts_rel"] >= 45][["mint", "wallet"]]
        if len(la):
            mset = la.groupby("mint")["wallet"].apply(set)
            eset = ea.groupby("mint")["wallet"].apply(set)
            F["a_fresh45"] = pd.Series({m: len(ws - eset.get(m, set())) for m, ws in mset.items()})
        aa = a[(p > 0)].copy()
        dlp = aa.groupby("mint")["p"].transform(lambda x: np.log(x).diff())
        ssol = np.where(dbuy[aa.index], aa["sol"], -aa["sol"])
        gnum = pd.Series(ssol * dlp).groupby(aa["mint"]).sum()
        gden = pd.Series(ssol ** 2).groupby(aa["mint"]).sum()
        F["a_impact"] = gnum / gden.replace(0, np.nan)
    return F.fillna(0)


def participants(a, c, k=MAX_PARTICIPANTS):
    """out[mint] = (amm_buyers: wallet->sol [all first-60s buyers], curve_top: wallet->sol [top-k])"""
    out = {}
    if a is not None and len(a):
        abuy = a["direction"].astype(str).str.upper().str.startswith("B")
        ab = a[abuy]
        wa = ab.groupby(["mint", "wallet"])["sol"].sum().reset_index()
        for m, w in wa.groupby("mint", sort=False):
            out.setdefault(m, [None, None])[0] = dict(zip(w["wallet"], w["sol"]))
    if c is not None and len(c):
        cc = c[c["failed"] == 0]
        cw = cc.groupby(["mint", "wallet"])["sol"].sum().reset_index()
        cw = cw.sort_values("sol", ascending=False, kind="mergesort")
        cw["rk"] = cw.groupby("mint").cumcount()
        for m, w in cw[cw["rk"] < k].groupby("mint", sort=False):
            out.setdefault(m, [None, None])[1] = dict(zip(w["wallet"], w["sol"]))
    for m in list(out):
        if out[m][0] is None:
            out[m][0] = {}
        if out[m][1] is None:
            out[m][1] = {}
    return out


class WalletTracker:
    """Cumulative per-wallet outcome stats, only counting tokens resolved >= 1h before t."""

    def __init__(self, cap=65536):
        self.wid = {}
        self.cap = cap
        self.n = np.zeros(cap, dtype=np.float32)
        self.n5 = np.zeros(cap, dtype=np.float32)
        self.n2 = np.zeros(cap, dtype=np.float32)
        self.n3 = np.zeros(cap, dtype=np.float32)
        self.n3d = np.zeros(cap, dtype=np.float32)
        self.n53d = np.zeros(cap, dtype=np.float32)
        self.events = []  # (mt, [wids...], hit5, hit2, hit3), kept sorted by mt
        self.ptr = 0
        self.wstart = 0
        self.wadded = 0

    def id(self, w):
        i = self.wid.get(w)
        if i is None:
            if len(self.wid) >= self.cap:
                newcap = self.cap * 2
                for a in ("n", "n5", "n2", "n3", "n3d", "n53d"):
                    old = getattr(self, a)
                    new = np.zeros(newcap, dtype=np.float32)
                    new[:self.cap] = old
                    setattr(self, a, new)
                self.cap = newcap
            i = self.wid[w] = len(self.wid)
        return i

    def add(self, mt, wids, hit5, hit2, hit3):
        self.events.append((mt, wids, hit5, hit2, hit3))

    def advance_to(self, t):
        limit = t - 3600.0
        ev = self.events
        ptr = self.ptr
        while ptr < len(ev) and ev[ptr][0] <= limit:
            _, wids, h5, h2, h3 = ev[ptr]
            n = self.n
            n[wids] += 1
            self.n5[wids] += h5
            self.n2[wids] += h2
            self.n3[wids] += h3
            ptr += 1
        self.ptr = ptr
        lo3 = t - 3600.0 - 259200.0
        while self.wstart < ptr and ev[self.wstart][0] < lo3:
            _, wids, h5, h2, h3 = ev[self.wstart]
            self.n3d[wids] -= 1
            self.n53d[wids] -= h5
            self.wstart += 1
        while self.wadded < ptr:
            _, wids, h5, h2, h3 = ev[self.wadded]
            self.n3d[wids] += 1
            self.n53d[wids] += h5
            self.wadded += 1


REP_COLS = ["amm_known", "amm_hitmax", "amm_hitmean", "amm_hittop3", "amm_eliteshare", "amm_eb5", "amm_eb2", "amm_eb3",
            "cur_known", "cur_hitmax", "cur_hitmean", "cur_hittop3", "cur_eliteshare", "cur_eb5", "cur_eb2", "cur_eb3",
            "cr_n", "cr_eb5"]
REP_COLS_V2 = ["amm_known3", "amm_eb5r", "cur_known3", "cur_eb5r"]
C_PRIOR = 10.0
P0_5 = 0.06
P0_2 = 0.28
P0_3 = 0.13


def rep_block(wids, weights, tr, v2=False):
    """Reputation features for a token's participant wallets.
    known: wallet with >=3 resolved tokens. elite: >=5 tokens and >=1 five-x hit."""
    n = tr.n
    known = 0
    known3 = 0
    eb5r_max = 0.0
    rates = []
    eb5_max = 0.0
    eb2_max = 0.0
    eb3_max = 0.0
    el_sol = 0.0
    tot = 0.0
    for w, sol in zip(wids, weights):
        i = tr.wid.get(w)
        if i is None:
            continue
        ni = n[i]
        tot += sol
        if ni >= 3:
            known += 1
            rates.append(tr.n5[i] / ni)
            eb5 = (tr.n5[i] + C_PRIOR * P0_5) / (ni + C_PRIOR)
            eb2 = (tr.n2[i] + C_PRIOR * P0_2) / (ni + C_PRIOR)
            eb3 = (tr.n3[i] + C_PRIOR * P0_3) / (ni + C_PRIOR)
            if eb5 > eb5_max:
                eb5_max = eb5
            if eb2 > eb2_max:
                eb2_max = eb2
            if eb3 > eb3_max:
                eb3_max = eb3
        if ni >= 5 and tr.n5[i] >= 1:
            el_sol += sol
        if v2:
            n3i = tr.n3d[i]
            if n3i >= 3:
                known3 += 1
                eb5r = (tr.n53d[i] + C_PRIOR * P0_5) / (n3i + C_PRIOR)
                if eb5r > eb5r_max:
                    eb5r_max = float(eb5r)
    if known == 0:
        base = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return base + ((float(known3), eb5r_max) if v2 else ())
    rs = sorted(rates, reverse=True)
    base = (float(known), rs[0], float(np.mean(rates)), float(np.mean(rs[:3])),
            el_sol / tot if tot > 0 else 0.0, eb5_max, eb2_max, eb3_max)
    return base + ((float(known3), eb5r_max) if v2 else ())


def load_meta(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    if "max_x" not in test.columns:
        test = test.assign(max_x=np.nan, final_x=np.nan)
    ap = os.path.join(data_dir, "train_archive.csv")
    archive = pd.read_csv(ap) if os.path.exists(ap) else pd.DataFrame(columns=["mint", "mt", "max_x", "final_x"])
    mig = pd.read_csv(os.path.join(data_dir, "pfamm_migrations.csv"))
    cre = pd.read_csv(os.path.join(data_dir, "pumpfun_token_creation.csv"))
    sample = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    return train, test, archive, mig, cre, sample


def creation_features(cre, mints):
    cre = cre[~cre["mint"].duplicated(keep="first")]
    cre = cre.set_index("mint")
    F = pd.DataFrame(index=pd.Index(mints, name="mint"))
    for c in ["bundle_size", "bundled_buys_count", "dev_balance", "direct_pf_invocation",
              "mayhem_mode", "gas_used", "amount_of_instructions", "bundled_buys",
              "amount_of_lookup_reads", "amount_of_lookup_writes", "creation_ix_index",
              "pf_program_index", "is_cashback_enabled", "bonding_curve", "initial_supply"]:
        if c in cre.columns:
            F[c] = pd.to_numeric(cre[c].reindex(mints), errors="coerce")
    if "created_ts" in cre.columns:
        F["created_ts"] = pd.to_numeric(cre["created_ts"].reindex(mints), errors="coerce")
    if "name" in cre.columns:
        nm = cre["name"].fillna("").astype(str).reindex(mints)
        F["name_len"] = nm.str.len()
        F["name_digit"] = nm.str.contains(r"\d").astype(np.int8)
        F["name_upper"] = nm.apply(lambda t: sum(ch.isupper() for ch in t) / max(len(t), 1))
        if "symbol" in cre.columns:
            F["name_eq_sym"] = (nm.str.lower().str[:4] == cre["symbol"].fillna("").astype(str).str.lower().str[:4].reindex(mints)).astype(np.int8)
    if "symbol" in cre.columns:
        F["sym_len"] = cre["symbol"].fillna("").astype(str).reindex(mints).str.len()
    if "token_program" in cre.columns:
        F["token2022"] = (~cre["token_program"].fillna("").astype(str).reindex(mints).str.startswith("Tokenkeg")).astype(np.int8)
    if "bundle_structure" in cre.columns:
        bs = cre["bundle_structure"].fillna("").astype(str).reindex(mints)
        F["bs_ops"] = bs.str.split().str.len()
        F["bs_buys"] = bs.apply(lambda s: len(re.findall(r"dB\d", s)))
        amts = bs.apply(lambda s: [int(x) for x in re.findall(r":(\d{6,})", s)])
        F["bs_amt_sum"] = amts.apply(lambda v: float(sum(v)) if v else 0.0)
        F["bs_amt_max"] = amts.apply(lambda v: float(max(v)) if v else 0.0)
    if "creator" in cre.columns:
        F["creator"] = cre["creator"].reindex(mints)
    return F


def migration_features(mig, mints):
    mig = mig[~mig["mint"].duplicated(keep="first")]
    mig = mig.set_index("mint")
    F = pd.DataFrame(index=pd.Index(mints, name="mint"))
    F["mig_sol"] = pd.to_numeric(mig["sol_amount"].reindex(mints), errors="coerce") / 1e9
    F["mig_tok"] = pd.to_numeric(mig["mint_amount"].reindex(mints), errors="coerce") / 1e6
    return F


WA_COLS = ["wa_n60", "wa_n10m", "wa_max10m", "wa_n1h", "wa_max1h", "wa_n3h", "wa_max3h"]
WA_WINDOWS = [(60.0, 0, None), (600.0, 1, 2), (3600.0, 3, 4), (10800.0, 5, 6)]


def wa_query(ts, wid, mint, W, mint_id, mt):
    # per window: wa_n* = participants active in window; wa_max* = max distinct other tokens touched
    out = np.zeros(len(WA_COLS))
    if not len(W) or not len(ts):
        return out
    hi0 = int(np.searchsorted(ts, mt, side="left"))
    if hi0 <= 0:
        return out
    for wsec, nidx, midx in WA_WINDOWS:
        lo = int(np.searchsorted(ts, mt - wsec, side="right"))
        if hi0 <= lo:
            continue
        seg_w = wid[lo:hi0]
        m = np.isin(seg_w, W)
        if not m.any():
            continue
        sw = seg_w[m]
        sm = mint[lo:hi0][m]
        if mint_id >= 0:
            k2 = sm != mint_id
            sw, sm = sw[k2], sm[k2]
        if not len(sw):
            continue
        vals = pd.Series(sm, index=sw).groupby(level=0).nunique().reindex(W, fill_value=0)
        out[nidx] = int((vals > 0).sum())
        if midx is not None:
            out[midx] = float(vals.max())
    return out


def build_features(data_dir, train, test, archive, cre, mig, history_days=None):
    v2 = feat_v2()
    days = list_tape_days(data_dir)
    if history_days is not None:
        days = dict(list(days.items())[-history_days:])
    log(f"tape days in scan: {len(days)} ({list(days)[0]} .. {list(days)[-1]})")
    labels = {}
    for df in (train, test, archive):
        if len(df):
            for m, mt, mx, fx in zip(df["mint"], df["mt"], df["max_x"], df["final_x"]):
                labels[m] = (mt, mx, fx)
    all_mints = list(labels.keys())
    cre_f = creation_features(cre, all_mints)
    mig_f = migration_features(mig, all_mints)

    tr = WalletTracker()
    frames = []
    rep_rows = []
    v4 = feat_v4()
    wa_ts = np.empty(0, dtype=float)
    wa_wid = np.empty(0, dtype=np.int64)
    wa_mint = np.empty(0, dtype=np.int64)
    wa_rows = []
    mint_id = {}

    def _wa_update(c, a, d):
        nonlocal wa_ts, wa_wid, wa_mint
        if not v4:
            return
        try:
            lo = pd.Timestamp(d, tz="UTC").timestamp() - 10800.0
            chunks = []
            for df in (c, a):
                if df is None or not len(df) or "ts" not in df.columns:
                    continue
                k = df["ts"] >= lo
                if k.any():
                    chunks.append(df.loc[k, ["ts", "wallet", "mint"]])
            if not chunks:
                return
            r = pd.concat(chunks, ignore_index=True)
            r["wid"] = r["wallet"].map(tr.id)
            r["mid"] = r["mint"].map(lambda x: mint_id.setdefault(x, len(mint_id)))
            wa_ts = np.concatenate([wa_ts, r["ts"].to_numpy(dtype=float)])
            wa_wid = np.concatenate([wa_wid, r["wid"].to_numpy(dtype=np.int64)])
            wa_mint = np.concatenate([wa_mint, r["mid"].to_numpy(dtype=np.int64)])
            o = np.argsort(wa_ts, kind="mergesort")
            wa_ts, wa_wid, wa_mint = wa_ts[o], wa_wid[o], wa_mint[o]
        except Exception as e:
            log("  v4 wa update failed:", repr(e))

    for d, paths in days.items():
        t0 = time.time()
        c = read_curve(paths["curve"], v2=v2) if "curve" in paths else None
        a = read_amm(paths["amm"], v2=v2) if "amm" in paths else None
        _wa_update(c, a, d)
        dayfeat = None
        if c is not None and len(c):
            dayfeat = curve_features(c, v2=v2)
        if a is not None and len(a):
            af = amm_features(a, v2=v2)
            dayfeat = af if dayfeat is None else dayfeat.join(af, how="outer")
        if dayfeat is not None:
            frames.append(dayfeat)
        parts = participants(a, c)
        # tokens of this day: those in parts (tape). mt from labels.
        day_lab = [(m, labels[m]) for m in parts if m in labels]
        day_lab.sort(key=lambda x: x[1][0])
        for m, (mt, mx, fx) in day_lab:
            tr.advance_to(float(mt))
            p_a, p_c = parts[m]
            aw = list(p_a.items())
            cw = list(p_c.items())
            creator = cre_f["creator"].get(m)
            cr_n = 0.0
            cr_eb = 0.0
            if creator is not None and not (isinstance(creator, float) and math.isnan(creator)):
                ci = tr.wid.get(creator)
                if ci is not None and tr.n[ci] > 0:
                    cr_n = float(tr.n[ci])
                    cr_eb = (tr.n5[ci] + C_PRIOR * P0_5) / (tr.n[ci] + C_PRIOR)
            row = [m, float(mt)]
            row.extend(rep_block([w for w, _ in aw], [s for _, s in aw], tr, v2=v2))
            row.extend(rep_block([w for w, _ in cw], [s for _, s in cw], tr, v2=v2))
            row.extend([cr_n, cr_eb])
            rep_rows.append(row)
            wids = set(p_a) | set(p_c)
            if creator is not None and not (isinstance(creator, float) and math.isnan(creator)):
                wids.add(creator)
            if v4 and wids:
                wid_arr = np.array([tr.id(w) for w in wids], dtype=np.int64)
                wa_rows.append([m] + list(wa_query(wa_ts, wa_wid, wa_mint, wid_arr,
                                                    mint_id.get(m, -1), float(mt))))
            if mx is not None and not (isinstance(mx, float) and math.isnan(mx)):
                tr.add(float(mt), np.array([tr.id(w) for w in wids], dtype=np.int64),
                       1.0 if mx >= 5 else 0.0, 1.0 if mx >= 2 else 0.0, 1.0 if mx >= 3 else 0.0)
        del c, a
        gc.collect()
        log(f"  {d}: parts={len(parts)} labeled={len(day_lab)} pend={len(tr.events)} {time.time()-t0:.0f}s")
    if frames:
        feat = pd.concat(frames)
    else:
        feat = pd.DataFrame()
    rep = pd.DataFrame(rep_rows, columns=["mint", "mt"] + REP_COLS + (REP_COLS_V2 if v2 else []))
    rep = rep.set_index("mint")
    feat = feat.join(rep, how="left")
    feat = feat.join(cre_f.drop(columns=["creator"], errors="ignore"), how="left")
    feat = feat.join(mig_f, how="left")
    if v2 and "a_plast" in feat.columns and "c_plast" in feat.columns:
        feat["mig_pop"] = feat["a_plast"] / np.exp(feat["c_plast"]).replace(0, np.nan)
    if v4 and wa_rows:
        wadf = pd.DataFrame(wa_rows, columns=["mint"] + WA_COLS).set_index("mint")
        feat = feat.join(wadf, how="left")
        log(f"v4 wa features: {len(wadf)} rows, index size {len(wa_ts)}")
    if feat_v3():
        lab_mt = {m: v[0] for m, v in labels.items()}
        lab_mx = {m: v[1] for m, v in labels.items()}
        lab_fx = {m: v[2] for m, v in labels.items()}
        all_mts = np.empty(len(lab_mt))
        all_mxs = np.empty(len(lab_mt))
        all_fxs = np.empty(len(lab_mt))
        ks = list(lab_mt.keys())
        for i, m in enumerate(ks):
            all_mts[i] = lab_mt[m]
            v = lab_mx[m]
            all_mxs[i] = np.nan if (isinstance(v, float) and math.isnan(v)) else v
            w = lab_fx[m]
            all_fxs[i] = np.nan if (isinstance(w, float) and math.isnan(w)) else w
        order = np.argsort(all_mts, kind="mergesort")
        g_mt = all_mts[order]
        g_mx = all_mxs[order]
        g_fx = all_fxs[order]
        q_mt = feat["mt"].to_numpy(dtype=float)
        out = np.zeros((len(feat), len(REG_COLS_V3)))
        # pointers over global array; queries in mt order (sort queries, then map back)
        qorder = np.argsort(q_mt, kind="mergesort")
        q = q_mt[qorder]
        hi = lo1 = lo3 = 0
        n_w = n5_w = n2_w = big_w = fxs_w = 0.0
        n3_w = n53_w = big3_w = 0.0
        fxs3_w = 0.0
        res = np.zeros((len(q), len(REG_COLS_V3)))
        for i in range(len(q)):
            t = q[i]
            while hi < len(g_mt) and g_mt[hi] <= t - 3600.0:
                mxh, fxh = g_mx[hi], g_fx[hi]
                n_w += 1; n3_w += 1
                if not np.isnan(mxh):
                    if mxh >= 5: n5_w += 1; n53_w += 1
                    if mxh >= 2: n2_w += 1
                    b = max(0.0, mxh - 5.0)
                    big_w += b; big3_w += b
                if not np.isnan(fxh):
                    fxs_w += fxh; fxs3_w += fxh
                hi += 1
            while lo1 < hi and g_mt[lo1] < t - 3600.0 - 86400.0:
                mxl = g_mx[lo1]
                n_w -= 1
                if not np.isnan(mxl):
                    if mxl >= 5: n5_w -= 1
                    if mxl >= 2: n2_w -= 1
                    big_w -= max(0.0, mxl - 5.0)
                if not np.isnan(g_fx[lo1]):
                    fxs_w -= g_fx[lo1]
                lo1 += 1
            while lo3 < hi and g_mt[lo3] < t - 3600.0 - 259200.0:
                mxl = g_mx[lo3]
                n3_w -= 1
                if not np.isnan(mxl):
                    if mxl >= 5: n53_w -= 1
                    big3_w -= max(0.0, mxl - 5.0)
                if not np.isnan(g_fx[lo3]):
                    fxs3_w -= g_fx[lo3]
                lo3 += 1
            res[i] = [n_w, (n5_w / n_w) if n_w else 0.0, (n2_w / n_w) if n_w else 0.0,
                      big_w, (fxs_w / n_w) if n_w else 0.0,
                      n3_w, (n53_w / n3_w) if n3_w else 0.0, big3_w]
        back = np.empty_like(res)
        back[qorder] = res
        feat[REG_COLS_V3] = back
    if "creator" in cre_f.columns:
        feat["creator"] = cre_f["creator"].reindex(feat.index)
    if "created_ts" in feat.columns:
        feat["age_s"] = feat["mt"].values - feat["created_ts"].values
    feat = feat[~feat.index.duplicated(keep="last")].sort_index()
    return feat, tr


FEATURE_SKIP = {"mint", "mt", "creator", "created_ts", "max_x", "final_x", "day"}


def feature_cols(frame):
    cols = []
    for c in frame.columns:
        if c in FEATURE_SKIP:
            continue
        s = frame[c]
        if s.dtype == object:
            continue
        if pd.api.types.is_numeric_dtype(s):
            cols.append(c)
    return cols


def train_models(df, fast=False, calib=None):
    """df: labeled feature table (index=mint, contains max_x). calib: optional holdout for early stopping."""
    frame = df
    nest = int(os.environ.get("WF_LGB_NEST", "400")) if fast else int(os.environ.get("LGB_NEST", "900"))
    lr = 0.05 if fast else 0.02
    clf_p = dict(LGB_CLF); clf_p["n_estimators"] = nest; clf_p["learning_rate"] = lr
    reg_p = dict(LGB_REG); reg_p["n_estimators"] = nest; reg_p["learning_rate"] = lr
    y = frame["max_x"].values
    cols = feature_cols(frame)
    X = frame[cols].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    REC_HL = float(os.environ.get("REC_HL", "0"))
    w = None
    if REC_HL > 0 and "mt" in frame.columns:
        mtv = frame["mt"].to_numpy(dtype=float)
        okm = ~np.isnan(mtv)
        w = np.ones(len(mtv))
        w[okm] = np.exp(-(mtv.max() - mtv[okm]) / (REC_HL * 86400.0))
    yfx = frame["final_x"].values
    ev = None
    if calib is not None and len(calib) > 200:
        Xc = calib[cols].astype(float).replace([np.inf, -np.inf], np.nan)
        ev = [(Xc, (calib["max_x"].values >= lv).astype(int)) for lv in TP_LEVELS]
        ev_reg = Xc
    seeds = [int(x) for x in os.environ.get("SEEDS", str(SEED)).split(",")]
    models = {}
    for j, lv in enumerate(TP_LEVELS):
        ylv = (y >= lv).astype(int)
        ms = []
        for sd in seeds:
            m = lgb.LGBMClassifier(random_state=sd, **clf_p)
            if ev is not None and ylv.sum() >= 20:
                m.fit(X, ylv, eval_set=[ev[j]],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            else:
                m.fit(X, ylv, sample_weight=w)
            ms.append(m)
        models[f"clf_{lv}"] = ms
    for q in Q_LEVELS:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=SEED, **reg_p)
        m.fit(X, np.log(np.clip(yfx, 1e-6, None)), sample_weight=w)
        models[f"qfin_{q}"] = m
    if PBLEND > 0:
        ymx = np.log(np.clip(y, 1e-6, None))
        for q in Q_LEVELS:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=SEED, **reg_p)
            m.fit(X, ymx, sample_weight=w)
            models[f"qmax_{q}"] = m
    if CASCADE_P > 0:
        yk = (y >= KILL).astype(int)
        if yk.sum() >= 400:
            mk = lgb.LGBMClassifier(random_state=SEED, **clf_p)
            mk.fit(X, yk)
            models["clf_kill"] = mk
            Xs = X[yk.astype(bool)]
            for lv in TP_LEVELS:
                ycs = (y >= lv).astype(int)[yk.astype(bool)]
                if ycs.sum() < 20 or (ycs.size - ycs.sum()) < 20:
                    models[f"cond_{lv}"] = None
                    continue
                mc = lgb.LGBMClassifier(random_state=SEED, **clf_p)
                mc.fit(Xs, ycs)
                models[f"cond_{lv}"] = mc
    if HGBW > 0:
        from sklearn.ensemble import HistGradientBoostingClassifier
        for j, lv in enumerate(TP_LEVELS):
            ylv = (y >= lv).astype(int)
            if ylv.sum() < 20:
                models[f"hgb_{lv}"] = None
                continue
            h = HistGradientBoostingClassifier(max_iter=int(os.environ.get("HGB_ITER", "300")),
                                               learning_rate=0.05, max_leaf_nodes=31,
                                               min_samples_leaf=30, l2_regularization=1.0,
                                               random_state=SEED)
            h.fit(X, ylv)
            models[f"hgb_{lv}"] = h
    return models


def clf_predict(models, lv, X):
    ms = models[f"clf_{lv}"]
    if isinstance(ms, list):
        return np.mean([m.predict_proba(X)[:, 1] for m in ms], axis=0)
    return ms.predict_proba(X)[:, 1]


class _Platt:
    def __init__(self, lr):
        self.lr = lr

    def transform(self, p):
        pc = np.clip(p, 1e-9, 1 - 1e-9)
        x = np.log(pc / (1 - pc))
        return self.lr.predict_proba(x[:, None])[:, 1]


def fit_calibrators(models, calib, cols):
    mode = os.environ.get("CALIB", "off")
    Xc = calib[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    calibs = {}
    for lv in TP_LEVELS:
        p = clf_predict(models, lv, Xc)
        y = (calib["max_x"].values >= lv).astype(int)
        if y.sum() < 20 or (len(y) - y.sum()) < 20:
            calibs[lv] = None
            continue
        if mode == "platt":
            from sklearn.linear_model import LogisticRegression
            pc = np.clip(p, 1e-9, 1 - 1e-9)
            x = np.log(pc / (1 - pc))
            lr = LogisticRegression(C=1.0).fit(x[:, None], y)
            calibs[lv] = _Platt(lr)
        else:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(p, y)
            calibs[lv] = ir
    return calibs


def apply_calibrators(models, calibs, X):
    n = len(X)
    P = np.zeros((n, len(TP_LEVELS)))
    for j, lv in enumerate(TP_LEVELS):
        p = clf_predict(models, lv, X)
        ir = calibs.get(lv) if calibs else None
        P[:, j] = ir.transform(p) if ir is not None else p
    return P


def quantile_survival(QM, lvls):
    # piecewise-linear CDF over Q_LEVELS (log scale) -> P(value >= lv) per token
    xs = np.array([0.0] + list(Q_LEVELS) + [1.0])
    lo = QM[:, 0] - 1.0
    hi = QM[:, -1] + 1.0
    ys = np.maximum.accumulate(np.column_stack([lo, QM, hi]), axis=1)
    n = ys.shape[0]
    S = np.empty((n, len(lvls)))
    for k, lv in enumerate(np.log(np.asarray(lvls, dtype=float))):
        idx = np.clip((ys <= lv).sum(axis=1) - 1, 0, ys.shape[1] - 2)
        y0 = ys[np.arange(n), idx]
        y1 = ys[np.arange(n), idx + 1]
        w = np.clip((lv - y0) / (y1 - y0 + 1e-9), 0.0, 1.0)
        Sk = 1.0 - (xs[idx] + (xs[idx + 1] - xs[idx]) * w)
        Sk = np.where(lv <= lo, 1.0, np.where(lv >= hi, 0.0, Sk))
        S[:, k] = Sk
    return S


def predict_models(models, feat, cols, calibs=None):
    X = feat[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    n = len(X)
    P = apply_calibrators(models, calibs, X)
    if PBLEND > 0 and f"qmax_{Q_LEVELS[0]}" in models:
        QM = np.zeros((n, len(Q_LEVELS)))
        for j, q in enumerate(Q_LEVELS):
            QM[:, j] = models[f"qmax_{q}"].predict(X)
        Pq = quantile_survival(QM, TP_LEVELS)
        P = (1.0 - PBLEND) * P + PBLEND * Pq
    if HGBW > 0:
        for j, lv in enumerate(TP_LEVELS):
            h = models.get(f"hgb_{lv}")
            if h is None:
                continue
            ph = h.predict_proba(X)[:, 1]
            P[:, j] = (1.0 - HGBW) * P[:, j] + HGBW * ph
    if CASCADE_P > 0 and "clf_kill" in models:
        pk = models["clf_kill"].predict_proba(X)[:, 1]
        Pc = P.copy()
        for j, lv in enumerate(TP_LEVELS):
            mc = models.get(f"cond_{lv}")
            if mc is None:
                continue
            Pc[:, j] = pk * mc.predict_proba(X)[:, 1]
        Pc = np.minimum.accumulate(Pc[:, ::-1], axis=1)[:, ::-1]
        P = (1.0 - CASCADE_P) * P + CASCADE_P * Pc
    P = np.minimum.accumulate(P[:, ::-1], axis=1)[:, ::-1]  # enforce monotone non-increasing
    QF = np.zeros((n, len(Q_LEVELS)))
    for j, q in enumerate(Q_LEVELS):
        QF[:, j] = models[f"qfin_{q}"].predict(X)
    return P, QF, None


def interp_rows(x, xp, fp):
    """Interpolate each row of fp (n, L) at scalars x (G,) -> (n, G). xp ascending."""
    idx = np.clip(np.searchsorted(xp, x, side="right") - 1, 0, len(xp) - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    w = (x[None, :] - x0) / (x1 - x0)
    out = np.empty((fp.shape[0], len(x)))
    for k in range(len(x)):
        out[:, k] = (1 - w[:, k]) * fp[:, idx[k]] + w[:, k] * fp[:, idx[k] + 1]
    return out


def ev_and_tp(P, QF):
    """EV-optimal tp per token. P: (n, len(TP_LEVELS)); QF: (n, len(Q_LEVELS)) log-final quantiles."""
    n = P.shape[0]
    rng = np.random.default_rng(SEED)
    u = np.concatenate([rng.random((n, N_MC)), 1.0 - rng.random((n, N_MC))], axis=1)  # antithetic
    # final_x samples from predicted quantile fn (log scale), flat-tail extrapolation
    lo = QF[:, 0] - 1.0
    hi = QF[:, -1] + 1.0
    xs = np.array([0.0] + list(Q_LEVELS) + [1.0])
    ys = np.column_stack([lo, QF, hi])  # (n, len(xs))
    F = np.zeros((n, u.shape[1]))
    for k in range(len(xs) - 1):
        mask = (u >= xs[k]) & (u < xs[k + 1])
        w = np.where(mask, (u - xs[k]) / (xs[k + 1] - xs[k]), 0.0)
        lo = ys[:, k][:, None]
        hi = ys[:, k + 1][:, None]
        F = np.where(mask, lo + w * (hi - lo), F)
    F = np.exp(F)  # (n, N_MC) final_x samples
    evcond = os.environ.get("EVCOND", "off") == "on"
    held = np.empty((n, len(TP_GRID)))
    CH = 2048
    for i0 in range(0, n, CH):
        i1 = min(i0 + CH, n)
        Fc = F[i0:i1]
        if evcond:
            # held branch is only reachable when final_x < tp (else the token reached tp)
            G = Fc[:, :, None] < TP_GRID[None, None, :]
            num = np.where(G, np.maximum(Fc[:, :, None] - 1.0 - COST, -1.0), 0.0)
            den = G.sum(axis=1)
            held[i0:i1] = np.where(den > 0, num.sum(axis=1) / np.maximum(den, 1), -1.0)
        else:
            held[i0:i1] = np.maximum(np.minimum(Fc[:, :, None], TP_GRID[None, None, :]) - 1.0 - COST, -1.0).mean(axis=1)
    Pg = interp_rows(TP_GRID, np.array(TP_LEVELS), P)
    EV = Pg * (TP_GRID[None, :] - 1.0 - COST) + (1.0 - Pg) * held
    EV = np.where(np.isnan(EV), -np.inf, EV)
    idx = np.argmax(EV, axis=1)
    tp = TP_GRID[idx]
    best = EV[np.arange(n), idx]
    tp = np.where((best > EVTHR) & (tp > 0), tp, 0.0)
    return tp, EV, Pg


def pnl(tp, mx, fx):
    tp = np.asarray(tp, float)
    mx = np.asarray(mx, float)
    fx = np.asarray(fx, float)
    hit = (mx >= tp) & (tp > 0)
    held = np.maximum(fx - 1.0 - COST, -1.0)
    return np.where(tp <= 0, 0.0, np.where(hit, tp - 1.0 - COST, held))


def walk_forward(feat, train, out, n_days=None):
    if n_days is None:
        n_days = int(os.environ.get("WF_DAYS", str(WF_DAYS_DEFAULT)))
    t = train.set_index("mint")
    cols = feature_cols(feat)
    df = feat.join(t[["max_x", "final_x"]], how="inner")
    df = df[df["mt"].notna()]
    df["day"] = pd.to_datetime(df["mt"], unit="s", utc=True).dt.date
    days = sorted(df["day"].unique())[-n_days:]
    rows = []
    all_days = sorted(df["day"].unique())
    for D in days:
        val = df[df["day"] == D]
        dstart = val["mt"].min()
        trn_all = df[(df["day"] < D) & (df["mt"] <= dstart - 7200)]
        if len(trn_all) < 1500 or len(val) < 30:
            continue
        use_calib = os.environ.get("CALIB", "off") in ("on", "platt")
        if use_calib:
            calib_days = all_days[all_days.index(D) - 3: all_days.index(D)]
            calib = trn_all[trn_all["day"].isin(calib_days)]
            trn = trn_all[~trn_all["day"].isin(calib_days)]
        else:
            calib = trn_all
            trn = trn_all
        models = train_models(trn, fast=True, calib=calib if use_calib else None)
        calibs = fit_calibrators(models, calib, cols) if use_calib else None
        P, QF, _ = predict_models(models, val, cols, calibs)
        tp_ev, EV, Pg = ev_and_tp(P, QF)
        mx = val["max_x"].values
        fx = val["final_x"].values
        pnl_ev = float(pnl(tp_ev, mx, fx).sum())
        p5 = Pg[:, list(TP_GRID).index(5.0)] if 5.0 in list(TP_GRID) else None
        rec = dict(day=str(D), n=len(val), n_trades=int((tp_ev > 0).sum()),
                   pnl_ev=pnl_ev, auc5=None, brier5=None, auc5_raw=None,
                   pnl_top1_5=None, pnl_top2_5=None, pnl_top05_5=None)
        from sklearn.metrics import roc_auc_score
        try:
            rec["auc5"] = float(roc_auc_score((mx >= 5).astype(int), p5))
            rec["brier5"] = float(np.mean((p5 - (mx >= 5).astype(int)) ** 2))
            Xv = val[cols].astype(float).replace([np.inf, -np.inf], np.nan)
            p5_raw = clf_predict(models, 5.0, Xv)
            rec["auc5_raw"] = float(roc_auc_score((mx >= 5).astype(int), p5_raw))
        except Exception:
            pass
        for cap in (10.0, 20.0, 31.0):
            tc = np.minimum(tp_ev, cap)
            rec[f"pnl_ev_c{int(cap)}"] = float(pnl(tc, mx, fx).sum())
        order = np.argsort(-p5)
        sc = Pg[:, list(TP_LEVELS).index(2.0)] * Pg[:, list(TP_LEVELS).index(5.0)]
        k1 = max(1, int(len(val) * 1.0 / 100.0))
        oc = np.argsort(-sc)[:k1]
        rec["pnl_cascade_top1_5"] = float(pnl(np.where(np.isin(np.arange(len(val)), oc), 5.0, 0.0), mx, fx).sum())
        for q, tag in [(1.0, "top1_5"), (2.0, "top2_5"), (0.5, "top05_5")]:
            k = max(1, int(len(val) * q / 100.0))
            rec[f"pnl_{tag}"] = float(pnl(np.where(np.arange(len(val)) < k, 5.0, 0.0), mx, fx).sum())
            tp_r = np.where(np.isin(np.arange(len(val)), order[:k]), tp_ev, 0.0)
            rec[f"pnl_ev_{tag}"] = float(pnl(tp_r, mx, fx).sum())
        bestrank = np.argsort(-EV)
        for q, tag in [(1.0, "best1_5"), (0.5, "best05_5")]:
            k = max(1, int(len(val) * q / 100.0))
            rec[f"pnl_{tag}"] = float(pnl(np.where(np.isin(np.arange(len(val)), bestrank[:k]), 5.0, 0.0), mx, fx).sum())
        tp_lo = np.where((tp_ev > 0) & (tp_ev <= 6.0), tp_ev, 0.0)
        tp_hi = np.where(tp_ev > 6.0, tp_ev, 0.0)
        rec["pnl_ev_lo"] = float(pnl(tp_lo, mx, fx).sum())
        rec["pnl_ev_hi"] = float(pnl(tp_hi, mx, fx).sum())
        rows.append(rec)
        log(f"  wf {D}: n={len(val)} trades={rec['n_trades']} pnl_ev={pnl_ev:+.2f} "
            f"auc5={rec['auc5'] and round(rec['auc5'], 4)} top1%_5={rec['pnl_top1_5']:+.2f} "
            f"evtop1%={rec['pnl_ev_top1_5']:+.2f}")
        del models
        gc.collect()
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "wf_results.csv"), index=False)
    if not len(res):
        log("WF: no eligible days (dataset too small)")
        return res
    log(f"WF TOTAL: pnl_ev={res['pnl_ev'].sum():+.2f} top1={res['pnl_top1_5'].sum():+.2f} "
        f"top2={res['pnl_top2_5'].sum():+.2f} top0.5={res['pnl_top05_5'].sum():+.2f} "
        f"evtop1={res['pnl_ev_top1_5'].sum():+.2f} evtop2={res['pnl_ev_top2_5'].sum():+.2f} "
        f"evc10={res['pnl_ev_c10'].sum():+.2f} evc20={res['pnl_ev_c20'].sum():+.2f} evc31={res['pnl_ev_c31'].sum():+.2f} "
        f"cascade={res['pnl_cascade_top1_5'].sum():+.2f}")
    return res


def make_submission(feat, train, test, sample, out):
    t = train.set_index("mint")
    cols = feature_cols(feat)
    df = feat.join(t[["max_x", "final_x"]], how="inner")
    df = df[df["mt"].notna()]
    df["day"] = pd.to_datetime(df["mt"], unit="s", utc=True).dt.date
    all_days = sorted(df["day"].unique())
    use_calib = os.environ.get("CALIB", "off") in ("on", "platt")
    calib_days = all_days[-4:]
    calib = df[df["day"].isin(calib_days)]
    trn = df[~df["day"].isin(calib_days)] if use_calib else df
    if len(trn) < 500:
        trn = df
    models = train_models(trn, fast=False, calib=calib if use_calib else None)
    calibs = fit_calibrators(models, calib, cols) if use_calib else None
    cols_test = [c for c in cols if c in feat.columns]
    P, QF, _ = predict_models(models, feat.loc[test["mint"].values], cols_test, calibs)
    tp_ev, EV, Pg = ev_and_tp(P, QF)
    log(f"submission EV stats: mean_best_EV={EV.max(1).mean():.3f} p5 mean={Pg[:, list(TP_GRID).index(5.0)].mean():.4f}")
    out_df = sample.copy()
    tp_map = dict(zip(test["mint"].values, np.round(tp_ev, 4)))
    out_df["tp"] = out_df["mint"].map(tp_map).fillna(0.0)
    # enforce contract
    out_df["tp"] = np.where((out_df["tp"] >= 3) & (out_df["tp"] <= 100), out_df["tp"], 0.0)
    assert len(out_df) == len(sample)
    out_df.to_csv(os.path.join(out, "submission.csv"), index=False)
    log(f"submission: {int((out_df['tp'] > 0).sum())}/{len(out_df)} trades; "
        f"tp value counts: {out_df['tp'].value_counts().head(8).to_dict()}")


def smoke(data_dir, train, test, cre, mig):
    days = list_tape_days(data_dir)
    log(f"days: {len(days)} first={list(days)[0]} last={list(days)[-1]}")
    log(f"train days: {sorted(set(pd.to_datetime(train['mt'], unit='s', utc=True).dt.date))[:3]} ... "
        f"{sorted(set(pd.to_datetime(train['mt'], unit='s', utc=True).dt.date))[-3:]}")
    log(f"test days: {sorted(set(pd.to_datetime(test['mt'], unit='s', utc=True).dt.date))}")
    last = list(days)[-2]
    p = days[last]
    c = pd.read_parquet(p["curve"], columns=CURVE_COLS) if "curve" in p else None
    a = read_amm(p["amm"]) if "amm" in p else None
    log(f"day {last}: curve={len(c) if c is not None else 0} rows, amm={len(a) if a is not None else 0} rows")
    if c is not None:
        log("curve cols sample:", c[["mint", "ts_rel", "direction", "sol", "failed", "new_price"]].head(3).to_dict("records"))
        log("curve direction values:", c["direction"].value_counts().to_dict())
        log("curve failed values:", c["failed"].value_counts().to_dict())
        log("curve ts_rel range:", c["ts_rel"].min(), c["ts_rel"].max())
        log("curve mints:", c["mint"].nunique())
    if a is not None:
        log("amm direction values:", a["direction"].value_counts().to_dict())
        log("amm ts_rel range:", a["ts_rel"].min(), a["ts_rel"].max())
        log("amm mints:", a["mint"].nunique())
        log("amm sample:", a.head(3).to_dict("records"))
    tc = pd.to_datetime(train["mt"], unit="s", utc=True)
    tt = pd.to_datetime(test["mt"], unit="s", utc=True)
    log(f"train mt range: {tc.min()} .. {tc.max()}")
    log(f"test mt range: {tt.min()} .. {tt.max()}")
    log(f"train mints in test: {len(set(train['mint']) & set(test['mint']))}")
    log(f"cre cols: {list(cre.columns)}")
    log(f"mig cols: {list(mig.columns)}")
    log("SMOKE DONE")




def pipe_full_check(c):
    return curve_features(c)

def diagnose(feat, train, test, data_dir):
    log("=== curve alignment debug (first tape day) ===")
    days = list_tape_days(data_dir)
    d0 = list(days)[0]
    c = pd.read_parquet(days[d0]["curve"], columns=CURVE_COLS)
    log("dtypes:", dict(c.dtypes.astype(str)))
    okm = c["failed"] == 0
    ok = c[okm]
    log(f"len(c)={len(c)} len(ok)={len(ok)}")
    log("c mint type:", type(c["mint"].iloc[0]), repr(c["mint"].iloc[0])[:60])
    log("ok mint type:", type(ok["mint"].iloc[0]), repr(ok["mint"].iloc[0])[:60])
    mints = pd.Index(c["mint"].unique(), name="mint")
    sz = ok.groupby("mint").size()
    log("mints[:2]:", list(mints[:2]), "sz.index[:2]:", list(sz.index[:2]))
    log("set overlap:", len(set(mints.tolist()) & set(sz.index.tolist())), "of", len(mints), len(sz.index))
    F = pd.DataFrame(index=mints)
    F["c_n"] = sz
    log("NaN rate after c_n assign:", F["c_n"].isna().mean())
    F2 = pipe_full_check(c)
    log("full curve_features NaN rate c_n:", F2["c_n"].isna().mean(), "rows:", len(F2))
    del c, ok
    t = train.set_index("mint")
    df = feat.join(t[["max_x", "final_x"]], how="inner")
    df["day"] = pd.to_datetime(df["mt"], unit="s", utc=True).dt.date
    log("=== daily hit rates (last 14 days) ===")
    days = sorted(df["day"].unique())
    for D in days[-14:]:
        v = df[df["day"] == D]
        mx = v["max_x"].values
        log(f"  {D}: n={len(v)} p5={100*(mx>=5).mean():.2f}% p3={100*(mx>=3).mean():.2f}% "
            f"p2={100*(mx>=2).mean():.2f}% p10={100*(mx>=10).mean():.2f}% "
            f"medmax={np.median(mx):.2f} medfin={np.median(v['final_x'].values):.3f}")
    cols = feature_cols(feat)
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y = df["max_x"].values
    log("=== feature sanity ===")
    for c in cols:
        s_ = X[c]
        nanr = s_.isna().mean()
        if nanr > 0.5:
            log(f"  HIGH-NAN {c}: {nanr:.2f}")
    log("=== spearman corr with max_x (top 30) ===")
    from scipy import stats as sps
    cors = {}
    for c in cols:
        x = X[c].values
        m = ~np.isnan(x)
        if m.sum() < 100:
            continue
        try:
            rho, _ = sps.spearmanr(x[m], y[m])
            cors[c] = rho
        except Exception:
            pass
    for c, r in sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:30]:
        log(f"  {c}: {r:+.4f}")
    log("=== manual recompute check (3 tokens, last train day) ===")
    last = days[-1]
    dpath = list_tape_days(data_dir)
    dstr = str(last)
    p = dpath[dstr]
    c = pd.read_parquet(p["curve"], columns=CURVE_COLS)
    a = read_amm(p["amm"])
    sample_mints = df[df["day"] == last].sample(3, random_state=1).index
    for m in sample_mints:
        cc = c[c["mint"] == m]
        aa = a[a["mint"] == m]
        ok = cc[cc["failed"] == 0]
        lab = df.loc[m]
        log(f"  {m}: label max_x={lab['max_x']:.3f} final_x={lab['final_x']:.4f} | "
            f"feat c_n={lab['c_n']:.0f} c_sol={lab['c_sol']:.2f} a_n={lab['a_n']:.0f} a_sol={lab['a_sol']:.2f} | "
            f"raw c_n={len(ok)} c_sol={ok['sol'].sum():.2f} a_n={len(aa)} a_sol={aa['sol'].sum():.2f}")
    del c, a
    log("DIAG DONE")


def main():
    mode = os.environ.get("MODE", "both")
    out = os.environ.get("OUT_DIR", os.environ.get("KAGGLE_WORKING_DIR", "out"))
    os.makedirs(out, exist_ok=True)
    data_dir = find_data_dir()
    log(f"DATA: {data_dir} MODE: {mode} OUT: {out} FEAT_V2={feat_v2()} FEAT_V3={feat_v3()} FEAT_V4={feat_v4()} "
        f"CALIB={os.environ.get('CALIB', 'off')} EVCOND={os.environ.get('EVCOND', 'off')} "
        f"EVTHR={EVTHR} PBLEND={PBLEND} HGBW={HGBW} CASCADE_P={CASCADE_P} N_MC={N_MC} "
        f"SEEDS={os.environ.get('SEEDS', SEED)} NJOBS={NJOBS}")
    train, test, archive, mig, cre, sample = load_meta(data_dir)
    log(f"train={len(train)} test={len(test)} archive={len(archive)} mig={len(mig)} cre={len(cre)}")
    if mode == "smoke":
        smoke(data_dir, train, test, cre, mig)
        return
    if mode == "diag":
        t0 = time.time()
        feat, tr = build_features(data_dir, train, test, archive, cre, mig, history_days=None)
        log(f"features: {feat.shape} in {time.time()-t0:.0f}s")
        diagnose(feat, train, test, data_dir)
        return
    t0 = time.time()
    hist = int(os.environ.get("HISTORY_DAYS", "0")) or None
    resume = os.environ.get("RESUME_FEATS", "")
    feat = None
    if resume and os.path.exists(resume):
        try:
            if resume.endswith(".csv"):
                feat = pd.read_csv(resume, index_col=0)
            else:
                feat = pd.read_parquet(resume)
            log(f"features resumed from {resume}: {feat.shape}")
            tr = None
        except Exception as e:
            log(f"resume failed ({e}); rebuilding")
    if feat is None:
        feat, tr = build_features(data_dir, train, test, archive, cre, mig, history_days=hist)
        log(f"features: {feat.shape} in {time.time()-t0:.0f}s; wallets={len(tr.wid) if tr is not None else '?'}")
    if os.environ.get("FEAT_SAVE", ""):
        feat.reset_index().to_parquet(os.environ["FEAT_SAVE"])
        log("features saved to", os.environ["FEAT_SAVE"])
    feat.reset_index().to_csv(os.path.join(out, "features.csv"), index=False)
    if mode in ("wf", "both"):
        walk_forward(feat, train, out)
    if mode in ("submit", "both"):
        make_submission(feat, train, test, sample, out)
    log("PIPELINE DONE total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
