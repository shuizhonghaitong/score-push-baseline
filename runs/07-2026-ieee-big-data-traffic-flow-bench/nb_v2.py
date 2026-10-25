# tfb-run-v1: reproduction of the 0.86517 public pipeline (anubisyy), single-file form.
# Private run notebook. Data + assets attached via kernel-metadata.json.
from pathlib import Path
import os, sys, json, gc, subprocess, time

T0 = time.time()
WORK = Path('/kaggle/working')
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
SOURCE = WORK / 'tfb_source'
SOURCE.mkdir(exist_ok=True)
INPUT = Path('/kaggle/input')

def find_unique(filename):
    matches = list(INPUT.rglob(filename))
    assert len(matches) == 1, f"Expected one {filename}, found {matches}. Check the attached inputs."
    return matches[0]

DATA = Path(os.environ['TFB_DATA_ROOT']) if 'TFB_DATA_ROOT' in os.environ else find_unique('submission_key.csv').parent
ASSETS = Path(os.environ['TFB_ASSET_ROOT']) if 'TFB_ASSET_ROOT' in os.environ else find_unique('onset_lgb.txt').parent
MODEL_DIR = ASSETS
OUTPUT = WORK / 'predictions'
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ.update(TFB_DATA_ROOT=str(DATA), TFB_OUTPUT_ROOT=str(OUTPUT), TFB_THREADS=str(min(4, os.cpu_count() or 1)))
print('Data:', DATA)
print('Assets:', ASSETS)

try:
    from IPython.display import display
except Exception:
    def display(x=None):
        print(x if x is not None else '')

MODULES = dict((
'base_panel.py', r'''
"""Train a causal queue classifier on observed train months and predict Task 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from config import RELEASE, OUT, THREADS
MONTHS = {"2030-06", "2030-11", "2031-01", "2031-02"}
HISTORY = 12
HORIZON = 6
SEED = 20260930


def panel_names():
    return [entry["corridor_id"] for entry in json.loads((RELEASE / "config" / "corridors.json").read_text())["panels"]
            if entry["corridor_id"] not in {"D12_I405_N", "D12_I405_S"}]


def avg(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    result = np.nansum(values, axis=0) / np.maximum(count, 1)
    return np.where(count > 0, result, np.nan)


def min_finite(values):
    result = np.where(np.isfinite(values), values, np.inf).min(axis=0)
    return np.where(np.isfinite(result), result, np.nan)


@dataclass
class Window:
    X: np.ndarray
    y: np.ndarray
    eligible: np.ndarray
    panel: str
    condition: str


class Panel:
    def __init__(self, name, panel_no):
        self.name = name
        self.panel_no = panel_no
        directory = RELEASE / "corridors" / name / "network"
        fd = pd.read_csv(directory / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")
        topo = pd.read_csv(directory / "lwr_mainline_topology.csv", dtype=str)
        self.links = fd.link_id.astype(str).tolist()
        self.link_index = {key: i for i, key in enumerate(self.links)}
        self.lanes = fd.lanes.to_numpy(dtype=np.float32)
        self.vf = fd.free_speed_kmh.to_numpy(dtype=np.float32)
        self.cap = fd.capacity_vph.to_numpy(dtype=np.float32)
        self.length = fd.length_km.to_numpy(dtype=np.float32)
        self.cut = (fd.v_cut if "v_cut" in fd else 0.6 * fd.free_speed_kmh).to_numpy(dtype=np.float32)
        self.incoming = self.adjacency(topo, "incoming_link_ids")
        self.outgoing = self.adjacency(topo, "outgoing_link_ids")

    def adjacency(self, topo, column):
        rows, cols = [], []
        for row in topo.itertuples(index=False):
            link = str(row.link_id)
            if link not in self.link_index:
                continue
            for other in str(getattr(row, column)).split(";"):
                if other in self.link_index:
                    rows.append(self.link_index[link])
                    cols.append(self.link_index[other])
        result = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                   shape=(len(self.links), len(self.links)))
        result.data[:] = 1.0
        return result

    def neighboring(self, graph, values):
        valid = np.isfinite(values).astype(np.float32)
        count = graph @ valid
        mean = graph @ np.nan_to_num(values, nan=0.0)
        return np.where(count > 0, mean / np.maximum(count, 1), np.nan)''',
'config.py', r'''
from pathlib import Path
import os
RELEASE = Path(os.environ["TFB_DATA_ROOT"])
OUT = Path(os.environ.get("TFB_OUTPUT_ROOT", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
THREADS = int(os.environ.get("TFB_THREADS", "4"))''',
'prepare_state.py', r'''
"""State features with lag/lead context and two independently synthesized masks."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from config import OUT, THREADS
import state_model as base

def features(panel,pi,times,v,q,e,rows,cols,regime):
    X,bv,bq=base.compute_features(panel,pi,times,v,q,rows,cols,regime)
    extra=[];lanes=np.maximum(panel.lanes[cols],1)
    for offset in (-12,-6,-3,-2,-1,1,2,3,6,12):
        rr=rows+offset;valid=(rr>=0)&(rr<len(times));rr=np.clip(rr,0,len(times)-1)
        vv=v[rr,cols];qq=q[rr,cols]/lanes
        extra.extend([np.where(valid,vv,np.nan),np.where(valid,qq,np.nan),
                      np.where(valid,qq/np.maximum(vv,1),np.nan),
                      np.where(valid&e[rr,cols]&np.isfinite(vv),1,0)])
    return np.column_stack([X,*extra]).astype(np.float32),bv,bq

def main():
    xx=[];yv=[];yq=[];val={k:[] for k in ('X','yv','yq','bv','bq','panel','regime')};audit=[]
    original_seed=base.SEED
    for pi,name in enumerate(base.all_panels()):
        p=base.Panel(name,pi)
        for month in (*base.TRAIN_MONTHS,base.VALID_MONTH):
            times,v,q,e=base.matrices(base.month_frame(name,month),p)
            isval=month==base.VALID_MONTH
            audit.append({'panel':name,'month':month,'cells':v.size,'speed_missing':float(np.isnan(v).mean()),'eligible':float(e.mean()),'finite_ineligible':float((np.isfinite(v)&~e).mean()),'negative_flow':int((q<0).sum())})
            for rep in range(1 if isval else 2):
                base.SEED=original_seed+rep*7919
                mask,regime=base.simulated_mask(name,times,v,q,e)
                rows,cols=base.sampled_targets(mask,times,1500 if isval else 750)
                X,bv,bq=features(p,pi,times,np.where(mask,np.nan,v),np.where(mask,np.nan,q),e,rows,cols,regime)
                rv=v[rows,cols]-bv;rq=q[rows,cols]/p.lanes[cols]-bq
                if isval:
                    for k,a in [('X',X),('yv',rv),('yq',rq),('bv',bv),('bq',bq),('panel',np.full(len(rows),pi)),('regime',regime[rows])]:val[k].append(a)
                else:xx.append(X);yv.append(rv);yq.append(rq)
            print(name,month,'done',flush=True)
    base.SEED=original_seed
    np.savez(OUT/'train.npz',X=np.concatenate(xx),yv=np.concatenate(yv),yq=np.concatenate(yq))
    np.savez(OUT/'valid.npz',**{k:np.concatenate(a) for k,a in val.items()})
    pd.DataFrame(audit).to_csv(OUT/'data_audit.csv',index=False)
    print('train',sum(len(a) for a in yv),'valid',sum(len(a) for a in val['yv']),'features',xx[0].shape[1],flush=True)
if __name__=='__main__':main()''',
'queue_features.py', r'''
"""Train a causal queue classifier on observed train months and predict Task 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from config import RELEASE, OUT, THREADS
MONTHS = {"2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02"}
HISTORY = 12
HORIZON = 6
SEED = 20260930


def panel_names():
    return [entry["corridor_id"] for entry in json.loads((RELEASE / "config" / "corridors.json").read_text())["panels"]
            if entry["corridor_id"] not in {"D12_I405_N", "D12_I405_S"}]


def avg(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    result = np.nansum(values, axis=0) / np.maximum(count, 1)
    return np.where(count > 0, result, np.nan)


def min_finite(values):
    result = np.where(np.isfinite(values), values, np.inf).min(axis=0)
    return np.where(np.isfinite(result), result, np.nan)


@dataclass
class Window:
    X: np.ndarray
    y: np.ndarray
    eligible: np.ndarray
    panel: str
    condition: str


class Panel:
    def __init__(self, name, panel_no):
        self.name = name
        self.panel_no = panel_no
        directory = RELEASE / "corridors" / name / "network"
        fd = pd.read_csv(directory / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")
        topo = pd.read_csv(directory / "lwr_mainline_topology.csv", dtype=str)
        self.links = fd.link_id.astype(str).tolist()
        self.link_index = {key: i for i, key in enumerate(self.links)}
        sample = sorted((RELEASE / "corridors" / name / "train" / "mainline_states" /
                         "year_month=2030-06").glob("*.parquet"))[0]
        location = pd.read_parquet(sample, columns=["link_id", "milepost"]).drop_duplicates("link_id")
        location["link_id"] = location.link_id.astype(str)
        milepost = location.set_index("link_id").reindex(self.links).milepost.to_numpy(dtype=np.float32)
        if not np.isfinite(milepost).all():
            raise ValueError(f"Missing mileposts for {name}")
        self.milepost = milepost
        self.sorted_links = np.argsort(milepost)
        self.link_rank = np.empty(len(self.links), dtype=np.int32)
        self.link_rank[self.sorted_links] = np.arange(len(self.links))
        self.lanes = fd.lanes.to_numpy(dtype=np.float32)
        self.vf = fd.free_speed_kmh.to_numpy(dtype=np.float32)
        self.cap = fd.capacity_vph.to_numpy(dtype=np.float32)
        self.length = fd.length_km.to_numpy(dtype=np.float32)
        self.cut = (fd.v_cut if "v_cut" in fd else 0.6 * fd.free_speed_kmh).to_numpy(dtype=np.float32)
        self.incoming = self.adjacency(topo, "incoming_link_ids")
        self.outgoing = self.adjacency(topo, "outgoing_link_ids")

    def adjacency(self, topo, column):
        rows, cols = [], []
        for row in topo.itertuples(index=False):
            link = str(row.link_id)
            if link not in self.link_index:
                continue
            for other in str(getattr(row, column)).split(";"):
                if other in self.link_index:
                    rows.append(self.link_index[link])
                    cols.append(self.link_index[other])
        result = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                   shape=(len(self.links), len(self.links)))
        result.data[:] = 1.0
        return result

    def neighboring(self, graph, values):
        valid = np.isfinite(values).astype(np.float32)
        count = graph @ valid
        mean = graph @ np.nan_to_num(values, nan=0.0)
        return np.where(count > 0, mean / np.maximum(count, 1), np.nan)

    def features(self, speed, flow, observed, origin, condition):
        """History arrays have exactly twelve rows ending five minutes before T."""
        if speed.shape != (HISTORY, len(self.links)) or flow.shape != speed.shape:
            raise ValueError((self.name, speed.shape, flow.shape))
        speed = speed.astype(np.float32) / self.vf[None, :]
        flow = flow.astype(np.float32) / self.cap[None, :]
        speed = np.where(np.isfinite(speed), speed, np.nan)
        flow = np.where(np.isfinite(flow), flow, np.nan)
        valid = np.isfinite(speed)
        positions = np.where(valid, np.arange(HISTORY)[:, None], -1).max(axis=0)
        columns = np.arange(len(self.links))
        last_speed = np.where(positions >= 0, speed[np.maximum(positions, 0), columns], np.nan)
        last_flow = np.where(positions >= 0, flow[np.maximum(positions, 0), columns], np.nan)
        age = HISTORY - 1 - positions
        age = np.where(positions >= 0, age, 12)
        recent_speed = avg(speed[-3:])
        prior_speed = avg(speed[-6:-3])
        recent_flow = avg(flow[-3:])
        prior_flow = avg(flow[-6:-3])
        mean_speed = avg(speed)
        mean_flow = avg(flow)
        queue_history = np.where(valid, speed <= (self.cut / self.vf)[None, :], False)
        last_queue = last_speed <= self.cut / self.vf
        hour = origin.hour + origin.minute / 60
        tod_sin = np.sin(2 * np.pi * hour / 24)
        tod_cos = np.cos(2 * np.pi * hour / 24)
        dow = origin.dayofweek
        constants = lambda value: np.full(len(self.links), value, dtype=np.float32)
        features = [
            constants(self.panel_no), np.arange(len(self.links), dtype=np.float32),
            self.vf, self.cap / np.maximum(self.lanes, 1), self.lanes, self.length,
            constants(hour), constants(tod_sin), constants(tod_cos), constants(dow),
            constants(1 if condition == "queue_ongoing" else 0),
            last_speed, last_flow, constants(0) + age, recent_speed, prior_speed,
            recent_speed - prior_speed, recent_flow, prior_flow, recent_flow - prior_flow,
            mean_speed, min_finite(speed), mean_flow, np.nanmax(np.where(np.isfinite(flow), flow, -np.inf), axis=0),
            valid.mean(axis=0), queue_history.sum(axis=0).astype(np.float32), last_queue.astype(np.float32),
            self.neighboring(self.incoming, last_speed), self.neighboring(self.outgoing, last_speed),
            self.neighboring(self.incoming, last_flow), self.neighboring(self.outgoing, last_flow),
            self.neighboring(self.incoming, last_queue.astype(np.float32)),
            self.neighboring(self.outgoing, last_queue.astype(np.float32)),
            constants(np.isfinite(speed).mean()), constants(np.nanmin(speed) if np.isfinite(speed).any() else 1.0),
        ]
        features[25] = np.where(np.isfinite(features[25]), features[25], np.nan)
        features[24] = np.where(np.isfinite(features[24]), features[24], np.nan)
        # Short lags preserve the shape of an approaching shock wave. A mean over
        # the whole history loses the timing of a rapid pre-onset slowdown.
        for lag in (1, 2, 3, 6, 9, 12):
            features.extend((speed[-lag], flow[-lag]))
        features.extend((
            min_finite(speed[-3:]), min_finite(speed[-6:]),
            avg(speed[-3:]) - avg(speed[-6:-3]),
            avg(flow[-3:]) - avg(flow[-6:-3]),
            self.neighboring(self.incoming, avg(speed[-3:])),
            self.neighboring(self.outgoing, avg(speed[-3:])),
            self.neighboring(self.incoming, avg(flow[-3:])),
            self.neighboring(self.outgoing, avg(flow[-3:])),
            self.neighboring(self.incoming, queue_history[-1].astype(np.float32)),
            self.neighboring(self.outgoing, queue_history[-1].astype(np.float32)),
        ))
        def milepost_neighbor(values, offset):
            ranks = self.link_rank + offset
            valid_neighbor = (ranks >= 0) & (ranks < len(self.links))
            links = self.sorted_links[np.clip(ranks, 0, len(self.links)-1)]
            return np.where(valid_neighbor, values[links], np.nan)

        features.append(self.milepost)
        for offset in (-2, -1, 1, 2):
            neighbor_milepost = milepost_neighbor(self.milepost, offset)
            features.extend((
                np.abs(neighbor_milepost-self.milepost),
                milepost_neighbor(last_speed, offset),
                milepost_neighbor(recent_speed, offset),
                milepost_neighbor(last_flow, offset),
                milepost_neighbor(recent_flow, offset),
                milepost_neighbor(last_queue.astype(np.float32), offset),
            ))
        base = np.column_stack(features).astype(np.float32)
        X = np.tile(base, (HORIZON, 1))
        X = np.column_stack([X, np.repeat(np.arange(1, HORIZON + 1), len(self.links))]).astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        return X

    def train_day(self, path):
        columns = ["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]
        frame = pd.read_parquet(path, columns=columns)
        frame["link_id"] = frame.link_id.astype(str)
        if frame.duplicated(["timestamp", "link_id"]).any():
            frame = frame.groupby(["timestamp", "link_id"], as_index=False).mean(numeric_only=True)
        times = sorted(frame.timestamp.unique())
        index = pd.Index(times)
        speed = frame.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=self.links).to_numpy(dtype=np.float32)
        flow = frame.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=self.links).to_numpy(dtype=np.float32)
        eligible = frame.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=self.links).fillna(0).to_numpy(dtype=bool)
        return times, speed, flow, eligible


def candidate_origins(panel, times, speed, eligible):
    queued = (speed <= panel.cut[None, :]) & eligible & np.isfinite(speed)
    active = queued.any(axis=1)
    candidates = {"queue_onset": [], "queue_ongoing": []}
    for t in range(HISTORY, len(times) - HORIZON):
        if np.isfinite(speed[t-HISTORY:t]).mean() < .7:
            continue
        if not active[t+1:t+HORIZON+1].any():
            continue
        if not active[t-HISTORY:t].any() and not active[t]:
            candidates["queue_onset"].append(t)
        if active[t]:
            candidates["queue_ongoing"].append(t)
    return candidates''',
'queue_ongoing.py', r'''
"""Train a causal queue classifier on observed train months and predict Task 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse

from config import RELEASE, OUT, THREADS
MONTHS = {"2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02"}
HISTORY = 12
HORIZON = 6
SEED = 20260930


def panel_names():
    return [entry["corridor_id"] for entry in json.loads((RELEASE / "config" / "corridors.json").read_text())["panels"]
            if entry["corridor_id"] not in {"D12_I405_N", "D12_I405_S"}]


def avg(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    result = np.nansum(values, axis=0) / np.maximum(count, 1)
    return np.where(count > 0, result, np.nan)


def min_finite(values):
    result = np.where(np.isfinite(values), values, np.inf).min(axis=0)
    return np.where(np.isfinite(result), result, np.nan)


@dataclass
class Window:
    X: np.ndarray
    y: np.ndarray
    eligible: np.ndarray
    panel: str
    condition: str


class Panel:
    def __init__(self, name, panel_no):
        self.name = name
        self.panel_no = panel_no
        directory = RELEASE / "corridors" / name / "network"
        fd = pd.read_csv(directory / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")
        topo = pd.read_csv(directory / "lwr_mainline_topology.csv", dtype=str)
        self.links = fd.link_id.astype(str).tolist()
        self.link_index = {key: i for i, key in enumerate(self.links)}
        self.lanes = fd.lanes.to_numpy(dtype=np.float32)
        self.vf = fd.free_speed_kmh.to_numpy(dtype=np.float32)
        self.cap = fd.capacity_vph.to_numpy(dtype=np.float32)
        self.length = fd.length_km.to_numpy(dtype=np.float32)
        self.cut = (fd.v_cut if "v_cut" in fd else 0.6 * fd.free_speed_kmh).to_numpy(dtype=np.float32)
        self.incoming = self.adjacency(topo, "incoming_link_ids")
        self.outgoing = self.adjacency(topo, "outgoing_link_ids")

    def adjacency(self, topo, column):
        rows, cols = [], []
        for row in topo.itertuples(index=False):
            link = str(row.link_id)
            if link not in self.link_index:
                continue
            for other in str(getattr(row, column)).split(";"):
                if other in self.link_index:
                    rows.append(self.link_index[link])
                    cols.append(self.link_index[other])
        result = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                   shape=(len(self.links), len(self.links)))
        result.data[:] = 1.0
        return result

    def neighboring(self, graph, values):
        valid = np.isfinite(values).astype(np.float32)
        count = graph @ valid
        mean = graph @ np.nan_to_num(values, nan=0.0)
        return np.where(count > 0, mean / np.maximum(count, 1), np.nan)

    def features(self, speed, flow, observed, origin, condition):
        """History arrays have exactly twelve rows ending five minutes before T."""
        if speed.shape != (HISTORY, len(self.links)) or flow.shape != speed.shape:
            raise ValueError((self.name, speed.shape, flow.shape))
        speed = speed.astype(np.float32) / self.vf[None, :]
        flow = flow.astype(np.float32) / self.cap[None, :]
        speed = np.where(np.isfinite(speed), speed, np.nan)
        flow = np.where(np.isfinite(flow), flow, np.nan)
        valid = np.isfinite(speed)
        positions = np.where(valid, np.arange(HISTORY)[:, None], -1).max(axis=0)
        columns = np.arange(len(self.links))
        last_speed = np.where(positions >= 0, speed[np.maximum(positions, 0), columns], np.nan)
        last_flow = np.where(positions >= 0, flow[np.maximum(positions, 0), columns], np.nan)
        age = HISTORY - 1 - positions
        age = np.where(positions >= 0, age, 12)
        recent_speed = avg(speed[-3:])
        prior_speed = avg(speed[-6:-3])
        recent_flow = avg(flow[-3:])
        prior_flow = avg(flow[-6:-3])
        mean_speed = avg(speed)
        mean_flow = avg(flow)
        queue_history = np.where(valid, speed <= (self.cut / self.vf)[None, :], False)
        last_queue = last_speed <= self.cut / self.vf
        hour = origin.hour + origin.minute / 60
        tod_sin = np.sin(2 * np.pi * hour / 24)
        tod_cos = np.cos(2 * np.pi * hour / 24)
        dow = origin.dayofweek
        constants = lambda value: np.full(len(self.links), value, dtype=np.float32)
        features = [
            constants(self.panel_no), np.arange(len(self.links), dtype=np.float32),
            self.vf, self.cap / np.maximum(self.lanes, 1), self.lanes, self.length,
            constants(hour), constants(tod_sin), constants(tod_cos), constants(dow),
            constants(1 if condition == "queue_ongoing" else 0),
            last_speed, last_flow, constants(0) + age, recent_speed, prior_speed,
            recent_speed - prior_speed, recent_flow, prior_flow, recent_flow - prior_flow,
            mean_speed, min_finite(speed), mean_flow, np.nanmax(np.where(np.isfinite(flow), flow, -np.inf), axis=0),
            valid.mean(axis=0), queue_history.sum(axis=0).astype(np.float32), last_queue.astype(np.float32),
            self.neighboring(self.incoming, last_speed), self.neighboring(self.outgoing, last_speed),
            self.neighboring(self.incoming, last_flow), self.neighboring(self.outgoing, last_flow),
            self.neighboring(self.incoming, last_queue.astype(np.float32)),
            self.neighboring(self.outgoing, last_queue.astype(np.float32)),
            constants(np.isfinite(speed).mean()), constants(np.nanmin(speed) if np.isfinite(speed).any() else 1.0),
        ]
        features[25] = np.where(np.isfinite(features[25]), features[25], np.nan)
        features[24] = np.where(np.isfinite(features[24]), features[24], np.nan)
        base = np.column_stack(features).astype(np.float32)
        X = np.tile(base, (HORIZON, 1))
        X = np.column_stack([X, np.repeat(np.arange(1, HORIZON + 1), len(self.links))]).astype(np.float32)
        X[~np.isfinite(X)] = np.nan
        return X

    def train_day(self, path):
        columns = ["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]
        frame = pd.read_parquet(path, columns=columns)
        frame["link_id"] = frame.link_id.astype(str)
        if frame.duplicated(["timestamp", "link_id"]).any():
            frame = frame.groupby(["timestamp", "link_id"], as_index=False).mean(numeric_only=True)
        times = sorted(frame.timestamp.unique())
        index = pd.Index(times)
        speed = frame.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=self.links).to_numpy(dtype=np.float32)
        flow = frame.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=self.links).to_numpy(dtype=np.float32)
        eligible = frame.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=self.links).fillna(0).to_numpy(dtype=bool)
        return times, speed, flow, eligible


def candidate_origins(panel, times, speed, eligible):
    queued = (speed <= panel.cut[None, :]) & eligible & np.isfinite(speed)
    active = queued.any(axis=1)
    candidates = {"queue_onset": [], "queue_ongoing": []}
    for t in range(HISTORY, len(times) - HORIZON):
        if np.isfinite(speed[t-HISTORY:t]).mean() < .7:
            continue
        if not active[t+1:t+HORIZON+1].any():
            continue
        if not active[t-HISTORY:t].any() and not active[t]:
            candidates["queue_onset"].append(t)
        if active[t]:
            candidates["queue_ongoing"].append(t)
    return candidates


def choose_origins(candidates, panel, date):
    output = []
    for condition, values in candidates.items():
        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|{panel}|{date}|{condition}".encode(), digest_size=8).digest(), "big")
        rng = np.random.default_rng(seed)
        chosen = []
        for t in rng.permutation(values):
            if all(abs(t - other) >= 72 for other in chosen):
                chosen.append(int(t))
            if len(chosen) >= 2:
                break
        output.extend((condition, t) for t in chosen)
    return output


def collect():
    train_X, train_y, train_w = [], [], []
    valid_windows = []
    feature_count = None
    for panel_no, name in enumerate(panel_names()):
        panel = Panel(name, panel_no)
        files = sorted((RELEASE / "corridors" / name / "train" / "mainline_states").glob("**/*.parquet"))
        files = [path for path in files if path.parent.name.removeprefix("year_month=") in MONTHS]
        print(f"data {name}: {len(files)} days", flush=True)
        panel_counts = {"train_onset": 0, "train_ongoing": 0, "valid_onset": 0, "valid_ongoing": 0}
        for path in files:
            times, speed, flow, eligible = panel.train_day(path)
            choices = choose_origins(candidate_origins(panel, times, speed, eligible), name, times[0][:10])
            is_valid_month = path.parent.name.endswith("2031-02")
            for condition, t in choices:
                origin = pd.Timestamp(times[t])
                X = panel.features(speed[t-HISTORY:t], flow[t-HISTORY:t], eligible[t-HISTORY:t], origin, condition)
                future_speed = speed[t+1:t+HORIZON+1]
                y = ((future_speed <= panel.cut[None, :]) & np.isfinite(future_speed)).reshape(-1)
                valid = (eligible[t+1:t+HORIZON+1] & np.isfinite(future_speed)).reshape(-1)
                if valid.sum() < .6 * len(valid):
                    continue
                if is_valid_month:
                    valid_windows.append(Window(X, y, valid, name, condition))
                    panel_counts[f"valid_{'onset' if condition == 'queue_onset' else 'ongoing'}"] += 1
                else:
                    train_X.append(X[valid])
                    train_y.append(y[valid].astype(np.uint8))
                    train_w.append(np.full(valid.sum(), 1000 / valid.sum(), dtype=np.float32))
                    panel_counts[f"train_{'onset' if condition == 'queue_onset' else 'ongoing'}"] += 1
                feature_count = X.shape[1]
        print(panel_counts, flush=True)
    X = np.vstack(train_X)
    y = np.concatenate(train_y)
    weights = np.concatenate(train_w)
    print(f"train: {X.shape}, positive={y.mean():.5f}; validation windows={len(valid_windows)}", flush=True)
    return X, y, weights, valid_windows, feature_count


def score_windows(windows, probabilities, threshold, onset_min):
    by_panel_condition = {}
    for window, p in zip(windows, probabilities):
        prediction = p >= threshold
        if window.condition == "queue_onset" and prediction.sum() < onset_min:
            index = np.argpartition(p, -onset_min)[-onset_min:]
            prediction[index] = True
        true = window.y & window.eligible
        predicted = prediction & window.eligible
        union = np.logical_or(true, predicted).sum()
        iou = np.logical_and(true, predicted).sum() / max(union, 1)
        by_panel_condition.setdefault((window.panel, window.condition), []).append(iou)
    panels = panel_names()
    condition_means = {key: float(np.mean(values)) for key, values in by_panel_condition.items()}
    overall = np.mean([np.mean([condition_means.get((panel, c), 0) for c in ("queue_onset", "queue_ongoing")]) for panel in panels])
    onset = np.mean([condition_means.get((panel, "queue_onset"), 0) for panel in panels])
    ongoing = np.mean([condition_means.get((panel, "queue_ongoing"), 0) for panel in panels])
    return float(overall), float(onset), float(ongoing)


def fit_model():
    X, y, weights, windows, feature_count = collect()
    persistence = [((w.X[:, 11] <= .60) & (w.X[:, 13] <= 3)).astype(np.float32) for w in windows]
    print("proxy persistence IoU, onset, ongoing:", score_windows(windows, persistence, .5, 0), flush=True)
    X_val = np.vstack([w.X[w.eligible] for w in windows])
    y_val = np.concatenate([w.y[w.eligible] for w in windows]).astype(np.uint8)
    w_val = np.concatenate([np.full(w.eligible.sum(), 1000 / w.eligible.sum(), dtype=np.float32) for w in windows])
    train_set = lgb.Dataset(X, label=y, weight=weights, free_raw_data=True)
    valid_set = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_set, free_raw_data=True)
    params = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.045,
        "num_leaves": 31, "max_depth": 8, "min_data_in_leaf": 250,
        "feature_fraction": .85, "bagging_fraction": .85, "bagging_freq": 1,
        "lambda_l2": 8.0, "max_bin": 127, "num_threads": THREADS,
        "verbosity": -1, "seed": SEED,
    }
    model = lgb.train(params, train_set, num_boost_round=700, valid_sets=[valid_set],
                      callbacks=[lgb.early_stopping(60, verbose=True), lgb.log_evaluation(50)])
    model.save_model(str(OUT / "queue_lgb.txt"))
    del X, X_val, train_set, valid_set
    probs = [model.predict(w.X, num_threads=THREADS) for w in windows]
    trials = []
    for threshold in np.arange(.05, .501, .025):
        for onset_min in (0, 3, 6, 12, 24):
            overall, onset, ongoing = score_windows(windows, probs, float(threshold), onset_min)
            trials.append((overall, onset, ongoing, float(threshold), onset_min))
    trials.sort(reverse=True)
    print("best threshold trials", trials[:10], flush=True)
    result = {"best_iteration": model.best_iteration, "feature_count": feature_count,
              "validation_windows": len(windows), "positive_fraction": float(y.mean()),
              "threshold": trials[0][3], "onset_min": trials[0][4],
              "proxy_iou": trials[0][0], "proxy_onset_iou": trials[0][1],
              "proxy_ongoing_iou": trials[0][2]}
    (OUT / "queue_model.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result, flush=True)
    return model, result''',
'state_inference.py', r'''
"""State residual ensemble; GPU prediction matches the original execution path."""
from pathlib import Path
import json,os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
from config import OUT,THREADS
from prepare_state import base,features

class StatePredictor:
    def __init__(self, assets):
        self.assets=Path(assets)
        self.settings={channel+'_mix': {'d8': 0.5, 'd10': 0.5} for channel in ('speed','flow')}
        self.task_type=os.environ.get('TFB_PREDICT_DEVICE', 'GPU' if get_gpu_device_count()>0 else 'CPU')
        self.models={}
        for channel in ('speed','flow'):
            for tag in self.settings[channel+'_mix']:
                model=CatBoostRegressor();model.load_model(str(self.assets/(channel+'_'+tag+'.cbm')))
                self.models[channel,tag]=model
        print('State inference device:',self.task_type,flush=True)
    def residual(self,X,channel):
        result=np.zeros(len(X))
        for tag,weight in self.settings[channel+'_mix'].items():
            model=self.models[channel,tag]
            for start in range(0,len(X),250000):
                batch=X[start:start+250000]
                pred=model.predict(batch,task_type=self.task_type,thread_count=THREADS)
                result[start:start+len(batch)]+=weight*pred
        return result

def predict_states(predictor):
    first=True
    for split in ('validation','private'):
        for pi,name in enumerate(base.all_panels()):
            panel=base.Panel(name,pi)
            frame=base.month_frame(name,'',split=split)
            times,v,q,e=base.matrices(frame,panel)
            target=pd.read_csv(base.RELEASE/'task1'/name/split/'sample_submission_state.csv',dtype={'station_id':str,'link_id':str})
            rows=times.get_indexer(target.timestamp);cols=pd.Index(panel.links).get_indexer(target.link_id)
            assert (rows>=0).all() and (cols>=0).all()
            regime=target.mask_regime.map({'R1':0,'R2':1,'R3':2}).to_numpy()
            X,bv,bq=features(panel,pi,times,v,q,e,rows,cols,regime)
            assert X.shape[1]==94
            target['speed_kmh']=np.clip(bv+predictor.residual(X,'speed'),1,180)
            target['flow_vph']=np.clip((bq+predictor.residual(X,'flow'))*panel.lanes[cols],0,150000)
            assert np.isfinite(target[['speed_kmh','flow_vph']].to_numpy()).all()
            columns=['panel','timestamp','station_id','link_id','mask_regime','speed_kmh','flow_vph']
            target[columns].to_csv(OUT/'state.csv',index=False,mode='w' if first else 'a',header=first)
            first=False;print(name,split,len(target),flush=True)''',
'state_model.py', r'''
"""LightGBM residual reconstruction with temporal and spatial context."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import RELEASE, OUT, THREADS
from base_panel import Panel as BasePanel

TRAIN_MONTHS = ("2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01")
VALID_MONTH = "2031-02"
SEED = 20261001
RATE = np.array([.20, .30, .50], dtype=np.float32)


class Panel(BasePanel):
    """Add milepost order from static, public training metadata."""

    def __init__(self, name, panel_no):
        super().__init__(name, panel_no)
        sample = sorted((RELEASE / "corridors" / name / "train" / "mainline_states" /
                         "year_month=2030-06").glob("*.parquet"))[0]
        location = pd.read_parquet(sample, columns=["link_id", "milepost"]).drop_duplicates("link_id")
        location["link_id"] = location.link_id.astype(str)
        milepost = location.set_index("link_id").reindex(self.links).milepost.to_numpy(dtype=np.float32)
        if not np.isfinite(milepost).all():
            raise ValueError(f"Missing mileposts for {name}")
        self.milepost = milepost
        self.sorted_links = np.argsort(milepost)
        self.link_rank = np.empty(len(self.links), dtype=np.int32)
        self.link_rank[self.sorted_links] = np.arange(len(self.links))


def all_panels():
    config = json.loads((RELEASE / "config" / "corridors.json").read_text())
    return [x["corridor_id"] for x in config["panels"]]


def month_frame(panel, month, split="train"):
    directory = RELEASE / "corridors" / panel / split / "mainline_states"
    if split != "train":
        directory = RELEASE / "corridors" / panel / split / "mainline_states_masked"
        paths = sorted(directory.glob("**/*.parquet"))
    else:
        paths = sorted(directory.glob(f"year_month={month}/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No state data: {panel} {split} {month}")
    cols = ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]
    frame = pd.concat((pd.read_parquet(path, columns=cols) for path in paths), ignore_index=True)
    frame["link_id"] = frame.link_id.astype(str)
    frame["station_id"] = frame.station_id.astype(str)
    if frame.duplicated(["timestamp", "link_id"]).any():
        raise ValueError(f"Duplicate link/time for {panel} {month}")
    return frame


def matrices(frame, panel):
    index = pd.Index(sorted(frame.timestamp.unique()))
    cols = panel.links
    speed = frame.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=cols).to_numpy(dtype=np.float32)
    flow = frame.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=cols).to_numpy(dtype=np.float32)
    eligible = frame.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=cols).fillna(0).to_numpy(dtype=bool)
    return index, speed, flow, eligible


def simulated_mask(panel_name, times, speed, flow, eligible):
    dates = pd.Index([text[:10] for text in times])
    result = np.zeros(speed.shape, dtype=bool)
    regimes = np.empty(speed.shape[0], dtype=np.uint8)
    for date in dates.unique():
        rows = np.flatnonzero(dates == date)
        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|{panel_name}|{date}".encode(), digest_size=8).digest(), "big")
        regime = seed % 3
        rng = np.random.default_rng(seed)
        regimes[rows] = regime
        result[rows] = (rng.random((len(rows), speed.shape[1])) < RATE[regime]) & eligible[rows] \
                       & np.isfinite(speed[rows]) & np.isfinite(flow[rows])
    return result, regimes


def sampled_targets(mask, times, per_day):
    dates = pd.Index([text[:10] for text in times])
    row_out, col_out = [], []
    for date in dates.unique():
        rows = np.flatnonzero(dates == date)
        coords = np.argwhere(mask[rows])
        if not len(coords):
            continue
        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|sample|{date}|{mask.shape[1]}".encode(), digest_size=8).digest(), "big")
        rng = np.random.default_rng(seed)
        chosen = coords[rng.choice(len(coords), size=min(per_day, len(coords)), replace=False)]
        row_out.append(rows[chosen[:, 0]])
        col_out.append(chosen[:, 1])
    return np.concatenate(row_out), np.concatenate(col_out)


def graph_mean(graph, values):
    observed = np.isfinite(values).astype(np.float32)
    count = (graph @ observed.T).T
    mean = (graph @ np.nan_to_num(values, nan=0.0).T).T / np.maximum(count, 1)
    mean = np.where(count > 0, mean, np.nan)
    return mean, count


def compute_features(panel, panel_no, times, observed_speed, observed_flow, rows, cols, regime):
    n_times, n_links = observed_speed.shape
    temporal_speed = pd.DataFrame(observed_speed).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    temporal_flow = pd.DataFrame(observed_flow).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)
    temporal_speed = np.where(np.isfinite(temporal_speed), temporal_speed, panel.vf[None, :])
    temporal_flow = np.where(np.isfinite(temporal_flow), temporal_flow, .6 * panel.cap[None, :])

    graph = (panel.incoming + panel.outgoing).tocsr()
    graph.data[:] = 1.0
    spatial_flow, spatial_count = graph_mean(graph, observed_flow)
    flow_base = np.where(spatial_count > 0, .5 * temporal_flow + .5 * spatial_flow, temporal_flow)
    up_speed, up_count = graph_mean(panel.incoming, observed_speed)
    down_speed, down_count = graph_mean(panel.outgoing, observed_speed)
    up_flow, _ = graph_mean(panel.incoming, observed_flow)
    down_flow, _ = graph_mean(panel.outgoing, observed_flow)

    valid = np.isfinite(observed_speed)
    positions = np.arange(n_times, dtype=np.int32)[:, None]
    prev_index = np.maximum.accumulate(np.where(valid, positions, -1), axis=0)
    next_index = np.minimum.accumulate(np.where(valid, positions, n_times)[::-1], axis=0)[::-1]
    p = prev_index[rows, cols]
    n = next_index[rows, cols]
    prev_v = observed_speed[np.maximum(p, 0), cols].copy()
    next_v = observed_speed[np.minimum(n, n_times - 1), cols].copy()
    prev_q = observed_flow[np.maximum(p, 0), cols].copy()
    next_q = observed_flow[np.minimum(n, n_times - 1), cols].copy()
    prev_v[p < 0] = np.nan
    prev_q[p < 0] = np.nan
    next_v[n >= n_times] = np.nan
    next_q[n >= n_times] = np.nan
    prev_gap = np.where(p >= 0, rows - p, 300).astype(np.float32)
    next_gap = np.where(n < n_times, n - rows, 300).astype(np.float32)

    stamp = pd.to_datetime(times, utc=True)
    tod = stamp.hour.to_numpy(dtype=np.float32) + stamp.minute.to_numpy(dtype=np.float32) / 60
    dow = stamp.dayofweek.to_numpy(dtype=np.float32)
    lanes = np.maximum(panel.lanes[cols], 1)
    base_speed = temporal_speed[rows, cols]
    base_flow = flow_base[rows, cols] / lanes
    temporal_flow_lane = temporal_flow[rows, cols] / lanes
    spatial_flow_lane = spatial_flow[rows, cols] / lanes
    up_q = up_flow[rows, cols] / lanes
    down_q = down_flow[rows, cols] / lanes
    regime_vector = regime if len(regime) == len(rows) else regime[rows]
    features = [
        np.full(len(rows), panel_no), cols, regime_vector,
        tod[rows], np.sin(2*np.pi*tod[rows]/24), np.cos(2*np.pi*tod[rows]/24), dow[rows],
        panel.vf[cols], panel.cap[cols] / lanes, lanes, panel.length[cols],
        base_speed, base_flow, temporal_flow_lane, spatial_flow_lane,
        base_speed / panel.vf[cols], base_flow * lanes / panel.cap[cols],
        prev_v, next_v, prev_q / lanes, next_q / lanes, prev_gap, next_gap,
        next_v - prev_v, (next_q - prev_q) / lanes,
        up_speed[rows, cols], down_speed[rows, cols], up_q, down_q,
        up_count[rows, cols], down_count[rows, cols], spatial_count[rows, cols],
        up_speed[rows, cols] - down_speed[rows, cols], up_q - down_q,
    ]
    for offset in (-2, -1, 1, 2):
        ranks = panel.link_rank[cols] + offset
        valid_neighbor = (ranks >= 0) & (ranks < n_links)
        neighbor = panel.sorted_links[np.clip(ranks, 0, n_links-1)]
        neighbor_lanes = np.maximum(panel.lanes[neighbor], 1)
        same_time_v = observed_speed[rows, neighbor]
        same_time_q = observed_flow[rows, neighbor] / neighbor_lanes
        filled_v = temporal_speed[rows, neighbor]
        filled_q = temporal_flow[rows, neighbor] / neighbor_lanes
        features.extend((
            np.where(valid_neighbor, np.abs(panel.milepost[neighbor]-panel.milepost[cols]), np.nan),
            np.where(valid_neighbor, same_time_v, np.nan),
            np.where(valid_neighbor, same_time_q, np.nan),
            np.where(valid_neighbor, filled_v, np.nan),
            np.where(valid_neighbor, filled_q, np.nan),
        ))
    X = np.column_stack(features).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, base_speed, base_flow''',
'train_ongoing.py', r'''
from queue_ongoing import fit_model
from config import OUT
model, settings = fit_model()
model.save_model(str(OUT/'ongoing_lgb.txt'))''',
'train_onset.py', r'''
"""Train the two-stage onset model using June–January and validate on February."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from config import RELEASE, OUT, THREADS
from queue_features import Panel, panel_names, HISTORY, HORIZON, candidate_origins
Xtrain=[];ytrain=[];valid=[]
for pi,name in enumerate(panel_names()):
    p=Panel(name,pi)
    for file in sorted((RELEASE/'corridors'/name/'train/mainline_states').glob('**/*.parquet')):
        times,v,q,e=p.train_day(file)
        cand=candidate_origins(p,times,v,e)['queue_onset'];cs=set(cand)
        for t in [a for a in cand if a-1 not in cs]:
            x=p.features(v[t-HISTORY:t],q[t-HISTORY:t],e[t-HISTORY:t],pd.Timestamp(times[t]),'queue_onset')[-len(p.links):]
            y=(v[t+HORIZON]<=p.cut)&np.isfinite(v[t+HORIZON]); eligible=e[t+HORIZON]&np.isfinite(v[t+HORIZON])
            if file.parent.name.endswith('2031-02'):
                valid.append((name,x,y,eligible))
            else:
                Xtrain.append(x[eligible]);ytrain.append(y[eligible])
    print(name,'train rows',sum(len(a) for a in ytrain),'valid windows',len(valid),flush=True)
X=np.concatenate(Xtrain);y=np.concatenate(ytrain)
Xv=np.concatenate([a[1][a[3]] for a in valid]);yv=np.concatenate([a[2][a[3]] for a in valid])
params={'objective':'binary','metric':'binary_logloss','learning_rate':.04,'num_leaves':15,'min_data_in_leaf':100,'lambda_l2':10,'feature_fraction':.9,'num_threads':THREADS,'verbosity':-1,'seed':20261004}
model=lgb.train(params,lgb.Dataset(X,label=y),num_boost_round=900,valid_sets=[lgb.Dataset(Xv,label=yv)],callbacks=[lgb.early_stopping(70),lgb.log_evaluation(100)])
model.save_model(str(OUT/'onset_stage1.txt'))

model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=1500,
                  init_model=str(OUT/'onset_stage1.txt'),
                  valid_sets=[lgb.Dataset(Xv, label=yv)],
                  callbacks=[lgb.early_stopping(70), lgb.log_evaluation(100)])
model.save_model(str(OUT/'onset_lgb.txt'))
print("Selected onset threshold: 0.50; predict the final horizon step with a one-link fallback.")''',
'train_onset_catboost.py', r'''
"""GPU onset classifier with causal history-dropout augmentation."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
from config import RELEASE, OUT, THREADS
from queue_features import Panel,panel_names,HISTORY,HORIZON,candidate_origins
xx=[];yy=[];valid=[];rng=np.random.default_rng(20261005)
for pi,name in enumerate(panel_names()):
    p=Panel(name,pi)
    for file in sorted((RELEASE/'corridors'/name/'train/mainline_states').glob('**/*.parquet')):
        times,v,q,e=p.train_day(file);cand=candidate_origins(p,times,v,e)['queue_onset'];cs=set(cand);isval=file.parent.name.endswith('2031-02')
        for t in [a for a in cand if a-1 not in cs]:
            truth=(v[t+1:t+HORIZON+1]<=p.cut)&np.isfinite(v[t+1:t+HORIZON+1]);eligible=e[t+1:t+HORIZON+1]&np.isfinite(v[t+1:t+HORIZON+1])
            for rep in range(1 if isval else 2):
                hv=v[t-HISTORY:t].copy();hq=q[t-HISTORY:t].copy();he=e[t-HISTORY:t].copy()
                if rep:
                    drop=rng.random(hv.shape)<.05;hv[drop]=np.nan;hq[drop]=np.nan;he[drop]=False
                x=p.features(hv,hq,he,pd.Timestamp(times[t]),'queue_onset')[-len(p.links):]
                if isval:valid.append((name,x,truth,eligible))
                else:xx.append(x[eligible[-1]]);yy.append(truth[-1,eligible[-1]])
    print(name,'train rows',sum(len(a) for a in yy),'valid windows',len(valid),flush=True)
X=np.concatenate(xx);y=np.concatenate(yy);Xv=np.concatenate([a[1][a[3][-1]] for a in valid]);yv=np.concatenate([a[2][-1,a[3][-1]] for a in valid])
m=CatBoostClassifier(iterations=3000,depth=7,learning_rate=.05,l2_leaf_reg=10,task_type='GPU',devices=os.environ.get('TFB_GPU_DEVICE', '0'),thread_count=THREADS,random_seed=20261005,loss_function='Logloss',early_stopping_rounds=150,verbose=100,train_dir=str(OUT/'catboost'))
m.fit(X,y,eval_set=(Xv,yv),use_best_model=True);m.save_model(str(OUT/'onset.cbm'))
old=lgb.Booster(model_file=str(OUT/'onset_lgb.txt'));records=[]
for name,x,truth,e in valid:
    p=m.predict_proba(x,thread_count=THREADS)[:,1];b=old.predict(x,num_threads=THREADS)
    for weight in (0,.5,1):
        prob=(1-weight)*b+weight*p
        for threshold in (.3,.35,.4,.45,.5,.55,.6):
            pred=np.zeros_like(truth);pred[-1]=prob>=threshold
            if not pred.any():pred[-1,np.argmax(prob)]=True
            iou=((pred&truth)&e).sum()/max(((pred|truth)&e).sum(),1)
            records.append({'panel':name,'gpu_weight':weight,'threshold':threshold,'iou':iou})
f=pd.DataFrame(records);f.to_csv(OUT/'window_scores.csv',index=False)
s=f.groupby(['gpu_weight','threshold','panel']).iou.mean().groupby(['gpu_weight','threshold']).mean().sort_values(ascending=False);s.to_csv(OUT/'scores.csv');print(s.to_string(),flush=True)
best=s.index[0];(OUT/'onset_settings.json').write_text(json.dumps({'gpu_weight':float(best[0]),'threshold':float(best[1]),'proxy':float(s.iloc[0]),'train_rows':len(y),'valid_windows':len(valid)},indent=2))''',
'train_state.py', r'''
from pathlib import Path
import argparse,json,time
import numpy as np
from catboost import CatBoostRegressor
from config import OUT, THREADS
ap=argparse.ArgumentParser();ap.add_argument('--channel',choices=['speed','flow'],required=True);ap.add_argument('--device',required=True);ap.add_argument('--iterations',type=int,default=9000);ap.add_argument('--depth',type=int,default=8);ap.add_argument('--tag',default='d8');a=ap.parse_args()
t=time.time();key='yv' if a.channel=='speed' else 'yq'
tr=np.load(OUT/'train.npz');va=np.load(OUT/'valid.npz')
model=CatBoostRegressor(iterations=a.iterations,depth=a.depth,learning_rate=.06,l2_leaf_reg=10,loss_function='RMSE',eval_metric='RMSE',task_type='GPU',devices=a.device,gpu_ram_part=.8,thread_count=THREADS,random_seed=20261005,bootstrap_type='Bernoulli',subsample=.85,early_stopping_rounds=150,verbose=100,train_dir=str(OUT/('catboost_'+a.channel+'_'+a.tag)))
model.fit(tr['X'],tr[key],eval_set=(va['X'],va[key]),use_best_model=True)
model.save_model(str(OUT/(a.channel+'_'+a.tag+'.cbm')))
pred=model.predict(va['X'],thread_count=THREADS);np.save(OUT/('valid_'+a.channel+'_'+a.tag+'.npy'),pred)
metrics={'best_iteration':model.get_best_iteration(),'rmse':float(np.sqrt(np.mean((va[key]-pred)**2))),'elapsed_seconds':time.time()-t,'train_rows':len(tr[key]),'valid_rows':len(va[key]),'device':a.device}
(OUT/(a.channel+'_'+a.tag+'_metrics.json')).write_text(json.dumps(metrics,indent=2));print(metrics,flush=True)'''
))
for name, text in MODULES.items():
    (SOURCE / name).write_text(text)
print('modules written:', sorted(MODULES))
sys.path[:0] = [str(SOURCE)]
os.environ['TFB_OUTPUT_ROOT'] = str(OUTPUT)


import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy import sparse
from scipy.linalg import solve
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import hashlib
import importlib.metadata
versions = {name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "lightgbm", "catboost", "pyarrow")}
print(versions)

manifest = json.loads((ASSETS / "manifest.json").read_text())
assert manifest["solution"] == "v20" and manifest["public_score"] == 0.86517
for filename, expected in manifest["files"].items():
    assert hashlib.sha256((ASSETS / filename).read_bytes()).hexdigest() == expected, filename
print("All asset checksums passed.")
display(pd.DataFrame({"this_run": versions, "original_run": manifest["original_environment"]}))
import state_model as base
# ================= STATE: v1 CatBoost ensemble, then physics post-processing (v2) =================
from state_inference import StatePredictor, predict_states
state_predictor = StatePredictor(MODEL_DIR)
predict_states(state_predictor)
del state_predictor; gc.collect()
(OUTPUT / 'state_v1.csv').write_text((OUTPUT / 'state.csv').read_text())
print('state v1 written', flush=True)

FD_ALPHA = 0.15
CONS_ALPHA = 0.40
CONS_PASSES = 5

def fd_params(name):
    fd = pd.read_csv(DATA / 'corridors' / name / 'network' / 'fd_parameters.csv', dtype={'link_id': str})
    have = [c for c in ('free_speed_kmh', 'capacity_vph', 'lanes', 'k_jam') if c in fd.columns]
    m = fd.groupby('link_id', as_index=False)[have].mean()
    m['lanes'] = m.lanes.fillna(1.0).clip(lower=1.0)
    m['critical_density'] = m.capacity_vph / m.free_speed_kmh.clip(lower=1.0)
    m['k_jam'] = m.k_jam.fillna(100.0 * m.lanes) if 'k_jam' in m.columns else 100.0 * m.lanes
    m['k_jam'] = np.maximum(m.k_jam, m.critical_density * 1.05)
    m['w'] = m.capacity_vph / (m.k_jam - m.critical_density)
    return m.set_index('link_id')

def physics_pass(name, split):
    panel = base.Panel(name, list(base.all_panels()).index(name))
    frame = base.month_frame(name, '', split=split)
    times, v, q, e = base.matrices(frame, panel)
    state = pd.read_csv(OUTPUT / 'state_v1.csv', dtype={'station_id': str, 'link_id': str},
                        usecols=['panel', 'timestamp', 'link_id', 'speed_kmh', 'flow_vph'])
    state = state[state.panel == name]
    state = state[state.timestamp.isin(times)]
    li = {l: i for i, l in enumerate(panel.links)}
    rows = times.get_indexer(state.timestamp)
    cols = state.link_id.map(li).to_numpy(dtype=int)
    v[rows, cols] = state.speed_kmh.to_numpy(float)
    q[rows, cols] = state.flow_vph.to_numpy(float)
    target = pd.read_csv(DATA / 'task1' / name / split / 'sample_submission_state.csv',
                         dtype={'station_id': str, 'link_id': str})
    trows = times.get_indexer(target.timestamp)
    tcols = target.link_id.map(li).to_numpy(dtype=int)
    fd = fd_params(name)
    vf = panel.vf.astype(float); cap = panel.cap.astype(float)
    k_crit = (cap / np.maximum(vf, 1.0)).astype(float)
    kj = fd.reindex(panel.links).k_jam.to_numpy(float)
    w = fd.reindex(panel.links).w.to_numpy(float)
    def soft_fd(V, Q):
        out = Q.copy()
        valid = np.isfinite(V) & np.isfinite(Q) & (V > 1.0)
        k = np.where(valid, Q / np.maximum(V, 1.0), 0.0)
        kc = k_crit[None, :]
        q_fd = np.where(k <= kc, vf[None, :] * k, w[None, :] * np.maximum(kj[None, :] - k, 0.0))
        reasonable = valid & (q_fd > 0) & (q_fd < cap[None, :] * 1.3)
        out[reasonable] = (1.0 - FD_ALPHA) * Q[reasonable] + FD_ALPHA * q_fd[reasonable]
        return out
    def conservation(Q):
        out = Q.copy()
        order = np.argsort(panel.milepost)
        pos = {int(i): r for r, i in enumerate(order)}
        links_sorted = [panel.links[int(i)] for i in order]
        idx_of = {l: i for i, l in enumerate(panel.links)}
        rank = np.array([pos[i] for i in range(len(panel.links))])
        for _pass in range(CONS_PASSES):
            for t in range(len(times)):
                fl = out[t]
                for r in range(len(order) - 1):
                    up = int(order[r]); dn = int(order[r + 1])
                    if rank[dn] != rank[up] + 1:
                        continue
                    q_up = fl[up]; q_dn = fl[dn]
                    if not (np.isfinite(q_up) and np.isfinite(q_dn)):
                        continue
                    residual = q_dn - q_up
                    correction = CONS_ALPHA * residual * 0.5
                    fl[up] = q_up + correction
                    fl[dn] = q_dn - correction
        return out
    q2 = soft_fd(v, q)
    q2 = conservation(q2)
    target['speed_kmh'] = v[trows, tcols]
    target['flow_vph'] = np.clip(q2[trows, tcols], 0, 150000)
    assert np.isfinite(target[['speed_kmh', 'flow_vph']].to_numpy()).all()
    return target[['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']]

first = True
for split in ('validation', 'private'):
    for name in base.all_panels():
        part = physics_pass(name, split)
        part.to_csv(OUTPUT / 'state.csv', index=False, mode='w' if first else 'a', header=first)
        first = False
        print('physics', name, split, len(part), flush=True)
print('state final written', flush=True)

# ================= QUEUE: v1 probabilities + newly trained LGBM members (v2) =================
import queue_features as qf
import queue_ongoing as qo
from queue_features import Panel, panel_names, HISTORY, HORIZON, candidate_origins
from queue_ongoing import Panel as OngoingPanel

onset_model_v1 = lgb.Booster(model_file=str(MODEL_DIR / 'onset_lgb.txt'))
ongoing_model_v1 = lgb.Booster(model_file=str(MODEL_DIR / 'ongoing_lgb.txt'))
onset_cat_v1 = CatBoostClassifier()
onset_cat_v1.load_model(str(MODEL_DIR / 'onset.cbm'))
assert onset_model_v1.num_feature() == 83
assert ongoing_model_v1.num_feature() == 36
ONSET_THR = 0.55
ONGOING_THR = 0.40
SEEDS_ONSET = (20270101, 20270102, 20270103)
ONGOING_VARIANTS = ((20270104, 31, 0.045), (20270105, 31, 0.045), (20270106, 63, 0.03))

windows_info = []
for split in ('validation', 'private'):
    for panel_no, name in enumerate(panel_names()):
        panel = Panel(name, panel_no)
        ongoing_panel = OngoingPanel(name, panel_no)
        folder = DATA / 'task2' / name / split
        windows = pd.read_csv(folder / 'window_index.csv')
        history = pd.read_parquet(folder / 'window_history.parquet')
        for window in windows.itertuples(index=False):
            origin = pd.Timestamp(window.forecast_origin)
            times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY,
                                  freq='5min', tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
            h = history[history.window_id == window.window_id].copy()
            h['link_id'] = h.link_id.astype(str)
            assert pd.to_datetime(h.timestamp, utc=True).max() < origin
            def matrix(column):
                return h.pivot(index='timestamp', columns='link_id', values=column).reindex(
                    index=times, columns=panel.links).to_numpy(dtype=np.float32)
            speed, flow, eligible = [matrix(c) for c in ('speed_kmh', 'flow_vph', 'is_score_eligible')]
            info = {'split': split, 'panel': name, 'panel_no': panel_no, 'links': panel.links,
                    'window_id': window.window_id, 'condition': window.condition, 'origin': origin}
            if window.condition == 'queue_onset':
                info['X'] = panel.features(speed, flow, eligible, origin, window.condition)[-len(panel.links):]
            else:
                observed = np.nan_to_num(eligible, nan=0).astype(bool)
                info['X'] = ongoing_panel.features(speed, flow, observed, origin, window.condition)
            windows_info.append(info)
print('windows collected:', len(windows_info), flush=True)

def onset_prob_v1(X):
    w = 0.5
    return (1 - w) * onset_model_v1.predict(X, num_threads=THREADS) + w * onset_cat_v1.predict_proba(X, thread_count=THREADS)[:, 1]

def ongoing_prob_v1(X):
    return ongoing_model_v1.predict(X, num_threads=THREADS)

for info in windows_info:
    info['p_v1'] = onset_prob_v1(info['X']) if info['condition'] == 'queue_onset' else ongoing_prob_v1(info['X'])

# ---- train onset members (two-stage protocol, 3 seeds) ----
onset_models = []
for seed in SEEDS_ONSET:
    Xtrain, ytrain, valid = [], [], []
    for pi, name in enumerate(panel_names()):
        p = Panel(name, pi)
        for file in sorted((DATA / 'corridors' / name / 'train' / 'mainline_states').glob('**/*.parquet')):
            times, v, q, e = p.train_day(file)
            cand = candidate_origins(p, times, v, e)['queue_onset']
            cs = set(cand)
            for t in [a for a in cand if a - 1 not in cs]:
                x = p.features(v[t-HISTORY:t], q[t-HISTORY:t], e[t-HISTORY:t], pd.Timestamp(times[t]), 'queue_onset')[-len(p.links):]
                y = (v[t+HORIZON] <= p.cut) & np.isfinite(v[t+HORIZON])
                elig = e[t+HORIZON] & np.isfinite(v[t+HORIZON])
                if file.parent.name.endswith('2031-02'):
                    valid.append((x, y, elig))
                else:
                    Xtrain.append(x[elig]); ytrain.append(y[elig])
    X = np.concatenate(Xtrain); y = np.concatenate(ytrain)
    Xv = np.concatenate([a[0][a[2]] for a in valid]); yv = np.concatenate([a[1][a[2]] for a in valid])
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': .04, 'num_leaves': 15,
              'min_data_in_leaf': 100, 'lambda_l2': 10, 'feature_fraction': .9, 'num_threads': THREADS,
              'verbosity': -1, 'seed': seed}
    m = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=900,
                  valid_sets=[lgb.Dataset(Xv, label=yv)], callbacks=[lgb.early_stopping(70), lgb.log_evaluation(500)])
    m = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=1500, init_model=m,
                  valid_sets=[lgb.Dataset(Xv, label=yv)], callbacks=[lgb.early_stopping(70), lgb.log_evaluation(500)])
    onset_models.append(m)
    del X, y, Xv, yv, Xtrain, ytrain, valid
    gc.collect()
    print('onset member trained', seed, flush=True)

# ---- train ongoing members (3 variants) ----
ongoing_models = []
X, y, weights, valid_windows, feature_count = qo.collect()
for seed, leaves, lr in ONGOING_VARIANTS:
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': lr, 'num_leaves': leaves,
              'max_depth': 8, 'min_data_in_leaf': 250, 'feature_fraction': .85, 'bagging_fraction': .85,
              'bagging_freq': 1, 'lambda_l2': 8.0, 'max_bin': 127, 'num_threads': THREADS, 'verbosity': -1,
              'seed': seed}
    X_val = np.vstack([w.X[w.eligible] for w in valid_windows])
    y_val = np.concatenate([w.y[w.eligible] for w in valid_windows]).astype(np.uint8)
    w_val = np.concatenate([np.full(w.eligible.sum(), 1000 / w.eligible.sum(), dtype=np.float32) for w in valid_windows])
    m = lgb.train(params, lgb.Dataset(X, label=y, weight=weights, free_raw_data=True), num_boost_round=700,
                  valid_sets=[lgb.Dataset(X_val, label=y_val, weight=w_val)],
                  callbacks=[lgb.early_stopping(60), lgb.log_evaluation(500)])
    ongoing_models.append(m)
    del X_val, y_val, w_val
    gc.collect()
    print('ongoing member trained', seed, leaves, lr, flush=True)
del X, y, weights, valid_windows
gc.collect()

# ---- blend probabilities and build final queue predictions ----
queue_pieces = []
for info in windows_info:
    if info['condition'] == 'queue_onset':
        probs = [info['p_v1']] + [m.predict(info['X'], num_threads=THREADS) for m in onset_models]
        p = np.mean(probs, axis=0)
        selected = p >= ONSET_THR
        if not selected.any():
            selected[np.argmax(p)] = True
        pred = np.zeros((HORIZON, info['X'].shape[0]), dtype=np.uint8)
        pred[-1] = selected.astype(np.uint8)
    else:
        probs = [info['p_v1']] + [m.predict(info['X'], num_threads=THREADS) for m in ongoing_models]
        p = np.mean(probs, axis=0)
        pred = (p >= ONGOING_THR).reshape(HORIZON, -1).astype(np.uint8)
    n_links = pred.shape[1]
    stamps = [(info['origin'] + pd.Timedelta(minutes=5 * step)).strftime('%Y-%m-%dT%H:%M:%SZ') for step in range(1, HORIZON + 1)]
    queue_pieces.append(pd.DataFrame({'window_id': info['window_id'], 'split': info['split'], 'panel': info['panel'],
                                      'timestamp': np.repeat(stamps, n_links),
                                      'link_id': np.tile(info['links'], HORIZON),
                                      'queue_pred': pred.ravel()}))

queue_rows = []
for split in ('validation', 'private'):
    for panel_no, name in enumerate(panel_names()):
        folder = DATA / 'task2' / name / split
        template = pd.read_csv(folder / 'sample_submission_queue.csv', dtype={'link_id': str})
        pieces = [q for q in queue_pieces if q['split'] == split and q['panel'] == name]
        pred = pd.concat(pieces, ignore_index=True)
        keys = ['window_id', 'timestamp', 'link_id']
        result = template[keys].merge(pred[keys + ['queue_pred']], on=keys, how='left', validate='one_to_one')
        assert result.queue_pred.notna().all() and len(result) == len(template)
        queue_rows.append(result)
queue = pd.concat(queue_rows, ignore_index=True)
queue.to_csv(OUTPUT / 'queue.csv', index=False)
print(f"Queue rows: {len(queue):,}; positive cells: {int(queue.queue_pred.sum()):,}", flush=True)
del queue_pieces, queue_rows, windows_info
gc.collect()

REBUILD_WARM_START = False
LAMBDA = 0.05
initial = pd.read_parquet(ASSETS / "odme_initial.parquet")
all_panels = [x["corridor_id"] for x in json.loads((DATA / "config/corridors.json").read_text())["panels"]]

def solve_odme(name, split):
    network = DATA / "corridors" / name / "network"
    folder = DATA / "task4" / name / split
    paths = pd.read_csv(network / "path_set.csv", dtype=str)
    incidence = pd.read_csv(network / "path_link_incidence.csv", dtype=str).drop_duplicates(["link_id", "path_id"])
    prior = pd.read_csv(folder / "synthetic_weak_prior.csv", dtype={"path_id": str, "departure_time": str})
    counts = pd.read_csv(folder / "synthetic_link_counts.csv", dtype={"link_id": str})
    ids = paths.path_id.tolist()
    path_index = {p: i for i, p in enumerate(ids)}
    supported = set(incidence.link_id)
    links = [link for link in counts.link_id if link in supported]
    link_index = {link: i for i, link in enumerate(links)}
    measured = incidence[incidence.link_id.isin(link_index)]
    A = sparse.csr_matrix((np.ones(len(measured)),
                          (measured.link_id.map(link_index), measured.path_id.map(path_index))),
                         shape=(len(links), len(ids)))
    c = counts.set_index("link_id").reindex(links)["count"].to_numpy(float)
    b = prior.set_index("path_id").reindex(ids).path_flow.to_numpy(float)
    assert np.isfinite(c).all() and np.isfinite(b).all()
    def objective(f):
        residual, change = A @ f - c, f - b
        return (float(0.5*(residual@residual + LAMBDA*(change@change))),
                np.asarray(A.T@residual + LAMBDA*change))
    def projected_gradient(f):
        gradient = objective(f)[1]
        return float(np.max(np.abs(np.where(f > 1e-7, gradient, np.minimum(gradient, 0)))))
    if REBUILD_WARM_START:
        gram = (A @ A.T).toarray()
        u = b + A.T @ solve(gram + LAMBDA*np.eye(len(c)), c-A@b, assume_a="pos")
        f0 = np.maximum(u, 0)
        if np.min(u) < -1e-7:
            first = minimize(objective, f0, jac=True, method="L-BFGS-B", bounds=[(0, None)]*len(ids),
                             options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-5})
            f0 = np.maximum(first.x, 0)
    else:
        departure = str(prior.departure_time.iloc[0])
        f0 = initial[(initial.panel == name) & (initial.departure_time == departure)].set_index(
            "path_id").reindex(ids).path_flow.to_numpy(float)
    assert np.isfinite(f0).all()
    fit = minimize(objective, f0, jac=True, method="L-BFGS-B", bounds=[(0, None)]*len(ids),
                   options={"maxiter": 8000, "ftol": 1e-14, "gtol": 1e-6, "maxcor": 30})
    f = np.maximum(fit.x, 0)
    before, after = objective(f0)[0], objective(f)[0]
    assert np.isfinite(f).all() and after <= before*(1+1e-8)
    diagnostics = {"panel": name, "split": split, "success": bool(fit.success),
                   "iterations": fit.nit, "objective_before": before, "objective_after": after,
                   "projected_gradient_before": projected_gradient(f0),
                   "projected_gradient_after": projected_gradient(f),
                   "count_fit": 1-np.abs(A@f-c).sum()/max(c.sum(), 1e-9)}
    template = pd.read_csv(folder / "sample_submission_path_flow.csv", dtype={"path_id": str})
    template["path_flow"] = template.path_id.map(dict(zip(ids, f)))
    assert template.path_flow.notna().all()
    return template, diagnostics


if RUN_INFERENCE:
    od_parts, diagnostics = [], []
    with threadpool_limits(limits=1, user_api="blas"):
        for split in ("validation", "private"):
            for name in all_panels:
                result, detail = solve_odme(name, split)
                od_parts.append(result)
                diagnostics.append(detail)
                print(name, split, "iterations:", detail["iterations"], "success:", detail["success"])
    odme = pd.concat(od_parts, ignore_index=True)
    odme.to_csv(OUTPUT / "odme.csv", index=False)
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(OUTPUT / "odme_diagnostics.csv", index=False)
    display(diagnostics.groupby("split")[["objective_before", "objective_after",
        "projected_gradient_before", "projected_gradient_after", "count_fit"]].mean())
else:
    print("Inference disabled. Set RUN_INFERENCE=True after training or providing the checkpoints.")
if RUN_INFERENCE:

    KEYS = {"state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
            "queue": ["window_id", "timestamp", "link_id"],
            "odme": ["panel", "departure_time", "path_id"]}
    VALUES = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}
    COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
    
    def indexed_predictions(task):
        frame = pd.read_csv(OUTPUT / f"{task}.csv", dtype={k: str for k in KEYS[task]})
        assert not frame.duplicated(KEYS[task]).any(), task
        assert np.isfinite(frame[VALUES[task]].to_numpy()).all(), task
        return frame.set_index(KEYS[task])[VALUES[task]]
    
    tables = {task: indexed_predictions(task) for task in KEYS}
    assert (tables["state"].speed_kmh > 0).all() and (tables["state"].flow_vph >= 0).all()
    assert tables["queue"].queue_pred.isin([0, 1]).all()
    assert (tables["odme"].path_flow >= 0).all()
    counts = {task: 0 for task in KEYS}
    row_count = 0
    submission = WORK / "submission.csv"
    for key in pd.read_csv(DATA / "submission_key.csv", dtype=str, chunksize=500_000):
        output = key[["submission_id", "task"]].copy()
        for column in COLUMNS[2:]:
            output[column] = 0.0
        assert output.task.isin(KEYS).all()
        for task in KEYS:
            mask = key.task == task
            if not mask.any():
                continue
            wanted = pd.MultiIndex.from_frame(key.loc[mask, KEYS[task]])
            matched = tables[task].reindex(wanted)
            assert matched.notna().all().all(), f"Missing {task} predictions"
            output.loc[mask, VALUES[task]] = matched.to_numpy()
            counts[task] += int(mask.sum())
        assert np.isfinite(output[COLUMNS[2:]].to_numpy()).all()
        output[COLUMNS].to_csv(submission, mode="w" if row_count == 0 else "a",
                               header=row_count == 0, index=False)
        row_count += len(output)
    assert counts == {"state": 6_735_795, "queue": 174_000, "odme": 70_708}
    assert row_count == 6_980_503
    with submission.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"rows": row_count, "task_rows": counts, "sha256": digest,
              "recorded_solution_score": 0.86517, "versions": versions,
              "rebuilt_warm_start": REBUILD_WARM_START}
    (WORK / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Saved:", submission)
    display(pd.read_csv(submission, nrows=5))
    del tables
    gc.collect();
else:
    print("Inference disabled. Set RUN_INFERENCE=True after training or providing the checkpoints.")
