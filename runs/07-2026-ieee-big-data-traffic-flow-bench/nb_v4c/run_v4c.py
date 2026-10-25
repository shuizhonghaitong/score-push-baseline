# tfb-run-v4c: v4b (t1extra + public queue @0.45, physics off) with REDUCED t1 (U-Net 24 epochs, LGBM 1200 rounds) to fit Kaggle 12h CPU limit.
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
RUN_TRAINING = False
RUN_INFERENCE = True
THREADS = min(4, os.cpu_count() or 1)

try:
    from IPython.display import display
except Exception:
    def display(x=None):
        print(x if x is not None else '')

MODULES = {
'base_panel.py': '\n"""Train a causal queue classifier on observed train months and predict Task 2."""\n\nfrom __future__ import annotations\n\nimport hashlib\nimport json\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\nimport lightgbm as lgb\nimport numpy as np\nimport pandas as pd\nfrom scipy import sparse\n\nfrom config import RELEASE, OUT, THREADS\nMONTHS = {"2030-06", "2030-11", "2031-01", "2031-02"}\nHISTORY = 12\nHORIZON = 6\nSEED = 20260930\n\n\ndef panel_names():\n    return [entry["corridor_id"] for entry in json.loads((RELEASE / "config" / "corridors.json").read_text())["panels"]\n            if entry["corridor_id"] not in {"D12_I405_N", "D12_I405_S"}]\n\n\ndef avg(values):\n    valid = np.isfinite(values)\n    count = valid.sum(axis=0)\n    result = np.nansum(values, axis=0) / np.maximum(count, 1)\n    return np.where(count > 0, result, np.nan)\n\n\ndef min_finite(values):\n    result = np.where(np.isfinite(values), values, np.inf).min(axis=0)\n    return np.where(np.isfinite(result), result, np.nan)\n\n\n@dataclass\nclass Window:\n    X: np.ndarray\n    y: np.ndarray\n    eligible: np.ndarray\n    panel: str\n    condition: str\n\n\nclass Panel:\n    def __init__(self, name, panel_no):\n        self.name = name\n        self.panel_no = panel_no\n        directory = RELEASE / "corridors" / name / "network"\n        fd = pd.read_csv(directory / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")\n        topo = pd.read_csv(directory / "lwr_mainline_topology.csv", dtype=str)\n        self.links = fd.link_id.astype(str).tolist()\n        self.link_index = {key: i for i, key in enumerate(self.links)}\n        self.lanes = fd.lanes.to_numpy(dtype=np.float32)\n        self.vf = fd.free_speed_kmh.to_numpy(dtype=np.float32)\n        self.cap = fd.capacity_vph.to_numpy(dtype=np.float32)\n        self.length = fd.length_km.to_numpy(dtype=np.float32)\n        self.cut = (fd.v_cut if "v_cut" in fd else 0.6 * fd.free_speed_kmh).to_numpy(dtype=np.float32)\n        self.incoming = self.adjacency(topo, "incoming_link_ids")\n        self.outgoing = self.adjacency(topo, "outgoing_link_ids")\n\n    def adjacency(self, topo, column):\n        rows, cols = [], []\n        for row in topo.itertuples(index=False):\n            link = str(row.link_id)\n            if link not in self.link_index:\n                continue\n            for other in str(getattr(row, column)).split(";"):\n                if other in self.link_index:\n                    rows.append(self.link_index[link])\n                    cols.append(self.link_index[other])\n        result = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),\n                                   shape=(len(self.links), len(self.links)))\n        result.data[:] = 1.0\n        return result\n\n    def neighboring(self, graph, values):\n        valid = np.isfinite(values).astype(np.float32)\n        count = graph @ valid\n        mean = graph @ np.nan_to_num(values, nan=0.0)\n        return np.where(count > 0, mean / np.maximum(count, 1), np.nan)',
'config.py': '\nfrom pathlib import Path\nimport os\nRELEASE = Path(os.environ["TFB_DATA_ROOT"])\nOUT = Path(os.environ.get("TFB_OUTPUT_ROOT", "/kaggle/working"))\nOUT.mkdir(parents=True, exist_ok=True)\nTHREADS = int(os.environ.get("TFB_THREADS", "4"))',
'prepare_state.py': '\n"""State features with lag/lead context and two independently synthesized masks."""\nfrom pathlib import Path\nimport sys,json\nimport numpy as np\nimport pandas as pd\nfrom config import OUT, THREADS\nimport state_model as base\n\ndef features(panel,pi,times,v,q,e,rows,cols,regime):\n    X,bv,bq=base.compute_features(panel,pi,times,v,q,rows,cols,regime)\n    extra=[];lanes=np.maximum(panel.lanes[cols],1)\n    for offset in (-12,-6,-3,-2,-1,1,2,3,6,12):\n        rr=rows+offset;valid=(rr>=0)&(rr<len(times));rr=np.clip(rr,0,len(times)-1)\n        vv=v[rr,cols];qq=q[rr,cols]/lanes\n        extra.extend([np.where(valid,vv,np.nan),np.where(valid,qq,np.nan),\n                      np.where(valid,qq/np.maximum(vv,1),np.nan),\n                      np.where(valid&e[rr,cols]&np.isfinite(vv),1,0)])\n    return np.column_stack([X,*extra]).astype(np.float32),bv,bq\n\ndef main():\n    xx=[];yv=[];yq=[];val={k:[] for k in (\'X\',\'yv\',\'yq\',\'bv\',\'bq\',\'panel\',\'regime\')};audit=[]\n    original_seed=base.SEED\n    for pi,name in enumerate(base.all_panels()):\n        p=base.Panel(name,pi)\n        for month in (*base.TRAIN_MONTHS,base.VALID_MONTH):\n            times,v,q,e=base.matrices(base.month_frame(name,month),p)\n            isval=month==base.VALID_MONTH\n            audit.append({\'panel\':name,\'month\':month,\'cells\':v.size,\'speed_missing\':float(np.isnan(v).mean()),\'eligible\':float(e.mean()),\'finite_ineligible\':float((np.isfinite(v)&~e).mean()),\'negative_flow\':int((q<0).sum())})\n            for rep in range(1 if isval else 2):\n                base.SEED=original_seed+rep*7919\n                mask,regime=base.simulated_mask(name,times,v,q,e)\n                rows,cols=base.sampled_targets(mask,times,1500 if isval else 750)\n                X,bv,bq=features(p,pi,times,np.where(mask,np.nan,v),np.where(mask,np.nan,q),e,rows,cols,regime)\n                rv=v[rows,cols]-bv;rq=q[rows,cols]/p.lanes[cols]-bq\n                if isval:\n                    for k,a in [(\'X\',X),(\'yv\',rv),(\'yq\',rq),(\'bv\',bv),(\'bq\',bq),(\'panel\',np.full(len(rows),pi)),(\'regime\',regime[rows])]:val[k].append(a)\n                else:xx.append(X);yv.append(rv);yq.append(rq)\n            print(name,month,\'done\',flush=True)\n    base.SEED=original_seed\n    np.savez(OUT/\'train.npz\',X=np.concatenate(xx),yv=np.concatenate(yv),yq=np.concatenate(yq))\n    np.savez(OUT/\'valid.npz\',**{k:np.concatenate(a) for k,a in val.items()})\n    pd.DataFrame(audit).to_csv(OUT/\'data_audit.csv\',index=False)\n    print(\'train\',sum(len(a) for a in yv),\'valid\',sum(len(a) for a in val[\'yv\']),\'features\',xx[0].shape[1],flush=True)\nif __name__==\'__main__\':main()',
'queue_features.py': '\n"""Train a causal queue classifier on observed train months and predict Task 2."""\n\nfrom __future__ import annotations\n\nimport hashlib\nimport json\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\nimport lightgbm as lgb\nimport numpy as np\nimport pandas as pd\nfrom scipy import sparse\n\nfrom config import RELEASE, OUT, THREADS\nMONTHS = {"2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02"}\nHISTORY = 12\nHORIZON = 6\nSEED = 20260930\n\n\ndef panel_names():\n    return [entry["corridor_id"] for entry in json.loads((RELEASE / "config" / "corridors.json").read_text())["panels"]\n            if entry["corridor_id"] not in {"D12_I405_N", "D12_I405_S"}]\n\n\ndef avg(values):\n    valid = np.isfinite(values)\n    count = valid.sum(axis=0)\n    result = np.nansum(values, axis=0) / np.maximum(count, 1)\n    return np.where(count > 0, result, np.nan)\n\n\ndef min_finite(values):\n    result = np.where(np.isfinite(values), values, np.inf).min(axis=0)\n    return np.where(np.isfinite(result), result, np.nan)\n\n\n@dataclass\nclass Window:\n    X: np.ndarray\n    y: np.ndarray\n    eligible: np.ndarray\n    panel: str\n    condition: str\n\n\nclass Panel:\n    def __init__(self, name, panel_no):\n        self.name = name\n        self.panel_no = panel_no\n        directory = RELEASE / "corridors" / name / "network"\n        fd = pd.read_csv(directory / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")\n        topo = pd.read_csv(directory / "lwr_mainline_topology.csv", dtype=str)\n        self.links = fd.link_id.astype(str).tolist()\n        self.link_index = {key: i for i, key in enumerate(self.links)}\n        sample = sorted((RELEASE / "corridors" / name / "train" / "mainline_states" /\n                         "year_month=2030-06").glob("*.parquet"))[0]\n        location = pd.read_parquet(sample, columns=["link_id", "milepost"]).drop_duplicates("link_id")\n        location["link_id"] = location.link_id.astype(str)\n        milepost = location.set_index("link_id").reindex(self.links).milepost.to_numpy(dtype=np.float32)\n        if not np.isfinite(milepost).all():\n            raise ValueError(f"Missing mileposts for {name}")\n        self.milepost = milepost\n        self.sorted_links = np.argsort(milepost)\n        self.link_rank = np.empty(len(self.links), dtype=np.int32)\n        self.link_rank[self.sorted_links] = np.arange(len(self.links))\n        self.lanes = fd.lanes.to_numpy(dtype=np.float32)\n        self.vf = fd.free_speed_kmh.to_numpy(dtype=np.float32)\n        self.cap = fd.capacity_vph.to_numpy(dtype=np.float32)\n        self.length = fd.length_km.to_numpy(dtype=np.float32)\n        self.cut = (fd.v_cut if "v_cut" in fd else 0.6 * fd.free_speed_kmh).to_numpy(dtype=np.float32)\n        self.incoming = self.adjacency(topo, "incoming_link_ids")\n        self.outgoing = self.adjacency(topo, "outgoing_link_ids")\n\n    def adjacency(self, topo, column):\n        rows, cols = [], []\n        for row in topo.itertuples(index=False):\n            link = str(row.link_id)\n            if link not in self.link_index:\n                continue\n            for other in str(getattr(row, column)).split(";"):\n                if other in self.link_index:\n                    rows.append(self.link_index[link])\n                    cols.append(self.link_index[other])\n        result = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),\n                                   shape=(len(self.links), len(self.links)))\n        result.data[:] = 1.0\n        return result\n\n    def neighboring(self, graph, values):\n        valid = np.isfinite(values).astype(np.float32)\n        count = graph @ valid\n        mean = graph @ np.nan_to_num(values, nan=0.0)\n        return np.where(count > 0, mean / np.maximum(count, 1), np.nan)\n\n    def features(self, speed, flow, observed, origin, condition):\n        """History arrays have exactly twelve rows ending five minutes before T."""\n        if speed.shape != (HISTORY, len(self.links)) or flow.shape != speed.shape:\n            raise ValueError((self.name, speed.shape, flow.shape))\n        speed = speed.astype(np.float32) / self.vf[None, :]\n        flow = flow.astype(np.float32) / self.cap[None, :]\n        speed = np.where(np.isfinite(speed), speed, np.nan)\n        flow = np.where(np.isfinite(flow), flow, np.nan)\n        valid = np.isfinite(speed)\n        positions = np.where(valid, np.arange(HISTORY)[:, None], -1).max(axis=0)\n        columns = np.arange(len(self.links))\n        last_speed = np.where(positions >= 0, speed[np.maximum(positions, 0), columns], np.nan)\n        last_flow = np.where(positions >= 0, flow[np.maximum(positions, 0), columns], np.nan)\n        age = HISTORY - 1 - positions\n        age = np.where(positions >= 0, age, 12)\n        recent_speed = avg(speed[-3:])\n        prior_speed = avg(speed[-6:-3])\n        recent_flow = avg(flow[-3:])\n        prior_flow = avg(flow[-6:-3])\n        mean_speed = avg(speed)\n        mean_flow = avg(flow)\n        queue_history = np.where(valid, speed <= (self.cut / self.vf)[None, :], False)\n        last_queue = last_speed <= self.cut / self.vf\n        hour = origin.hour + origin.minute / 60\n        tod_sin = np.sin(2 * np.pi * hour / 24)\n        tod_cos = np.cos(2 * np.pi * hour / 24)\n        dow = origin.dayofweek\n        constants = lambda value: np.full(len(self.links), value, dtype=np.float32)\n        features = [\n            constants(self.panel_no), np.arange(len(self.links), dtype=np.float32),\n            self.vf, self.cap / np.maximum(self.lanes, 1), self.lanes, self.length,\n            constants(hour), constants(tod_sin), constants(tod_cos), constants(dow),\n            constants(1 if condition == "queue_ongoing" else 0),\n            last_speed, last_flow, constants(0) + age, recent_speed, prior_speed,\n            recent_speed - prior_speed, recent_flow, prior_flow, recent_flow - prior_flow,\n            mean_speed, min_finite(speed), mean_flow, np.nanmax(np.where(np.isfinite(flow), flow, -np.inf), axis=0),\n            valid.mean(axis=0), queue_history.sum(axis=0).astype(np.float32), last_queue.astype(np.float32),\n            self.neighboring(self.incoming, last_speed), self.neighboring(self.outgoing, last_speed),\n            self.neighboring(self.incoming, last_flow), self.neighboring(self.outgoing, last_flow),\n            self.neighboring(self.incoming, last_queue.astype(np.float32)),\n            self.neighboring(self.outgoing, last_queue.astype(np.float32)),\n            constants(np.isfinite(speed).mean()), constants(np.nanmin(speed) if np.isfinite(speed).any() else 1.0),\n        ]\n        features[25] = np.where(np.isfinite(features[25]), features[25], np.nan)\n        features[24] = np.where(np.isfinite(features[24]), features[24], np.nan)\n        # Short lags preserve the shape of an approaching shock wave. A mean over\n        # the whole history loses the timing of a rapid pre-onset slowdown.\n        for lag in (1, 2, 3, 6, 9, 12):\n            features.extend((speed[-lag], flow[-lag]))\n        features.extend((\n            min_finite(speed[-3:]), min_finite(speed[-6:]),\n            avg(speed[-3:]) - avg(speed[-6:-3]),\n            avg(flow[-3:]) - avg(flow[-6:-3]),\n            self.neighboring(self.incoming, avg(speed[-3:])),\n            self.neighboring(self.outgoing, avg(speed[-3:])),\n            self.neighboring(self.incoming, avg(flow[-3:])),\n            self.neighboring(self.outgoing, avg(flow[-3:])),\n            self.neighboring(self.incoming, queue_history[-1].astype(np.float32)),\n            self.neighboring(self.outgoing, queue_history[-1].astype(np.float32)),\n        ))\n        def milepost_neighbor(values, offset):\n            ranks = self.link_rank + offset\n            valid_neighbor = (ranks >= 0) & (ranks < len(self.links))\n            links = self.sorted_links[np.clip(ranks, 0, len(self.links)-1)]\n            return np.where(valid_neighbor, values[links], np.nan)\n\n        features.append(self.milepost)\n        for offset in (-2, -1, 1, 2):\n            neighbor_milepost = milepost_neighbor(self.milepost, offset)\n            features.extend((\n                np.abs(neighbor_milepost-self.milepost),\n                milepost_neighbor(last_speed, offset),\n                milepost_neighbor(recent_speed, offset),\n                milepost_neighbor(last_flow, offset),\n                milepost_neighbor(recent_flow, offset),\n                milepost_neighbor(last_queue.astype(np.float32), offset),\n            ))\n        base = np.column_stack(features).astype(np.float32)\n        X = np.tile(base, (HORIZON, 1))\n        X = np.column_stack([X, np.repeat(np.arange(1, HORIZON + 1), len(self.links))]).astype(np.float32)\n        X[~np.isfinite(X)] = np.nan\n        return X\n\n    def train_day(self, path):\n        columns = ["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]\n        frame = pd.read_parquet(path, columns=columns)\n        frame["link_id"] = frame.link_id.astype(str)\n        if frame.duplicated(["timestamp", "link_id"]).any():\n            frame = frame.groupby(["timestamp", "link_id"], as_index=False).mean(numeric_only=True)\n        times = sorted(frame.timestamp.unique())\n        index = pd.Index(times)\n        speed = frame.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=self.links).to_numpy(dtype=np.float32)\n        flow = frame.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=self.links).to_numpy(dtype=np.float32)\n        eligible = frame.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=self.links).fillna(0).to_numpy(dtype=bool)\n        return times, speed, flow, eligible\n\n\ndef candidate_origins(panel, times, speed, eligible):\n    queued = (speed <= panel.cut[None, :]) & eligible & np.isfinite(speed)\n    active = queued.any(axis=1)\n    candidates = {"queue_onset": [], "queue_ongoing": []}\n    for t in range(HISTORY, len(times) - HORIZON):\n        if np.isfinite(speed[t-HISTORY:t]).mean() < .7:\n            continue\n        if not active[t+1:t+HORIZON+1].any():\n            continue\n        if not active[t-HISTORY:t].any() and not active[t]:\n            candidates["queue_onset"].append(t)\n        if active[t]:\n            candidates["queue_ongoing"].append(t)\n    return candidates',
'queue_ongoing.py': '\n"""Train a causal queue classifier on observed train months and predict Task 2."""\n\nfrom __future__ import annotations\n\nimport hashlib\nimport json\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\nimport lightgbm as lgb\nimport numpy as np\nimport pandas as pd\nfrom scipy import sparse\n\nfrom config import RELEASE, OUT, THREADS\nMONTHS = {"2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02"}\nHISTORY = 12\nHORIZON = 6\nSEED = 20260930\n\n\ndef panel_names():\n    return [entry["corridor_id"] for entry in json.loads((RELEASE / "config" / "corridors.json").read_text())["panels"]\n            if entry["corridor_id"] not in {"D12_I405_N", "D12_I405_S"}]\n\n\ndef avg(values):\n    valid = np.isfinite(values)\n    count = valid.sum(axis=0)\n    result = np.nansum(values, axis=0) / np.maximum(count, 1)\n    return np.where(count > 0, result, np.nan)\n\n\ndef min_finite(values):\n    result = np.where(np.isfinite(values), values, np.inf).min(axis=0)\n    return np.where(np.isfinite(result), result, np.nan)\n\n\n@dataclass\nclass Window:\n    X: np.ndarray\n    y: np.ndarray\n    eligible: np.ndarray\n    panel: str\n    condition: str\n\n\nclass Panel:\n    def __init__(self, name, panel_no):\n        self.name = name\n        self.panel_no = panel_no\n        directory = RELEASE / "corridors" / name / "network"\n        fd = pd.read_csv(directory / "fd_parameters.csv", dtype={"link_id": str}).drop_duplicates("link_id")\n        topo = pd.read_csv(directory / "lwr_mainline_topology.csv", dtype=str)\n        self.links = fd.link_id.astype(str).tolist()\n        self.link_index = {key: i for i, key in enumerate(self.links)}\n        self.lanes = fd.lanes.to_numpy(dtype=np.float32)\n        self.vf = fd.free_speed_kmh.to_numpy(dtype=np.float32)\n        self.cap = fd.capacity_vph.to_numpy(dtype=np.float32)\n        self.length = fd.length_km.to_numpy(dtype=np.float32)\n        self.cut = (fd.v_cut if "v_cut" in fd else 0.6 * fd.free_speed_kmh).to_numpy(dtype=np.float32)\n        self.incoming = self.adjacency(topo, "incoming_link_ids")\n        self.outgoing = self.adjacency(topo, "outgoing_link_ids")\n\n    def adjacency(self, topo, column):\n        rows, cols = [], []\n        for row in topo.itertuples(index=False):\n            link = str(row.link_id)\n            if link not in self.link_index:\n                continue\n            for other in str(getattr(row, column)).split(";"):\n                if other in self.link_index:\n                    rows.append(self.link_index[link])\n                    cols.append(self.link_index[other])\n        result = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),\n                                   shape=(len(self.links), len(self.links)))\n        result.data[:] = 1.0\n        return result\n\n    def neighboring(self, graph, values):\n        valid = np.isfinite(values).astype(np.float32)\n        count = graph @ valid\n        mean = graph @ np.nan_to_num(values, nan=0.0)\n        return np.where(count > 0, mean / np.maximum(count, 1), np.nan)\n\n    def features(self, speed, flow, observed, origin, condition):\n        """History arrays have exactly twelve rows ending five minutes before T."""\n        if speed.shape != (HISTORY, len(self.links)) or flow.shape != speed.shape:\n            raise ValueError((self.name, speed.shape, flow.shape))\n        speed = speed.astype(np.float32) / self.vf[None, :]\n        flow = flow.astype(np.float32) / self.cap[None, :]\n        speed = np.where(np.isfinite(speed), speed, np.nan)\n        flow = np.where(np.isfinite(flow), flow, np.nan)\n        valid = np.isfinite(speed)\n        positions = np.where(valid, np.arange(HISTORY)[:, None], -1).max(axis=0)\n        columns = np.arange(len(self.links))\n        last_speed = np.where(positions >= 0, speed[np.maximum(positions, 0), columns], np.nan)\n        last_flow = np.where(positions >= 0, flow[np.maximum(positions, 0), columns], np.nan)\n        age = HISTORY - 1 - positions\n        age = np.where(positions >= 0, age, 12)\n        recent_speed = avg(speed[-3:])\n        prior_speed = avg(speed[-6:-3])\n        recent_flow = avg(flow[-3:])\n        prior_flow = avg(flow[-6:-3])\n        mean_speed = avg(speed)\n        mean_flow = avg(flow)\n        queue_history = np.where(valid, speed <= (self.cut / self.vf)[None, :], False)\n        last_queue = last_speed <= self.cut / self.vf\n        hour = origin.hour + origin.minute / 60\n        tod_sin = np.sin(2 * np.pi * hour / 24)\n        tod_cos = np.cos(2 * np.pi * hour / 24)\n        dow = origin.dayofweek\n        constants = lambda value: np.full(len(self.links), value, dtype=np.float32)\n        features = [\n            constants(self.panel_no), np.arange(len(self.links), dtype=np.float32),\n            self.vf, self.cap / np.maximum(self.lanes, 1), self.lanes, self.length,\n            constants(hour), constants(tod_sin), constants(tod_cos), constants(dow),\n            constants(1 if condition == "queue_ongoing" else 0),\n            last_speed, last_flow, constants(0) + age, recent_speed, prior_speed,\n            recent_speed - prior_speed, recent_flow, prior_flow, recent_flow - prior_flow,\n            mean_speed, min_finite(speed), mean_flow, np.nanmax(np.where(np.isfinite(flow), flow, -np.inf), axis=0),\n            valid.mean(axis=0), queue_history.sum(axis=0).astype(np.float32), last_queue.astype(np.float32),\n            self.neighboring(self.incoming, last_speed), self.neighboring(self.outgoing, last_speed),\n            self.neighboring(self.incoming, last_flow), self.neighboring(self.outgoing, last_flow),\n            self.neighboring(self.incoming, last_queue.astype(np.float32)),\n            self.neighboring(self.outgoing, last_queue.astype(np.float32)),\n            constants(np.isfinite(speed).mean()), constants(np.nanmin(speed) if np.isfinite(speed).any() else 1.0),\n        ]\n        features[25] = np.where(np.isfinite(features[25]), features[25], np.nan)\n        features[24] = np.where(np.isfinite(features[24]), features[24], np.nan)\n        base = np.column_stack(features).astype(np.float32)\n        X = np.tile(base, (HORIZON, 1))\n        X = np.column_stack([X, np.repeat(np.arange(1, HORIZON + 1), len(self.links))]).astype(np.float32)\n        X[~np.isfinite(X)] = np.nan\n        return X\n\n    def train_day(self, path):\n        columns = ["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]\n        frame = pd.read_parquet(path, columns=columns)\n        frame["link_id"] = frame.link_id.astype(str)\n        if frame.duplicated(["timestamp", "link_id"]).any():\n            frame = frame.groupby(["timestamp", "link_id"], as_index=False).mean(numeric_only=True)\n        times = sorted(frame.timestamp.unique())\n        index = pd.Index(times)\n        speed = frame.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=self.links).to_numpy(dtype=np.float32)\n        flow = frame.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=self.links).to_numpy(dtype=np.float32)\n        eligible = frame.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=self.links).fillna(0).to_numpy(dtype=bool)\n        return times, speed, flow, eligible\n\n\ndef candidate_origins(panel, times, speed, eligible):\n    queued = (speed <= panel.cut[None, :]) & eligible & np.isfinite(speed)\n    active = queued.any(axis=1)\n    candidates = {"queue_onset": [], "queue_ongoing": []}\n    for t in range(HISTORY, len(times) - HORIZON):\n        if np.isfinite(speed[t-HISTORY:t]).mean() < .7:\n            continue\n        if not active[t+1:t+HORIZON+1].any():\n            continue\n        if not active[t-HISTORY:t].any() and not active[t]:\n            candidates["queue_onset"].append(t)\n        if active[t]:\n            candidates["queue_ongoing"].append(t)\n    return candidates\n\n\ndef choose_origins(candidates, panel, date):\n    output = []\n    for condition, values in candidates.items():\n        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|{panel}|{date}|{condition}".encode(), digest_size=8).digest(), "big")\n        rng = np.random.default_rng(seed)\n        chosen = []\n        for t in rng.permutation(values):\n            if all(abs(t - other) >= 72 for other in chosen):\n                chosen.append(int(t))\n            if len(chosen) >= 2:\n                break\n        output.extend((condition, t) for t in chosen)\n    return output\n\n\ndef collect():\n    train_X, train_y, train_w = [], [], []\n    valid_windows = []\n    feature_count = None\n    for panel_no, name in enumerate(panel_names()):\n        panel = Panel(name, panel_no)\n        files = sorted((RELEASE / "corridors" / name / "train" / "mainline_states").glob("**/*.parquet"))\n        files = [path for path in files if path.parent.name.removeprefix("year_month=") in MONTHS]\n        print(f"data {name}: {len(files)} days", flush=True)\n        panel_counts = {"train_onset": 0, "train_ongoing": 0, "valid_onset": 0, "valid_ongoing": 0}\n        for path in files:\n            times, speed, flow, eligible = panel.train_day(path)\n            choices = choose_origins(candidate_origins(panel, times, speed, eligible), name, times[0][:10])\n            is_valid_month = path.parent.name.endswith("2031-02")\n            for condition, t in choices:\n                origin = pd.Timestamp(times[t])\n                X = panel.features(speed[t-HISTORY:t], flow[t-HISTORY:t], eligible[t-HISTORY:t], origin, condition)\n                future_speed = speed[t+1:t+HORIZON+1]\n                y = ((future_speed <= panel.cut[None, :]) & np.isfinite(future_speed)).reshape(-1)\n                valid = (eligible[t+1:t+HORIZON+1] & np.isfinite(future_speed)).reshape(-1)\n                if valid.sum() < .6 * len(valid):\n                    continue\n                if is_valid_month:\n                    valid_windows.append(Window(X, y, valid, name, condition))\n                    panel_counts[f"valid_{\'onset\' if condition == \'queue_onset\' else \'ongoing\'}"] += 1\n                else:\n                    train_X.append(X[valid])\n                    train_y.append(y[valid].astype(np.uint8))\n                    train_w.append(np.full(valid.sum(), 1000 / valid.sum(), dtype=np.float32))\n                    panel_counts[f"train_{\'onset\' if condition == \'queue_onset\' else \'ongoing\'}"] += 1\n                feature_count = X.shape[1]\n        print(panel_counts, flush=True)\n    X = np.vstack(train_X)\n    y = np.concatenate(train_y)\n    weights = np.concatenate(train_w)\n    print(f"train: {X.shape}, positive={y.mean():.5f}; validation windows={len(valid_windows)}", flush=True)\n    return X, y, weights, valid_windows, feature_count\n\n\ndef score_windows(windows, probabilities, threshold, onset_min):\n    by_panel_condition = {}\n    for window, p in zip(windows, probabilities):\n        prediction = p >= threshold\n        if window.condition == "queue_onset" and prediction.sum() < onset_min:\n            index = np.argpartition(p, -onset_min)[-onset_min:]\n            prediction[index] = True\n        true = window.y & window.eligible\n        predicted = prediction & window.eligible\n        union = np.logical_or(true, predicted).sum()\n        iou = np.logical_and(true, predicted).sum() / max(union, 1)\n        by_panel_condition.setdefault((window.panel, window.condition), []).append(iou)\n    panels = panel_names()\n    condition_means = {key: float(np.mean(values)) for key, values in by_panel_condition.items()}\n    overall = np.mean([np.mean([condition_means.get((panel, c), 0) for c in ("queue_onset", "queue_ongoing")]) for panel in panels])\n    onset = np.mean([condition_means.get((panel, "queue_onset"), 0) for panel in panels])\n    ongoing = np.mean([condition_means.get((panel, "queue_ongoing"), 0) for panel in panels])\n    return float(overall), float(onset), float(ongoing)\n\n\ndef fit_model():\n    X, y, weights, windows, feature_count = collect()\n    persistence = [((w.X[:, 11] <= .60) & (w.X[:, 13] <= 3)).astype(np.float32) for w in windows]\n    print("proxy persistence IoU, onset, ongoing:", score_windows(windows, persistence, .5, 0), flush=True)\n    X_val = np.vstack([w.X[w.eligible] for w in windows])\n    y_val = np.concatenate([w.y[w.eligible] for w in windows]).astype(np.uint8)\n    w_val = np.concatenate([np.full(w.eligible.sum(), 1000 / w.eligible.sum(), dtype=np.float32) for w in windows])\n    train_set = lgb.Dataset(X, label=y, weight=weights, free_raw_data=True)\n    valid_set = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_set, free_raw_data=True)\n    params = {\n        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.045,\n        "num_leaves": 31, "max_depth": 8, "min_data_in_leaf": 250,\n        "feature_fraction": .85, "bagging_fraction": .85, "bagging_freq": 1,\n        "lambda_l2": 8.0, "max_bin": 127, "num_threads": THREADS,\n        "verbosity": -1, "seed": SEED,\n    }\n    model = lgb.train(params, train_set, num_boost_round=700, valid_sets=[valid_set],\n                      callbacks=[lgb.early_stopping(60, verbose=True), lgb.log_evaluation(50)])\n    model.save_model(str(OUT / "queue_lgb.txt"))\n    del X, X_val, train_set, valid_set\n    probs = [model.predict(w.X, num_threads=THREADS) for w in windows]\n    trials = []\n    for threshold in np.arange(.05, .501, .025):\n        for onset_min in (0, 3, 6, 12, 24):\n            overall, onset, ongoing = score_windows(windows, probs, float(threshold), onset_min)\n            trials.append((overall, onset, ongoing, float(threshold), onset_min))\n    trials.sort(reverse=True)\n    print("best threshold trials", trials[:10], flush=True)\n    result = {"best_iteration": model.best_iteration, "feature_count": feature_count,\n              "validation_windows": len(windows), "positive_fraction": float(y.mean()),\n              "threshold": trials[0][3], "onset_min": trials[0][4],\n              "proxy_iou": trials[0][0], "proxy_onset_iou": trials[0][1],\n              "proxy_ongoing_iou": trials[0][2]}\n    (OUT / "queue_model.json").write_text(json.dumps(result, indent=2), encoding="utf-8")\n    print(result, flush=True)\n    return model, result',
'state_inference.py': '\n"""State residual ensemble; GPU prediction matches the original execution path."""\nfrom pathlib import Path\nimport json,os\nimport numpy as np\nimport pandas as pd\nfrom catboost import CatBoostRegressor\nfrom catboost.utils import get_gpu_device_count\nfrom config import OUT,THREADS\nfrom prepare_state import base,features\n\nclass StatePredictor:\n    def __init__(self, assets):\n        self.assets=Path(assets)\n        self.settings={channel+\'_mix\': {\'d8\': 0.5, \'d10\': 0.5} for channel in (\'speed\',\'flow\')}\n        self.task_type=os.environ.get(\'TFB_PREDICT_DEVICE\', \'GPU\' if get_gpu_device_count()>0 else \'CPU\')\n        self.models={}\n        for channel in (\'speed\',\'flow\'):\n            for tag in self.settings[channel+\'_mix\']:\n                model=CatBoostRegressor();model.load_model(str(self.assets/(channel+\'_\'+tag+\'.cbm\')))\n                self.models[channel,tag]=model\n        print(\'State inference device:\',self.task_type,flush=True)\n    def residual(self,X,channel):\n        result=np.zeros(len(X))\n        for tag,weight in self.settings[channel+\'_mix\'].items():\n            model=self.models[channel,tag]\n            for start in range(0,len(X),250000):\n                batch=X[start:start+250000]\n                pred=model.predict(batch,task_type=self.task_type,thread_count=THREADS)\n                result[start:start+len(batch)]+=weight*pred\n        return result\n\ndef predict_states(predictor):\n    first=True\n    for split in (\'validation\',\'private\'):\n        for pi,name in enumerate(base.all_panels()):\n            panel=base.Panel(name,pi)\n            frame=base.month_frame(name,\'\',split=split)\n            times,v,q,e=base.matrices(frame,panel)\n            target=pd.read_csv(base.RELEASE/\'task1\'/name/split/\'sample_submission_state.csv\',dtype={\'station_id\':str,\'link_id\':str})\n            rows=times.get_indexer(target.timestamp);cols=pd.Index(panel.links).get_indexer(target.link_id)\n            assert (rows>=0).all() and (cols>=0).all()\n            regime=target.mask_regime.map({\'R1\':0,\'R2\':1,\'R3\':2}).to_numpy()\n            X,bv,bq=features(panel,pi,times,v,q,e,rows,cols,regime)\n            assert X.shape[1]==94\n            target[\'speed_kmh\']=np.clip(bv+predictor.residual(X,\'speed\'),1,180)\n            target[\'flow_vph\']=np.clip((bq+predictor.residual(X,\'flow\'))*panel.lanes[cols],0,150000)\n            assert np.isfinite(target[[\'speed_kmh\',\'flow_vph\']].to_numpy()).all()\n            columns=[\'panel\',\'timestamp\',\'station_id\',\'link_id\',\'mask_regime\',\'speed_kmh\',\'flow_vph\']\n            target[columns].to_csv(OUT/\'state.csv\',index=False,mode=\'w\' if first else \'a\',header=first)\n            first=False;print(name,split,len(target),flush=True)',
'state_model.py': '\n"""LightGBM residual reconstruction with temporal and spatial context."""\n\nfrom __future__ import annotations\n\nimport hashlib\nimport json\nimport sys\nfrom pathlib import Path\n\nimport lightgbm as lgb\nimport numpy as np\nimport pandas as pd\n\nfrom config import RELEASE, OUT, THREADS\nfrom base_panel import Panel as BasePanel\n\nTRAIN_MONTHS = ("2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01")\nVALID_MONTH = "2031-02"\nSEED = 20261001\nRATE = np.array([.20, .30, .50], dtype=np.float32)\n\n\nclass Panel(BasePanel):\n    """Add milepost order from static, public training metadata."""\n\n    def __init__(self, name, panel_no):\n        super().__init__(name, panel_no)\n        sample = sorted((RELEASE / "corridors" / name / "train" / "mainline_states" /\n                         "year_month=2030-06").glob("*.parquet"))[0]\n        location = pd.read_parquet(sample, columns=["link_id", "milepost"]).drop_duplicates("link_id")\n        location["link_id"] = location.link_id.astype(str)\n        milepost = location.set_index("link_id").reindex(self.links).milepost.to_numpy(dtype=np.float32)\n        if not np.isfinite(milepost).all():\n            raise ValueError(f"Missing mileposts for {name}")\n        self.milepost = milepost\n        self.sorted_links = np.argsort(milepost)\n        self.link_rank = np.empty(len(self.links), dtype=np.int32)\n        self.link_rank[self.sorted_links] = np.arange(len(self.links))\n\n\ndef all_panels():\n    config = json.loads((RELEASE / "config" / "corridors.json").read_text())\n    return [x["corridor_id"] for x in config["panels"]]\n\n\ndef month_frame(panel, month, split="train"):\n    directory = RELEASE / "corridors" / panel / split / "mainline_states"\n    if split != "train":\n        directory = RELEASE / "corridors" / panel / split / "mainline_states_masked"\n        paths = sorted(directory.glob("**/*.parquet"))\n    else:\n        paths = sorted(directory.glob(f"year_month={month}/*.parquet"))\n    if not paths:\n        raise FileNotFoundError(f"No state data: {panel} {split} {month}")\n    cols = ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]\n    frame = pd.concat((pd.read_parquet(path, columns=cols) for path in paths), ignore_index=True)\n    frame["link_id"] = frame.link_id.astype(str)\n    frame["station_id"] = frame.station_id.astype(str)\n    if frame.duplicated(["timestamp", "link_id"]).any():\n        raise ValueError(f"Duplicate link/time for {panel} {month}")\n    return frame\n\n\ndef matrices(frame, panel):\n    index = pd.Index(sorted(frame.timestamp.unique()))\n    cols = panel.links\n    speed = frame.pivot(index="timestamp", columns="link_id", values="speed_kmh").reindex(index=index, columns=cols).to_numpy(dtype=np.float32)\n    flow = frame.pivot(index="timestamp", columns="link_id", values="flow_vph").reindex(index=index, columns=cols).to_numpy(dtype=np.float32)\n    eligible = frame.pivot(index="timestamp", columns="link_id", values="is_score_eligible").reindex(index=index, columns=cols).fillna(0).to_numpy(dtype=bool)\n    return index, speed, flow, eligible\n\n\ndef simulated_mask(panel_name, times, speed, flow, eligible):\n    dates = pd.Index([text[:10] for text in times])\n    result = np.zeros(speed.shape, dtype=bool)\n    regimes = np.empty(speed.shape[0], dtype=np.uint8)\n    for date in dates.unique():\n        rows = np.flatnonzero(dates == date)\n        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|{panel_name}|{date}".encode(), digest_size=8).digest(), "big")\n        regime = seed % 3\n        rng = np.random.default_rng(seed)\n        regimes[rows] = regime\n        result[rows] = (rng.random((len(rows), speed.shape[1])) < RATE[regime]) & eligible[rows] \\\n                       & np.isfinite(speed[rows]) & np.isfinite(flow[rows])\n    return result, regimes\n\n\ndef sampled_targets(mask, times, per_day):\n    dates = pd.Index([text[:10] for text in times])\n    row_out, col_out = [], []\n    for date in dates.unique():\n        rows = np.flatnonzero(dates == date)\n        coords = np.argwhere(mask[rows])\n        if not len(coords):\n            continue\n        seed = int.from_bytes(hashlib.blake2b(f"{SEED}|sample|{date}|{mask.shape[1]}".encode(), digest_size=8).digest(), "big")\n        rng = np.random.default_rng(seed)\n        chosen = coords[rng.choice(len(coords), size=min(per_day, len(coords)), replace=False)]\n        row_out.append(rows[chosen[:, 0]])\n        col_out.append(chosen[:, 1])\n    return np.concatenate(row_out), np.concatenate(col_out)\n\n\ndef graph_mean(graph, values):\n    observed = np.isfinite(values).astype(np.float32)\n    count = (graph @ observed.T).T\n    mean = (graph @ np.nan_to_num(values, nan=0.0).T).T / np.maximum(count, 1)\n    mean = np.where(count > 0, mean, np.nan)\n    return mean, count\n\n\ndef compute_features(panel, panel_no, times, observed_speed, observed_flow, rows, cols, regime):\n    n_times, n_links = observed_speed.shape\n    temporal_speed = pd.DataFrame(observed_speed).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)\n    temporal_flow = pd.DataFrame(observed_flow).interpolate(axis=0, limit_direction="both").to_numpy(dtype=np.float32)\n    temporal_speed = np.where(np.isfinite(temporal_speed), temporal_speed, panel.vf[None, :])\n    temporal_flow = np.where(np.isfinite(temporal_flow), temporal_flow, .6 * panel.cap[None, :])\n\n    graph = (panel.incoming + panel.outgoing).tocsr()\n    graph.data[:] = 1.0\n    spatial_flow, spatial_count = graph_mean(graph, observed_flow)\n    flow_base = np.where(spatial_count > 0, .5 * temporal_flow + .5 * spatial_flow, temporal_flow)\n    up_speed, up_count = graph_mean(panel.incoming, observed_speed)\n    down_speed, down_count = graph_mean(panel.outgoing, observed_speed)\n    up_flow, _ = graph_mean(panel.incoming, observed_flow)\n    down_flow, _ = graph_mean(panel.outgoing, observed_flow)\n\n    valid = np.isfinite(observed_speed)\n    positions = np.arange(n_times, dtype=np.int32)[:, None]\n    prev_index = np.maximum.accumulate(np.where(valid, positions, -1), axis=0)\n    next_index = np.minimum.accumulate(np.where(valid, positions, n_times)[::-1], axis=0)[::-1]\n    p = prev_index[rows, cols]\n    n = next_index[rows, cols]\n    prev_v = observed_speed[np.maximum(p, 0), cols].copy()\n    next_v = observed_speed[np.minimum(n, n_times - 1), cols].copy()\n    prev_q = observed_flow[np.maximum(p, 0), cols].copy()\n    next_q = observed_flow[np.minimum(n, n_times - 1), cols].copy()\n    prev_v[p < 0] = np.nan\n    prev_q[p < 0] = np.nan\n    next_v[n >= n_times] = np.nan\n    next_q[n >= n_times] = np.nan\n    prev_gap = np.where(p >= 0, rows - p, 300).astype(np.float32)\n    next_gap = np.where(n < n_times, n - rows, 300).astype(np.float32)\n\n    stamp = pd.to_datetime(times, utc=True)\n    tod = stamp.hour.to_numpy(dtype=np.float32) + stamp.minute.to_numpy(dtype=np.float32) / 60\n    dow = stamp.dayofweek.to_numpy(dtype=np.float32)\n    lanes = np.maximum(panel.lanes[cols], 1)\n    base_speed = temporal_speed[rows, cols]\n    base_flow = flow_base[rows, cols] / lanes\n    temporal_flow_lane = temporal_flow[rows, cols] / lanes\n    spatial_flow_lane = spatial_flow[rows, cols] / lanes\n    up_q = up_flow[rows, cols] / lanes\n    down_q = down_flow[rows, cols] / lanes\n    regime_vector = regime if len(regime) == len(rows) else regime[rows]\n    features = [\n        np.full(len(rows), panel_no), cols, regime_vector,\n        tod[rows], np.sin(2*np.pi*tod[rows]/24), np.cos(2*np.pi*tod[rows]/24), dow[rows],\n        panel.vf[cols], panel.cap[cols] / lanes, lanes, panel.length[cols],\n        base_speed, base_flow, temporal_flow_lane, spatial_flow_lane,\n        base_speed / panel.vf[cols], base_flow * lanes / panel.cap[cols],\n        prev_v, next_v, prev_q / lanes, next_q / lanes, prev_gap, next_gap,\n        next_v - prev_v, (next_q - prev_q) / lanes,\n        up_speed[rows, cols], down_speed[rows, cols], up_q, down_q,\n        up_count[rows, cols], down_count[rows, cols], spatial_count[rows, cols],\n        up_speed[rows, cols] - down_speed[rows, cols], up_q - down_q,\n    ]\n    for offset in (-2, -1, 1, 2):\n        ranks = panel.link_rank[cols] + offset\n        valid_neighbor = (ranks >= 0) & (ranks < n_links)\n        neighbor = panel.sorted_links[np.clip(ranks, 0, n_links-1)]\n        neighbor_lanes = np.maximum(panel.lanes[neighbor], 1)\n        same_time_v = observed_speed[rows, neighbor]\n        same_time_q = observed_flow[rows, neighbor] / neighbor_lanes\n        filled_v = temporal_speed[rows, neighbor]\n        filled_q = temporal_flow[rows, neighbor] / neighbor_lanes\n        features.extend((\n            np.where(valid_neighbor, np.abs(panel.milepost[neighbor]-panel.milepost[cols]), np.nan),\n            np.where(valid_neighbor, same_time_v, np.nan),\n            np.where(valid_neighbor, same_time_q, np.nan),\n            np.where(valid_neighbor, filled_v, np.nan),\n            np.where(valid_neighbor, filled_q, np.nan),\n        ))\n    X = np.column_stack(features).astype(np.float32)\n    X[~np.isfinite(X)] = np.nan\n    return X, base_speed, base_flow',
'train_ongoing.py': "\nfrom queue_ongoing import fit_model\nfrom config import OUT\nmodel, settings = fit_model()\nmodel.save_model(str(OUT/'ongoing_lgb.txt'))",
'train_onset.py': '\n"""Train the two-stage onset model using June–January and validate on February."""\nimport numpy as np\nimport pandas as pd\nimport lightgbm as lgb\nfrom config import RELEASE, OUT, THREADS\nfrom queue_features import Panel, panel_names, HISTORY, HORIZON, candidate_origins\nXtrain=[];ytrain=[];valid=[]\nfor pi,name in enumerate(panel_names()):\n    p=Panel(name,pi)\n    for file in sorted((RELEASE/\'corridors\'/name/\'train/mainline_states\').glob(\'**/*.parquet\')):\n        times,v,q,e=p.train_day(file)\n        cand=candidate_origins(p,times,v,e)[\'queue_onset\'];cs=set(cand)\n        for t in [a for a in cand if a-1 not in cs]:\n            x=p.features(v[t-HISTORY:t],q[t-HISTORY:t],e[t-HISTORY:t],pd.Timestamp(times[t]),\'queue_onset\')[-len(p.links):]\n            y=(v[t+HORIZON]<=p.cut)&np.isfinite(v[t+HORIZON]); eligible=e[t+HORIZON]&np.isfinite(v[t+HORIZON])\n            if file.parent.name.endswith(\'2031-02\'):\n                valid.append((name,x,y,eligible))\n            else:\n                Xtrain.append(x[eligible]);ytrain.append(y[eligible])\n    print(name,\'train rows\',sum(len(a) for a in ytrain),\'valid windows\',len(valid),flush=True)\nX=np.concatenate(Xtrain);y=np.concatenate(ytrain)\nXv=np.concatenate([a[1][a[3]] for a in valid]);yv=np.concatenate([a[2][a[3]] for a in valid])\nparams={\'objective\':\'binary\',\'metric\':\'binary_logloss\',\'learning_rate\':.04,\'num_leaves\':15,\'min_data_in_leaf\':100,\'lambda_l2\':10,\'feature_fraction\':.9,\'num_threads\':THREADS,\'verbosity\':-1,\'seed\':20261004}\nmodel=lgb.train(params,lgb.Dataset(X,label=y),num_boost_round=900,valid_sets=[lgb.Dataset(Xv,label=yv)],callbacks=[lgb.early_stopping(70),lgb.log_evaluation(100)])\nmodel.save_model(str(OUT/\'onset_stage1.txt\'))\n\nmodel = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=1500,\n                  init_model=str(OUT/\'onset_stage1.txt\'),\n                  valid_sets=[lgb.Dataset(Xv, label=yv)],\n                  callbacks=[lgb.early_stopping(70), lgb.log_evaluation(100)])\nmodel.save_model(str(OUT/\'onset_lgb.txt\'))\nprint("Selected onset threshold: 0.50; predict the final horizon step with a one-link fallback.")',
'train_onset_catboost.py': '\n"""GPU onset classifier with causal history-dropout augmentation."""\nfrom pathlib import Path\nimport sys,json\nimport numpy as np\nimport pandas as pd\nimport lightgbm as lgb\nfrom catboost import CatBoostClassifier\nimport os\nfrom config import RELEASE, OUT, THREADS\nfrom queue_features import Panel,panel_names,HISTORY,HORIZON,candidate_origins\nxx=[];yy=[];valid=[];rng=np.random.default_rng(20261005)\nfor pi,name in enumerate(panel_names()):\n    p=Panel(name,pi)\n    for file in sorted((RELEASE/\'corridors\'/name/\'train/mainline_states\').glob(\'**/*.parquet\')):\n        times,v,q,e=p.train_day(file);cand=candidate_origins(p,times,v,e)[\'queue_onset\'];cs=set(cand);isval=file.parent.name.endswith(\'2031-02\')\n        for t in [a for a in cand if a-1 not in cs]:\n            truth=(v[t+1:t+HORIZON+1]<=p.cut)&np.isfinite(v[t+1:t+HORIZON+1]);eligible=e[t+1:t+HORIZON+1]&np.isfinite(v[t+1:t+HORIZON+1])\n            for rep in range(1 if isval else 2):\n                hv=v[t-HISTORY:t].copy();hq=q[t-HISTORY:t].copy();he=e[t-HISTORY:t].copy()\n                if rep:\n                    drop=rng.random(hv.shape)<.05;hv[drop]=np.nan;hq[drop]=np.nan;he[drop]=False\n                x=p.features(hv,hq,he,pd.Timestamp(times[t]),\'queue_onset\')[-len(p.links):]\n                if isval:valid.append((name,x,truth,eligible))\n                else:xx.append(x[eligible[-1]]);yy.append(truth[-1,eligible[-1]])\n    print(name,\'train rows\',sum(len(a) for a in yy),\'valid windows\',len(valid),flush=True)\nX=np.concatenate(xx);y=np.concatenate(yy);Xv=np.concatenate([a[1][a[3][-1]] for a in valid]);yv=np.concatenate([a[2][-1,a[3][-1]] for a in valid])\nm=CatBoostClassifier(iterations=3000,depth=7,learning_rate=.05,l2_leaf_reg=10,task_type=\'GPU\',devices=os.environ.get(\'TFB_GPU_DEVICE\', \'0\'),thread_count=THREADS,random_seed=20261005,loss_function=\'Logloss\',early_stopping_rounds=150,verbose=100,train_dir=str(OUT/\'catboost\'))\nm.fit(X,y,eval_set=(Xv,yv),use_best_model=True);m.save_model(str(OUT/\'onset.cbm\'))\nold=lgb.Booster(model_file=str(OUT/\'onset_lgb.txt\'));records=[]\nfor name,x,truth,e in valid:\n    p=m.predict_proba(x,thread_count=THREADS)[:,1];b=old.predict(x,num_threads=THREADS)\n    for weight in (0,.5,1):\n        prob=(1-weight)*b+weight*p\n        for threshold in (.3,.35,.4,.45,.5,.55,.6):\n            pred=np.zeros_like(truth);pred[-1]=prob>=threshold\n            if not pred.any():pred[-1,np.argmax(prob)]=True\n            iou=((pred&truth)&e).sum()/max(((pred|truth)&e).sum(),1)\n            records.append({\'panel\':name,\'gpu_weight\':weight,\'threshold\':threshold,\'iou\':iou})\nf=pd.DataFrame(records);f.to_csv(OUT/\'window_scores.csv\',index=False)\ns=f.groupby([\'gpu_weight\',\'threshold\',\'panel\']).iou.mean().groupby([\'gpu_weight\',\'threshold\']).mean().sort_values(ascending=False);s.to_csv(OUT/\'scores.csv\');print(s.to_string(),flush=True)\nbest=s.index[0];(OUT/\'onset_settings.json\').write_text(json.dumps({\'gpu_weight\':float(best[0]),\'threshold\':float(best[1]),\'proxy\':float(s.iloc[0]),\'train_rows\':len(y),\'valid_windows\':len(valid)},indent=2))',
'train_state.py': "\nfrom pathlib import Path\nimport argparse,json,time\nimport numpy as np\nfrom catboost import CatBoostRegressor\nfrom config import OUT, THREADS\nap=argparse.ArgumentParser();ap.add_argument('--channel',choices=['speed','flow'],required=True);ap.add_argument('--device',required=True);ap.add_argument('--iterations',type=int,default=9000);ap.add_argument('--depth',type=int,default=8);ap.add_argument('--tag',default='d8');a=ap.parse_args()\nt=time.time();key='yv' if a.channel=='speed' else 'yq'\ntr=np.load(OUT/'train.npz');va=np.load(OUT/'valid.npz')\nmodel=CatBoostRegressor(iterations=a.iterations,depth=a.depth,learning_rate=.06,l2_leaf_reg=10,loss_function='RMSE',eval_metric='RMSE',task_type='GPU',devices=a.device,gpu_ram_part=.8,thread_count=THREADS,random_seed=20261005,bootstrap_type='Bernoulli',subsample=.85,early_stopping_rounds=150,verbose=100,train_dir=str(OUT/('catboost_'+a.channel+'_'+a.tag)))\nmodel.fit(tr['X'],tr[key],eval_set=(va['X'],va[key]),use_best_model=True)\nmodel.save_model(str(OUT/(a.channel+'_'+a.tag+'.cbm')))\npred=model.predict(va['X'],thread_count=THREADS);np.save(OUT/('valid_'+a.channel+'_'+a.tag+'.npy'),pred)\nmetrics={'best_iteration':model.get_best_iteration(),'rmse':float(np.sqrt(np.mean((va[key]-pred)**2))),'elapsed_seconds':time.time()-t,'train_rows':len(tr[key]),'valid_rows':len(va[key]),'device':a.device}\n(OUT/(a.channel+'_'+a.tag+'_metrics.json')).write_text(json.dumps(metrics,indent=2));print(metrics,flush=True)"
}
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

# ================= TASK 1 EXTRA (v3): LODO weekday x tod profiles, LGBM x3, U-Net x2 =================
USE_T1_EXTRA = True
T1_TREES_SHARE = 0.70
T1_PUB_SHARE = 0.50
LGBM_VARIANTS = ((20280101, 127, 0.04), (20280102, 63, 0.05), (20280103, 255, 0.03))
UNET_SEEDS = (20280201, 20280202)
UNET_WIDTH = 48
UNET_EPOCHS = 24
PAD_MIN = 48
SPEED_SCALE = 25.0
FLOW_SCALE = 600.0
t1_share = {'speed': T1_TREES_SHARE, 'flow': T1_TREES_SHARE}
STATE_BASE_FILE = 'state_v1.csv'
T1_EXTRA_OK = False

def _fill(v, fallback):
    return np.where(np.isfinite(v), v, fallback)

try:
    import prepare_state as pstate
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    lanes_of = lambda p: np.maximum(p.lanes, 1.0)

    def _padf(arr, pad):
        a = np.ascontiguousarray(arr).astype(np.float16)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        n = int(a.shape[1])
        if n > pad:
            raise ValueError(f'_padf: {a.shape} wider than pad {pad}')
        if n < pad:
            a = np.concatenate([a, np.zeros((a.shape[0], pad - n), dtype=np.float16)], axis=1)
        return a

    def _static_of(p, lanes, pad):
        L = len(p.links)
        cols = np.stack([p.vf / 100.0, p.cap / lanes / 100.0, p.cut / 100.0,
                         np.ones(L, dtype=np.float64)], axis=1).astype(np.float16)
        st1 = np.concatenate([cols.T, np.zeros((4, pad - L), dtype=np.float16)], axis=1)
        return np.repeat(st1[:, None, :], 288, axis=1)

    class PanelProfiles:
        """Weekday x tod profiles from unmasked train days, with leave-one-day-out variants."""
        def __init__(self, name):
            p = base.Panel(name, 0)
            self.panel = p
            self.lanes = lanes_of(p)
            L = len(p.links)
            sumv = np.zeros((7, 288, L), dtype=np.float64)
            sumq = np.zeros((7, 288, L), dtype=np.float64)
            cnt = np.zeros((7, 288, L), dtype=np.int64)
            self.dow = []
            self.day_v = []
            self.day_q = []
            for month in base.TRAIN_MONTHS:
                times, v, q, e = base.matrices(base.month_frame(name, month), p)
                assert len(times) % 288 == 0, (name, month, len(times))
                n_days = len(times) // 288
                dow = np.asarray(pd.to_datetime(times, utc=True).dayofweek, dtype=np.int64)
                vd = v.reshape(n_days, 288, L)
                qd = (q / self.lanes[None, None, :]).reshape(n_days, 288, L)
                ed = e.reshape(n_days, 288, L)
                for d in range(n_days):
                    dd = int(dow[288 * d])
                    assert dow[288 * d] == dow[288 * d + 287], (name, month, d)
                    self.dow.append(dd)
                    self.day_v.append(vd[d])
                    self.day_q.append(qd[d])
                    fin = ed[d] & np.isfinite(vd[d]) & np.isfinite(qd[d])
                    sumv[dd] += np.where(fin, vd[d], 0.0)
                    sumq[dd] += np.where(fin, qd[d], 0.0)
                    cnt[dd] += fin
            self.sumv = sumv
            self.sumq = sumq
            self.cnt = cnt
            infv = sumv / np.maximum(cnt, 1)
            self.prof_in = np.where(np.isfinite(infv), infv, p.vf[None, None, :])
            infq = sumq / np.maximum(cnt, 1)
            self.prof_q_in = np.where(np.isfinite(infq), infq, 0.6 * (p.cap / self.lanes)[None, None, :])
            self.n_days = len(self.dow)
            print('profiles', name, 'days:', self.n_days, flush=True)

        def lodo(self, d):
            dd = self.dow[d]
            vv = self.day_v[d]
            qq = self.day_q[d]
            fin = np.isfinite(vv) & np.isfinite(qq)
            c = self.cnt[dd]
            den = np.maximum(c - fin.astype(np.int64), 1)
            pv = (self.sumv[dd] - np.where(fin, vv, 0.0)) / den
            pq = (self.sumq[dd] - np.where(fin, qq, 0.0)) / den
            return (np.where(np.isfinite(pv), pv, self.panel.vf[None, :]),
                    np.where(np.isfinite(pq), pq, 0.6 * (self.panel.cap / self.lanes)))

        def month_profile(self, times):
            """(n_rows, L) inference profile for a non-train month."""
            n_rows = len(times)
            day_of = np.arange(n_rows) // 288
            tod = np.arange(n_rows) % 288
            dow = np.asarray(pd.to_datetime(times, utc=True).dayofweek, dtype=np.int64)
            pv = self.prof_in[dow[day_of], tod, :]
            pq = self.prof_q_in[dow[day_of], tod, :]
            return pv, pq

    def ext_X(p, pi, times, v, q, e, rows, cols, regime, pv_row, pq_row):
        X, bv, bq = pstate.features(p, pi, times, v, q, e, rows, cols, regime)
        extra = np.column_stack([pv_row, pq_row,
                                 bv / np.maximum(pv_row, 1e-3),
                                 bq / np.maximum(pq_row, 1e-3)])
        return np.column_stack([X, extra]).astype(np.float32), bv, bq

    # ---- extended feature sets (mirrors prepare_state) ----
    prof_cache = {}
    xx, yv, yq, xv, yvv, yqv = [], [], [], [], [], []
    original_seed = base.SEED
    for pi, name in enumerate(base.all_panels()):
        p = base.Panel(name, pi)
        prof = PanelProfiles(name)
        prof_cache[name] = prof
        n_days = prof.n_days
        day_dow = np.asarray(prof.dow, dtype=np.int64)
        for month in (*base.TRAIN_MONTHS, base.VALID_MONTH):
            times, v, q, e = base.matrices(base.month_frame(name, month), p)
            isval = month == base.VALID_MONTH
            for rep in range(1 if isval else 2):
                base.SEED = original_seed + rep * 7919
                mask, regime = base.simulated_mask(name, times, v, q, e)
                rows, cols = base.sampled_targets(mask, times, 1500 if isval else 750)
                v_masked = np.where(mask, np.nan, v)
                q_masked = np.where(mask, np.nan, q)
                day_of = rows // 288
                tod = rows % 288
                if isval:
                    pv_full, pq_full = prof_cache[name].month_profile(times)
                    pv_row = pv_full[rows, cols]
                    pq_row = pq_full[rows, cols]
                    del pv_full, pq_full
                else:
                    pv_full = np.empty((n_days, 288, len(p.links)), dtype=np.float32)
                    pq_full = np.empty_like(pv_full)
                    for d in np.unique(day_of):
                        a, b = prof.lodo(int(d))
                        pv_full[d] = a
                        pq_full[d] = b
                    pv_row = pv_full[day_of, tod, cols]
                    pq_row = pq_full[day_of, tod, cols]
                    del pv_full, pq_full
                X, bv, bq = ext_X(p, pi, times, v_masked, q_masked, e, rows, cols, regime, pv_row, pq_row)
                rv = v[rows, cols] - bv
                rq = q[rows, cols] / p.lanes[cols] - bq
                if isval:
                    xv.append(X); yvv.append(rv); yqv.append(rq)
                else:
                    xx.append(X); yv.append(rv); yq.append(rq)
                del X, rv, rq, v_masked, q_masked
            gc.collect()
            print('ext features', name, month, flush=True)
    base.SEED = original_seed
    Xtr = np.concatenate(xx); del xx
    yvtr = np.concatenate(yv); yqtr = np.concatenate(yq); del yv, yq
    Xva = np.concatenate(xv); yvv_arr = np.concatenate(yvv); yqv_arr = np.concatenate(yqv); del xv
    yvv = yvv_arr
    yqv = yqv_arr
    print('ext features ready', Xtr.shape, Xva.shape, flush=True)

    # ---- 3 LGBM variants x 2 channels ----
    lgbm_models = {'speed': [], 'flow': []}
    valid_sets = {'speed': lgb.Dataset(Xva, label=yvv, free_raw_data=False), 'flow': lgb.Dataset(Xva, label=yqv, free_raw_data=False)}
    for seed, leaves, lr in LGBM_VARIANTS:
        for channel, ytr in (('speed', yvtr), ('flow', yqtr)):
            params = {'objective': 'regression_l2', 'metric': 'rmse', 'learning_rate': lr,
                      'num_leaves': leaves, 'min_data_in_leaf': 200, 'feature_fraction': 0.9,
                      'bagging_fraction': 0.9, 'bagging_freq': 1, 'lambda_l2': 10.0,
                      'num_threads': THREADS, 'verbosity': -1, 'seed': seed}
            m = lgb.train(params, lgb.Dataset(Xtr, label=ytr, free_raw_data=False), num_boost_round=1200,
                          valid_sets=[valid_sets[channel]],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
            lgbm_models[channel].append(m)
            print('lgbm', channel, seed, 'best_iter:', m.best_iteration, flush=True)
    for channel, models, ytrue in (('speed', lgbm_models['speed'], yvv), ('flow', lgbm_models['flow'], yqv)):
        pred = np.mean([m.predict(Xva, num_threads=THREADS) for m in models], axis=0)
        base_rmse = float(np.sqrt(np.mean(ytrue ** 2)))
        lgbm_rmse = float(np.sqrt(np.mean((ytrue - pred) ** 2)))
        if base_rmse > 0 and lgbm_rmse / base_rmse >= 0.95:
            t1_share[channel] = 0.5
        print(f'valid residual RMSE {channel}: zero={base_rmse:.4f} lgbm={lgbm_rmse:.4f} share={t1_share[channel]}', flush=True)
    del Xtr, yvtr, yqtr, Xva, yvv, yqv, valid_sets
    gc.collect()

    # ---- U-Net day grids for train + validation months ----
    class UNet(nn.Module):
        def __init__(self, c_in=9, w=48):
            super().__init__()
            block = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.GELU())
            self.e1 = block(c_in, w); self.e2 = block(w, 2 * w); self.e3 = block(2 * w, 4 * w)
            self.mid = block(4 * w, 4 * w)
            self.u2 = block(8 * w, 2 * w); self.u1 = block(4 * w, w)
            self.head = nn.Conv2d(2 * w, 2, 1)
            self.pool = nn.MaxPool2d(2)
        def forward(self, x):
            a1 = self.e1(x)
            a2 = self.e2(self.pool(a1))
            a3 = self.e3(self.pool(a2))
            b = self.mid(a3)
            b = self.u2(torch.cat([F.interpolate(b, size=a3.shape[2:], mode='nearest'), a3], 1))
            b = self.u1(torch.cat([F.interpolate(b, size=a2.shape[2:], mode='nearest'), a2], 1))
            b = torch.cat([F.interpolate(b, size=a1.shape[2:], mode='nearest'), a1], 1)
            return self.head(b)

    max_links = max(len(prof_cache[n].panel.links) for n in prof_cache)
    pad = int(((max(PAD_MIN, max_links) + 7) // 8) * 8)
    print('pad links:', pad, 'max_links:', max_links, flush=True)

    unet_data = []
    base.SEED = original_seed
    for name in base.all_panels():
        p = prof_cache[name].panel
        lanes = lanes_of(p)
        static = _static_of(p, lanes, pad)
        frames = {}
        for month in (*base.TRAIN_MONTHS, base.VALID_MONTH):
            frames[month] = base.matrices(base.month_frame(name, month), p)
        days = []
        for month in (*base.TRAIN_MONTHS, base.VALID_MONTH):
            times = frames[month][0]
            n_days = len(times) // 288
            for d in range(n_days):
                days.append((month, d))
        dyn = np.zeros((len(days), 5, 288, pad), dtype=np.float16)
        tidx, tdv, tdq = [], [], []
        tday_off = [0]
        for k, (month, d) in enumerate(days):
            times, v, q, e = frames[month]
            d0 = 288 * d
            vd = v[d0:d0 + 288]; qd = q[d0:d0 + 288]; ed = e[d0:d0 + 288]
            if month == base.VALID_MONTH:
                pv_d = prof_cache[name].prof_in[int(np.asarray(pd.to_datetime(times, utc=True).dayofweek)[d0])]
                pq_d = prof_cache[name].prof_q_in[int(np.asarray(pd.to_datetime(times, utc=True).dayofweek)[d0])]
            else:
                pv_d, pq_d = prof_cache[name].lodo(d)
            fin_v = np.isfinite(vd)
            fin_q = np.isfinite(qd)
            dyn[k, 0] = _padf(_fill(vd, pv_d), pad)
            dyn[k, 1] = _padf(_fill(qd / lanes[None, :], pq_d), pad)
            dyn[k, 2] = _padf(pv_d, pad)
            dyn[k, 3] = _padf(pq_d, pad)
            dyn[k, 4] = _padf((fin_v & fin_q & ed), pad)
            mask, _reg = base.simulated_mask(name, times, v, q, e)
            target = mask[d0:d0 + 288] & ed & fin_v & fin_q
            if target.any():
                ti = np.argwhere(target)
                tidx.append(ti.astype(np.int16))
                tdv.append(((vd - pv_d) / SPEED_SCALE)[target].astype(np.float16))
                tdq.append(((qd / lanes[None, :] - pq_d) / FLOW_SCALE)[target].astype(np.float16))
                tday_off.append(tday_off[-1] + len(ti))
        unet_data.append({'dyn': dyn, 'static': static,
                          'tidx': np.concatenate(tidx) if tidx else np.zeros((0, 2), dtype=np.int16),
                          'tdv': np.concatenate(tdv) if tdv else np.zeros(0, dtype=np.float16),
                          'tdq': np.concatenate(tdq) if tdq else np.zeros(0, dtype=np.float16),
                          'off': np.asarray(tday_off, dtype=np.int32),
                          'val_days': [k for k, (month, _d) in enumerate(days) if month == base.VALID_MONTH],
                          'n_days': len(days)})
        del frames
        gc.collect()
        print('unet data', name, len(days), flush=True)
    base.SEED = original_seed

    flat = []
    val_items = []
    for pi_d, data in enumerate(unet_data):
        for d in range(data['n_days']):
            if d in data['val_days']:
                val_items.append((pi_d, d))
            else:
                flat.append((pi_d, d))
    val_days_set = set(val_items)

    def batch_loss(model, items):
        x = torch.empty((len(items), 9, 288, pad), dtype=torch.float32)
        for i, (pi_d, d) in enumerate(items):
            data = unet_data[pi_d]
            x[i, :5] = torch.from_numpy(data['dyn'][d])
            x[i, 5:] = torch.from_numpy(data['static'])
        out = model(x)
        loss = torch.zeros(())
        for i, (pi_d, d) in enumerate(items):
            data = unet_data[pi_d]
            a, b = data['off'][d], data['off'][d + 1]
            if b > a:
                ti = data['tidx'][a:b]
                loss = loss + ((out[i, 0, ti[:, 0], ti[:, 1]] - torch.from_numpy(data['tdv'][a:b].astype(np.float32))).pow(2).mean()
                               + (out[i, 1, ti[:, 0], ti[:, 1]] - torch.from_numpy(data['tdq'][a:b].astype(np.float32))).pow(2).mean())
        return loss / max(1, len(items))

    unets = []
    for seed in UNET_SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = UNet(9, UNET_WIDTH).float()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best = None
        hist = []
        for epoch in range(UNET_EPOCHS):
            model.train()
            idx = rng.permutation(len(flat))
            order = [flat[j] for j in idx]
            tot, nbatch = 0.0, 0
            for k in range(0, len(order), 4):
                items = order[k:k + 4]
                loss = batch_loss(model, items)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
                nbatch += 1
            model.eval()
            with torch.no_grad():
                val = float(sum(batch_loss(model, val_items[k:k + 4]).item() for k in range(0, len(val_items), 4)) / max(1, len(val_items) // 4 + 1))
            hist.append(val)
            print(f'unet seed {seed} epoch {epoch} train {tot / max(1, nbatch):.5f} val {val:.5f}', flush=True)
            if best is None or val < best[0] - 1e-5:
                best = (val, {kk: vv.detach().clone() for kk, vv in model.state_dict().items()})
            if len(hist) > 12 and val >= min(hist[-12:]):
                break
        model.load_state_dict(best[1])
        model.eval()
        unets.append(model)
        gc.collect()
        print('unet trained', seed, 'best val', best[0], flush=True)

    # ---- inference on task1 target rows: LGBM + U-Net ----
    pieces = []
    for split in ('validation', 'private'):
        for pi, name in enumerate(base.all_panels()):
            p = prof_cache[name].panel
            lanes = lanes_of(p)
            times, v, q, e = base.matrices(base.month_frame(name, '', split=split), p)
            target = pd.read_csv(DATA / 'task1' / name / split / 'sample_submission_state.csv',
                                 dtype={'station_id': str, 'link_id': str})
            rows = times.get_indexer(target.timestamp)
            cols = target.link_id.map({l: i for i, l in enumerate(p.links)}).to_numpy(dtype=int)
            regime = target.mask_regime.map({'R1': 0, 'R2': 1, 'R3': 2}).to_numpy()
            assert (rows >= 0).all() and (cols >= 0).all()
            pv_full, pq_full = prof_cache[name].month_profile(times)
            X, bv, bq = ext_X(p, pi, times, v, q, e, rows, cols, regime,
                              pv_full[rows, cols], pq_full[rows, cols])
            sp = np.mean([m.predict(X, num_threads=THREADS) for m in lgbm_models['speed']], axis=0)
            fp = np.mean([m.predict(X, num_threads=THREADS) for m in lgbm_models['flow']], axis=0)
            n_days = len(times) // 288
            dow = np.asarray(pd.to_datetime(times, utc=True).dayofweek, dtype=np.int64)
            din = np.zeros((n_days, 5, 288, pad), dtype=np.float16)
            for d in range(n_days):
                d0 = 288 * d
                vd = v[d0:d0 + 288]; qd = q[d0:d0 + 288]; ed = e[d0:d0 + 288]
                pv_d = prof_cache[name].prof_in[dow[d0]]
                pq_d = prof_cache[name].prof_q_in[dow[d0]]
                din[d, 0] = _padf(_fill(vd, pv_d), pad)
                din[d, 1] = _padf(_fill(qd / lanes[None, :], pq_d), pad)
                din[d, 2] = _padf(pv_d, pad)
                din[d, 3] = _padf(pq_d, pad)
                din[d, 4] = _padf(np.isfinite(vd) & np.isfinite(qd) & ed, pad)
            static = _static_of(p, lanes, pad)
            outs = []
            for m in unets:
                chunk = []
                with torch.no_grad():
                    for d in range(0, n_days, 4):
                        days_b = list(range(d, min(d + 4, n_days)))
                        x = torch.empty((len(days_b), 9, 288, pad), dtype=torch.float32)
                        for j, dd in enumerate(days_b):
                            x[j, :5] = torch.from_numpy(din[dd])
                            x[j, 5:] = torch.from_numpy(static)
                        chunk.append(m(x).numpy())
                outs.append(np.concatenate(chunk, axis=0))
            dv = np.mean(outs, axis=0)
            day_of = rows // 288
            tod = rows % 288
            unet_speed = pv_full[rows, cols] + SPEED_SCALE * dv[day_of, 0, tod, cols]
            unet_flow = (pq_full[rows, cols] + FLOW_SCALE * dv[day_of, 1, tod, cols]) * lanes[cols]
            pieces.append(pd.DataFrame({
                'panel': name, 'timestamp': target.timestamp, 'link_id': target.link_id,
                'speed_kmh': np.clip(bv + sp, 1, 180),
                'flow_vph': np.clip((bq + fp) * lanes[cols], 0, 150000),
                'unet_speed': np.clip(unet_speed, 1, 180),
                'unet_flow': np.clip(unet_flow, 0, 150000)}))
            del X, sp, fp, din, outs, dv
            gc.collect()
            print('t1extra infer', name, split, flush=True)
    extra_state = pd.concat(pieces, ignore_index=True)
    extra_state.to_csv(OUTPUT / 'state_t1extra.csv', index=False)
    print('t1extra state written:', len(extra_state), flush=True)

    # ---- blend: 50% v1 CatBoost + 50% (70% LGBM + 30% U-Net) ----
    v1s = pd.read_csv(OUTPUT / 'state_v1.csv', dtype={'station_id': str, 'link_id': str})
    assert len(v1s) == len(extra_state)
    assert (v1s.panel.values == extra_state.panel.values).all()
    assert (v1s.timestamp.values == extra_state.timestamp.values).all()
    assert (v1s.link_id.values == extra_state.link_id.values).all()
    blend = v1s.copy()
    for col, lcol, ucol, chan in (('speed_kmh', 'speed_kmh', 'unet_speed', 'speed'), ('flow_vph', 'flow_vph', 'unet_flow', 'flow')):
        mine = t1_share[chan] * extra_state[lcol].to_numpy() + (1.0 - t1_share[chan]) * extra_state[ucol].to_numpy()
        blend[col] = T1_PUB_SHARE * v1s[col].to_numpy() + (1.0 - T1_PUB_SHARE) * mine
    assert np.isfinite(blend[['speed_kmh', 'flow_vph']].to_numpy()).all()
    blend.to_csv(OUTPUT / 'state_blend.csv', index=False)
    STATE_BASE_FILE = 'state_blend.csv'
    T1_EXTRA_OK = True
    print('state blend written', flush=True)
except Exception as err:
    import traceback
    traceback.print_exc()
    print('T1 EXTRA FAILED, falling back to v1 state:', repr(err), flush=True)
    T1_EXTRA_OK = False


USE_PHYSICS = False
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
    state = pd.read_csv(OUTPUT / STATE_BASE_FILE, dtype={'station_id': str, 'link_id': str},
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
if USE_PHYSICS:
    for split in ('validation', 'private'):
        for name in base.all_panels():
            part = physics_pass(name, split)
            part.to_csv(OUTPUT / 'state.csv', index=False, mode='w' if first else 'a', header=first)
            first = False
            print('physics', name, split, len(part), flush=True)
else:
    (OUTPUT / 'state.csv').write_text((OUTPUT / STATE_BASE_FILE).read_text())
    print('physics disabled; state copied from', STATE_BASE_FILE, flush=True)
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
ONGOING_THR = 0.45
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
ONSET_PUBLIC_ONLY = True
ONGOING_PUBLIC_ONLY = True
queue_pieces = []
for info in windows_info:
    if info['condition'] == 'queue_onset':
        probs = [info['p_v1']] if ONSET_PUBLIC_ONLY else [info['p_v1']] + [m.predict(info['X'], num_threads=THREADS) for m in onset_models]
        p = np.mean(probs, axis=0)
        selected = p >= ONSET_THR
        if not selected.any():
            selected[np.argmax(p)] = True
        pred = np.zeros((HORIZON, info['X'].shape[0]), dtype=np.uint8)
        pred[-1] = selected.astype(np.uint8)
    else:
        probs = [info['p_v1']] if ONGOING_PUBLIC_ONLY else [info['p_v1']] + [m.predict(info['X'], num_threads=THREADS) for m in ongoing_models]
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
        pieces = [q for q in queue_pieces if q['split'].iloc[0] == split and q['panel'].iloc[0] == name]
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
              "rebuilt_warm_start": REBUILD_WARM_START,
              "t1_extra": T1_EXTRA_OK, "state_base": STATE_BASE_FILE, "ongoing_thr": ONGOING_THR, "use_physics": USE_PHYSICS, "t1_share": t1_share}
    (WORK / "run_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("Saved:", submission)
    display(pd.read_csv(submission, nrows=5))
    del tables
    gc.collect();
else:
    print("Inference disabled. Set RUN_INFERENCE=True after training or providing the checkpoints.")
