"""Load g.tec Unicorn Hybrid EEG CSVs and segment trials.

CSV layout (per host example notebook):
  - Named EEG channel columns: Fz C3 Cz C4 PO7 Pz PO8 Oz (plus aux channels)
  - A Marker column: "Marker_val" (and possibly a timestamp column)
  - 250 Hz.
Training cues: Marker_val in {'move','rest'}.
Test cues: Marker_val == 'cue_start'.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

EEG_COLS = ['Fz', 'C3', 'Cz', 'C4', 'PO7', 'Pz', 'PO8', 'Oz']
FS = 250
CUE_SEC = 5.0          # cue window length (cue onset -> cue_stop)
BASELINE_SEC = 3.0     # fixation period before cue onset
CUE_SAMPLES = int(CUE_SEC * FS)
BASELINE_SAMPLES = int(BASELINE_SEC * FS)


def find_columns(df: pd.DataFrame):
    eeg = [c for c in EEG_COLS if c in df.columns]
    marker_col = None
    for c in df.columns:
        if 'Marker_val' in c:
            marker_col = c
            break
    if marker_col is None:
        cand = [c for c in df.columns if 'arker' in c and ('val' in c.lower() or 'value' in c.lower())]
        if cand:
            marker_col = cand[0]
    return eeg, marker_col


def load_trial_windows(path: str):
    """Return list of dicts: {cue, baseline, subject, run, idx, next_marker, gap}.

    cue: (FS*5, n_ch) float32 window starting at cue onset.
    baseline: (FS*3, n_ch) window ending at cue onset (may be None if file start).
    """
    df = pd.read_csv(path)
    eeg, marker_col = find_columns(df)
    if len(eeg) != 8 or marker_col is None:
        raise ValueError(f'Unexpected columns in {path}: {list(df.columns)[:25]}')
    data = df[eeg].to_numpy(dtype=np.float32)
    markers = df[marker_col]
    vals = markers.tolist()
    n = data.shape[0]
    trials = []
    prev_idx = None
    for i, v in enumerate(vals):
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == '':
            continue
        v = str(v).strip()
        if v in ('move', 'rest', 'cue_start'):
            cue_end = min(i + CUE_SAMPLES, n)
            if cue_end - i < CUE_SAMPLES - 2 * FS:  # truncated window at file end
                continue
            base = data[max(0, i - BASELINE_SAMPLES):i, :]
            trials.append({
                'label': v,
                'cue': data[i:cue_end, :],
                'baseline': base,
                'start': i,
            })
    return trials


def subject_from_name(path: str) -> str:
    base = os.path.basename(path)
    return base.split('_')[0].split('-')[0]


def iter_dataset(root: str, kind: str = 'train'):
    """Yield (subject, run_id, path, trials). kind in {'train','test','all'}."""
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if not f.endswith('.csv'):
                continue
            is_test = 'test' in f
            if kind == 'train' and is_test:
                continue
            if kind == 'test' and not is_test:
                continue
            yield subject_from_name(os.path.join(dirpath, f)), f, os.path.join(dirpath, f)
