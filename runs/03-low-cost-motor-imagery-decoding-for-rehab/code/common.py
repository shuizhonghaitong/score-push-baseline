"""Shared data loading & preprocessing for LCMID (single subject)."""
import numpy as np
import pandas as pd
import glob, os, re
from scipy import signal

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
FS = 250.0
CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'PO7', 'Pz', 'PO8', 'Oz']
CUE_LEN = 5.0      # seconds after cue_start
BASE_LEN = 3.0     # seconds before cue_start (fixation)
TEST_SUBJECTS = ['S%03d' % i for i in [1,2,3,4,5,6,7,9,10,11,12,14,16,17,18,19,20]]

_NOTCH_CACHE = [None]
def notch(x):
    if _NOTCH_CACHE[0] is None:
        _NOTCH_CACHE[0] = signal.iirnotch(50.0 / (FS / 2.0), 30.0)
    b, a = _NOTCH_CACHE[0]
    return signal.filtfilt(b, a, x, axis=0)

def car(x):
    return x - x.mean(axis=1, keepdims=True)

def detrend_linear(x):
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    A = np.stack([t, np.ones(n)], axis=1)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef

def read_epochs(path, labeled=True):
    """Return (epochs_cue (N,T,8), epochs_base (N,Tb,8), labels (N,) or None)."""
    df = pd.read_csv(path, low_memory=False)
    x = df[CHANNELS].to_numpy(np.float64)
    m = df['Marker_val'].astype(str)
    x = car(x)
    x = notch(x)
    if labeled:
        cue_idx = np.flatnonzero(m.isin(['rest', 'move']).to_numpy())
        labels = np.array([1 if v == 'move' else 0 for v in m.iloc[cue_idx]])
    else:
        cue_idx = np.flatnonzero((m == 'cue_start').to_numpy())
        labels = None
    tcue = int(CUE_LEN * FS)
    tbase = int(BASE_LEN * FS)
    cues, bases = [], []
    for i in cue_idx:
        s, e = i, i + tcue
        if e > len(x):
            continue
        cues.append(x[s:e])
        b0 = max(0, i - tbase)
        bases.append(x[b0:i][-tbase:])
    cues = np.stack(cues)
    bases = np.stack(bases)
    return cues, bases, labels

def load_subject(sub):
    """sub: 'S001'. Returns dict with arrays for each train run and test."""
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, 'Training set', f'{sub}-*_eeg.csv'))):
        run = re.search(r'-0(\d+)_eeg', f).group(1)
        cues, bases, labels = read_epochs(f, labeled=True)
        out[f'tr{run}'] = (cues, bases, labels, run)
    ft = os.path.join(DATA, 'Single subject test set', f'{sub}_test.csv')
    cues, bases, _ = read_epochs(ft, labeled=False)
    out['test'] = (cues, bases, None, 'test')
    return out

if __name__ == '__main__':
    d = load_subject('S001')
    for k, (c, b, y, r) in d.items():
        print(k, 'cues', c.shape, 'bases', b.shape, 'labels', None if y is None else (int(y.sum()), len(y)))
