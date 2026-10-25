"""DL track: per-subject EEGNet on raw 8-channel EEG (competition-rule compliant).
Self-contained for Kaggle notebook (uses /kaggle/input) or local data dir."""
import os, sys, glob, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

def _find_data():
    slug = 'low-cost-motor-imagery-decoding-for-rehab'
    cands = [
        '/kaggle/input/competitions/' + slug,
        '/kaggle/input/' + slug,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
    ]
    if os.environ.get('LCMID_DATA'):
        cands = [os.environ['LCMID_DATA']] + cands
    for p in cands:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, 'Training set')):
            return p
    if os.path.isdir('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'Training set' in dirs:
                return root
    import shutil
    listing = os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'no /kaggle/input'
    raise SystemExit('DATA_NOT_FOUND. /kaggle/input: %r' % listing)

DATA = _find_data()
FS = 250.0
CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'PO7', 'Pz', 'PO8', 'Oz']
CUE_S, BASE_S = 5.0, 3.0
TCUE, TBASE = int(CUE_S * FS), int(BASE_S * FS)
TEST_SUBJECTS = ['S%03d' % i for i in [1,2,3,4,5,6,7,9,10,11,12,14,16,17,18,19,20]]
SEED = 42

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _notch(x):
    from scipy import signal
    b, a = signal.iirnotch(50.0 / (FS / 2.0), 30.0)
    return signal.filtfilt(b, a, x, axis=0)

def read_epochs(path, labeled=True):
    df = pd.read_csv(path, low_memory=False)
    x = df[CHANNELS].to_numpy(np.float64)
    m = df['Marker_val'].astype(str)
    x = x - x.mean(axis=1, keepdims=True)
    x = _notch(x)
    if labeled:
        idx = np.flatnonzero(m.isin(['rest', 'move']).to_numpy())
        y = np.array([1 if v == 'move' else 0 for v in m.iloc[idx]])
    else:
        idx = np.flatnonzero((m == 'cue_start').to_numpy())
        y = None
    cues, bases = [], []
    for i in idx:
        if i + TCUE > len(x):
            continue
        cues.append(x[i:i + TCUE])
        b0 = max(0, i - TBASE)
        bases.append(x[b0:i][-TBASE:])
    return np.stack(cues), np.stack(bases), y

def load_subject(sub):
    runs = {}
    for f in sorted(glob.glob(os.path.join(DATA, 'Training set', f'{sub}-*_eeg.csv'))):
        run = os.path.basename(f).split('-')[1][:3]
        c, b, y = read_epochs(f, labeled=True)
        runs[run] = (c, b, y)
    tc, tb, _ = read_epochs(os.path.join(DATA, 'Single subject test set', f'{sub}_test.csv'), labeled=False)
    return runs, tc, tb

class EEGNet(nn.Module):
    def __init__(self, n_ch=8, T=1250, F1=8, D=2, F2=16, kl=64, p=0.25):
        super().__init__()
        self.c1 = nn.Conv2d(1, F1, (1, kl), padding=(0, kl // 2), bias=False)
        self.b1 = nn.BatchNorm2d(F1)
        self.dw = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.b2 = nn.BatchNorm2d(F1 * D)
        self.d1 = nn.Dropout(p)
        self.sd = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sp = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.b3 = nn.BatchNorm2d(F2)
        self.d2 = nn.Dropout(p)
        self.fc = nn.Linear(F2 * (T // 32), 2)

    def forward(self, x):  # x: (B, 1, C, T)
        x = torch.nn.functional.elu(self.b2(self.dw(self.b1(self.c1(x)))))
        x = self.d1(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 4))
        x = torch.nn.functional.elu(self.b3(self.sp(self.sd(x))))
        x = self.d2(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 8))
        return self.fc(x.flatten(1))

def to_tensors(cues, y=None, augment=False, rng=None):
    X = cues.transpose(0, 2, 1).astype(np.float32)  # (N, C, T)
    X = X - X.mean(axis=(1, 2), keepdims=True)
    X = X / (X.std(axis=(1, 2), keepdims=True) + 1e-6)
    if augment and rng is not None:
        N = len(X)
        extra = []
        for i in range(N):
            for j in range(2):
                xx = X[i].copy()
                if rng.random() < 0.7:
                    shift = int(rng.uniform(-0.10, 0.10) * xx.shape[-1])
                    xx = np.roll(xx, shift, axis=-1)
                if rng.random() < 0.6:
                    t0 = rng.integers(0, xx.shape[-1] - 250)
                    xx[:, t0:t0 + 250] = 0.0
                if rng.random() < 0.3:
                    c = rng.integers(0, xx.shape[0])
                    xx[c] = 0.0
                extra.append(xx[None])
        X = np.concatenate([X] + extra, axis=0)
        if y is not None:
            pass
    t = torch.from_numpy(X)  # (N*3, C, T)
    if y is None:
        return t
    ty = torch.from_numpy(np.asarray(y, dtype=np.int64))
    if augment:
        ty = torch.cat([ty, ty, ty], 0)
    return t, ty

def train_fold(cues, y, val_cues, val_y, device, epochs=150, bs=24, lr=1.5e-3, seed=SEED):
    seed_everything(seed)
    tr, tyr = to_tensors(cues, y, augment=True, rng=np.random.default_rng(seed))
    va, vy = to_tensors(val_cues, val_y)
    lt = TensorDataset(tr, tyr)
    vt = TensorDataset(va, vy)
    dt = DataLoader(lt, bs, shuffle=True)
    dv = DataLoader(vt, bs)
    model = EEGNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()
    best = None
    for ep in range(epochs):
        model.train()
        for xb, yb in dt:
            xb, yb = xb.unsqueeze(1).to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        vp = []
        with torch.no_grad():
            for xb, yb in dv:
                vp.append(torch.softmax(model(xb.unsqueeze(1).to(device)), 1)[:, 1].cpu().numpy())
        vp = np.concatenate(vp)
        acc = ((vp >= 0.5).astype(int) == val_y).mean()
        if best is None or acc > best[0] + 1e-9:
            best = (acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model, float(best[0])

def predict(model, cues, device):
    model.eval()
    t = to_tensors(cues)
    out = []
    with torch.no_grad():
        for i in range(0, len(t), 64):
            out.append(torch.softmax(model(t[i:i + 64].unsqueeze(1).to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from sklearn.model_selection import StratifiedKFold
    results = {}
    for sub in TEST_SUBJECTS:
        runs, tc, tb = load_subject(sub)
        cues = np.concatenate([runs[r][0] for r in sorted(runs)])
        y = np.concatenate([runs[r][2] for r in sorted(runs)])
        t0 = time.time()
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(y))
        for tr_i, va_i in skf.split(cues, y):
            m, acc = train_fold(cues[tr_i], y[tr_i], cues[va_i], y[va_i], device)
            oof[va_i] = predict(m, cues[va_i], device)
            del m
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        cv5 = float(((oof >= 0.5).astype(int) == y).mean())
        # run3 holdout
        pos = 0
        r3 = []
        for r in sorted(runs):
            n = len(runs[r][2])
            if r == '003':
                r3 = list(range(pos, pos + n))
            pos += n
        acc3 = np.nan
        if len(r3) >= 4:
            r3 = np.array(r3)
            m, _ = train_fold(cues[~np.isin(np.arange(len(y)), r3)], y[~np.isin(np.arange(len(y)), r3)], cues[r3], y[r3], device, seed=SEED + 1)
            p = predict(m, cues[r3], device)
            acc3 = float(((p >= 0.5).astype(int) == y[r3]).mean())
            del m
        # final: 3 seeds on all data
        models = []
        for s in (SEED, SEED + 100, SEED + 200, SEED + 300, SEED + 400):
            m, _ = train_fold(cues, y, cues[-10:], y[-10:], device, epochs=200, seed=s)
            models.append(m)
        test_p = np.mean([predict(m, tc, device) for m in models], axis=0)
        results[sub] = {'cv5': cv5, 'run3': None if acc3 != acc3 else acc3, 'test_p': test_p.tolist(),
                        'sec': round(time.time() - t0, 1)}
        print(sub, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[sub].items() if k != 'test_p'}, flush=True)
        del models
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    os.makedirs('outputs', exist_ok=True)
    json.dump(results, open('outputs/dl_results.json', 'w'))
    rows = []
    i = 0
    for sub in TEST_SUBJECTS:
        for q in results[sub]['test_p']:
            rows.append((i, 'move' if q >= 0.5 else 'rest'))
            i += 1
    pd.DataFrame(rows, columns=['ID', 'TARGET']).to_csv('outputs/submission_dl.csv', index=False)
    print('SUBMISSION_WRITTEN', i, flush=True)

if __name__ == '__main__':
    main()
